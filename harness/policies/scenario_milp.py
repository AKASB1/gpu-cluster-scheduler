"""scenario_milp: a two-stage stochastic reservation with recourse
(docs/milp.md section 6).

At each invocation it takes the first K pending jobs in FIFO order and S
run-time scenarios sampled from the empirical distribution of
runtime / estimate in the completion history (the user's ratios when the user
has at least `min_history` completed jobs, otherwise all users'; ratio 1
without any history). The first stage decides which of the K jobs start now
(the same in every scenario); the second stage chooses, per scenario, the
start slots of the others (recourse). The objective is the expected weighted
flow plus lambda times its CVaR at level alpha (Rockafellar and Uryasev):

    min  (1/S) sum_s L_s + lambda * (eta + 1/((1 - alpha) S) sum_s u_s)
    s.t. u_s >= L_s - eta,  u_s >= 0,
         L_s = sum_j w_j (C_js - now),
         capacity per class, slot, and scenario (running jobs hold their GPUs
         for a scenario-specific remaining time),
         y_{j,0,s} = x_j for every scenario s (non-anticipativity).

Jobs started now go through the first-fit placement routine (skipped if they
do not place). Deterministic: scenarios come from
numpy.random.default_rng(SeedSequence([seed, fnv1a64("scenario_milp")])).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse

from .. import placement as P
from ..model import View, sec
from ..milp.worker import MilpWorker
from .baselines import fifo_order, start_action
from .milp_rolling import solver_info, usable_capacity

TAU = 10.0
DEFAULTS = {"k": 8, "h": 16, "slot_s": 600.0, "scenarios": 8, "lambda": 0.5, "alpha": 0.9, "min_history": 5,
            "mip_rel_gap": 0.01, "node_limit": 2000, "time_limit_s": 20.0}


def fnv1a64(s: str) -> int:
    h = 0xcbf29ce484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


class ScenarioMilp:
    name = "scenario_milp"

    def __init__(self, params, seed, cluster):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"scenario_milp: unknown parameters {sorted(unknown)}")
        p = dict(DEFAULTS, **params)
        self.k, self.h, self.s = int(p["k"]), int(p["h"]), int(p["scenarios"])
        self.slot_ms = int(round(float(p["slot_s"]) * 1000))
        self.lam, self.alpha, self.min_hist = float(p["lambda"]), float(p["alpha"]), int(p["min_history"])
        self.gap, self.node_limit, self.cap_s = float(p["mip_rel_gap"]), int(p["node_limit"]), float(p["time_limit_s"])
        if min(self.k, self.h, self.s, self.slot_ms, self.node_limit) < 1 or not (0 <= self.alpha < 1) or self.lam < 0:
            raise ValueError("scenario_milp: invalid parameters")
        self.worker = MilpWorker()
        self.rng = np.random.default_rng(np.random.SeedSequence([seed & 0xFFFFFFFFFFFFFFFF, fnv1a64(self.name)]))
        self.user_ratios: dict = {}
        self.all_ratios: list = []

    def _learn(self, view: View):
        for h in view.history_new:
            rt, est = float(h["runtime_s"]), float(h["estimate_s"])
            if est > 0:
                r = rt / est
                self.user_ratios.setdefault(h["user"], []).append(r)
                self.all_ratios.append(r)

    def _ratios(self, user: str):
        u = self.user_ratios.get(user, [])
        if len(u) >= self.min_hist:
            return u
        return self.all_ratios

    def _sample(self, user: str, n: int, floor_ratio: float = 0.0):
        """n ratios from the empirical distribution, conditioned on ratio > floor_ratio."""
        pool = self._ratios(user)
        if floor_ratio > 0:
            pool = [r for r in pool if r > floor_ratio]
        if not pool:
            return np.full(n, max(1.0, floor_ratio * 1.05))
        return np.asarray(pool)[self.rng.integers(0, len(pool), size=n)]

    def schedule(self, view: View) -> dict:
        t0, c0 = time.perf_counter(), time.process_time()
        self._learn(view)
        order = fifo_order(view)
        cand = order[: self.k]
        classes = sorted(view.cluster.classes)
        speed = view.cluster.classes
        cap = usable_capacity(view, classes)
        H, D, S = self.h, self.slot_ms, self.s
        # running jobs: scenario-specific remaining slots
        busy = np.zeros((S, len(classes), H))
        ci = {c: i for i, c in enumerate(classes)}
        for r in view.running:
            est = sec(r.estimate_ms)
            done = r.work_done
            ratios = self._sample(r.user, S, floor_ratio=done / est if est > 0 else 0.0)
            for s in range(S):
                remaining_work = max(ratios[s] * est - done, 0.0)
                over_ms = max(0, r.run_start_ms + r.overhead_ms - view.now_ms)
                slots = max(1, math.ceil((over_ms + remaining_work * 1000 / r.rate) / D))
                for n, w in r.placement:
                    busy[s, ci[view.nodes[n].gpu_class], : min(slots, H)] += r.gpus * w
        reps = []
        for j in cand:
            g = j.gpus * j.workers
            elig = [c for c in classes if (not j.gpu_class or j.gpu_class == c) and g <= cap[c]]
            if not elig:
                continue
            est = sec(j.estimate_ms)
            over = math.ceil(view.cluster.restart_overhead_ms / D) if j.started else 0
            fastest = max(speed[c] for c in classes if not j.gpu_class or j.gpu_class == c)
            w = 1.0 / max(est / fastest, TAU)
            # a preempted job keeps its retained work: only ratios that leave work remaining
            ratios = self._sample(j.user, S, floor_ratio=j.retained / est if est > 0 else 0.0)
            ps = {c: [max(1, math.ceil(max(ratios[s] * est - j.retained, 0.0) * 1000 / (speed[c] * D) - 1e-9) + over)
                      for s in range(S)] for c in elig}
            reps.append((j, g, ps, w))
        starts_now, info = [], {"status": "skipped", "gap": 0.0, "capped": False}
        if reps:
            # variables: x_j (n), y_{j,t,c,s}, d_{j,s}, eta, u_s
            nj = len(reps)
            cols_y = [(ji, t, c, s) for ji, (_, _, ps, _) in enumerate(reps) for c in ps for s in range(S) for t in range(H)]
            iy0 = nj
            id0 = iy0 + len(cols_y)
            ieta = id0 + nj * S
            iu0 = ieta + 1
            nv = iu0 + S
            cost = np.zeros(nv)
            # scenario loss L_s = sum w (C - now); expectation weight 1/S plus CVaR through u_s >= L_s - eta
            lrow_cols = [[] for _ in range(S)]
            lrow_vals = [[] for _ in range(S)]
            for k, (ji, t, c, s) in enumerate(cols_y):
                w, p = reps[ji][3], reps[ji][2][c][s]
                v = w * (t + p) * D / 1000
                cost[iy0 + k] += v / S
                lrow_cols[s].append(iy0 + k)
                lrow_vals[s].append(v)
            for ji in range(nj):
                w, ps = reps[ji][3], reps[ji][2]
                for s in range(S):
                    v = w * (H + min(ps[c][s] for c in ps)) * D / 1000
                    cost[id0 + ji * S + s] += v / S
                    lrow_cols[s].append(id0 + ji * S + s)
                    lrow_vals[s].append(v)
            cost[ieta] = self.lam
            cost[iu0:iu0 + S] = self.lam / ((1 - self.alpha) * S)
            rows, cols, vals, lb, ub = [], [], [], [], []
            r_i = 0

            def add_row(cs, vs, lo, hi):
                nonlocal r_i
                rows.extend([r_i] * len(cs))
                cols.extend(cs)
                vals.extend(vs)
                lb.append(lo)
                ub.append(hi)
                r_i += 1

            # assignment per scenario: sum_t,c y + d = 1
            per = {}
            for k, (ji, t, c, s) in enumerate(cols_y):
                per.setdefault((ji, s), []).append(iy0 + k)
            for ji in range(nj):
                for s in range(S):
                    cs = per[(ji, s)] + [id0 + ji * S + s]
                    add_row(cs, [1.0] * len(cs), 1, 1)
            # non-anticipativity: sum_c y_{j,0,c,s} = x_j
            for ji in range(nj):
                for s in range(S):
                    cs = [iy0 + k for k, (jj, t, c, ss) in enumerate(cols_y) if jj == ji and ss == s and t == 0]
                    add_row(cs + [ji], [1.0] * len(cs) + [-1.0], 0, 0)
            # capacity per scenario, class, slot
            capm = {}
            for k, (ji, t, c, s) in enumerate(cols_y):
                g, p = reps[ji][1], reps[ji][2][c][s]
                for tau in range(t, min(t + p, H)):
                    capm.setdefault((s, c, tau), ([], []))
                    capm[(s, c, tau)][0].append(iy0 + k)
                    capm[(s, c, tau)][1].append(g)
            for (s, c, tau), (cs, vs) in sorted(capm.items()):
                add_row(cs, vs, -np.inf, max(cap[c] - busy[s, ci[c], tau], 0.0))
            # CVaR: u_s - L_s + eta >= 0
            for s in range(S):
                add_row(lrow_cols[s] + [ieta, iu0 + s], [-v for v in lrow_vals[s]] + [1.0, 1.0], 0, np.inf)
            A = sparse.csr_matrix((vals, (rows, cols)), shape=(r_i, nv))
            integ = np.ones(nv)
            integ[ieta:] = 0
            lo = np.zeros(nv)
            hi = np.ones(nv)
            lo[ieta], hi[ieta] = -np.inf, np.inf
            hi[iu0:] = np.inf
            res = self.worker.solve(cost, [(A, np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))], integ, lo, hi,
                                    {"disp": False, "mip_rel_gap": self.gap, "node_limit": self.node_limit,
                                     "time_limit": self.cap_s, "presolve": True}, self.cap_s * 1.5 + 5)
            info, x = solver_info(res)
            if x is not None:
                chosen = {reps[ji][0].job_id for ji in range(nj) if x[ji] > 0.5}
                starts_now = [j for j in cand if j.job_id in chosen]
        free = P.free_of(view.nodes)
        actions = []
        for j in starts_now:
            r = P.Req.of(j)
            pl = P.first_fit(r, view.nodes, free)
            if pl is not None:
                P.take(free, r, pl)
                actions.append(start_action(j, pl, view))
        if not actions and not view.running and order:  # work-conserving safeguard (as milp_rolling)
            j = order[0]
            pl = P.first_fit(P.Req.of(j), view.nodes, free)
            if pl is not None:
                actions.append(start_action(j, pl, view))
        info["solve_s"] = time.perf_counter() - t0
        info["cpu_s"] = info.pop("worker_cpu_s", 0.0) + (time.process_time() - c0)
        # an invocation with nothing to model reports no solve (not a non-optimal one)
        return {"actions": actions, "solver": None if info["status"] == "skipped" else info}
