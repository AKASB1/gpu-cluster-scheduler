"""milp_rolling: a rolling-horizon MILP policy (docs/milp.md section 4).

At each invocation it plans the first K pending jobs (FIFO order) over H slots
with a pool model built from the user estimates (never the true run times),
starts the jobs the plan starts in slot 0 through the first-fit placement
routine, and re-plans at the next invocation.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse

from .. import placement as P
from ..milp.worker import MilpWorker
from ..model import View, sec
from .baselines import fifo_order, start_action

TAU = 10.0
DEFAULTS = {"k": 12, "h": 24, "slot_s": 300.0, "mip_rel_gap": 0.01, "node_limit": 2000, "time_limit_s": 10.0}


class MilpRolling:
    name = "milp_rolling"

    def __init__(self, params, seed, cluster):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"milp_rolling: unknown parameters {sorted(unknown)}")
        p = dict(DEFAULTS, **params)
        self.k, self.h = int(p["k"]), int(p["h"])
        self.slot_ms = int(round(float(p["slot_s"]) * 1000))
        self.gap, self.node_limit, self.cap_s = float(p["mip_rel_gap"]), int(p["node_limit"]), float(p["time_limit_s"])
        if self.k < 1 or self.h < 1 or self.slot_ms < 1 or self.node_limit < 1:
            raise ValueError("milp_rolling: k, h, slot_s, node_limit must be positive")
        self.worker = MilpWorker()

    def schedule(self, view: View) -> dict:
        t0, c0 = time.perf_counter(), time.process_time()
        order = fifo_order(view)
        cand = order[: self.k]
        classes = sorted(view.cluster.classes)
        speed = view.cluster.classes
        cap = usable_capacity(view, classes)
        H, D = self.h, self.slot_ms
        busy = {c: [0] * H for c in classes}
        for r in view.running:
            slots = max(1, math.ceil((r.est_end_ms - view.now_ms) / D))
            for n, w in r.placement:
                c = view.nodes[n].gpu_class
                for tau in range(min(slots, H)):
                    busy[c][tau] += r.gpus * w
        cols, cost = [], []
        defer_cost = []
        reps = []  # jobs the model can represent
        for j in cand:
            g = j.gpus * j.workers
            elig = [c for c in classes if (not j.gpu_class or j.gpu_class == c) and g <= cap[c]]
            if not elig:
                continue  # needs a mix of classes: handled by the safeguard below
            work = max(sec(j.estimate_ms) - j.retained, 0.0)
            over = math.ceil(view.cluster.restart_overhead_ms / D) if j.started else 0
            fastest = max(speed[c] for c in classes if not j.gpu_class or j.gpu_class == c)
            w = 1.0 / max(sec(j.estimate_ms) / fastest, TAU)
            ps = {c: max(1, math.ceil(work * 1000 / (speed[c] * D) - 1e-9) + over) for c in elig}
            ji = len(reps)
            reps.append((j, g, ps))
            for c in elig:
                for t in range(H):
                    cols.append((ji, t, c))
                    cost.append(w * (t + ps[c]) * D / 1000)
            defer_cost.append(w * (H + min(ps.values())) * D / 1000)
        starts_now = []
        info = {"status": "skipped", "gap": 0.0, "capped": False}
        if reps:
            nv = len(cols) + len(reps)
            c_vec = np.array(cost + defer_cost)
            rows, cidx, vals = [], [], []
            ci = {c: i for i, c in enumerate(classes)}
            for k, (ji, t, c) in enumerate(cols):
                g, ps = reps[ji][1], reps[ji][2]
                for tau in range(t, min(t + ps[c], H)):
                    rows.append(ci[c] * H + tau)
                    cidx.append(k)
                    vals.append(g)
            A_cap = sparse.csr_matrix((vals, (rows, cidx)), shape=(len(classes) * H, nv))
            ub = np.array([cap[c] - busy[c][tau] for c in classes for tau in range(H)], dtype=float)
            arow = [ji for ji, _, _ in cols] + list(range(len(reps)))
            A_as = sparse.csr_matrix((np.ones(nv), (arow, np.arange(nv))), shape=(len(reps), nv))
            res = self.worker.solve(c_vec, [(A_cap, -np.inf, np.maximum(ub, 0)), (A_as, 1, 1)], np.ones(nv), 0, 1,
                                    {"disp": False, "mip_rel_gap": self.gap, "node_limit": self.node_limit,
                                     "time_limit": self.cap_s, "presolve": True}, self.cap_s * 1.5 + 5)
            info, x = solver_info(res)
            if x is not None:
                planned = {reps[ji][0].job_id for k, (ji, t, c) in enumerate(cols) if t == 0 and x[k] > 0.5}
                starts_now = [j for j in cand if j.job_id in planned]
        free = P.free_of(view.nodes)
        actions = []
        for j in starts_now:  # FIFO order
            r = P.Req.of(j)
            pl = P.first_fit(r, view.nodes, free)
            if pl is not None:
                P.take(free, r, pl)
                actions.append(start_action(j, pl, view))
        # Work-conserving safeguard: if nothing runs and the plan started
        # nothing that places, start the FIFO head greedily (it fits the empty
        # cluster), so the policy can never deadlock.
        if not actions and not view.running and order:
            j = order[0]
            r = P.Req.of(j)
            pl = P.first_fit(r, view.nodes, free)
            if pl is not None:
                actions.append(start_action(j, pl, view))
        info["solve_s"] = time.perf_counter() - t0
        info["cpu_s"] = info.pop("worker_cpu_s", 0.0) + (time.process_time() - c0)
        # an invocation with nothing to model reports no solve (not a non-optimal one)
        return {"actions": actions, "solver": None if info["status"] == "skipped" else info}


def usable_capacity(view: View, classes) -> dict:
    """GPUs per class that are free or held by running jobs: nodes under
    repair (no free capacity, nothing running) and GPUs held during a
    preemption grace period are left out."""
    cap = {c: 0 for c in classes}
    for n in view.nodes:
        cap[n.gpu_class] += n.free_gpus
    for r in view.running:
        for n, w in r.placement:
            cap[view.nodes[n].gpu_class] += r.gpus * w
    return cap


def solver_info(res):
    """Status, gap, cap flag, and the worker's CPU time of a worker answer (None = hard cap)."""
    if res is None:
        return {"status": "wall_cap", "gap": 0.0, "capped": True, "worker_cpu_s": 0.0}, None
    msg = (res["message"] or "").lower()
    capped = "time limit" in msg
    status = "optimal" if res["status"] == 0 else ("time_limit" if capped else ("node_limit" if res["status"] == 1 else "error"))
    return {"status": status, "gap": float(res["mip_gap"] or 0.0), "capped": capped, "worker_cpu_s": res["cpu_s"]}, res["x"]
