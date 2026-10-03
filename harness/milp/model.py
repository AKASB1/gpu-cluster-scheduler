"""Time-indexed MILP of the pool and node models, solved with
scipy.optimize.milp (HiGHS). See docs/milp.md section 3."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .brute import greedy
from .instance import Instance


@dataclass
class Result:
    status: str          # optimal, node_limit, time_limit (capped), infeasible, error
    objective: float     # sum of weighted flow of the incumbent (nan if none)
    bound: float         # dual bound
    gap: float
    nodes: int
    capped: bool
    wall_solve_s: float  # wall-clock, non-deterministic
    cpu_solve_s: float   # process CPU time of the solve, non-deterministic
    starts: list         # starts[j] = (slot, machine index) or None
    variables: int
    horizon: int
    whole_slot: bool

    def to_dict(self, inst: Instance) -> dict:
        out = {"status": self.status, "objective": self.objective, "bound": self.bound, "gap": self.gap, "nodes": self.nodes,
               "capped": self.capped, "wall_solve_s": self.wall_solve_s, "cpu_solve_s": self.cpu_solve_s, "variables": self.variables, "horizon": self.horizon,
               "whole_slot": self.whole_slot, "model": inst.model, "slot_s": inst.slot_ms / 1000, "starts": []}
        for j, s in enumerate(self.starts):
            if s is None:
                continue
            t, m = s
            e = {"job_id": inst.jobs[j].job_id, "start_s": t * inst.slot_ms / 1000, "machine": inst.names[m]}
            if inst.model == "node":
                e["placement"] = [{"node": inst.names[m], "workers": 1}]
            out["starts"].append(e)
        return out


def latest_starts(inst: Instance, T: int, ub: float) -> list:
    """Per job and machine, the latest start slot any schedule with objective
    <= ub can use (docs/milp.md section 3.3)."""
    pmin = [min(p for p in row if p is not None) for row in inst.proc]
    # the smallest possible w_j (C_j - submit_j) of each job in the model
    floor_flow = [inst.weight[j] * ((inst.release[j] + pmin[j]) * inst.slot_ms - inst.jobs[j].submit_ms) / 1000
                  for j in range(inst.n)]
    total = sum(floor_flow)
    out = []
    for j in range(inst.n):
        slack = ub - (total - floor_flow[j])  # max of w_j (C_j - submit_j)
        c_max_ms = inst.jobs[j].submit_ms + slack / inst.weight[j] * 1000
        c_max = math.floor(c_max_ms / inst.slot_ms + 1e-7)  # completion slot bound, small tolerance
        row = []
        for p in inst.proc[j]:
            row.append(None if p is None else min(T - p, c_max - p))
        out.append(row)
    return out


@dataclass
class Formulation:
    """min c x  s.t.  A_cap x <= cap,  A_as x = 1,  x binary; column k is (job, slot, machine) cols[k]."""
    cols: list
    c: np.ndarray
    A_cap: sparse.csr_matrix
    cap: np.ndarray
    A_as: sparse.csr_matrix
    horizon: int


def formulate(inst: Instance) -> Formulation:
    T = inst.horizon()
    ub, _ = greedy(inst, T)
    last = latest_starts(inst, T, ub)
    cols = []  # (j, t, m)
    for j in range(inst.n):
        for m, p in enumerate(inst.proc[j]):
            if p is None:
                continue
            for t in range(inst.release[j], last[j][m] + 1):
                cols.append((j, t, m))
    nv = len(cols)
    c = np.array([inst.flow_cost(j, t, m) for j, t, m in cols])
    # capacity rows: (machine, slot[, resource]) -> row index
    nm = len(inst.machines)
    res_kinds = [("g", inst.demand, [m.gpus for m in inst.machines])]
    if inst.use_cpu_mem:
        res_kinds += [("c", inst.cpu, [m.cpus for m in inst.machines]), ("m", inst.mem, [m.mem_mb for m in inst.machines])]
    rows, cidx, vals = [], [], []
    for k, (j, t, m) in enumerate(cols):
        p = inst.proc[j][m]
        for r, (_, dem, _) in enumerate(res_kinds):
            if dem[j] == 0:
                continue
            base = (r * nm + m) * T
            for tau in range(t, t + p):
                rows.append(base + tau)
                cidx.append(k)
                vals.append(dem[j])
    ncap = len(res_kinds) * nm * T
    A_cap = sparse.csr_matrix((vals, (rows, cidx)), shape=(ncap, nv))
    cap = np.concatenate([np.repeat(np.array(caps, dtype=float), T) for _, _, caps in res_kinds])
    A_as = sparse.csr_matrix((np.ones(nv), ([j for j, _, _ in cols], np.arange(nv))), shape=(inst.n, nv))
    return Formulation(cols, c, A_cap, cap, A_as, T)


def starts_of(inst: Instance, cols: list, x) -> tuple:
    """The schedule of a 0/1 solution vector and its sum of weighted flow."""
    starts = [None] * inst.n
    for k in np.flatnonzero(np.asarray(x) > 0.5):
        j, t, m = cols[k]
        starts[j] = (t, m)
    return starts, float(sum(inst.flow_cost(j, s[0], s[1]) for j, s in enumerate(starts)))


def solve(inst: Instance, mip_rel_gap: float = 0.0, node_limit: int | None = None, time_limit: float = 120.0) -> Result:
    f = formulate(inst)
    cols, c, nv, T = f.cols, f.c, len(f.cols), f.horizon
    cons = [LinearConstraint(f.A_cap, -np.inf, f.cap), LinearConstraint(f.A_as, 1, 1)]
    opts = {"disp": False, "mip_rel_gap": mip_rel_gap, "time_limit": time_limit, "presolve": True}
    if node_limit is not None:
        opts["node_limit"] = int(node_limit)
    t0, c0 = time.perf_counter(), time.process_time()
    res = milp(c, constraints=cons, integrality=np.ones(nv), bounds=Bounds(0, 1), options=opts)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    msg = (res.message or "").lower()
    capped = "time limit" in msg
    if res.status == 0:
        status = "optimal"
    elif capped:
        status = "time_limit"
    elif res.status == 1:
        status = "node_limit"
    elif res.status == 2:
        status = "infeasible"
    else:
        status = "error"
    starts, obj = [None] * inst.n, float("nan")
    if res.x is not None:
        starts, obj = starts_of(inst, cols, res.x)
    bound = float(getattr(res, "mip_dual_bound", float("nan")) or float("nan"))
    if status == "optimal" and math.isnan(bound):
        bound = obj
    gap = float(getattr(res, "mip_gap", float("nan")) or 0.0) if res.x is not None else float("nan")
    nodes = int(getattr(res, "mip_node_count", 0) or 0)
    return Result(status, obj, bound, gap, nodes, capped, wall, cpu, starts, nv, T, inst.whole_slot)
