"""Serial schedule generation (SSGS), the greedy upper bound, and the
exhaustive search of check 7 (docs/milp.md section 3.5)."""
from __future__ import annotations

from .instance import Instance


class Usage:
    """Per-machine usage of GPUs (and CPUs, memory when the model uses them) per slot."""

    def __init__(self, inst: Instance, horizon: int):
        self.inst = inst
        self.T = horizon
        k = len(inst.machines)
        self.g = [[0] * horizon for _ in range(k)]
        self.c = [[0] * horizon for _ in range(k)] if inst.use_cpu_mem else None
        self.m = [[0] * horizon for _ in range(k)] if inst.use_cpu_mem else None

    def fits(self, j: int, m: int, t: int) -> bool:
        inst, p = self.inst, self.inst.proc[j][m]
        if t + p > self.T:
            return False
        mach = inst.machines[m]
        g = self.g[m]
        cap, d = mach.gpus, inst.demand[j]
        for tau in range(t, t + p):
            if g[tau] + d > cap:
                return False
        if self.c is not None:
            for tau in range(t, t + p):
                if self.c[m][tau] + inst.cpu[j] > mach.cpus or self.m[m][tau] + inst.mem[j] > mach.mem_mb:
                    return False
        return True

    def earliest(self, j: int, m: int):
        for t in range(self.inst.release[j], self.T - self.inst.proc[j][m] + 1):
            if self.fits(j, m, t):
                return t
        return None

    def add(self, j: int, m: int, t: int, sign: int = 1):
        inst = self.inst
        for tau in range(t, t + inst.proc[j][m]):
            self.g[m][tau] += sign * inst.demand[j]
            if self.c is not None:
                self.c[m][tau] += sign * inst.cpu[j]
                self.m[m][tau] += sign * inst.mem[j]


def greedy(inst: Instance, horizon: int | None = None):
    """SSGS in order of w_j / min p_j (descending), each job at its earliest
    start over all machines (ties: lower machine index). Returns (cost, starts)
    with starts[j] = (slot, machine)."""
    T = horizon or inst.horizon()
    order = sorted(range(inst.n), key=lambda j: (-inst.weight[j] / min(p for p in inst.proc[j] if p is not None), j))
    u = Usage(inst, T)
    starts, cost = [None] * inst.n, 0.0
    for j in order:
        best = None
        for m, p in enumerate(inst.proc[j]):
            if p is None:
                continue
            t = u.earliest(j, m)
            if t is not None and (best is None or t + p < best[0] + inst.proc[j][best[1]]):
                best = (t, m)
        if best is None:
            raise RuntimeError("greedy: horizon too short")
        u.add(j, best[1], best[0])
        starts[j] = best
        cost += inst.flow_cost(j, best[0], best[1])
    return cost, starts


def brute_force(inst: Instance, max_jobs: int = 6):
    """Exhaustive search over job orders and machine choices with SSGS,
    branch-and-bound pruning, and symmetry breaking between identical empty
    machines. Returns (optimal cost, starts)."""
    if inst.n > max_jobs:
        raise ValueError(f"brute force is limited to {max_jobs} jobs")
    T = inst.horizon()
    best_cost, best_starts = greedy(inst, T)
    best = [best_cost, list(best_starts)]
    lb = [inst.weight[j] * ((inst.release[j] + min(p for p in inst.proc[j] if p is not None)) * inst.slot_ms
                            - inst.jobs[j].submit_ms) / 1000 for j in range(inst.n)]
    u = Usage(inst, T)
    starts = [None] * inst.n
    used = [0] * len(inst.machines)
    key = [(m.gpu_class, m.speed, m.gpus, m.cpus, m.mem_mb) for m in inst.machines]

    def dfs(remaining: list, cost: float, rest_lb: float):
        if not remaining:
            if cost < best[0] - 1e-9:
                best[0], best[1] = cost, list(starts)
            return
        for k, j in enumerate(remaining):
            others = remaining[:k] + remaining[k + 1:]
            seen_empty = set()
            for m, p in enumerate(inst.proc[j]):
                if p is None:
                    continue
                if used[m] == 0:
                    if key[m] in seen_empty:
                        continue  # an identical empty machine was already tried
                    seen_empty.add(key[m])
                t = u.earliest(j, m)
                if t is None:
                    continue
                c = cost + inst.flow_cost(j, t, m)
                rl = rest_lb - lb[j]
                if c + rl >= best[0] - 1e-9:
                    continue
                u.add(j, m, t)
                used[m] += 1
                starts[j] = (t, m)
                dfs(others, c, rl)
                starts[j] = None
                used[m] -= 1
                u.add(j, m, t, -1)

    dfs(list(range(inst.n)), 0.0, sum(lb))
    return best[0], best[1]
