"""Python implementations of two Go baselines, for the conformance check
(check 8): fifo+first_fit+none and fifo+first_fit+easy. They follow
internal/policy/composite.go rule by rule.
"""
from __future__ import annotations

from .. import placement as P
from ..model import View, estimated_duration_ms, sec


def fifo_order(view: View):
    return sorted(view.pending, key=lambda p: (p.submit_ms, p.job_id))


def start_action(p, pl, view: View):
    return {"op": "start", "job_id": p.job_id,
            "placement": [{"node": view.nodes[n].name, "workers": w} for n, w in pl]}


class FifoFirstFit:
    """FIFO order, first-fit placement, no backfilling (stop at the first job that does not fit)."""

    name = "fifo+first_fit+none"

    def __init__(self, params, seed, cluster):
        if params:
            raise ValueError(f"{self.name} takes no parameters")

    def schedule(self, view: View) -> dict:
        free = P.free_of(view.nodes)
        actions = []
        for p in fifo_order(view):
            r = P.Req.of(p)
            pl = P.first_fit(r, view.nodes, free)
            if pl is None:
                break
            P.take(free, r, pl)
            actions.append(start_action(p, pl, view))
        return {"actions": actions}


class FifoFirstFitEasy:
    """FIFO order, first-fit placement, EASY backfilling (see docs/contracts.md section 7)."""

    name = "fifo+first_fit+easy"

    def __init__(self, params, seed, cluster):
        if params:
            raise ValueError(f"{self.name} takes no parameters")

    def schedule(self, view: View) -> dict:
        free = P.free_of(view.nodes)
        actions, reservations = [], []
        # releases: (end ms, job id, request, placement)
        rel = [(r.est_end_ms, r.job_id, P.Req(r.gpus, r.cpus, r.mem_mb, r.workers), r.placement) for r in view.running]
        head_found = False
        shadow = 0
        extra = None
        for p in fifo_order(view):
            r = P.Req.of(p)
            if not head_found:
                pl = P.first_fit(r, view.nodes, free)
                if pl is not None:
                    P.take(free, r, pl)
                    actions.append(start_action(p, pl, view))
                    dur = estimated_duration_ms(p, pl, view, sec(p.estimate_ms) - p.retained)
                    rel.append((view.now_ms + dur, p.job_id, r, pl))
                    continue
                head_found = True
                res = self._shadow(r, view, free, rel)
                if res is None:
                    break  # no reservation possible: no backfilling
                shadow, extra = res
                reservations.append({"job_id": p.job_id, "shadow_ms": shadow})
                continue
            pl = P.first_fit(r, view.nodes, free)
            if pl is None:
                continue
            if view.now_ms + estimated_duration_ms(p, pl, view, sec(p.estimate_ms) - p.retained) <= shadow:
                P.take(free, r, pl)
                actions.append(start_action(p, pl, view))
                continue
            cap_x = [[min(a, b) for a, b in zip(free[n], extra[n])] for n in range(len(free))]
            pl2 = P.first_fit(r, view.nodes, cap_x)
            if pl2 is not None:
                P.take(free, r, pl2)
                P.take(extra, r, pl2)
                actions.append(start_action(p, pl2, view))
        return {"actions": actions, "reservations": reservations}

    @staticmethod
    def _shadow(r, view, free, rel):
        rel = sorted(rel, key=lambda e: (e[0], e[1]))
        cap = [list(c) for c in free]
        k = 0
        while k < len(rel):
            P.give(cap, rel[k][2], rel[k][3])
            pl = P.first_fit(r, view.nodes, cap)
            if pl is None:
                k += 1
                continue
            while k + 1 < len(rel) and rel[k + 1][0] == rel[k][0]:
                k += 1
                P.give(cap, rel[k][2], rel[k][3])
            P.take(cap, r, pl)
            return rel[k][0], cap
        return None
