"""A MILP worker process with a hard wall-clock cap.

The HiGHS build vendored by SciPy 1.13 does not always stop at its own
time limit (docs/milp.md section 3.4). Online policies therefore solve in a
long-lived worker process; a solve that is not answered within the hard cap
is abandoned: the worker is terminated and replaced, and the caller gets
None (reported as status "wall_cap", capped).
"""
from __future__ import annotations

import multiprocessing as mp
import time


def _serve(conn):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return  # the parent is gone
        if job is None:
            return
        c, cons, integrality, lb, ub, options = job
        c0 = time.process_time()
        res = milp(np.asarray(c), constraints=[LinearConstraint(A, lo, hi) for A, lo, hi in cons], integrality=integrality,
                   bounds=Bounds(lb, ub), options=options)
        out = {"status": int(res.status), "message": res.message or "", "x": None if res.x is None else res.x,
               "mip_gap": getattr(res, "mip_gap", None), "cpu_s": time.process_time() - c0}
        conn.send(out)


class MilpWorker:
    """Solves scipy.optimize.milp problems in a child process with a hard cap."""

    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._proc = None

    def _start(self):
        self._conn, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_serve, args=(child,), daemon=True)
        self._proc.start()
        child.close()

    def solve(self, c, cons, integrality, lb, ub, options, hard_cap_s: float):
        """cons: list of (sparse matrix, lower, upper). Returns a dict or None (cap)."""
        if self._proc is None or not self._proc.is_alive():
            self._start()
        self._conn.send((c, cons, integrality, lb, ub, options))
        if self._conn.poll(hard_cap_s):
            return self._conn.recv()
        self._proc.terminate()
        self._proc.join(30)
        self._proc = None
        return None

    def close(self):
        if self._proc is not None and self._proc.is_alive():
            try:
                self._conn.send(None)
                self._proc.join(10)
            finally:
                if self._proc.is_alive():
                    self._proc.terminate()
