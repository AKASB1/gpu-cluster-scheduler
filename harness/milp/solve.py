"""Offline MILP solver CLI used by the benchmark and the replay test.

    python -m harness.milp.solve --tasks tasks.json --out results.jsonl

tasks.json is a list of {"id", "trace", "cluster", "model", "slot_s",
"mip_rel_gap", "node_limit", "time_limit_s"}. One JSON line per task is
written to --out (the schedule, status, objective, bound, gap, and the
wall-clock solve time). A bound is reported only for whole-slot instances.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import scipy

from .instance import build, load_cluster, load_trace
from .model import solve


def highs_version() -> str:
    """The HiGHS banner version of SciPy's vendored solver (as printed by HiGHS)."""
    import os
    import subprocess
    code = ("import numpy as np; from scipy.optimize import milp, Bounds; "
            "milp(np.array([1.0]), integrality=np.array([1]), bounds=Bounds([0],[1]), options={'disp': True})")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                             env=dict(os.environ, PYTHONUTF8="1")).stdout
        for line in out.splitlines():
            if line.startswith("Running HiGHS"):
                return line.split("[")[0].replace("Running HiGHS", "").strip()
    except Exception:
        pass
    return "unknown"


def clean(x):
    return None if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x


def run_task(t: dict) -> dict:
    jobs = load_trace(t["trace"])
    nodes = load_cluster(t["cluster"])
    inst = build(jobs, nodes, t["model"], float(t["slot_s"]))
    r = solve(inst, mip_rel_gap=float(t.get("mip_rel_gap", 0.0)), node_limit=t.get("node_limit"),
              time_limit=float(t.get("time_limit_s", 120.0)))
    d = r.to_dict(inst)
    if not inst.whole_slot:
        d["bound"] = None  # never call a rounded optimum a bound (docs/milp.md section 2)
    d["id"] = t.get("id")
    d["jobs"] = len(jobs)
    return {k: clean(v) for k, v in d.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", help="JSON list of tasks")
    ap.add_argument("--out", help="JSON-lines output")
    ap.add_argument("--versions", action="store_true", help="print SciPy and HiGHS versions as JSON and exit")
    a = ap.parse_args(argv)
    if a.versions:
        print(json.dumps({"python": sys.version.split()[0], "scipy": scipy.__version__, "highs": highs_version()}))
        return 0
    with open(a.tasks, encoding="utf-8") as f:
        tasks = json.load(f)
    runner = CappedRunner()
    try:
        with open(a.out, "w", encoding="utf-8", newline="\n") as out:
            for t in tasks:
                out.write(json.dumps(runner.run(t), sort_keys=True) + "\n")
                out.flush()
    finally:
        runner.close()
    return 0


def _serve(conn):
    """Worker loop: solve tasks sent over conn until None arrives."""
    while True:
        t = conn.recv()
        if t is None:
            return
        try:
            if t.get("_test_sleep_s"):  # test hook: a solve that ignores its time limit
                import time as _time
                _time.sleep(float(t["_test_sleep_s"]))
            conn.send(("ok", run_task(t)))
        except Exception as e:  # reported by the parent
            conn.send(("err", f"{type(e).__name__}: {e}"))


class CappedRunner:
    """Runs tasks in one long-lived worker process with a hard wall-clock cap.

    HiGHS (as vendored by SciPy 1.13) does not always honour its own time
    limit promptly. A task that is not answered within 1.5 x time_limit_s +
    10 s is reported as status "wall_cap" (capped, no solution, no bound); the
    worker is terminated and replaced.
    """

    def __init__(self):
        import multiprocessing as mp
        self._ctx = mp.get_context("spawn")
        self._start()

    def _start(self):
        self._conn, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_serve, args=(child,), daemon=True)
        self._proc.start()
        child.close()

    def run(self, t: dict) -> dict:
        import time as _time
        cap = float(t.get("time_limit_s", 120.0))
        t0 = _time.perf_counter()
        self._conn.send(t)
        if self._conn.poll(cap * 1.5 + 10):
            kind, val = self._conn.recv()
            if kind == "err":
                raise RuntimeError(f"task {t.get('id')}: {val}")
            return val
        self._proc.terminate()
        self._proc.join(30)
        self._start()
        jobs = load_trace(t["trace"])
        inst = build(jobs, load_cluster(t["cluster"]), t["model"], float(t["slot_s"]))
        return {"id": t.get("id"), "status": "wall_cap", "capped": True, "objective": None, "bound": None, "gap": None, "nodes": 0,
                "variables": 0, "horizon": inst.horizon(), "whole_slot": inst.whole_slot, "model": t["model"],
                "slot_s": float(t["slot_s"]), "jobs": len(jobs), "starts": [], "wall_solve_s": _time.perf_counter() - t0, "cpu_solve_s": None}

    def close(self):
        try:
            self._conn.send(None)
            self._proc.join(30)
        finally:
            if self._proc.is_alive():
                self._proc.terminate()


def run_capped(t: dict) -> dict:
    """One task with the hard cap (a fresh worker; used by tests)."""
    r = CappedRunner()
    try:
        return r.run(t)
    finally:
        r.close()


if __name__ == "__main__":
    sys.exit(main())
