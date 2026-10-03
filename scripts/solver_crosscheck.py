"""Optional solver cross-check of the S8 MILP instances (Tier 2, item 8).

    python scripts/solver_crosscheck.py --tasks benchmarks/outputs/benchmark/milp \
        --out benchmarks/results/solver_crosscheck

Solves every task of the benchmark's S8 MILP experiments (small, medium, the
solve-time sweep; the milp_tasks.json files written by `go run
./cmd/benchmark`) with the same formulation (harness.milp.model.formulate)
and up to three solvers, one instance at a time, in a worker process with
the hard cap of the solver CLI (1.5 x time limit + 10 s):

  - scipy: scipy.optimize.milp, the HiGHS build vendored by SciPy (the
    default solver of every committed result);
  - highspy: a current HiGHS through its own Python package;
  - gurobi: gurobipy, when it imports and its license works.

It is NOT part of the default path, CI, or any number the README relies on.
Run it in a separate throwaway environment that has the extra packages, never
in the environment the committed results come from. Every solver gets the
task's time limit, a relative gap of 0, and one thread. Times are wall-clock
and process CPU on a shared machine: indicative only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
from importlib import metadata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from harness.milp.instance import build, load_cluster, load_trace  # noqa: E402
from harness.milp.model import formulate, starts_of  # noqa: E402

SETS = ["s8-small", "s8-medium", "sweep"]


def solve_scipy(f, tl):
    from scipy.optimize import Bounds, LinearConstraint, milp
    res = milp(f.c, constraints=[LinearConstraint(f.A_cap, -np.inf, f.cap), LinearConstraint(f.A_as, 1, 1)],
               integrality=np.ones(len(f.cols)), bounds=Bounds(0, 1),
               options={"disp": False, "mip_rel_gap": 0.0, "time_limit": tl, "presolve": True})
    msg = (res.message or "").lower()
    status = "optimal" if res.status == 0 else ("time_limit" if "time limit" in msg else f"other:{res.status}")
    return status, res.x, getattr(res, "mip_dual_bound", None)


def solve_highspy(f, tl):
    import highspy
    from scipy import sparse
    A = sparse.vstack([f.A_cap, f.A_as]).tocsc()
    nv, ncap, nas = len(f.cols), f.A_cap.shape[0], f.A_as.shape[0]
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = nv, A.shape[0]
    lp.col_cost_ = np.asarray(f.c, dtype=float)
    lp.col_lower_, lp.col_upper_ = np.zeros(nv), np.ones(nv)
    lp.row_lower_ = np.concatenate([np.full(ncap, -inf), np.ones(nas)])
    lp.row_upper_ = np.concatenate([np.asarray(f.cap, dtype=float), np.ones(nas)])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger] * nv
    h = highspy.Highs()
    for k, v in (("output_flag", False), ("mip_rel_gap", 0.0), ("time_limit", float(tl)), ("threads", 1)):
        h.setOptionValue(k, v)
    h.passModel(lp)
    h.run()
    ms = h.getModelStatus()
    status = ("optimal" if ms == highspy.HighsModelStatus.kOptimal else
              "time_limit" if ms == highspy.HighsModelStatus.kTimeLimit else f"other:{h.modelStatusToString(ms)}")
    sol = h.getSolution()
    x = np.array(sol.col_value) if sol.value_valid else None
    return status, x, h.getInfo().mip_dual_bound


def solve_gurobi(f, tl):
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        m = gp.Model(env=env)
        m.Params.TimeLimit, m.Params.MIPGap, m.Params.Threads = float(tl), 0.0, 1
        x = m.addMVar(len(f.cols), vtype=gp.GRB.BINARY)
        m.setObjective(np.asarray(f.c, dtype=float) @ x)
        m.addConstr(f.A_cap @ x <= np.asarray(f.cap, dtype=float))
        m.addConstr(f.A_as @ x == np.ones(f.A_as.shape[0]))
        m.optimize()
        status = ("optimal" if m.Status == gp.GRB.OPTIMAL else "time_limit" if m.Status == gp.GRB.TIME_LIMIT
                  else f"other:{m.Status}")
        return status, (x.X if m.SolCount > 0 else None), m.ObjBound
    finally:
        env.dispose()


SOLVERS = {"scipy": solve_scipy, "highspy": solve_highspy, "gurobi": solve_gurobi}


def versions():
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    import scipy
    from harness.milp.solve import highs_version
    out["scipy"] = f"scipy {scipy.__version__} (vendored HiGHS {highs_version()})"
    try:
        out["highspy"] = f"highspy {metadata.version('highspy')}"
    except metadata.PackageNotFoundError:
        out["highspy"] = None
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        env.dispose()
        out["gurobi"] = "gurobipy %d.%d.%d" % gp.gurobi.version()
    except Exception as e:  # not installed, or no working license
        out["gurobi"] = None
        out["gurobi_unavailable"] = f"{type(e).__name__}"
    return out


def _serve(conn):
    while True:
        item = conn.recv()
        if item is None:
            return
        solver, t = item
        try:
            inst = build(load_trace(t["trace"]), load_cluster(t["cluster"]), t["model"], float(t["slot_s"]))
            f = formulate(inst)
            t0, c0 = time.perf_counter(), time.process_time()
            status, x, bound = SOLVERS[solver](f, float(t.get("time_limit_s", 120.0)))
            wall, cpu = time.perf_counter() - t0, time.process_time() - c0
            obj = starts_of(inst, f.cols, x)[1] if x is not None else None
            conn.send(("ok", {"status": status, "objective": obj, "bound": None if bound is None else float(bound),
                              "wall_s": wall, "cpu_s": cpu, "variables": len(f.cols), "jobs": inst.n}))
        except Exception as e:
            conn.send(("err", f"{type(e).__name__}: {e}"))


class Worker:
    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.start()

    def start(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_serve, args=(child,), daemon=True)
        self.proc.start()
        child.close()

    def run(self, solver, t):
        cap = float(t.get("time_limit_s", 120.0)) * 1.5 + 10
        t0 = time.perf_counter()
        self.conn.send((solver, t))
        if self.conn.poll(cap):
            kind, val = self.conn.recv()
            return val if kind == "ok" else {"status": "error", "error": val}
        self.proc.terminate()
        self.proc.join(30)
        self.start()
        return {"status": "wall_cap", "objective": None, "bound": None, "wall_s": time.perf_counter() - t0, "cpu_s": None}

    def close(self):
        try:
            self.conn.send(None)
            self.proc.join(30)
        finally:
            if self.proc.is_alive():
                self.proc.terminate()


def summarize(rows, vers, solvers):
    lines = ["<!-- generated by scripts/solver_crosscheck.py; simulated instances, wall-clock and CPU on a shared machine -->", "",
             "| solver | version | needs |", "|---|---|---|"]
    needs = {"scipy": "nothing extra (default solver of every committed result)",
             "highspy": "`pip install highspy` in a throwaway environment", "gurobi": "gurobipy and a Gurobi license"}
    for s in solvers:
        lines.append(f"| {s} | {vers.get(s) or 'not available'} | {needs[s]} |")
    lines += ["", "Every solver: same formulation, the task's time limit, relative gap 0, one thread requested (the vendored "
              "HiGHS of SciPy 1.13 takes no thread option). Times include passing the model to the solver; CPU is the "
              "process CPU time of the solving process.", ""]
    by, njobs = {}, {}
    for r in rows:
        by[(r["set"], r["id"], r["solver"])] = r
        if r.get("jobs") not in ("", None):
            njobs[(r["set"], r["id"])] = int(r["jobs"])
    lines += ["| set | solver | instances | optimal | objective = scipy (both optimal) | max rel. diff | median wall (s) | "
              "median CPU (s) | median wall ratio to scipy |", "|---|---|---|---|---|---|---|---|---|"]
    for st in SETS:
        ids = sorted({r["id"] for r in rows if r["set"] == st}, key=lambda x: (len(x), x))
        if not ids:
            continue
        for s in solvers:
            rs = [by[(st, i, s)] for i in ids if (st, i, s) in by]
            opt = [r for r in rs if r["status"] == "optimal"]
            agree, diffs, ratios = 0, [], []
            for r in opt:
                ref = by.get((st, r["id"], "scipy"))
                if ref is None or ref["status"] != "optimal":
                    continue
                d = abs(float(r["objective"]) - float(ref["objective"])) / max(1e-9, abs(float(ref["objective"])))
                diffs.append(d)
                agree += d <= 1e-6
                if float(ref["wall_s"]) > 0:
                    ratios.append(float(r["wall_s"]) / float(ref["wall_s"]))
            walls = [float(r["wall_s"]) for r in rs if r["wall_s"] not in ("", None)]
            cpus = [float(r["cpu_s"]) for r in rs if r["cpu_s"] not in ("", None)]
            med = lambda v, d=3: f"{statistics.median(v):.{d}f}" if v else "-"  # noqa: E731
            lines.append(f"| {st} | {s} | {len(rs)} | {len(opt)} | {agree}/{len(diffs)} | {max(diffs):.1e} | " if diffs else
                         f"| {st} | {s} | {len(rs)} | {len(opt)} | 0/0 | - | ")
            lines[-1] += f"{med(walls)} | {med(cpus)} | {med(ratios, 2)} |"
    # tractability on the sweep: per slot length, the largest job count at which every instance is optimal
    sw = [r for r in rows if r["set"] == "sweep"]
    if sw:
        slots = sorted({float(r["slot_s"]) for r in sw}, reverse=True)
        lines += ["", "Solve-time sweep: the largest job count at which every instance was solved to optimality within the "
                  "time limit (more jobs or a finer grid did not all finish).", "",
                  "| solver | " + " | ".join(f"slot {s:g} s" for s in slots) + " |", "|---" * (len(slots) + 1) + "|"]
        for s in solvers:
            cells = []
            for sl in slots:
                ok = {}
                for r in sw:
                    if r["solver"] == s and float(r["slot_s"]) == sl:
                        n = njobs[(r["set"], r["id"])]
                        ok[n] = ok.get(n, True) and r["status"] == "optimal"
                good = [n for n in sorted(ok) if all(ok[m] for m in ok if m <= n)]
                cells.append(str(good[-1]) if good else "none")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", default="benchmarks/outputs/benchmark/milp", help="folder with <set>/milp_tasks.json")
    ap.add_argument("--out", default="benchmarks/results/solver_crosscheck")
    ap.add_argument("--solvers", default="scipy,highspy,gurobi")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--summary-only", action="store_true", help="rewrite summary.md from crosscheck.csv and manifest.json in --out")
    a = ap.parse_args(argv)
    if a.summary_only:
        with open(os.path.join(a.out, "crosscheck.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(os.path.join(a.out, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(summarize(rows, man["versions"], man["solvers"]))
        print("wrote", os.path.join(a.out, "summary.md"))
        return 0
    vers = versions()
    solvers = [s for s in a.solvers.split(",") if s == "scipy" or vers.get(s)]
    print(json.dumps(vers), "solvers:", solvers, flush=True)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    w = Worker()
    try:
        for st in a.sets.split(","):
            path = os.path.join(a.tasks, st, "milp_tasks.json")
            if not os.path.exists(path):
                print(f"{path} missing: run `go run ./cmd/benchmark` first", flush=True)
                continue
            with open(path, encoding="utf-8") as f:
                tasks = json.load(f)
            for t in tasks:
                n_jobs = len(load_trace(t["trace"]))
                for s in solvers:  # interleaved, so machine load affects every solver alike
                    r = w.run(s, t)
                    rows.append({"set": st, "id": str(t["id"]), "model": t["model"], "slot_s": t["slot_s"],
                                 "jobs": n_jobs, "solver": s, "status": r["status"],
                                 "objective": "" if r.get("objective") is None else repr(float(r["objective"])),
                                 "bound": "" if r.get("bound") is None or not math.isfinite(r["bound"]) else repr(float(r["bound"])),
                                 "wall_s": f"{r['wall_s']:.4f}", "cpu_s": "" if r.get("cpu_s") is None else f"{r['cpu_s']:.4f}",
                                 "time_limit_s": t.get("time_limit_s", 120.0)})
                    print(st, t["id"], s, r["status"], rows[-1]["objective"], rows[-1]["wall_s"], flush=True)
    finally:
        w.close()
    cols = ["set", "id", "model", "slot_s", "jobs", "solver", "status", "objective", "bound", "wall_s", "cpu_s", "time_limit_s"]
    with open(os.path.join(a.out, "crosscheck.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"versions": vers, "solvers": solvers, "threads": 1, "mip_rel_gap": 0, "platform": platform.platform(),
                   "processor": platform.processor(), "note": "shared workstation; times indicative only; not used by the README"},
                  f, indent=1, sort_keys=True)
        f.write("\n")
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(summarize(rows, vers, solvers))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
