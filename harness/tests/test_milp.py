"""Check 7 (Python side): the MILP optimum equals exhaustive search on random
whole-slot instances of at most 6 jobs, for the node and the pool model; two
solves of one instance in fresh processes agree.

The random instances are in-memory test fixtures built with a seeded numpy
generator (not workload traces; traces are generated only by Go)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from harness.milp.brute import brute_force, greedy
from harness.milp.instance import Job, Node, build
from harness.milp.model import solve

SLOT = 60.0


def random_instance(seed: int, model: str, n_jobs: int):
    rng = np.random.default_rng([seed, 7])
    n_nodes = int(rng.integers(1, 5))
    nodes = []
    for i in range(n_nodes):
        hetero = model == "node" and i > 0 and rng.random() < 0.3
        nodes.append(Node(f"n{i}", "r0", "slow" if hetero else "fast", 0.5 if hetero else 1.0,
                          int(rng.choice([2, 4, 8])), 64, 64000))
    max_g = max(n.gpus for n in nodes)
    jobs = []
    for k in range(n_jobs):
        workers = int(rng.integers(1, 3)) if model == "pool" else 1
        gpus = int(rng.choice([g for g in (1, 2, 4) if g * workers <= sum(n.gpus for n in nodes if n.gpu_class == "fast") and g <= max_g]))
        jobs.append(Job(f"j{k}", int(rng.integers(0, 5)) * 60000, gpus, workers, 0, 0, "",
                        int(rng.integers(1, 6)) * 60000, 0))
    jobs.sort(key=lambda j: j.submit_ms)
    return jobs, nodes


@pytest.mark.parametrize("model", ["node", "pool"])
def test_milp_equals_exhaustive_search(model):
    checked = 0
    for seed in range(30):
        jobs, nodes = random_instance(seed, model, 3 + seed % 4)
        try:
            inst = build(jobs, nodes, model, SLOT)
        except ValueError:
            continue  # a job fits no machine of this random cluster
        assert inst.whole_slot
        best, starts = brute_force(inst)
        res = solve(inst)
        assert res.status == "optimal"
        assert abs(res.objective - best) <= 1e-6 * max(1.0, best), (seed, res.objective, best)
        # the brute-force schedule really has that cost, and greedy is never better
        assert abs(sum(inst.flow_cost(j, s[0], s[1]) for j, s in enumerate(starts)) - best) < 1e-9
        assert greedy(inst)[0] >= best - 1e-9
        checked += 1
    assert checked >= 25


def test_rounded_instances_are_flagged():
    jobs = [Job("a", 0, 1, 1, 0, 0, "", 90000, 0)]  # 1.5 slots of 60 s
    inst = build(jobs, [Node("n0", "r", "fast", 1.0, 8, 8, 8000)], "node", SLOT)
    assert not inst.whole_slot and inst.proc[0][0] == 2


def test_rounded_release_never_precedes_the_submission():
    # submitted at 90 s on a 60 s grid: the earliest start is slot 2 (120 s), so
    # the rounded optimum is an approximation from above (true minimum flow 60 s)
    jobs = [Job("a", 90000, 1, 1, 0, 0, "", 60000, 0)]
    inst = build(jobs, [Node("n0", "r", "fast", 1.0, 8, 8, 8000)], "node", SLOT)
    r = solve(inst)
    assert inst.release[0] == 2 and r.starts[0][0] == 2
    assert r.objective >= inst.weight[0] * 60 - 1e-9


SCRIPT = """
import json, sys
sys.path.insert(0, %r)
from harness.tests.test_milp import random_instance
from harness.milp.instance import build
from harness.milp.model import solve
jobs, nodes = random_instance(11, "node", 6)
inst = build(jobs, nodes, "node", 60.0)
r = solve(inst)
print(json.dumps({"status": r.status, "objective": r.objective, "starts": r.starts}))
"""


def test_two_solves_in_fresh_processes_agree():
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outs = [subprocess.run([sys.executable, "-c", SCRIPT % root], capture_output=True, text=True, timeout=120, check=True).stdout
            for _ in range(2)]
    a, b = json.loads(outs[0]), json.loads(outs[1])
    assert a == b and a["status"] == "optimal"


def test_hard_wall_cap_terminates_a_runaway_solve(tmp_path):
    """A solve that overruns its cap (here: a test hook that sleeps) is terminated and reported as wall_cap."""
    import os
    import time
    from harness.milp.solve import run_capped
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    trace = tmp_path / "t.csv"
    trace.write_text("job_id,submit_s,tenant,user,priority,gpus,workers,gpu_class,topology,cpus,mem_gb,runtime_s,estimate_s,"
                     "preemptible,checkpoint_interval_s,max_wait_s\nj0,0,t,u,1,1,1,,any,0,0,300,300,0,0,\n", encoding="utf-8")
    task = {"id": "x", "trace": str(trace), "cluster": os.path.join(root, "configs", "clusters", "s8-small.json"), "model": "node",
            "slot_s": 300, "time_limit_s": 0, "_test_sleep_s": 120}
    t0 = time.perf_counter()
    r = run_capped(task)
    assert r["status"] == "wall_cap" and r["capped"] and r["objective"] is None
    assert time.perf_counter() - t0 < 40
    task.pop("_test_sleep_s")
    task["time_limit_s"] = 30
    assert run_capped(task)["status"] == "optimal"
