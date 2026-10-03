"""Tests of the view model, the Python baselines, and the protocol server."""
import io
import json

import pytest

from harness import placement as P
from harness.model import (Cluster, fmt_s, ms, parse_view, rate_of, span_of, topology_factor,
                           work_to_duration_ms)
from harness.policies import make_policy
from harness.policy_server import serve

CLUSTER = {
    "classes": [{"name": "a", "speed": 1.0}], "racks": ["r"], "cross_node_factor": 1.1, "cross_rack_factor": 1.25,
    "restart_overhead_s": 60, "preempt_grace_s": 0,
    "nodes": [{"name": f"n{i}", "rack": "r", "class": "a", "speed": 1.0, "gpus": 8, "cpus": 1000, "mem_mb": 10**9} for i in range(3)],
}


def pending(job_id, submit, gpus, workers, est):
    return {"job_id": job_id, "tenant": "t", "user": "u", "submit_s": submit, "priority": 1, "gpus": gpus, "workers": workers,
            "cpus": 0, "mem_mb": 0, "gpu_class": "", "topology": "any", "estimate_s": est, "wait_s": 0, "retained_s": 0,
            "max_wait_s": None, "preemptible": False, "checkpoint_interval_s": 0, "preemptions": 0, "started": False}


def running(job_id, gpus, node, end):
    return {"job_id": job_id, "tenant": "t", "user": "u", "priority": 1, "gpus": gpus, "workers": 1, "cpus": 0, "mem_mb": 0,
            "gpu_class": "", "topology": "any", "submit_s": 0, "first_start_s": 0, "run_start_s": 0, "overhead_s": 0,
            "placement": [{"node": node, "workers": 1}], "rate": 1.0, "estimate_s": end, "retained_at_start_s": 0,
            "work_done_s": 0, "est_remaining_work_s": end, "est_end_s": end, "preemptible": False,
            "checkpoint_interval_s": 0, "preemptions": 0}


def view_dict(free, run, pend, now=0):
    return {"now_s": now, "nodes": [{"name": f"n{i}", "free_gpus": g, "free_cpus": 1000, "free_mem_mb": 10**9, "running": []}
                                    for i, g in enumerate(free)],
            "running": run, "pending": pend, "tenants": [], "history_new": []}


def test_time_and_rate_rules():
    assert ms(12.345) == 12345 and ms(0.001) == 1 and ms(86400) == 86400000
    assert fmt_s(12345) == 12.345 and fmt_s(12000) == 12 and fmt_s(1) == 0.001
    # same rounding as Go: 1000 s of work at rate 1/1.15 is exactly 1150 s
    assert work_to_duration_ms(1000, 1 / 1.15) == 1150000
    assert work_to_duration_ms(1.0005, 1) == 1001
    assert work_to_duration_ms(0, 0.5) == 0
    assert topology_factor("rack", "cluster", 1.1, 1.25) == 1.25
    assert topology_factor("node", "rack", 1.1, 1.25) == 1.1
    assert topology_factor("rack", "rack", 1.1, 1.25) == 1.0
    assert topology_factor("any", "cluster", 1.1, 1.25) == 1.0


def test_span_and_rate():
    c = Cluster.parse(dict(CLUSTER, racks=["r", "s"], nodes=CLUSTER["nodes"][:2] + [dict(CLUSTER["nodes"][2], rack="s", speed=0.5)]))
    nodes = c.nodes
    assert span_of([(0, 1)], nodes) == "node"
    assert span_of([(0, 1), (1, 1)], nodes) == "rack"
    assert span_of([(0, 1), (2, 1)], nodes) == "cluster"
    assert rate_of("rack", [(0, 1), (2, 1)], nodes, c) == 0.5 / 1.25


def test_first_fit_respects_class_cpu_memory():
    c = Cluster.parse(CLUSTER)
    v = parse_view(view_dict([6, 4, 8], [], []), c)
    free = P.free_of(v.nodes)
    assert P.first_fit(P.Req(4, 0, 0, 2), v.nodes, free) == [(0, 1), (1, 1)]
    assert P.first_fit(P.Req(8, 0, 0, 2), v.nodes, free) is None
    assert P.first_fit(P.Req(1, 0, 0, 1, gpu_class="b"), v.nodes, free) is None
    free[2][1] = 1
    assert P.first_fit(P.Req(4, 2, 0, 2), v.nodes, free) == [(0, 1), (1, 1)]
    assert P.first_fit(P.Req(4, 2, 0, 3), v.nodes, free) is None  # node 2 has 1 CPU: no worker of 2 CPUs
    free[2][1] = 4
    assert P.first_fit(P.Req(4, 2, 0, 3), v.nodes, free) == [(0, 1), (1, 1), (2, 1)]


def test_easy_by_hand_matches_the_go_test():
    """Same case as TestEASYByHand in Go: H reserved at 100; C on extra (n2); D ends before the shadow."""
    c = Cluster.parse(CLUSTER)
    v = parse_view(view_dict([8, 0, 4], [running("R", 8, "n1", 100), running("S", 4, "n2", 200)],
                             [pending("H", 1, 8, 2, 10), pending("C", 2, 4, 1, 500), pending("D", 3, 8, 1, 50), pending("E", 4, 8, 1, 500)]), c)
    d = make_policy("fifo+first_fit+easy", {}, 1, c).schedule(v)
    assert [(a["job_id"], a["placement"]) for a in d["actions"]] == [
        ("C", [{"node": "n2", "workers": 1}]), ("D", [{"node": "n0", "workers": 1}])]
    assert d["reservations"] == [{"job_id": "H", "shadow_ms": 100000}]
    d = make_policy("fifo+first_fit+none", {}, 1, c).schedule(v)
    assert d["actions"] == []


def run_server(lines):
    inp = io.BytesIO(b"".join(json.dumps(m).encode() + b"\n" for m in lines))
    out = io.BytesIO()
    code = serve(inp, out)
    return code, [json.loads(x) for x in out.getvalue().splitlines()]


def test_server_session():
    v = view_dict([8, 8, 8], [], [pending("a", 0, 8, 2, 10)])
    code, replies = run_server([
        {"type": "hello", "protocol_version": 1, "policy": {"name": "fifo+first_fit+none"}, "seed": 3, "cluster": CLUSTER},
        {"type": "schedule", "seq": 1, "view": v},
        {"type": "bye"}])
    assert code == 0
    assert replies[0] == {"type": "hello", "name": "fifo+first_fit+none", "version": "gcs-harness 1"}
    assert replies[1]["type"] == "decision" and replies[1]["seq"] == 1
    assert replies[1]["actions"] == [{"op": "start", "job_id": "a", "placement": [{"node": "n0", "workers": 1}, {"node": "n1", "workers": 1}]}]
    assert replies[2] == {"type": "bye"}


def test_server_rejects_wrong_version_and_unknown_messages():
    code, replies = run_server([{"type": "hello", "protocol_version": 2, "policy": {"name": "x"}, "seed": 0, "cluster": CLUSTER}])
    assert code == 1 and replies[0]["type"] == "error"
    code, replies = run_server([{"type": "dance"}])
    assert code == 1 and "unknown message" in replies[0]["message"]


def test_unknown_policy_is_an_error():
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("nope", {}, 0, Cluster.parse(CLUSTER))
