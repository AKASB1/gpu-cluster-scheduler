"""milp_rolling on hand-made views."""
from harness.model import Cluster, parse_view
from harness.policies import make_policy
from harness.tests.test_protocol import CLUSTER, pending, running, view_dict


def test_short_job_first_when_capacity_is_scarce():
    # One 8-GPU node free; A (8 GPUs, 3000 s) and B (8 GPUs, 300 s) both wait.
    # Weighted flow prefers B first: start B now, A later.
    c = Cluster.parse(CLUSTER)
    v = parse_view(view_dict([8, 0, 0], [running("R1", 8, "n1", 10000), running("R2", 8, "n2", 10000)],
                             [pending("A", 0, 8, 1, 3000), pending("B", 1, 8, 1, 300)]), c)
    d = make_policy("milp_rolling", {"k": 4, "h": 12, "slot_s": 300}, 1, c).schedule(v)
    assert [a["job_id"] for a in d["actions"]] == ["B"]
    assert d["solver"]["status"] == "optimal" and not d["solver"]["capped"]


def test_deterministic_and_never_idle_on_an_empty_cluster():
    c = Cluster.parse(CLUSTER)
    v = parse_view(view_dict([8, 8, 8], [], [pending(f"j{i}", i, 4, 1, 600 * (i + 1)) for i in range(6)]), c)
    p = make_policy("milp_rolling", {}, 1, c)
    a, b = p.schedule(v), p.schedule(v)
    assert a["actions"] == b["actions"] and len(a["actions"]) == 6  # 24 GPUs fit all six 4-GPU jobs


def test_unknown_parameter_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        make_policy("milp_rolling", {"bogus": 1}, 1, Cluster.parse(CLUSTER))


def test_worker_hard_cap_and_recovery():
    import numpy as np
    from scipy import sparse
    from harness.milp.worker import MilpWorker
    from harness.policies.milp_rolling import solver_info
    w = MilpWorker()
    try:
        A = sparse.csr_matrix(np.ones((1, 2)))
        args = (np.array([1.0, 2.0]), [(A, 1, 1)], np.ones(2), 0, 1, {"disp": False})
        assert w.solve(*args, hard_cap_s=0) is None  # the answer cannot arrive in 0 s: capped
        info, x = solver_info(None)
        assert info["status"] == "wall_cap" and info["capped"] and x is None
        res = w.solve(*args, hard_cap_s=60)  # a fresh worker answers
        info, x = solver_info(res)
        assert info["status"] == "optimal" and list(np.round(x)) == [1, 0]
    finally:
        w.close()
