"""scenario_milp on hand-made views."""
import pytest

from harness.model import Cluster, parse_view
from harness.policies import make_policy
from harness.tests.test_protocol import CLUSTER, pending, running, view_dict


def hist(user, runtime, estimate, k=0):
    return {"job_id": f"h{k}", "tenant": "t", "user": user, "submit_s": 0, "runtime_s": runtime, "estimate_s": estimate, "finish_s": 1}


def test_without_history_it_plans_like_the_deterministic_model():
    # One free 8-GPU node; A (8 GPUs, 3000 s) and B (8 GPUs, 300 s): B first.
    c = Cluster.parse(CLUSTER)
    v = parse_view(view_dict([8, 0, 0], [running("R1", 8, "n1", 10000), running("R2", 8, "n2", 10000)],
                             [pending("A", 0, 8, 1, 3000), pending("B", 1, 8, 1, 300)]), c)
    d = make_policy("scenario_milp", {"k": 4, "h": 12, "slot_s": 300, "scenarios": 4}, 1, c).schedule(v)
    assert [a["job_id"] for a in d["actions"]] == ["B"]
    assert d["solver"]["status"] == "optimal"


def test_deterministic_given_seed_and_messages():
    c = Cluster.parse(CLUSTER)
    d = view_dict([8, 8, 0], [running("R", 8, "n2", 2000)], [pending(f"j{i}", i, 4, 1, 600 * (i + 1)) for i in range(6)])
    d["history_new"] = [hist("u", 100 * (k + 1), 300, k) for k in range(8)]
    outs = []
    for _ in range(2):
        p = make_policy("scenario_milp", {"scenarios": 6, "lambda": 1.0}, 42, c)
        outs.append(p.schedule(parse_view(d, c))["actions"])
    assert outs[0] == outs[1] and len(outs[0]) >= 1


def test_history_changes_the_scenarios():
    # The user's jobs ran 4x longer than estimated: the learned ratios are 4.
    c = Cluster.parse(CLUSTER)
    p = make_policy("scenario_milp", {"min_history": 2}, 1, c)
    d = view_dict([8, 8, 8], [], [pending("a", 0, 1, 1, 100)])
    d["history_new"] = [hist("u", 400, 100, k) for k in range(3)]
    p.schedule(parse_view(d, c))
    assert p._ratios("u") == [4.0, 4.0, 4.0]
    assert all(r == 4.0 for r in p._sample("u", 5))
    # conditioning on the work already done: no ratio above 4 exists, so the job is "about to end"
    assert all(r >= 4.0 for r in p._sample("u", 3, floor_ratio=5.0))


def test_invalid_parameters():
    c = Cluster.parse(CLUSTER)
    with pytest.raises(ValueError):
        make_policy("scenario_milp", {"alpha": 1.0}, 1, c)
    with pytest.raises(ValueError):
        make_policy("scenario_milp", {"nope": 1}, 1, c)
