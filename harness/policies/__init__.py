"""Policy registry of the Python harness."""
from __future__ import annotations

from . import baselines, misbehave

_SIMPLE = {cls.name: cls for cls in (baselines.FifoFirstFit, baselines.FifoFirstFitEasy,
                                     misbehave.Sleep, misbehave.BadJSON, misbehave.Invalid, misbehave.Crash)}
_SOLVER = ("milp_rolling", "scenario_milp")  # need numpy/scipy: imported only when requested


def _solver_class(name: str):
    if name == "milp_rolling":
        from .milp_rolling import MilpRolling
        return MilpRolling
    from .scenario_milp import ScenarioMilp
    return ScenarioMilp


def make_policy(name: str, params, seed: int, cluster):
    if name in _SIMPLE:
        return _SIMPLE[name](params or {}, seed, cluster)
    if name in _SOLVER:
        return _solver_class(name)(params or {}, seed, cluster)
    raise ValueError(f"unknown policy {name!r}; known: {sorted(list(_SIMPLE) + list(_SOLVER))}")
