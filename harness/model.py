"""The scheduling view of external-policy protocol v1 (docs/contracts.md section 6).

Times arrive as decimal seconds with at most three decimals and are kept as
integer milliseconds. Memory is in whole MB. The arithmetic helpers repeat
the Go rules bit for bit (same float operations in the same order).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


def ms(x) -> int:
    """Decimal seconds (JSON number) -> integer milliseconds (exact for <= 3 decimals)."""
    return int(round(float(x) * 1000))


def sec(m: int) -> float:
    """Integer milliseconds -> float seconds (correctly rounded, as Go's float64(ms)/1000)."""
    return m / 1000


def fmt_s(m: int):
    """Integer milliseconds -> the shortest decimal seconds value for JSON."""
    if m % 1000 == 0:
        return m // 1000
    return float(f"{m // 1000}.{m % 1000:03d}".rstrip("0"))


def ceil_ms(x: float) -> int:
    """Round a float number of ms up to a whole ms, with the 1e-6 ms tolerance."""
    if x <= 0:
        return 0
    return math.ceil(x - 1e-6)


def work_to_duration_ms(work: float, rate: float) -> int:
    """ceil(work * 1000 / rate - 1e-6) ms; 0 for work <= 0."""
    if work <= 0:
        return 0
    return ceil_ms(work * 1000 / rate)


@dataclass
class Node:
    index: int
    name: str
    rack: str
    gpu_class: str
    speed: float
    gpus: int
    cpus: int
    mem_mb: int
    free_gpus: int = 0
    free_cpus: int = 0
    free_mem_mb: int = 0


@dataclass
class Cluster:
    classes: dict
    racks: list
    cross_node: float
    cross_rack: float
    restart_overhead_ms: int
    preempt_grace_ms: int
    nodes: list  # static Node templates in index order

    @staticmethod
    def parse(d: dict) -> "Cluster":
        nodes = [Node(i, n["name"], n["rack"], n["class"], float(n["speed"]), int(n["gpus"]), int(n["cpus"]), int(n["mem_mb"]))
                 for i, n in enumerate(d["nodes"])]
        return Cluster({c["name"]: float(c["speed"]) for c in d["classes"]}, list(d["racks"]),
                       float(d["cross_node_factor"]), float(d["cross_rack_factor"]),
                       ms(d["restart_overhead_s"]), ms(d["preempt_grace_s"]), nodes)


@dataclass
class Running:
    job_id: str
    tenant: str
    user: str
    priority: int
    gpus: int
    workers: int
    cpus: int
    mem_mb: int
    gpu_class: str
    topology: str
    run_start_ms: int
    overhead_ms: int
    placement: list  # [(node index, workers)]
    rate: float
    estimate_ms: int
    retained_at_start: float
    work_done: float
    est_end_ms: int
    preemptible: bool
    checkpoint_interval_ms: int


@dataclass
class Pending:
    job_id: str
    tenant: str
    user: str
    submit_ms: int
    priority: int
    gpus: int
    workers: int
    cpus: int
    mem_mb: int
    gpu_class: str
    topology: str
    estimate_ms: int
    wait_ms: int
    retained: float
    max_wait_ms: Optional[int]
    preemptible: bool
    checkpoint_interval_ms: int
    preemptions: int
    started: bool


@dataclass
class View:
    now_ms: int
    cluster: Cluster
    nodes: list
    running: list
    pending: list
    tenants: list
    history_new: list = field(default_factory=list)


def parse_view(d: dict, cluster: Cluster) -> View:
    index = {n.name: n.index for n in cluster.nodes}
    nodes = []
    for i, nd in enumerate(d["nodes"]):
        t = cluster.nodes[i]
        if nd["name"] != t.name:
            raise ValueError(f"node {i} is {nd['name']}, the handshake said {t.name}")
        nodes.append(Node(i, t.name, t.rack, t.gpu_class, t.speed, t.gpus, t.cpus, t.mem_mb,
                          int(nd["free_gpus"]), int(nd["free_cpus"]), int(nd["free_mem_mb"])))
    running = [Running(r["job_id"], r["tenant"], r["user"], int(r["priority"]), int(r["gpus"]), int(r["workers"]), int(r["cpus"]),
                       int(r["mem_mb"]), r["gpu_class"], r["topology"], ms(r["run_start_s"]), ms(r["overhead_s"]),
                       [(index[p["node"]], int(p["workers"])) for p in r["placement"]], float(r["rate"]), ms(r["estimate_s"]),
                       float(r["retained_at_start_s"]), float(r["work_done_s"]), ms(r["est_end_s"]), bool(r["preemptible"]),
                       ms(r["checkpoint_interval_s"]))
               for r in d["running"]]
    pending = [Pending(p["job_id"], p["tenant"], p["user"], ms(p["submit_s"]), int(p["priority"]), int(p["gpus"]), int(p["workers"]),
                       int(p["cpus"]), int(p["mem_mb"]), p["gpu_class"], p["topology"], ms(p["estimate_s"]), ms(p["wait_s"]),
                       float(p["retained_s"]), None if p["max_wait_s"] is None else ms(p["max_wait_s"]), bool(p["preemptible"]),
                       ms(p["checkpoint_interval_s"]), int(p["preemptions"]), bool(p["started"]))
               for p in d["pending"]]
    return View(ms(d["now_s"]), cluster, nodes, running, pending, list(d["tenants"]), list(d["history_new"]))


def span_of(placement, nodes) -> str:
    if len(placement) <= 1:
        return "node"
    rack = nodes[placement[0][0]].rack
    for n, _ in placement[1:]:
        if nodes[n].rack != rack:
            return "cluster"
    return "rack"


def topology_factor(topology: str, span: str, cross_node: float, cross_rack: float) -> float:
    if span == "node" or topology == "any":
        return 1.0
    if span == "rack" and topology == "node":
        return cross_node
    if span == "rack":
        return 1.0
    return cross_rack


def rate_of(topology: str, placement, nodes, cluster: Cluster) -> float:
    s_min = min(nodes[n].speed for n, _ in placement)
    f = topology_factor(topology, span_of(placement, nodes), cluster.cross_node, cluster.cross_rack)
    return s_min / f


def estimated_duration_ms(p: Pending, placement, view: View, work: float) -> int:
    """overhead (if the job ran before) + ceil((work) * 1000 / rate) ms, work clamped at 0."""
    rate = rate_of(p.topology, placement, view.nodes, view.cluster)
    overhead = view.cluster.restart_overhead_ms if p.started else 0
    return overhead + work_to_duration_ms(max(work, 0.0), rate)
