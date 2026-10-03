"""Instances of the offline MILP models (docs/milp.md section 3): a trace of
schema v1 on a cluster of configuration v1, on a time grid of slot length
Delta. Python reads traces; it never generates them."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field

TAU = 10.0
HEADER = ("job_id,submit_s,tenant,user,priority,gpus,workers,gpu_class,topology,cpus,mem_gb,runtime_s,estimate_s,"
          "preemptible,checkpoint_interval_s,max_wait_s").split(",")


def to_ms(s: str) -> int:
    return int(round(float(s) * 1000))


@dataclass
class Job:
    job_id: str
    submit_ms: int
    gpus: int
    workers: int
    cpus: int
    mem_mb: int
    gpu_class: str
    runtime_ms: int
    estimate_ms: int

    @property
    def total_gpus(self) -> int:
        return self.gpus * self.workers


@dataclass
class Node:
    name: str
    rack: str
    gpu_class: str
    speed: float
    gpus: int
    cpus: int
    mem_mb: int


def load_trace(path: str) -> list:
    jobs = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        head = next(rd)
        if head != HEADER:
            raise ValueError(f"{path}: not a schema-v1 trace")
        for row in rd:
            d = dict(zip(HEADER, row))
            jobs.append(Job(d["job_id"], to_ms(d["submit_s"]), int(d["gpus"]), int(d["workers"]), int(d["cpus"]),
                            int(round(float(d["mem_gb"]) * 1000)), d["gpu_class"], to_ms(d["runtime_s"]), to_ms(d["estimate_s"])))
    return jobs


def load_cluster(path: str) -> list:
    """Nodes of a cluster configuration v1, sorted by (rack, name) as in Go."""
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    speed = {k["name"]: float(k["speed"]) for k in c["classes"]}
    nodes = [Node(n["name"], n["rack"], n["class"], speed[n["class"]], n["gpus"], n["cpus"], int(round(n["mem_gb"] * 1000)))
             for n in c.get("nodes", [])]
    for g in c.get("node_groups", []):
        for i in range(g["count"]):
            nodes.append(Node(f"{g['prefix']}{i:02d}", g["rack"], g["class"], speed[g["class"]], g["gpus"], g["cpus"],
                              int(round(g["mem_gb"] * 1000))))
    nodes.sort(key=lambda n: (n.rack, n.name))
    return nodes


@dataclass
class Machine:
    """A capacity unit of a model: a node (node model) or a class pool (pool model)."""
    name: str
    gpu_class: str
    speed: float
    gpus: int
    cpus: int
    mem_mb: int


@dataclass
class Instance:
    jobs: list
    machines: list
    model: str            # "node" or "pool"
    slot_ms: int
    release: list         # r_j (slots)
    proc: list            # proc[j][m] = slots, or None if j cannot use m
    demand: list          # GPUs per machine-slot: gpus (node) or gpus*workers (pool)
    cpu: list
    mem: list
    weight: list
    whole_slot: bool
    use_cpu_mem: bool = False
    names: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.jobs)

    def horizon(self) -> int:
        return max(self.release) + sum(max(p for p in row if p is not None) for row in self.proc)

    def flow_cost(self, j: int, start_slot: int, m: int) -> float:
        """w_j * (C_j - submit_j) in seconds for a start in slot start_slot on machine m."""
        c_ms = (start_slot + self.proc[j][m]) * self.slot_ms
        return self.weight[j] * (c_ms - self.jobs[j].submit_ms) / 1000


def build(jobs: list, nodes: list, model: str, slot_s: float) -> Instance:
    """Build the node or pool model instance. Durations are rounded up to
    whole slots; whole_slot tells whether no rounding happened."""
    slot_ms = int(round(slot_s * 1000))
    if model == "node":
        if any(j.workers != 1 for j in jobs):
            raise ValueError("the node model needs single-worker jobs")
        machines = [Machine(n.name, n.gpu_class, n.speed, n.gpus, n.cpus, n.mem_mb) for n in nodes]
    elif model == "pool":
        classes = {}
        for n in nodes:
            m = classes.setdefault(n.gpu_class, Machine(n.gpu_class, n.gpu_class, n.speed, 0, 0, 0))
            m.gpus += n.gpus
            m.cpus += n.cpus
            m.mem_mb += n.mem_mb
        machines = [classes[k] for k in sorted(classes)]
    else:
        raise ValueError(f"unknown model {model!r}")
    fastest_any = max(n.speed for n in nodes)
    whole = True
    release, proc, demand, cpu, mem, weight = [], [], [], [], [], []
    for j in jobs:
        if j.submit_ms % slot_ms:
            whole = False
        release.append(-(-j.submit_ms // slot_ms))  # rounded up: a job never starts before its submission
        row = []
        for m in machines:
            ok = (not j.gpu_class or j.gpu_class == m.gpu_class)
            if model == "node":
                ok = ok and j.gpus <= m.gpus and j.cpus <= m.cpus and j.mem_mb <= m.mem_mb
            else:
                ok = ok and j.total_gpus <= m.gpus
            if not ok:
                row.append(None)
                continue
            exact = j.runtime_ms / (m.speed * slot_ms)
            p = max(1, math.ceil(exact - 1e-9))
            if abs(exact - round(exact)) > 1e-9:
                whole = False
            row.append(p)
        if all(p is None for p in row):
            raise ValueError(f"job {j.job_id} fits no {model} of the cluster")
        proc.append(row)
        demand.append(j.gpus if model == "node" else j.total_gpus)
        cpu.append(j.cpus if model == "node" else j.cpus * j.workers)
        mem.append(j.mem_mb if model == "node" else j.mem_mb * j.workers)
        fastest = max(n.speed for n in nodes if n.gpu_class == j.gpu_class) if j.gpu_class else fastest_any
        ideal = j.runtime_ms / 1000 / fastest
        weight.append(1 / max(ideal, TAU))
    use = model == "node" and any(c > 0 or m > 0 for c, m in zip(cpu, mem))
    return Instance(jobs, machines, model, slot_ms, release, proc, demand, cpu, mem, weight, whole, use,
                    [m.name for m in machines])
