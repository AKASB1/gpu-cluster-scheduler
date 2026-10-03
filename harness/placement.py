"""Placement routines (same rules as internal/placement in Go)."""
from __future__ import annotations


class Req:
    __slots__ = ("gpus", "cpus", "mem_mb", "workers", "gpu_class", "topology")

    def __init__(self, gpus, cpus, mem_mb, workers, gpu_class="", topology="any"):
        self.gpus, self.cpus, self.mem_mb, self.workers = gpus, cpus, mem_mb, workers
        self.gpu_class, self.topology = gpu_class, topology

    @staticmethod
    def of(job) -> "Req":
        return Req(job.gpus, job.cpus, job.mem_mb, job.workers, getattr(job, "gpu_class", ""), getattr(job, "topology", "any"))


def free_of(nodes):
    """Mutable free capacity per node: [gpus, cpus, mem_mb]."""
    return [[n.free_gpus, n.free_cpus, n.free_mem_mb] for n in nodes]


def fits(r: Req, node_class: str, cap) -> int:
    if r.gpu_class and r.gpu_class != node_class:
        return 0
    w = cap[0] // r.gpus
    if r.cpus > 0:
        w = min(w, cap[1] // r.cpus)
    if r.mem_mb > 0:
        w = min(w, cap[2] // r.mem_mb)
    return max(w, 0)


def take(free, r: Req, placement):
    for n, w in placement:
        free[n][0] -= r.gpus * w
        free[n][1] -= r.cpus * w
        free[n][2] -= r.mem_mb * w


def give(free, r: Req, placement):
    for n, w in placement:
        free[n][0] += r.gpus * w
        free[n][1] += r.cpus * w
        free[n][2] += r.mem_mb * w


def first_fit(r: Req, nodes, free):
    """Fill nodes in index order (rack, name), as many workers per node as fit."""
    need = r.workers
    out = []
    for i, n in enumerate(nodes):
        w = min(fits(r, n.gpu_class, free[i]), need)
        if w > 0:
            out.append((i, w))
            need -= w
            if need == 0:
                return out
    return None
