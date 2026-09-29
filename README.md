# GPU Cluster Scheduler

A scheduler for heterogeneous GPU clusters with dynamic job arrivals, multi-GPU placement, queue priorities, preemption, topology constraints, and SLO-aware policies.

**Status:** implementation scaffold.

## Scope

- cluster and GPU inventory
- queueing and priority classes
- single- and multi-GPU jobs
- gang admission for distributed jobs
- topology-aware placement
- preemption and backfilling
- fairness and quota policies
- utilization / fragmentation metrics
- policy simulator for repeatable experiments
- Kubernetes scheduler integration later

## Proposed stack

Go · Kubernetes scheduling APIs · Prometheus · gRPC · optional Python simulator/analysis tools

## Scheduling loop

```text
Pending Jobs
     │
     ▼
Admission / Queue
     │
     ▼
Cluster Snapshot
     │
     ▼
Scheduling Policy
     │
     ├──► placement
     ├──► preemption
     └──► backfill
             │
             ▼
        Kubernetes
             │
             ▼
Metrics / Events / Replay
```

## Repository layout

```text
cmd/scheduler/
internal/
  api/
  cluster/
  queue/
  policy/
  placement/
  preemption/
  metrics/
simulator/
benchmarks/
deploy/
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Reference projects

- [kubernetes-sigs/kueue](https://github.com/kubernetes-sigs/kueue) — workload queueing and admission
- [volcano-sh/volcano](https://github.com/volcano-sh/volcano) — batch and gang scheduling
- [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes) — scheduling framework and scheduler internals
- [NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) — GPU discovery and resource exposure in Kubernetes

## License

MIT
