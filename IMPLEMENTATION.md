# Implementation plan

## Job model

A job includes:

- requested GPU count and optional GPU class
- CPU / memory request
- priority
- queue / tenant
- gang size when applicable
- topology constraints
- preemptibility
- optional start-deadline or SLO

## Cluster model

Represent each node with allocatable CPU, memory, GPU devices, GPU type, and topology labels. Keep allocated and reserved resources separate.

## Baseline policies

Implement these before any learned or optimization-based policy:

1. FIFO
2. priority FIFO
3. best-fit / bin packing
4. least-fragmentation placement
5. dominant-resource fairness across tenants
6. backfilling
7. preemption by priority

## Metrics

Measure:

- queueing delay
- job completion time
- GPU utilization
- GPU fragmentation
- preemption count
- fairness by tenant
- SLO violation rate

## Delivery order

1. in-memory job and cluster model
2. deterministic scheduling loop
3. FIFO + best-fit baselines
4. event simulator
5. multi-GPU / gang jobs
6. topology constraints
7. preemption and backfilling
8. Prometheus metrics
9. Kubernetes scheduling-framework adapter
10. optimization-based policy experiments

## Scaffold checkpoint

The in-memory job/cluster model and deterministic FIFO loop are implemented. Best-fit is exposed as a helper. Gang placement, preemption execution, metrics exporters, and Kubernetes integration remain planned.
