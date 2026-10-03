# Implementation plan

## Job model

A job (trace schema v1, docs/contracts.md §2) has:

- GPUs per worker and a worker count (gang: all workers start together or the job does not start)
- an optional GPU class (the only class it may use)
- CPU and memory per worker
- priority (0–9) and preemptibility, with a checkpoint interval
- tenant (the unit of fairness) and user (the unit of runtime prediction)
- a topology sensitivity (`any`, `rack`, `node`)
- an optional start-wait target (`max_wait_s`)
- the true run time (hidden from policies) and the user's estimate

## Cluster model

Nodes with GPUs of one class, CPUs, and memory, grouped into racks (cluster configuration v1, docs/contracts.md §3). Classes have a speed relative to the reference class. A placement spanning nodes or racks beyond a job's sensitivity runs slower by a topology factor. Allocated, free, and grace-held resources are tracked separately by the simulator.

## Baseline policies

Built from four independent parts (docs/contracts.md §7), all implemented and unit-tested:

1. order: FIFO, priority (with aging), shortest estimate first, dominant-resource fairness across tenants
2. placement: first fit, best fit / bin packing, least fragmentation, topology aware, heterogeneity aware (`hetero_ect`, Tier 2)
3. backfill: none, EASY, conservative, EASY with learned predictions, risk-aware EASY with a run-time quantile (`risk`, Tier 2), EASY with true run times (oracle)
4. preemption: none, priority preemption with victims that minimize lost work
5. quotas (Tier 2): nominal per-tenant quotas, strict or with borrowing, and reclaim by preemption

Optimization-based policies live in the Python harness: the offline MILP reference (pool and node models), the rolling-horizon `milp_rolling`, and the two-stage stochastic `scenario_milp` with a CVaR term (Tier 2) (docs/milp.md).

## Metrics

Queueing delay, completion time, bounded slowdown (mean and tails, per tenant and priority), GPU utilization, goodput and wasted GPU-hours, fragmentation (blocked time and stranded GPUs), preemption count, fairness by tenant (Jain), SLO attainment, makespan, broken reservations, scheduling and solver time (wall-clock and CPU, reported apart); for Tier 2 scenarios also quota satisfaction, borrowed GPU-hours, quota-weighted fairness, failure kills, and node downtime. Definitions: docs/contracts.md §5.

## Delivery order

Revised: the simulator and benchmark come before external integrations, and optimization-based policies before Prometheus and Kubernetes.

- [x] 1. in-memory job and cluster model (trace schema v1, cluster configuration v1, loaders and validators)
- [x] 2. deterministic scheduling loop
- [x] 3. event simulator, trace generator, and metrics
- [x] 4. FIFO + best-fit baselines, first benchmark table
- [x] 5. multi-GPU / gang jobs
- [x] 6. topology constraints
- [x] 7. preemption and backfilling
- [x] 8. optimization-based policies (MILP reference, online variant) on the same traces
- [x] 9. Prometheus-format metrics of a replayed simulation (`/metrics` in replay mode; no Prometheus server, nothing from a real cluster)
- [ ] 10. Kubernetes scheduling-framework adapter (planned)

## Decision problem

- Decision: start job `i` with a worker count per node at time `t`, preempt a running job, or keep jobs queued.
- State: queue, free GPUs, CPUs, memory and GPU class per node, running jobs and their estimated ends, tenant usage, completion history.
- Uncertainty: arrival times, true run time versus the user estimate.
- Objective: tail bounded slowdown and wait, waste, SLO violations, and tenant fairness (the score J of docs/contracts.md §5); the MILP minimizes the sum of weighted flow.

Policy families, in implementation order: FIFO, best fit, backfilling, priority preemption, MILP (offline and rolling horizon). Online policies that handle runtime uncertainty explicitly (risk-aware backfilling, a scenario-based stochastic MILP) were added in Tier 2 and evaluated under a distribution shift (S10, S11).

Go carries the scheduler core. The MILP reference and the online MILP live in a Python harness, since solver support is strongest there. Both read the same trace format, and Python policies run in the Go simulator through the external-policy protocol.

## Evaluation and acceptance

Workloads: low / medium / high load; homogeneous / heterogeneous GPUs; accurate / noisy / user-style runtime estimates and shifts between them; bursty and diurnal arrivals; multi-tenant with and without quotas; high-priority bursts with preemption; node failures; small instances with an exact MILP optimum; replayed jobs of a public trace (Helios) on a simulated cluster; a capacity-planning sweep. The protocol (frozen tuning on disjoint seeds, confidence intervals, paired comparisons, an oracle, an optimality gap, a sensitivity check) is in benchmarks/README.md.

The benchmark is done when:

- [x] one command reproduces all tables and figures from fixed seeds (`go run ./cmd/benchmark`, after the committed tuning)
- [x] at least four policies run on the same traces
- [x] assumptions (arrival model, runtime model, node model) are written down (docs/simulator.md)
- [x] the README has the framework figure, at least two result figures, and one result table

No result numbers are added to this repository before the benchmark produces them.

## Cross-project contracts

The general conventions — time, randomness, units, determinism, and the CSV and manifest rules — are those of version 1 of the conventions of `llm-serving-control` (its `docs/contracts.md`, sections 1 and 2). This project follows them and defines, in its own `docs/contracts.md`, the job trace schema, the cluster configuration, the scheduling metrics, and the external-policy protocol. Other projects follow that document rather than importing these packages.

A Go and a Python implementation of the same baseline policy give identical assignments on shared traces: this is check 8, tested in `internal/extpolicy` (`TestConformanceGoPython`) for FIFO with first fit and for EASY backfilling.

## Scaffold checkpoint

Replaced. The repository now holds a deterministic discrete-event simulator with gang placement, GPU classes, topology factors, preemption with checkpoints and restart overhead; the Go baseline policies; the Python harness with the protocol server, the offline MILP, `milp_rolling`, and `scenario_milp`; Tier 2 extensions (risk-aware backfilling, heterogeneity-aware placement, quotas, node failures, the Helios converter and replay scenario, the capacity sweep, replay mode with `/metrics`); the benchmark with frozen tuning and committed results; and the model checks (Lindley, M/G/1, M/M/c, accounting identities, EASY reservations, invariants, MILP against exhaustive search, Go/Python conformance) as tests. Kubernetes integration and gRPC remain planned.
