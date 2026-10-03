# Simulator model

The simulator is a **model**: every result it produces is *simulated, with assumed parameters*. It does not claim anything about production schedulers or real clusters. This document lists every rule and every parameter, each marked **assumed** (a modelling choice) or **derived** (with its source).

Code: `simulator/simulator.go` (event loop, actions, accounting), `internal/metrics` (records, invariants, identities, metrics), `internal/trace` (workload generator), `internal/cluster` (cluster configuration).

## 1. Time and events

- Discrete-event simulation on a virtual clock; no sleeps. Time is a whole number of milliseconds (`time.Duration`). Trace times are decimal seconds with at most three decimals, so they convert exactly.
- A duration computed from a rate is rounded **up** to the next millisecond: `ceil(work * 1000 / rate - 1e-6)` ms. The `1e-6` ms tolerance absorbs float error, so that, for example, 1000 s of work at rate `1/1.15` takes exactly 1150 s and not 1150.001 s. Event order therefore never depends on float accumulation.
- Event kinds: job submission, job completion, end of a preemption grace period, periodic tick (optional), and policy wake-up (optional, requested by the policy).
- Events at the same instant are applied in a fixed order: grace releases, then completions (by `job_id`), then submissions (in trace order). Then the policy is invoked **once** for that instant, if at least one job is pending. Ticks (`schedule_interval_s`, default off) and wake-ups invoke it as well.
- Policy decisions take no virtual time. Their wall-clock time is measured separately (`wall_` columns) and never influences the simulation.
- After the policy acted, the simulator records one **sample** of the cluster state (allocated and free GPUs, stranded GPUs, the blocked indicator); it holds until the next instant.
- One run is single-threaded and deterministic. Runs are independent; the benchmark runs them in parallel.

## 2. Resources and placement

- A node has GPUs of one class, CPUs, and memory. Memory is handled in whole MB (0.001 GB): `round(mem_gb * 1000)`; every capacity check (simulator and Go and Python policies) compares whole MB, so all sides agree exactly.
- A worker needs `gpus` GPUs, `cpus` CPUs, and `mem_gb` of memory on **one** node. The GPUs of a node are interchangeable (no NVLink or PCIe topology inside a node). Several workers of one job may share a node.
- A job with `gpu_class` set runs only on nodes of that class; a job without it may use any class, and may even mix classes.
- A start places all workers atomically (gang) or not at all. A placement is a worker count per node.
- A job that cannot fit even the empty cluster (respecting its class) is **infeasible**: it is recorded, counted, never submitted to the policy, and excluded from every metric. Generated traces contain none.

## 3. Speed and topology

- Work is measured in reference-seconds: seconds of progress on a GPU of speed 1.0. A job is done when its work reaches `runtime_s`.
- A running job progresses at `rate = min(speed of the classes it occupies) / f` work-seconds per second (synchronous training runs at the pace of its slowest worker).
- The **span** of a placement is `node` (one node), `rack` (several nodes of one rack), or `cluster` (several racks).
- The topology factor `f` is 1 when the span is no wider than the job's sensitivity allows (`node` allows the node span only; `rack` allows node and rack; `any` allows everything). Otherwise `f` is the cluster's `cross_node_factor` (a `node`-sensitive job over several nodes of one rack) or `cross_rack_factor` (a `node`- or `rack`-sensitive job over several racks). `1 <= cross_node_factor <= cross_rack_factor`.
- `f` and the rate are fixed at the start of a run; there is no migration.

## 4. Preemption, checkpoints, restarts

- A policy may preempt a running job only if `preemptible` is 1.
- Work done at preemption = work retained at the start of the run + progress of this run (progress starts after the restart overhead).
- Retained work = work done rounded **down** to a multiple of `checkpoint_interval_s` (checkpoints are taken every `checkpoint_interval_s` reference-seconds of progress, counted from the job's first start); nothing is retained when it is 0. A `1e-9` relative tolerance in the rounding absorbs float error, and the retained work is capped at the work done. The difference is **lost work**.
- With `preempt_grace_s = 0` (default) the GPUs are free at once and the job returns to the queue at the same instant. With `preempt_grace_s > 0` the job keeps holding its GPUs (counted as allocated, consumed, and wasted) until the grace period ends, and only then re-enters the queue; the pending time (`queue_total`) counts from the preemption.
- Every start after a preemption first holds its GPUs for `restart_overhead_s` without progress (counted as allocated and as waste). The first start pays nothing.
- No kill at the estimate: a job that outlives its estimate keeps running.

## 5. What policies see

The view (`internal/api`, wire form in `docs/contracts.md` §6): time; nodes (name, rack, class, speed, total and free GPUs/CPUs/memory, jobs running there with worker counts); running jobs (request, placement, start of the current run, restart overhead, rate, estimate, retained work, work done, estimated remaining work, estimated end, preemptibility, checkpoint interval); pending jobs sorted by priority (high first), submit time, `job_id` (request, estimate, wait so far, retained work, `max_wait_s`, whether it ran before); per-tenant usage (GPU-seconds so far, running GPUs, CPUs, memory); the history of completed jobs (user, submit time, true run time, revealed only at completion).

The view never contains the true run time of an unfinished job. Oracle policies get it through a separate lookup injected by the benchmark; they are labelled oracles and never presented as deployable.

**Estimated end of a running job** (the documented overrun rule): `est_end = run_start + overhead + ceil_ms((estimate - retained_at_start) / rate)`; when that is already in the past (the job outlived its estimate), `est_end = now`. Backfilling policies therefore treat an overrun job as "about to finish"; the reservation guarantee of backfilling does not hold then, and the benchmark counts broken reservations.

## 6. Policy actions and errors

- A decision is an ordered list of `start(job, placement)` and `preempt(job)` actions, optionally a wake-up time and the shadow times of backfilling reservations (used only for the reservation check). Actions apply in order, so a preempt can free resources for a later start in the same list.
- The simulator checks every action: the job exists and is in the right state, the nodes exist and are listed once, worker counts are >= 1 and sum to the gang size, capacities (GPU, CPU, memory) and the class constraint hold, and preempted jobs are preemptible and were not started at the same instant (a start and a preemption of one job in one decision would count the job as started without giving it GPU time). An invalid action aborts the run with an error that names the policy and the action; nothing is skipped silently.
- **Termination.** The run continues until every feasible job has completed. A **deadlock** — nothing running, jobs pending, no future submission, no pending grace release or wake-up, and the policy starts nothing when called again at the same instant — is a run error that names the policy (a policy bug, not a result). A run that exceeds `1000 * (jobs + 1)` invocations is also an error.
- Node failures (Tier 2) are described in section 12.

## 7. Accounting and identities

For one run of a job with placement `P` (workers `w_n` on node `n` of speed `s_n`, `g` GPUs per worker, `G = g * sum w_n`), duration `D`, restart overhead `O' = min(D, O)`, progress time `Dp = D - O'`, minimum speed `s_min`, and factor `f`, in speed-weighted GPU-seconds (reference-GPU-seconds):

| Term | Definition |
|---|---|
| consumed | `sum_n g w_n s_n D` (+ grace: `sum_n g w_n s_n grace`) |
| restart overhead | `sum_n g w_n s_n O'` |
| mixed-speed slack | `sum_n g w_n (s_n - s_min) Dp` — faster GPUs waiting for the slowest worker |
| topology slowdown | `G s_min Dp (1 - 1/f)` |
| progress | `G (s_min / f) Dp = G * work of the run` |
| lost (preempted run) | `G * (work done - retained)` |
| useful (job) | `G * runtime_s` |
| rounding | `G * (retained at start + work of the last run - runtime_s)` — progress beyond the run time caused by rounding the end up to a millisecond; >= 0 and below `G * rate * 0.001` |
| grace | `sum_n g w_n s_n grace` for each preemption with a grace period |

Identity per job: **consumed = useful + lost + overhead + topology + slack + rounding + grace** (float, relative tolerance `1e-9`). Two more identities are exact (integer milliseconds): the integral over time of the number of jobs in the system equals the sum of their JCTs (Little's law as an identity), and the integral of allocated GPUs equals the sum of allocated GPU-milliseconds over all runs (restart overhead and grace included). `metrics.CheckIdentities` checks all three on every benchmark run. The per-job identity alone would hold by construction (rounding is the remainder), so `metrics.Verify` also recomputes every run from its placement, the node speeds and racks, and the cluster parameters, independently of the simulator's bookkeeping: the slowest speed, the topology factor, and the rate; the restart overhead (none on the first start); the work done; the retained work (checkpoint rule); the duration of the completing run (overhead plus the remaining work at its rate, rounded up to a millisecond, so rounding lies in `[0, G * rate * 0.001)`); every waste term; and the allocated GPU-milliseconds. It further checks the invariants: no over-allocation of GPUs, CPUs, or memory on any node, with a node under repair counted as fully allocated; every feasible job completes exactly once, after its submission; one start instant per run; segments ordered in time; class and preemptibility constraints. A test corrupts records in ways that keep the identities and checks that `Verify` rejects them.

## 8. Theory checks (tests in `simulator/theory_test.go`)

| Check | Setup | Criterion |
|---|---|---|
| Lindley (exact) | one node, one GPU, FIFO, lognormal service, Poisson arrivals, 20,000 jobs, seeds 1–3 | every simulated wait equals `W[n+1] = max(0, W[n] + S[n] - A[n+1])` on the realised ms-rounded times |
| M/G/1 | one GPU, rho = 0.7, mean service 100 s; exponential, deterministic, lognormal (sigma 1); 10 seeds x 50,000 jobs | mean wait within 5%, 5%, 8% of `lambda E[S^2] / (2 (1 - rho))` |
| M/M/c | one node with c = 4 (50,000 jobs) and c = 16 (100,000 jobs) GPUs, rho = 0.7, 10 seeds | mean wait within 6% and 10% of Erlang C |

Tolerances were fixed before the first run; the tests log the measured error and its standard error.

## 9. Parameters and their sources

| Parameter | Value(s) | Status and source |
|---|---|---|
| Node shape | 8 GPUs, 128 CPU cores, 1024 GB | derived: NVIDIA's published DGX A100 system specification (8 A100 GPUs, 2 x 64-core CPUs, 1 TB memory) |
| A100-class speed | 1.0 | reference class (definition) |
| V100-class speed | 0.40 | derived: ratio of NVIDIA's published peak dense FP16 tensor-core throughput (V100 125 TFLOPS, A100 312 TFLOPS); assumed to apply uniformly to all jobs (real speed-ups vary by model) |
| `cross_node_factor` | 1.10 | assumed |
| `cross_rack_factor` | 1.25 | assumed |
| `restart_overhead_s` | 120 | assumed |
| `preempt_grace_s` | 0 | assumed |
| Checkpoint intervals | 0 (none), 600 s (short), 1800 s (default), 3600 s (long) | assumed |
| Racks | 4 nodes per rack | assumed |

The sensitivity sweep (benchmarks/README.md) varies the topology penalty `(f - 1)`, the restart overhead, and the checkpoint interval by x0.5 and x2, and the estimate-error level.

## 10. Workload generator (assumed)

All distributions are assumptions, shaped after the qualitative findings of the public trace studies (most jobs use one GPU, run times are heavy-tailed, a few jobs are large gangs, users resubmit similar jobs); no number from those studies is reproduced. Configuration: `configs/workloads/base.json` and the scenario overrides.

- **Arrivals.** Poisson; gamma renewal with a coefficient of variation above 1 (bursty); a two-state Markov-modulated Poisson process (bursty); diurnal by thinning, `rate(t) = mean * (1 + A sin(2 pi t / period))`; batch (every job at time 0). The mean rate is `load * work_capacity / E[gpus * workers * runtime | users]`: offered load is defined against the cluster's work capacity (sum of GPUs x speed), and the expectation is taken given the user population drawn for the seed (every user's typical run time is drawn first; the expectation over the per-job noise and the truncation is a Monte Carlo average on a fixed stream). So every seed has the configured load in expectation; the heavy-tailed run times of individual jobs still make the realised load of a 2000-job trace vary (base workload at load 0.8: mean 0.78 over 20 seeds, single seeds within 15%; test `TestOfferedLoadCalibration`).
- **Bursts** (S6): burst starts form a Poisson process; each burst submits a fixed number of high-priority, non-preemptible jobs uniformly within a spread; they add load on top of the main stream.
- **Sizes.** A categorical mix of (GPUs per worker, workers): 1, 2, 4, 8 GPUs single-worker; 4x2 node-sensitive; 8x2 and 8x4 gangs (rack-sensitive with probability 0.7); 1x16. Large gangs are a few percent of jobs.
- **Run times.** Each user has a typical run time `lognormal(ln 1200 s, 1.0)` drawn once; a job's run time is that value times `lognormal(0, 0.8)`, truncated to [60 s, 24 h]. Users therefore resubmit jobs of similar length, which is what runtime predictors exploit.
- **Tenants and users.** Four tenants with unequal submission shares (0.4/0.3/0.2/0.1 by default; S5 uses one heavy tenant), eight users each.
- **Priorities.** Classes 1 (50%, all preemptible, no wait target), 4 (40%, half preemptible, `max_wait_s` 7200), 8 (10%, not preemptible, `max_wait_s` 900).
- **CPU and memory** per worker: 12 CPUs and 96 GB per GPU; 10% of jobs CPU-heavy (16 per GPU), 5% memory-heavy (128 GB per GPU).
- **Class constraints** (heterogeneous scenarios): a fraction of jobs is pinned to one class.
- **Estimates**, `estimate = runtime * m`:
  - `exact`: m = 1;
  - `lognormal`: m = exp(N(0, sigma)), sigma 0.5 by default (half of the estimates are underestimates);
  - `user` (after the model of Mu'alem and Feitelson, with underestimates added): 15% exact; 10% underestimates with m ~ U(0.5, 0.95); otherwise the accuracy runtime/estimate ~ U(0.05, 1) and the estimate is rounded up to the next "popular" value (15 min, 30 min, 1, 2, 4, 8, 12, 24, 48 h). In the original setting jobs were killed at their estimate, so that data has no underestimates; here they are added deliberately.
- **Slot grid** (S8 MILP instances): run times are a whole number of slots and submit times are on the slot grid.

## 11. Simplifications (what the model leaves out)

Node failures only as in section 12 (whole nodes, independent, no correlated or partial failures), no network contention between jobs, no intra-node GPU topology, no elastic or fractional GPUs, no job dependencies, no migration, no kill at the estimate, quotas only as a policy rule (section 12), no data loading or storage effects, speeds independent of the model being trained, and topology penalties that depend only on the span. Results are labelled "simulated, assumed parameters".

## 12. Tier 2 extensions: node failures and quotas

**Node failures.** A run may carry an explicit failure schedule (`simulator.Config.Failures`: node, time, repair duration; none by default). The benchmark draws it per node and seed from the stream `failures/<node>`: exponential times between failures with mean `mtbf_s` and exponential repair times with mean `repair_s` (rounded to whole ms, at least 1 ms), the same schedule for every policy (common random numbers). At an instant, completions are processed first (a job that completes when its node fails completes), then repairs, then failures, then submissions. A failure kills every job with a worker on the node, preemptible or not: like a preemption, the work beyond the last checkpoint is lost (`lost_` in the accounting), the job is requeued with its retained work and pays the restart overhead at its next start, and the run segment is marked killed (`failure_kills`). The node then has no free capacity until the repair ends; a failure of a node already under repair is absorbed. Failures are considered while the run has jobs to submit, pending, or running. The deadlock check does not fire while a node is down (a job may be waiting for the repair). Failures together with `preempt_grace_s > 0` are rejected. The schedule is an assumption (independent nodes, memoryless failures), not a model of any real cluster.

**Quotas** are a policy rule, not a simulator feature: the simulator only reports per-tenant usage in the view; admission, borrowing, and reclaim are defined in `docs/contracts.md` §7 and the quota metrics in §5.
