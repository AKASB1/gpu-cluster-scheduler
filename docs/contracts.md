# Contracts, version 1

This document is the binding, language-neutral definition of the formats, metric definitions, and interfaces of `gpu-cluster-scheduler`. Implementations in other languages follow this document; they do not import the Go packages. Changes are deliberate, versioned, and logged.

Version: **1**.

The general conventions (time, randomness, units, determinism, CSV and manifest rules) follow version 1 of the conventions of the sibling project `llm-serving-control` (its `docs/contracts.md`, sections 1 and 2). They are restated below so that this document stands alone; where this project needs more precision, the extra rule is stated here.

## 1. Conventions

- **Time.** Policies never read the wall clock and never sleep; they read time from the view (or from an injected clock). Trace and report times are decimal seconds from the start of the trace, with at most three decimals. Inside the simulator, time is a whole number of milliseconds (`time.Duration` in Go). A duration computed from a rate is rounded up to the next millisecond: `ceil(work * 1000 / rate - 1e-6)` ms, where the `1e-6` ms tolerance absorbs float error (so 1000 s of work at rate `1/1.15` takes exactly 1150 s). Event order therefore never depends on float accumulation.
- **Randomness.** Every component draws from its own stream, derived from the run seed and the component name: `x = seed XOR fnv1a64(name)`, `s1 = splitmix64(x)`, `s2 = splitmix64(s1)`, stream = PCG(`s1`, `s2`) of Go's `math/rand/v2`. `fnv1a64` is 64-bit FNV-1a over the UTF-8 bytes; `splitmix64` is the standard SplitMix64 output function (add `0x9e3779b97f4a7c15`, then xor-shift-multiply by `0xbf58476d1ce4e5b9` and `0x94d049bb133111eb`). The trace a seed produces depends only on the generator configuration and the seed, never on which policy runs (common random numbers). User runtime estimates are part of the trace. Python never generates traces; its own draws use `numpy.random.default_rng(numpy.random.SeedSequence([seed, fnv1a64(policy_name)]))`.
- **Units.** GPUs, CPUs, workers, and jobs are integers; time is seconds; consumption is GPU-seconds or GPU-hours (raw GPUs); work is reference-GPU-seconds (seconds of progress on a GPU of speed 1.0, times GPUs); memory is GB. There is no currency.
- **Determinism.** The same cluster configuration, trace, policy configuration, and seed give identical outputs. Ties break by an explicit total order — jobs: priority (high first), submit time, `job_id`; nodes: rack, node name — never by map iteration order. The only non-deterministic outputs are wall-clock measurements; they live in columns whose names start with `wall_` (in separate files, see section 6), and byte-comparison tests exclude them.

## 2. Job trace schema v1

A trace is a CSV file with exactly this header (16 columns). Rows are sorted by `submit_s` (non-decreasing; rows with equal times keep their file order).

```text
job_id,submit_s,tenant,user,priority,gpus,workers,gpu_class,topology,cpus,mem_gb,runtime_s,estimate_s,preemptible,checkpoint_interval_s,max_wait_s
```

| Column | Type | Rule |
|---|---|---|
| `job_id` | string | non-empty, unique, no comma |
| `submit_s` | decimal | finite, >= 0, at most three decimals |
| `tenant` | string | non-empty; the unit of fairness and quota |
| `user` | string | may be empty (then it is the tenant); used by runtime predictors |
| `priority` | integer | 0 to 9; higher is more important |
| `gpus` | integer | >= 1; GPUs per worker |
| `workers` | integer | >= 1; gang size: all workers start together or the job does not start |
| `gpu_class` | string | may be empty (any class); otherwise the only class the job may use |
| `topology` | string | `any`, `rack`, or `node`: the tightest domain the job is sensitive to |
| `cpus` | integer | >= 0, per worker |
| `mem_gb` | decimal | finite, >= 0, per worker |
| `runtime_s` | decimal | > 0, at most three decimals; true run time on the reference class (speed 1.0), without topology penalty or restart overhead; hidden from policies (only oracles read it) |
| `estimate_s` | decimal | > 0, at most three decimals; the user's runtime estimate in the same unit; the only run-time information policies get |
| `preemptible` | integer | 0 or 1 |
| `checkpoint_interval_s` | decimal | >= 0, in work (reference) seconds; 0 means no checkpoint: a preemption loses all progress |
| `max_wait_s` | decimal | may be empty (no wait target); otherwise the target for the wait before the first start |

Decimal time columns (`submit_s`, `runtime_s`, `estimate_s`, `checkpoint_interval_s`, `max_wait_s`) are plain decimals: digits, optionally a point and one to three digits. Signs, exponents, and a fourth decimal are rejected. Writers use the shortest form (`12`, `12.5`, `0.001`). Line endings LF or CRLF; fields may be quoted as in RFC 4180.

A loader rejects: a missing or different header (line 1), a wrong field count, an unparsable or out-of-range value, an empty required field, a duplicate `job_id`, and a row whose `submit_s` is smaller than the previous row's. Each error names the 1-based line number. A header with no rows is a valid empty trace; a zero-byte file is invalid.

**Manifest.** Beside `<name>.csv` lies `<name>.manifest.json`:

```json
{
  "schema_version": 1,
  "generator": {"name": "gcs-gen", "version": 1, "params": {"...": "the full generator configuration"}},
  "seed": 7,
  "jobs": 3000,
  "duration_s": 320123.456,
  "content_sha256": "<hex SHA-256 of the CSV bytes>"
}
```

`duration_s` is the last submit time. A replayer checks `schema_version`, `content_sha256`, and `jobs` before use.

## 3. Cluster configuration v1

JSON, unknown fields rejected:

```json
{
  "schema_version": 1,
  "name": "homogeneous",
  "classes": [{"name": "a100", "speed": 1.0}],
  "node_groups": [{"count": 4, "prefix": "r0-n", "rack": "r0", "class": "a100", "gpus": 8, "cpus": 128, "mem_gb": 1024}],
  "nodes": [{"name": "spare", "rack": "r9", "class": "a100", "gpus": 8, "cpus": 128, "mem_gb": 1024}],
  "cross_node_factor": 1.1,
  "cross_rack_factor": 1.25,
  "restart_overhead_s": 120,
  "preempt_grace_s": 0,
  "sources": {"speed": "where each parameter comes from (assumed or derived, with the source)"}
}
```

- `classes`: unique non-empty names, `speed` > 0 relative to the reference class (speed 1.0).
- `node_groups` expand to `count` nodes named `<prefix><index>` with a two-digit index from 00; `nodes` lists single nodes. Together they must give at least one node.
- Every node has a unique non-empty name, a non-empty rack, a known class, and positive `gpus`, `cpus`, and `mem_gb`.
- `1 <= cross_node_factor <= cross_rack_factor`; `restart_overhead_s >= 0` and `preempt_grace_s >= 0` with at most three decimals.
- Nodes are ordered by (rack, name); that order is the node index used everywhere (views, placements, tie-breaks).
- The **work capacity** of a cluster is the sum over nodes of GPUs times class speed (reference-GPU-seconds per second). Offered load is defined against it.

## 4. Simulator model (summary)

The binding model is described in `docs/simulator.md`: whole-millisecond virtual time, gang placement on nodes, class constraints, speed and topology factor (`rate = min speed / f`), preemption with checkpoint loss and restart overhead, no kill at the estimate, the overrun rule for estimated ends, termination and deadlock, and the accounting identities. Memory is compared in whole MB (`round(mem_gb * 1000)`).

## 5. Metrics

All feasible jobs of a run complete (the run drains). The **measurement window** is the feasible jobs in trace (submit) order without the first and the last `floor(n / 10)` jobs; those jobs still run. The window is defined by the trace, so it is the same job set for every policy. Infeasible jobs are excluded from everything.

Per job (seconds): `wait` = first start − submit; `jct` = completion − submit; `queue_total` = all pending time, including after preemptions; `ideal` = `runtime_s` / speed of the fastest class the job may use in this cluster (its class if constrained); bounded slowdown `bsld = max(1, jct / max(ideal, tau))` with `tau = 10 s`; weighted flow `wflow = jct / max(ideal, tau)` (unclamped). A job with `max_wait_s` meets its target when `wait <= max_wait_s`.

Per run (window jobs unless stated otherwise; column names as in `runs.csv`):

| Column(s) | Definition |
|---|---|
| `wait_*`, `jct_*`, `bsld_*` (`mean`, `p50`, `p95`, `p99`) | over window jobs; percentiles are nearest rank: the `ceil(p n)`-th smallest |
| `tenant_<t>_*`, `prio_<p>_*` | `wait_`, `jct_`, `bsld_` (`mean`, `p50`, `p95`, `p99`) per tenant and per priority, plus the job count `jobs` |
| `wflow_mean`; `wflow_sum_all` | mean weighted flow of window jobs; sum over **all** feasible jobs (the MILP objective) |
| `large_wait_mean`, `large_wait_p95` | wait of window jobs with `gpus * workers >= 8` |
| `utilization` | allocated GPU-seconds / (total GPUs x span), over the window's time span: submit time of the first to that of the last window job (batch scenarios, where that span is empty: first submission to last completion). GPUs held during restart overhead or a grace period count as allocated |
| `goodput_ratio` | sum of useful work (`runtime_s * gpus * workers`) over sum of speed-weighted GPU-seconds consumed, window jobs (1 = no waste) |
| `wasted_gpu_hours` and its parts `lost_`, `overhead_`, `topology_`, `slack_` (incl. ms rounding), `grace_gpu_hours` | consumed − useful in reference-GPU-hours, decomposed as in `docs/simulator.md` §7 |
| `preemptions` | preemptions of window jobs |
| `frag_blocked_frac` | fraction of the span in which some pending job fits the free GPU, CPU, memory, and class totals of the cluster but cannot be placed (state after the policy acted at each instant, held until the next instant). Workers of a job are identical, so "placeable" is exact: the sum over nodes of `min(free_gpus / gpus, free_cpus / cpus, free_mb / mb)` reaches `workers` |
| `stranded_frac` | integral over the span of the expected stranded free GPUs divided by the integral of free GPUs. A node with `f` free GPUs strands `sum over g of p(g) * (f mod g)` GPUs (`f` when `f < g`), where `p(g)` is the per-worker GPU-size distribution of the whole trace (weighted by workers). Inspired by, and not identical to, the fragmentation measure of the FGD paper ("Beware of Fragmentation", Weng et al.) |
| `jain_bsld`, `tenant_worst_best_ratio` | Jain's index `(sum x)^2 / (n sum x^2)` over the per-tenant mean bounded slowdown (tenants with at least one window job), and the ratio of the worst to the best tenant mean |
| `slo_attainment`, `slo_violation_rate` | jobs meeting `max_wait_s` / jobs with a target (empty when no window job has one); violation = 1 − attainment |
| `makespan` | last completion − first submission over all feasible jobs (meaningful for batch scenarios) |
| `reservations`, `broken_reservations` | window jobs for which a backfilling policy reported a shadow time as head of its queue, and those whose first start after that report came later than the **first** shadow time reported |
| `invocations` | policy invocations |
| `failure_kills`, `node_downtime_gpu_hours` | runs of window jobs ended by a node failure (Tier 2, `docs/simulator.md` §12); GPU-hours of failed nodes under repair within the span |
| `quota_satisfaction`, `borrowed_gpu_hours`, `jain_weighted_bsld` | quota scenarios only (Tier 2), with nominal quotas `floor(share * total GPUs)` and integrals over the span: served entitled demand `sum_t ∫ min(usage_t, demand_t, quota_t)` over `sum_t ∫ min(demand_t, quota_t)` (usage = allocated GPUs of tenant `t`, demand = allocated plus pending GPUs); `∫ max(0, usage_t − quota_t)` summed over tenants in GPU-hours; the quota-weighted Jain index `(sum w b)^2 / (sum w * sum w b^2)` over the per-tenant mean bounded slowdown `b` with the shares as weights `w` |
| `wall_*` (separate file `wall.csv`) | decision wall time per invocation (mean, P95, max); for solver-based policies the solver calls, the wall-clock solve time (`wall_solve_*`) and the process CPU time of the solve (`wall_solve_cpu_*`, solver worker plus policy process), capped and non-optimal solves, the largest gap. Non-deterministic; never used in comparisons |

**Objective** (tuning and summary score, column `J`):

```text
J = alpha * P95(bsld)/R_s + beta * P95(wait)/R_w + gamma * (1 - goodput_ratio) + delta * slo_violation_rate + epsilon * (1 - jain_bsld)
```

`R_s` and `R_w` are the mean over the tuning seeds of P95(bsld) and P95(wait) of `fifo+first_fit+none` in the same scenario (`R_w` floored at 1 s), computed once and recorded in `configs/tuned/`. A weight is 0 when its term does not apply (no wait targets, a single tenant), and in a scenario where the J-term check on the tuning seeds (`go run ./cmd/benchmark -jterm-check`, `configs/tuned/jterm_check.csv`) found the term no more variable across policies than across seeds (scenario field `zero_weights`). Means are never used in J.

## 6. External-policy protocol v1

The simulator (Go) starts the policy as a child process without a shell — by default `python -m harness.policy_server` in the repository root, the interpreter taken from `$PYTHON` or `python` on `PATH` — and talks to it over stdin and stdout: **one JSON object per line, UTF-8, LF only**. Pipes are binary on both sides (Python reads `sys.stdin.buffer` and writes `sys.stdout.buffer`; the child gets `PYTHONUTF8=1`), so no newline translation or code-page conversion happens. stdout carries protocol lines only; the policy logs to stderr, which the simulator copies into a log file.

Encoding rules: times are decimal seconds with at most three decimals (exact milliseconds); memory is in whole MB (`mem_mb`); rates, speeds, and work are JSON floats (shortest round-trip form); placements name nodes; optional values are `null`. Unknown fields in a reply are an error.

**Messages.**

1. `hello` (simulator → policy) and its reply:

```json
{"type":"hello","protocol_version":1,"policy":{"name":"fifo+first_fit+easy","params":{}},"seed":7,
 "cluster":{"classes":[{"name":"a100","speed":1}],"racks":["r0"],"cross_node_factor":1.1,"cross_rack_factor":1.25,
            "restart_overhead_s":120,"preempt_grace_s":0,
            "nodes":[{"name":"r0-n00","rack":"r0","class":"a100","speed":1,"gpus":8,"cpus":128,"mem_mb":1024000}]}}
{"type":"hello","name":"fifo+first_fit+easy","version":"gcs-harness 1"}
```

2. `schedule` (one per policy invocation; `seq` counts from 1) and its reply. `history_new` holds only the completions since the previous `schedule` message; the policy accumulates them.

```json
{"type":"schedule","seq":1,"view":{"now_s":12.5,
 "nodes":[{"name":"r0-n00","free_gpus":4,"free_cpus":80,"free_mem_mb":640000,"running":[{"job_id":"j000001","workers":1}]}],
 "running":[{"job_id":"j000001","tenant":"t0","user":"t0-u1","priority":1,"gpus":4,"workers":1,"cpus":48,"mem_mb":384000,
             "gpu_class":"","topology":"any","submit_s":0,"first_start_s":0,"run_start_s":0,"overhead_s":0,
             "placement":[{"node":"r0-n00","workers":1}],"rate":1,"estimate_s":600,"retained_at_start_s":0,
             "work_done_s":12.5,"est_remaining_work_s":587.5,"est_end_s":600,"preemptible":true,
             "checkpoint_interval_s":1800,"preemptions":0}],
 "pending":[{"job_id":"j000002","tenant":"t1","user":"t1-u0","submit_s":12.5,"priority":4,"gpus":8,"workers":1,"cpus":96,
             "mem_mb":768000,"gpu_class":"","topology":"any","estimate_s":3600,"wait_s":0,"retained_s":0,"max_wait_s":7200,
             "preemptible":false,"checkpoint_interval_s":1800,"preemptions":0,"started":false}],
 "tenants":[{"tenant":"t0","gpu_seconds":50,"running_gpus":4,"running_cpus":48,"running_mem_mb":384000}],
 "history_new":[]}}
{"type":"decision","seq":1,"actions":[{"op":"start","job_id":"j000002","placement":[{"node":"r0-n01","workers":1}]}],
 "wake_at_s":null,"reservations":[],"solver":null}
```

   Reply fields: `actions` — an ordered list of `{"op":"start","job_id","placement":[{"node","workers"}]}` and `{"op":"preempt","job_id"}`; `wake_at_s` — optional time at which the policy wants to be invoked again; `reservations` — optional `[{"job_id","shadow_s"}]` (shadow times of backfilling reservations, used only for the reservation check); `solver` — optional `{"status","solve_s","cpu_s","gap","capped"}` for solver-based policies: `solve_s` wall-clock and `cpu_s` process CPU seconds of the solve (reported only in `wall_` columns); `null` when the invocation solved nothing.

   A node under repair after a failure (Tier 2) appears with zero free GPUs, CPUs, and memory and no running jobs; the policy needs no other signal.

3. `bye` (simulator → policy) and its reply `{"type":"bye"}`; the policy then exits with status 0.

At any point the policy may answer `{"type":"error","message":"..."}` instead; the simulator aborts the run with that message.

**Rules.**

- The policy is a pure function of the messages it received and its seed. Its own random draws use `numpy.random.default_rng(SeedSequence([seed, fnv1a64(policy name)]))`.
- Timeouts are wall-clock and generous: 60 s per call by default (configurable). A timeout, a crash (stdout closed), a malformed line (not JSON, an unknown field, a wrong type, more than three decimals in a time), a reply of the wrong type or `seq`, an unknown node, or an invalid action aborts the run with an error that names the policy and includes the tail of its stderr. There is no silent fallback.
- The simulator closes the child on every exit path (normal end with `bye`, error, timeout, panic, interrupt) and reaps it; a test asserts that no child is left.
- Solver limits inside a policy are deterministic work limits (node limits, relative gap). A wall-clock safety cap is allowed; a run in which it was hit is marked `solver_capped` (`wall_solver_capped > 0`) and is left out of byte-identical claims. The harness's MILP policies solve in a separate worker process and abandon a solve after a hard cap of 1.5 x their time limit + 5 s (status `wall_cap`), so one call stays below the 60 s protocol timeout.

## 7. Policy semantics

A scheduling policy has a name and a JSON parameter object (`{"name": "...", "params": {...}}`), takes the read-only view plus private internal state, is deterministic given its seed, and does not depend on the simulator, HTTP, or the wall clock. One factory (`internal/policy.New`) builds the policies for the simulator, the benchmark, and the CLI; `py:<name>` selects a policy of the Python harness through the protocol.

**View** (Go: `internal/api.View`; wire form in §6): the time; per node its name, rack, class, speed, total and free GPUs, CPUs, and memory, and the jobs running there with their worker counts; the running jobs (request, placement, start of the current run, restart overhead, rate, estimate, retained work, work done, estimated remaining work, estimated end — see the overrun rule in `docs/simulator.md` §5 — preemptibility, checkpoint interval); the pending jobs sorted by priority (high first), submit time, `job_id`, with request, estimate, pending time so far, retained work, `max_wait_s`, and whether the job ran before; per-tenant usage (GPU-seconds so far, running GPUs, CPUs, memory); the history of completed jobs (user, submit time, true run time) for predictors. Never the true run time of an unfinished job.

**Actions** apply in order (a preempt can free resources for a later start in the same list). The simulator checks every action; an invalid action aborts the run with an error naming the policy and the action. A decision may also carry a wake-up time and reservation reports.

**Composition of the Go baselines.** A policy name is `order+placement+backfill[+preempt][+quota|+quota_strict][+reclaim]`:

- **Order** — `fifo`: submit time, then `job_id` (priority ignored); `priority`: effective priority (high first) = priority + `aging_per_hour` x pending hours, then submit time, `job_id`; `shortest_estimate`: remaining estimated work (estimate − retained), then submit time, `job_id`; `drf`: dominant resource fairness over tenants across GPUs, CPUs, and memory, built by progressive filling (repeatedly the next job, in pending order, of the tenant with the lowest dominant share, ties by tenant name; the job's demand is added to that share as if it started).
- **Placement** — `first_fit`: nodes in index order, as many workers per node as fit; `best_fit`: worker by worker, the node left with the fewest free GPUs; `least_fragmentation`: worker by worker, the node where the worker increases the expected stranded GPUs (`sum p(g) (free mod g)`, with `p` the size distribution of the jobs the policy has seen) the least, ties by best fit; `topology_aware`: the smallest span first — one node (best fit), else one rack (the fitting rack with the fewest free GPUs; nodes with the most room first), else the cluster (racks with the most room first, then the rack with the most room on one node). Ties go to the lower node index. Every placement respects classes, CPUs, and memory, and is complete: it finds a placement whenever one exists (workers are identical). `hetero_ect` (Tier 2) is the one deliberate exception: for a job without a class constraint it estimates, per class, the completion time `ECT(c) = wait(c) + estimated remaining work / speed(c)`, where `wait(c)` is 0 when the job fits on class `c` now, else the time until the running jobs on class-`c` nodes, released in order of estimated end, leave room for it; it picks the class with the smallest ECT (ties: faster class, then class name), places first-fit within that class, never mixes classes, and leaves the job pending when the chosen class has no room yet (it waits for faster GPUs when that is expected to finish sooner). Class-constrained jobs are placed first-fit on their class.
- **Backfill** — `none`: start jobs in order and stop at the first one that does not fit. `easy` (EASY backfilling): start jobs in order while they fit; the first job that does not fit is the **head**; its **shadow time** is found by releasing the running jobs (and the jobs just started) in order of estimated end (ties by `job_id`) and testing the placement routine after each release; the reservation holds the nodes that placement chose; the **extra** capacity is the capacity free at the shadow time (every job ending at that instant released) minus the reservation. A later job starts now if its estimated end (now + restart overhead if it ran before + estimated remaining work / rate of its placement) is not after the shadow time, or if it can be placed within `min(free now, extra)` (then it consumes extra). An overrun job's estimated end is "now" (`docs/simulator.md` §5): the reservation guarantee does not hold then, and broken reservations are counted. `conservative`: every queued job (at most `depth` per invocation, default 64) gets the earliest reservation at which its placement fits for its whole estimated duration on the capacity profile left by the running jobs and the earlier reservations; reservations at "now" start now; overrun jobs are assumed to end 1 ms after now. `easy_predicted`: EASY in which the release times of the shadow computation use predicted run times — the mean of the user's last `history` (default 2) completed run times (Tsafrir, Etsion, and Feitelson), never above the estimate, the estimate without history; a running job that outlived its prediction falls back to its estimate (then the overrun rule). Candidates are still tested with user estimates. `easy_oracle`: EASY with the true run times everywhere — an **oracle**, benchmark only, never deployable. `risk` (Tier 2, parameters `quantile` default 0.9 and `min_history` default 5): EASY in which every estimated duration — release times in the shadow computation and the backfill test of candidates — is the estimate times the `quantile`-quantile (nearest rank) of the completed jobs' `runtime / estimate` ratios of the same user when the user has at least `min_history` completions, else of all users, else 1; the prediction can exceed the estimate (users who underestimate get longer predictions). A running job that outlived its prediction falls back to its estimate, then the overrun rule.
- **Preemption** — `+preempt` (priority preemption, parameters `min_gap` default 1 and `max_victims` default 8): after the backfill pass, each pending job in order that was not started and does not fit may preempt running preemptible jobs with priority at most its own minus `min_gap`. Victims are taken in order of the work they would lose (`gpus * workers * (work done − work retained by the checkpoint rule)`), then lower priority, then latest start, then `job_id`, until the job fits (at most `max_victims`); then each victim, last added first, is dropped if the job still fits without it. Requires `preempt_grace_s = 0`.
- **Quotas** (Tier 2; built from the concepts of the Kueue documentation — nominal quota, cohort borrowing, reclaim — and not claimed to behave like Kueue). The scenario gives each tenant a share; the nominal quota is `floor(share * total GPUs)`, shares sum to at most 1. `+quota_strict`: a job is admitted only when its tenant's running GPUs plus the job's GPUs stay within the nominal quota (a job larger than the whole quota is admitted when its tenant runs nothing, else it could never run). `+quota`: jobs beyond the quota are admitted too (borrowing unused quota of the other tenants; the cohort is all tenants). In both, the jobs within quota at the start of the invocation are considered before the others (stable partition of the base order), and a job that is not admitted is skipped and does not block other tenants. `+reclaim` (needs a quota suffix): a pending job within its tenant's quota that does not fit may preempt preemptible jobs of tenants running above their quota, victims chosen as for `+preempt` (least lost work, at most 8).

`plan` replays a fixed plan (`{"starts": [{"job_id","start_s","placement":[{"node","workers"}]}]}`), asking for a wake-up at each planned start; a missed start is an error. It checks the offline MILP (check 7).
