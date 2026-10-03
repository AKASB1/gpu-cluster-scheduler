# MILP models

The optimization-based policies live in the Python harness (`harness/milp/`, `harness/policies/milp_rolling.py`) and are solved with `scipy.optimize.milp`, which wraps the HiGHS solver. Everything here is **simulated**; the offline models are clairvoyant references, not deployable schedulers.

## 1. Objective

Minimize the **sum of weighted flow** of the instance's jobs:

```text
sum_j wflow_j = sum_j w_j * (C_j - submit_j),   w_j = 1 / max(ideal_j, tau),   tau = 10 s
```

where `C_j` is the completion time and `ideal_j = runtime_j / speed of the fastest class j may use` (docs/contracts.md §5). This is exactly the `wflow_sum_all` column of the simulator results, so MILP objectives and simulated policies are compared on the same quantity.

## 2. Time grid

Time is a grid of slots of length `Delta` (seconds, a parameter). A job's processing time on class `c` (speed `s_c`) is `p_jc = ceil(runtime_j / (s_c * Delta))` slots, and its release slot is `r_j = ceil(submit_j / Delta)`.

- **Whole-slot data** (every `runtime_j / s_c` and `submit_j` a multiple of `Delta`): no rounding happens. The time-indexed model is then *exact*: with integer data there is an optimal non-preemptive schedule in which every start is a release time or a completion time of another job (left-shift argument), hence on the grid. The S8 instances are generated on the grid (`slot_grid` in the generator config) and the code refuses to report a **bound** for an instance that is not whole-slot.
- **Rounded data** (online use in `milp_rolling`): durations and release slots are rounded **up**, so every schedule of the rounded problem is feasible for the original one. The optimum is then exact for the rounded problem. Note that rounding up makes it an *approximation from above*, not a lower bound of the original problem (one job of 1.5 slots: original optimum 1.5, rounded 2); a valid lower bound would need rounding down. We therefore never call a rounded optimum a bound.

## 3. Offline models (clairvoyant: true run times, known submit times, no preemption)

### 3.1 Pool model (lower bound)

Binary `s[j,t,c] = 1` iff job `j` starts in slot `t` on class `c` (all of its `G_j = gpus_j * workers_j` GPUs from class `c`; classes allowed by `gpu_class`).

```text
min   sum_{j,t,c} w_j * ((t + p_jc) * Delta - submit_j) * s[j,t,c]
s.t.  sum_{t,c} s[j,t,c] = 1                                        for every job j
      sum_j sum_{t' in (tau - p_jc, tau]} G_j * s[j,t',c] <= Cap_c       for every class c and slot tau
      s[j,t,c] = 0 for t < r_j;  s binary
```

`Cap_c` is the number of GPUs of class `c`. The model **ignores** node boundaries (a gang may be split arbitrarily), the gang spread over nodes, the topology factor, mixed-class placements, CPU, and memory. Every schedule the simulator can produce without preemption, on a **single-class** cluster with whole-slot data, is a feasible solution of the pool model with an equal or larger objective (topology slowdowns only lengthen jobs). So the pool optimum is a **lower bound** for every non-preemptive policy on such instances (the S8 medium set). On a multi-class cluster a simulated job may mix classes, which the pool model cannot represent; we do not claim the bound there.

### 3.2 Node model (exact for single-worker jobs)

For instances whose jobs all have `workers = 1`, binary `s[j,t,n]` (job `j` starts in slot `t` on node `n`; nodes of a class `j` may use, with enough GPUs, CPUs, and memory for the job):

```text
min   sum_{j,t,n} w_j * ((t + p_jn) * Delta - submit_j) * s[j,t,n]
s.t.  sum_{t,n} s[j,t,n] = 1                                         for every job j
      sum_j sum_{t' in (tau - p_jn, tau]} gpus_j * s[j,t',n] <= GPUs_n    for every node n and slot tau
      (the same rows for CPUs and memory when some job requests them)
```

A single worker always has span `node`, so there is no topology factor; with whole-slot data the model is **exact**: its optimum is the best non-preemptive schedule of the instance. It leaves out nothing the simulator models for such jobs except preemption.

### 3.3 Horizon and variable pruning

- **Horizon.** `T = max_j r_j + sum_j max_c p_jc`. After the last release, an active schedule always runs at least one job, so every active schedule (and an optimal one is active) ends by `T`.
- **Upper bound pruning.** A feasible schedule from a greedy heuristic (serial schedule generation in order of `w_j / p_j`, each job on the node or class where it completes earliest, at its earliest feasible start there, ties by the lower node or class index) gives an upper bound `UB`. In any schedule with objective `<= UB`, `w_j (C_j - submit_j) <= UB - sum_{i != j} w_i p_i^min Delta`, which bounds `C_j` and removes all later start variables. This keeps every optimal solution.

### 3.4 Solving and repeatability

`scipy.optimize.milp` with deterministic limits: a node limit and a relative gap (`mip_rel_gap`, default 0 for the reference solutions), plus a wall-clock safety cap (`time_limit`). The HiGHS build vendored by SciPy 1.13 does not always stop at its time limit (one sweep instance ran 289 s against a 15 s limit), so the offline solver CLI runs the solves in a worker process and terminates it at `1.5 x time_limit + 10 s`; such a task is reported as `wall_cap` (no solution, no bound). A solve that stops at the cap is marked `capped` and is left out of byte-identical claims; its incumbent is reported with the dual bound and the gap. Results report status, objective, dual bound, gap, nodes, and the solve time (a `wall_` value). Two solves of the same instance in fresh processes must agree on status, objective, and schedule (tested).

### 3.5 Exhaustive search (check 7)

For at most 6 jobs, `harness/milp/brute.py` enumerates every job order and every node (or class) choice and builds each schedule with the serial schedule generation scheme (each job at its earliest feasible start on its node given the jobs already placed). For a fixed assignment of jobs to nodes this enumerates all active schedules, and an optimal schedule of a regular objective is active, so the enumeration finds the optimum. Branch-and-bound pruning (partial cost + each remaining job's release-plus-processing lower bound) and symmetry breaking between identical empty nodes keep it fast; neither removes an optimal schedule.

Check 7 (tests): on random whole-slot instances of at most 6 jobs, the MILP optimum equals the exhaustive optimum (node and pool models); the MILP schedule, replayed in the simulator as a fixed plan (`plan` policy), reproduces the MILP objective; and no Go baseline beats the optimum. A violation means a bug in one of them.

## 4. Online model: `milp_rolling`

A policy of the protocol (`py:milp_rolling`). At each invocation:

1. Take the first `K` pending jobs in FIFO order (submit time, `job_id`).
2. Build a pool model over the next `H` slots of length `Delta`, starting now. Durations come from the **user estimates** (never the true run times): `p_jc = ceil((estimate_j - retained_j) / (s_c * Delta))`, plus the restart overhead in slots for a job that ran before. Running jobs are fixed capacity profiles: their GPUs of class `c` are busy for `ceil((est_end - now) / Delta)` slots (at least one slot for an overrun job). The capacity of class `c` counts the GPUs that are free or held by running jobs; a node under repair and GPUs held during a preemption grace period are left out (the same in `scenario_milp`). Weights use the estimate: `w_j = 1 / max(estimate_j / fastest speed, tau)`.
3. A job may also be **deferred** beyond the horizon (variable `d_j`, cost `w_j * (H + min_c p_jc) * Delta`), so the model is always feasible. Start slots range over `0..H-1`; capacity rows cover slots `0..H-1` (a job may run past the horizon).
4. Solve with the deterministic limits (`node_limit`, `mip_rel_gap`) and the safety cap; start, in FIFO order, the jobs the plan starts in slot 0, each through the first-fit placement routine, skipping those that do not place (the pool model ignores node boundaries).
5. Re-plan at the next invocation.

Parameters (tuned like any other policy, same budget): `K`, `H`, `slot_s` (`Delta`), `mip_rel_gap`, `node_limit`. The policy solves in a long-lived worker process with a hard cap of `1.5 x time_limit + 5 s` (status `wall_cap`, the plan of that invocation is empty and the work-conserving safeguard applies), so one call cannot exceed the 60 s protocol timeout even when HiGHS overruns its time limit. It reports its solver status, gap, wall-clock solve time, and process CPU time (worker plus policy process) in the decision (`wall_` columns), so decision times can be compared with the decision interval (the mean time between invocations).

## 5. Experiments

- **Optimality gap** (S8): every Go baseline (exact estimates, so the comparison is fair) on at least 50 small instances (at most 12 single-worker jobs on at most 4 nodes; node-model optimum) and 50 medium instances (30 to 60 jobs, single-class cluster; pool-model lower bound). Gap = `(policy - reference) / reference`; on the medium set it is a gap to a lower bound, so it over-states the true gap.
- **Solve time** against the number of jobs and the grid size: where the MILP stops being tractable under the node limit and the safety cap.
- **Online** (S9): `milp_rolling` against the best baselines at a size where it solves within its decision interval. A result in which it does not pay off is reported as such.

## 6. Stochastic model: `scenario_milp` (Tier 2)

A two-stage stochastic reservation with recourse (`harness/policies/scenario_milp.py`, a policy of the protocol, `py:scenario_milp`). At each invocation:

1. Take the first `K` pending jobs in FIFO order (the candidates) and draw `S` run-time **scenarios**. A scenario multiplies each job's user estimate by a ratio `runtime / estimate` drawn from the empirical distribution of the completion history: the user's own ratios when the user has at least `min_history` completed jobs, otherwise all users' ratios, and ratio 1 when there is no history. For a running job the draw is conditioned on the work already done, and for a preempted pending job on its retained work (only ratios that leave work remaining). Draws come from `numpy.random.default_rng(SeedSequence([seed, fnv1a64("scenario_milp")]))`, so the policy is a pure function of its messages and its seed.
2. **First stage**: binary `x_j` — job `j` starts now (identical in every scenario: non-anticipativity). **Second stage**, per scenario `s`: binary `y[j,t,c,s]` (start slot `t` on class `c`, with `y[j,0,c,s]` summing to `x_j`) and `d[j,s]` (deferred beyond the horizon). Running jobs hold their GPUs for a scenario-specific number of slots.
3. With the loss of a scenario `L_s = sum_j w_j (C_js - now)` (weighted flow over the horizon, deferred jobs charged `w_j (H + min_c p_jcs) Delta`), the objective is the expectation plus `lambda` times the conditional value at risk at level `alpha`, linearized as in Rockafellar and Uryasev:

```text
min  (1/S) sum_s L_s + lambda * ( eta + 1/((1 - alpha) S) * sum_s u_s )
s.t. u_s >= L_s - eta,  u_s >= 0                                  for every scenario s
     sum_{t,c} y[j,t,c,s] + d[j,s] = 1                             for every job j, scenario s
     sum_c y[j,0,c,s] = x_j                                        for every job j, scenario s
     sum_j sum_{t' in (tau - p_jcs, tau]} G_j y[j,t',c,s] <= Cap_c - busy_s(c, tau)   for every s, c, tau < H
     x, y, d binary; eta free; u >= 0
```

4. Start the jobs with `x_j = 1`, in FIFO order, through the first-fit placement routine (skipping those that do not place); the same work-conserving safeguard as `milp_rolling` applies.

Parameters (tuned with the same budget as every other policy): `k`, `h`, `slot_s`, `scenarios`, `lambda`, `alpha`, `min_history`, `mip_rel_gap`, `node_limit`, and the wall-clock safety cap `time_limit_s`. The model has `K * H * S` second-stage binaries per class; it is meant for the small online setting of S11, not for the 128-GPU cluster.

The Go policy `risk` backfilling (EASY with a quantile of the same empirical ratio distribution, docs/contracts.md §7) is the non-solver counterpart.

## 7. Solver cross-check (optional, Tier 2)

The default solver of every committed result is the HiGHS 1.2.0 that SciPy 1.13.1 vendors, whose time limit proved unreliable (section 3.4). `scripts/solver_crosscheck.py` re-solved all S8 tasks (50 small, 50 medium, the 63 sweep instances; the same instances as the evaluation) with the same formulation (`harness.milp.model.formulate`) and three solvers, one instance at a time and interleaved, gap 0, one thread requested, the task's time limit, and the hard cap of section 3.4. It runs only in a separate throwaway environment with the extra packages; nothing on the default path, in CI, or behind any README number depends on it. Results: `benchmarks/results/solver_crosscheck/` (`summary.md`, per-solve `crosscheck.csv`, `manifest.json`).

| Solver | Needs | Optimal objectives vs the default | S8 medium median solve | Sweep: largest job count all solved at slot 300 / 150 / 60 s |
|---|---|---|---|---|
| SciPy 1.13.1 (vendored HiGHS 1.2.0) | nothing (default) | — | 2.98 s | 24 / 16 / 8 |
| highspy 1.15.1 (current HiGHS) | `pip install highspy` | equal on every instance both solved (max relative difference 1.2e-16) | 3.30 s | 32 / 24 / 16 |
| Gurobi 12.0.2 | `gurobipy` and a license | equal on every instance both solved | 0.40 s | 32 / 32 / 32 |

Wall-clock values from a shared workstation while other benchmark processes ran (median ratios to the default: highspy 1.3-1.7x slower on the instances both solve to optimality, Gurobi 0.15-0.2x on S8); they are indicative only. The tractability limit of the solve-time figure is therefore a property of the default solver: a current HiGHS doubles the job count at the finest grid, and Gurobi solved every sweep instance within the 15 s limit.
