"""Writes configs/experiments/benchmark.json (the frozen scenario list). Run from the repository root:
    python scripts/make_experiment.py
The JSON file is the source of truth; this script only documents how it was built."""
import json

ORDER = ["fifo+{placement}+{backfill}", "priority+{placement}+{backfill}", "shortest_estimate+{placement}+{backfill}", "drf+{placement}+{backfill}"]
BACKFILL = ["{order}+{placement}+none", "{order}+{placement}+easy", "{order}+{placement}+conservative", "{order}+{placement}+easy_predicted", "{order}+{placement}+easy_oracle"]
PLACEMENT = ["{order}+first_fit+{backfill}", "{order}+best_fit+{backfill}", "{order}+least_fragmentation+{backfill}", "{order}+topology_aware+{backfill}"]
ALL = ORDER + BACKFILL + PLACEMENT
POP = [900, 1800, 3600, 7200, 14400, 28800, 43200, 86400, 172800]
EST = {"exact": {"model": "exact"},
       "lognormal": {"model": "lognormal", "mu": 0, "sigma": 0.5},
       "user": {"model": "user", "p_exact": 0.15, "p_under": 0.10, "under_min": 0.5, "under_max": 0.95, "acc_min": 0.05, "popular_s": POP}}
H, HET = "configs/clusters/homogeneous.json", "configs/clusters/heterogeneous.json"
BASE = "configs/workloads/base.json"
FRAG_SIZES = [{"weight": 0.35, "gpus": 1, "workers": 1}, {"weight": 0.25, "gpus": 2, "workers": 1}, {"weight": 0.20, "gpus": 4, "workers": 1},
              {"weight": 0.08, "gpus": 8, "workers": 1}, {"weight": 0.04, "gpus": 4, "workers": 2, "topology": "node", "topo_frac": 1.0},
              {"weight": 0.05, "gpus": 8, "workers": 2, "topology": "rack", "topo_frac": 0.7},
              {"weight": 0.03, "gpus": 8, "workers": 4, "topology": "rack", "topo_frac": 0.7}]
S9_SIZES = [{"weight": 0.48, "gpus": 1, "workers": 1}, {"weight": 0.15, "gpus": 2, "workers": 1}, {"weight": 0.15, "gpus": 4, "workers": 1},
            {"weight": 0.14, "gpus": 8, "workers": 1}, {"weight": 0.02, "gpus": 4, "workers": 2, "topology": "node", "topo_frac": 1.0},
            {"weight": 0.06, "gpus": 8, "workers": 2, "topology": "rack", "topo_frac": 0.7}]
BURST = {"mean_gap_s": 21600, "jobs": 30, "spread_s": 600, "priority": 9, "max_wait_s": 600}
S6_POL = ["priority+{placement}+{backfill}", "priority+{placement}+{backfill}+preempt", "fifo+{placement}+{backfill}"]

sc = []
for L in (60, 80, 95):
    sc.append({"id": f"s1-load{L}", "group": "S1", "note": "order, backfill, and placement comparison, exact estimates",
               "cluster": H, "workload": BASE, "overrides": {"arrival": {"load": L / 100}}, "policies": ALL})
for L in (80, 95):
    for e in ("exact", "lognormal", "user"):
        sc.append({"id": f"s2-{L}-{e}", "group": "S2", "note": "backfilling under estimate error",
                   "cluster": H, "workload": BASE, "overrides": {"arrival": {"load": L / 100}, "estimate": EST[e]}, "policies": BACKFILL})
sc.append({"id": "s3-stream", "group": "S3", "note": "fragmentation: small jobs plus 8-32 GPU gangs", "cluster": H, "workload": BASE,
           "overrides": {"arrival": {"load": 0.85}, "sizes": FRAG_SIZES}, "policies": PLACEMENT})
sc.append({"id": "s3-batch", "group": "S3", "note": "fragmentation, batch: every job at time 0 (makespan)", "cluster": H, "workload": BASE,
           "overrides": {"jobs": 300, "arrival": {"process": "batch", "load": None}, "sizes": FRAG_SIZES}, "policies": PLACEMENT})
sc.append({"id": "s4-hetero", "group": "S4", "note": "heterogeneous cluster with class-constrained jobs", "cluster": HET, "workload": BASE,
           "overrides": {"arrival": {"load": 0.8}, "class_constraints": [{"class": "a100", "frac": 0.15}, {"class": "v100", "frac": 0.15}]},
           "policies": PLACEMENT})
sc.append({"id": "s5-tenants", "group": "S5", "note": "multi-tenant with one heavy tenant", "cluster": H, "workload": BASE,
           "overrides": {"arrival": {"load": 0.9}, "tenants": [{"name": "t0", "weight": 0.55, "users": 8}, {"name": "t1", "weight": 0.15, "users": 8},
                                                               {"name": "t2", "weight": 0.15, "users": 8}, {"name": "t3", "weight": 0.15, "users": 8}]},
           "policies": ["fifo+{placement}+{backfill}", "priority+{placement}+{backfill}", "drf+{placement}+{backfill}"]})
for name, ck in (("none", 0), ("short", 600), ("long", 3600)):
    sc.append({"id": f"s6-ckpt-{name}", "group": "S6", "note": "high-priority bursts and preemption", "cluster": H, "workload": BASE,
               "overrides": {"arrival": {"load": 0.75}, "burst": BURST, "checkpoint_interval_s": ck, "estimate": EST["lognormal"]},
               "policies": S6_POL})
sc.append({"id": "s7-bursty", "group": "S7", "note": "bursty arrivals (gamma renewal, CV 4)", "cluster": H, "workload": BASE,
           "overrides": {"arrival": {"process": "gamma", "load": 0.8, "cv": 4}}, "policies": ALL})
sc.append({"id": "s7-diurnal", "group": "S7", "note": "diurnal arrivals (amplitude 0.8, period 24 h)", "cluster": H, "workload": BASE,
           "overrides": {"arrival": {"process": "diurnal", "load": 0.8, "amplitude": 0.8, "period_s": 86400}}, "policies": ALL})
sc.append({"id": "s8-small", "group": "S8", "kind": "milp-small", "note": "MILP gap, node-model optimum", "cluster": "configs/clusters/s8-small.json",
           "workload": "configs/workloads/s8-small.json", "policies": ALL, "seeds": list(range(1, 51)), "slot_s": 300, "time_limit_s": 120})
sc.append({"id": "s8-medium", "group": "S8", "kind": "milp-medium", "note": "MILP gap to the pool-model lower bound", "cluster": "configs/clusters/s8-medium.json",
           "workload": "configs/workloads/s8-medium.json", "policies": ALL, "seeds": list(range(1, 51)), "slot_s": 300, "node_limit": 20000, "time_limit_s": 60})
sc.append({"id": "s9-online", "group": "S9", "note": "online MILP against the best baselines", "cluster": "configs/clusters/s8-medium.json", "workload": BASE,
           "overrides": {"jobs": 200, "arrival": {"load": 0.85}, "sizes": S9_SIZES, "estimate": EST["lognormal"]},
           "policies": ["py:milp_rolling", "{order}+{placement}+{backfill}", "shortest_estimate+{placement}+{backfill}",
                        "{order}+{placement}+easy", "{order}+{placement}+none"], "seeds": list(range(1, 11))})
SEEDS10 = list(range(1, 11))


def variant(base, vid, assumption, scale, extra_over=None, **kw):
    b = next(s for s in sc if s["id"] == base)
    v = {"id": vid, "group": "SENS", "note": f"sensitivity of {base}: {assumption} x{scale}", "cluster": b["cluster"], "workload": b["workload"],
         "overrides": json.loads(json.dumps(b["overrides"])), "policies": b["policies"], "seeds": SEEDS10, "tuned_from": base,
         "assumption": assumption, "scale": scale}
    for k, val in (extra_over or {}).items():
        if isinstance(val, dict):
            v["overrides"].setdefault(k, {}).update(val)
        else:
            v["overrides"][k] = val
    v.update(kw)
    return v


sens = []
for s, f in ((0.5, "x0.5"), (2, "x2")):
    sens.append(variant("s2-80-lognormal", f"sens-s2-est-{f}", "estimate error (lognormal sigma)", s, {"estimate": {"sigma": 0.5 * s}}))
    sens.append(variant("s2-80-lognormal", f"sens-s2-topo-{f}", "topology penalty (f - 1)", s, penalty_scale=s))
for d, f in ((-10, "m10"), (10, "p10")):
    sens.append(variant("s2-80-lognormal", f"sens-s2-load-{f}", "offered load (points)", d, {"arrival": {"load": round(0.8 + d / 100, 2)}}))
for s, f in ((0.5, "x0.5"), (2, "x2")):
    sens.append(variant("s6-ckpt-short", f"sens-s6-est-{f}", "estimate error (lognormal sigma)", s, {"estimate": {"sigma": 0.5 * s}}))
    sens.append(variant("s6-ckpt-short", f"sens-s6-ovh-{f}", "restart overhead and checkpoint interval", s, {"checkpoint_interval_s": 600 * s}, overhead_scale=s))
    sens.append(variant("s6-ckpt-short", f"sens-s6-topo-{f}", "topology penalty (f - 1)", s, penalty_scale=s))
sc += sens

# Tier 2 (1): uncertainty-aware policies and distribution shift. The shift
# variants reuse the frozen tuning of their base scenario (error model fitted
# on one regime, evaluated under a worse one).
UNC = ["{order}+{placement}+easy", "{order}+{placement}+easy_predicted", "{order}+{placement}+risk",
       "{order}+{placement}+conservative", "{order}+{placement}+easy_oracle"]
USER_BAD = dict(EST["user"], p_under=0.20, acc_min=0.025)
for name, est, shifted, what in (("lognormal", EST["lognormal"], dict(EST["lognormal"], sigma=1.0), "lognormal sigma 0.5 -> 1.0"),
                                 ("user", EST["user"], USER_BAD, "user-style: underestimates 10% -> 20%, accuracy floor 0.05 -> 0.025")):
    base_id = f"s10-{name}"
    sc.append({"id": base_id, "group": "S10", "note": "uncertainty-aware backfilling (risk = EASY with a run-time quantile)",
               "cluster": H, "workload": BASE, "overrides": {"arrival": {"load": 0.9}, "estimate": est}, "policies": UNC})
    sc.append({"id": f"s10-shift-{name}", "group": "S10", "note": f"distribution shift: {what}; tuning frozen on {base_id}",
               "cluster": H, "workload": BASE, "overrides": {"arrival": {"load": 0.9}, "estimate": shifted}, "policies": UNC,
               "tuned_from": base_id, "assumption": f"estimate error shift ({what})", "scale": 2})
S11_POL = ["py:scenario_milp", "py:milp_rolling", "{order}+{placement}+risk", "{order}+{placement}+easy_predicted", "{order}+{placement}+easy"]
s9 = next(s for s in sc if s["id"] == "s9-online")
sc.append({"id": "s11-online", "group": "S11", "note": "two-stage stochastic MILP against the deterministic MILP and the baselines",
           "cluster": s9["cluster"], "workload": BASE, "overrides": json.loads(json.dumps(s9["overrides"])), "policies": S11_POL,
           "seeds": list(range(1, 11))})
shift11 = json.loads(json.dumps(s9["overrides"]))
shift11["estimate"] = dict(EST["lognormal"], sigma=1.0)
sc.append({"id": "s11-shift", "group": "S11", "note": "distribution shift: lognormal sigma 0.5 -> 1.0; tuning frozen on s11-online",
           "cluster": s9["cluster"], "workload": BASE, "overrides": shift11, "policies": S11_POL, "seeds": list(range(1, 11)),
           "tuned_from": "s11-online", "assumption": "estimate error shift (lognormal sigma 0.5 -> 1.0)", "scale": 2})

# Tier 2 (3): heterogeneity-aware placement, with and without class constraints.
HET_POL = ["{order}+first_fit+{backfill}", "{order}+best_fit+{backfill}", "{order}+topology_aware+{backfill}", "{order}+hetero_ect+{backfill}"]
sc.append({"id": "s12-hetero-free", "group": "S12", "note": "heterogeneous cluster, no class constraints: class-blind vs hetero_ect",
           "cluster": HET, "workload": BASE, "overrides": {"arrival": {"load": 0.8}}, "policies": HET_POL})
sc.append({"id": "s12-hetero-pinned", "group": "S12", "note": "heterogeneous cluster, 15% of jobs pinned to each class",
           "cluster": HET, "workload": BASE, "overrides": {"arrival": {"load": 0.8}, "class_constraints": [{"class": "a100", "frac": 0.15},
                                                                                                       {"class": "v100", "frac": 0.15}]},
           "policies": HET_POL})

# Tier 2 (4): quotas with cohort borrowing and reclaim; one heavy tenant, equal quotas.
QUOTA_TENANTS = [{"name": "t0", "weight": 0.55, "users": 8}, {"name": "t1", "weight": 0.15, "users": 8},
                 {"name": "t2", "weight": 0.15, "users": 8}, {"name": "t3", "weight": 0.15, "users": 8}]
ALL_PREEMPTIBLE = [{"priority": 1, "weight": 0.5, "preemptible_frac": 1.0},
                   {"priority": 4, "weight": 0.4, "preemptible_frac": 1.0, "max_wait_s": 7200},
                   {"priority": 8, "weight": 0.1, "preemptible_frac": 1.0, "max_wait_s": 900}]
sc.append({"id": "s14-quota", "group": "S14", "note": "quotas: one heavy tenant, equal nominal quotas, all jobs preemptible",
           "cluster": H, "workload": BASE, "overrides": {"arrival": {"load": 0.9}, "tenants": QUOTA_TENANTS, "priorities": ALL_PREEMPTIBLE,
                                                         "checkpoint_interval_s": 600},
           "quota_shares": {"t0": 0.25, "t1": 0.25, "t2": 0.25, "t3": 0.25},
           "policies": ["{order}+{placement}+{backfill}", "{order}+{placement}+{backfill}+quota_strict", "{order}+{placement}+{backfill}+quota",
                        "{order}+{placement}+{backfill}+quota+reclaim", "drf+{placement}+{backfill}"]})

# Tier 2 (5): node failures (MTBF 4 days per node, mean repair 2 h) across checkpoint intervals.
for name, ck in (("none", 0), ("short", 600), ("long", 3600)):
    sc.append({"id": f"s15-fail-ckpt-{name}", "group": "S15", "note": "node failures: jobs on a failed node are killed and requeued",
               "cluster": H, "workload": BASE, "overrides": {"arrival": {"load": 0.8}, "checkpoint_interval_s": ck},
               "failures": {"mtbf_s": 345600, "repair_s": 7200},
               "policies": ["{order}+{placement}+{backfill}", "fifo+{placement}+{backfill}"]})
# The J-term check on the tuning seeds (go run ./cmd/benchmark -jterm-check,
# configs/tuned/jterm_check.csv) found J_waste uninformative without
# checkpoints under failures (spread 0.0022 across policies, seed noise
# 0.0156): weight 0 there (benchmarks/README.md); goodput is still reported.
next(s for s in sc if s["id"] == "s15-fail-ckpt-none")["zero_weights"] = ["J_waste"]

# Tier 2 (2): replayed jobs of the Helios Venus cluster (CC-BY-4.0, downloaded by
# scripts/fetch_helios.sh; not committed) on a simulated cluster derived from the
# log's own capacity; synthetic user-style estimates; tuning on the week before.
sc.append({"id": "s13-helios-venus", "group": "S13", "note": "replayed jobs on a simulated cluster (Helios Venus, 3 days, synthetic estimates)",
           "cluster": "", "workload": "", "replay": {"format": "helios", "log": "benchmarks/outputs/helios/data/Venus/cluster_log.csv",
                                                    "capacity": "benchmarks/outputs/helios/data/Venus/cluster_gpu_number.csv",
                                                    "start": "2020-08-10 00:00:00", "tuning_start": "2020-08-03 00:00:00", "days": 3,
                                                    "estimate": EST["user"]},
           "policies": ["fifo+first_fit+easy", "{order}+{placement}+none", "{order}+{placement}+easy", "{order}+{placement}+easy_predicted",
                        "{order}+{placement}+risk", "fifo+{placement}+{backfill}", "drf+{placement}+{backfill}"],
           "seeds": list(range(1, 11))})

exp = {
    "name": "benchmark",
    "baseline": "fifo+first_fit+none",
    "tuning_seeds": [1001, 1002, 1003, 1004, 1005],
    "eval_seeds": list(range(1, 21)),
    "weights": {"alpha": 1, "beta": 1, "gamma": 1, "delta": 1, "epsilon": 1},
    "tuning": {"configurations": 8, "search_seed": 7},
    "defaults_scenario": "s1-load80",
    "scenarios": sc,
    "quick": {"scenarios": ["s1-load80", "s2-95-user", "s6-ckpt-short", "s8-small", "s9-online"], "seeds": [1, 2], "jobs": 300, "milp_instances": 3},
    "capacity": {"workload": BASE, "loads": [0.6, 0.7, 0.8, 0.9, 1.0], "tuned_from": "s1-load80", "target_p95_wait_s": 3600,
                 "policies": ["fifo+first_fit+none", "fifo+{placement}+{backfill}", "{order}+{placement}+{backfill}"],
                 "seeds": list(range(1, 11)), "min_nodes": 8, "max_nodes": 32},
    "sweep": {"cluster": "configs/clusters/s8-small.json", "workload": "configs/workloads/s8-small.json", "jobs": [4, 6, 8, 12, 16, 24, 32],
              "slots_s": [300, 150, 60], "instances": 3, "node_limit": 200000, "time_limit_s": 15},
    "key_metrics": ["wait_mean", "wait_p95", "wait_p99", "jct_mean", "jct_p95", "bsld_mean", "bsld_p95", "bsld_p99", "wflow_mean",
                    "wflow_sum_all", "utilization", "goodput_ratio", "wasted_gpu_hours", "lost_gpu_hours", "overhead_gpu_hours",
                    "topology_gpu_hours", "slack_gpu_hours", "preemptions", "frag_blocked_frac", "stranded_frac", "large_wait_mean",
                    "large_wait_p95", "jain_bsld", "tenant_worst_best_ratio", "slo_attainment", "makespan", "reservations",
                    "broken_reservations", "prio_9_wait_p95", "prio_8_wait_p95", "prio_1_wait_p95", "prio_1_bsld_p95",
                    "tenant_t0_bsld_mean", "tenant_t1_bsld_mean", "tenant_t2_bsld_mean", "tenant_t3_bsld_mean",
                    "tenant_t0_bsld_p95", "tenant_t1_bsld_p95", "tenant_t2_bsld_p95", "tenant_t3_bsld_p95",
                    "quota_satisfaction", "borrowed_gpu_hours", "jain_weighted_bsld", "failure_kills", "node_downtime_gpu_hours"],
}
with open("configs/experiments/benchmark.json", "w", newline="\n") as f:
    f.write(json.dumps(exp, indent=1) + "\n")
print(len(sc), "scenarios")
