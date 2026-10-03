"""Figures and the result table from the committed benchmark results.

    python scripts/plot_results.py --results benchmarks/results/benchmark

Reads only files under --results (aggregate.csv, paired.csv, runs.csv,
wall.csv, sensitivity.csv, manifest.json) and writes PNG figures to
docs/figures/ and a Markdown table to <results>/table.md. Every value is
simulated with assumed parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots (validated reference palette, light mode), fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SIM = "simulated, assumed parameters"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
})


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def agg_index(rows):
    """(scenario, policy, metric) -> (mean, lo, hi, n)."""
    out = {}
    for r in rows:
        out[(r["scenario"], r["policy"], r["metric"])] = (num(r["mean"]), num(r["ci95_low"]), num(r["ci95_high"]), int(r["n"]))
    return out


def short(policy: str) -> str:
    return policy.replace("shortest_estimate", "sjf").replace("least_fragmentation", "least_frag").replace(
        "topology_aware", "topo_aware").replace("easy_predicted", "easy_pred").replace("conservative", "conserv")


def save(fig, out_dir, name, note):
    fig.text(0.01, 0.01, note, fontsize=7, color=INK2, ha="left", va="bottom")
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    size = os.path.getsize(path)
    assert size < 300_000, f"{name} is {size} bytes (limit 300 KB)"
    print(f"wrote {path} ({size // 1024} KB)")


def dot_ci(ax, labels, vals, color, offset=0.0, label=None):
    ys = [i + offset for i in range(len(labels))]
    for y, (m, lo, hi, _n) in zip(ys, vals):
        if not math.isnan(lo):
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
    ax.scatter([v[0] for v in vals], ys, s=36, color=color, zorder=3, edgecolors=SURFACE, linewidths=1.5, label=label)


def fig_s2(res, A, out_dir):
    """S2: P95 bounded slowdown per backfill policy under three estimate models (load 0.95)."""
    scen = [("s2-95-exact", "exact"), ("s2-95-lognormal", "lognormal (sigma 0.5)"), ("s2-95-user", "user-style (10% under)")]
    pols = sorted({p for (s, p, m) in A if s == "s2-95-exact" and m == "bsld_p95"})
    if not pols:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for k, (s, name) in enumerate(scen):
        vals = [A.get((s, p, "bsld_p95"), (float("nan"),) * 4) for p in pols]
        dot_ci(ax, pols, vals, SLOTS[k], offset=(k - 1) * 0.22, label=name)
    ax.set_yticks(range(len(pols)), [short(p) for p in pols])
    ax.set_xscale("log")
    ax.set_xlabel("P95 bounded slowdown (log; lower is better), mean and 95% CI over 20 seeds")
    ax.set_title("S2 at offered load 0.95: backfilling under runtime-estimate error")
    ax.invert_yaxis()
    ax.legend(loc="lower right", title="estimate model", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_dir, "s2_backfill_estimates.png", SIM + "; oracle = true run times, not deployable")


def fig_s1_load(res, A, out_dir):
    """S1: P95 wait against offered load for the main policies."""
    loads = [("s1-load60", 0.60), ("s1-load80", 0.80), ("s1-load95", 0.95)]
    pols = sorted({p for (s, p, m) in A if s == "s1-load80" and m == "wait_p95"})
    if not pols:
        return
    # Four series that each change one part: the baseline, the best policy by J
    # at 0.95, the same without backfilling, and the same with FIFO order.
    others = [p for p in pols if p != "fifo+first_fit+none" and "oracle" not in p]
    others.sort(key=lambda p: A[("s1-load95", p, "J")][0] if ("s1-load95", p, "J") in A else 1e9)
    best = others[0]
    for p in others:  # the best policy whose no-backfill and FIFO-order variants were both run
        o, pl, b = p.split("+")[:3]
        if f"{o}+{pl}+none" in pols and f"fifo+{pl}+{b}" in pols:
            best = p
            break
    o, pl, b = best.split("+")[:3]
    show = [p for p in ["fifo+first_fit+none", f"{o}+{pl}+none", f"fifo+{pl}+{b}", best] if p in pols]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, p in enumerate(show):
        xs, ms, los, his = [], [], [], []
        for s, L in loads:
            m, lo, hi, _ = A.get((s, p, "wait_p95"), (float("nan"),) * 4)
            xs.append(L), ms.append(m / 3600), los.append(max(lo, 1e-3) / 3600), his.append(hi / 3600)
        ax.fill_between(xs, los, his, color=SLOTS[k], alpha=0.15, linewidth=0)
        ax.plot(xs, ms, color=SLOTS[k], marker="o", markersize=5, label=short(p))
    ax.set_yscale("log")
    ax.set_xlabel("offered load")
    ax.set_ylabel("P95 wait (hours, log)")
    ax.set_xticks([0.6, 0.8, 0.95])
    ax.set_title("S1: P95 wait against load, one part changed at a time")
    ax.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_dir, "s1_wait_vs_load.png", SIM + "; band = 95% CI over 20 seeds")


def fig_milp_gap(res, A, out_dir):
    """S8: optimality gap of every Go baseline (small: to the optimum; medium: to the lower bound)."""
    panels = [("s8-small", "small: gap to the optimum (node model)"),
              ("s8-medium", "medium: gap to a lower bound (pool model)")]
    pols = sorted({p for (s, p, m) in A if s == "s8-small" and m == "gap"})
    if not pols:
        return
    order = sorted(pols, key=lambda p: A[("s8-small", p, "gap")][0])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
    for ax, (s, title) in zip(axes, panels):
        vals = [A.get((s, p, "gap"), (float("nan"),) * 4) for p in order]
        vals = [(m * 100, lo * 100, hi * 100, n) for (m, lo, hi, n) in vals]
        ax.barh(range(len(order)), [v[0] for v in vals], height=0.6, color=SLOTS[0])
        for y, (m, lo, hi, _n) in enumerate(vals):
            if not math.isnan(lo):
                ax.plot([lo, hi], [y, y], color=INK, linewidth=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("gap in sum of weighted flow (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [short(p) for p in order])
    axes[0].invert_yaxis()
    fig.suptitle("S8: how far the Go baselines are from the MILP reference (exact estimates, 50 instances each)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, out_dir, "s8_milp_gap.png", SIM + "; whiskers = 95% CI; the medium gap is to a lower bound and over-states the true gap")


def fig_solve_time(res, out_dir):
    """MILP solve time against the number of jobs and the grid size."""
    det = read_csv(os.path.join(res, "S8", "sweep", "runs.csv"))
    wall = {(r["policy"], r["seed"]): num(r["wall_solve_s"]) for r in read_csv(os.path.join(res, "S8", "sweep", "wall.csv"))}
    if not det:
        return
    series = defaultdict(lambda: defaultdict(list))
    capped = []
    for r in det:
        slot, n = num(r["slot_s"]), int(num(r["jobs"]))
        t = wall.get((r["policy"], r["seed"]), float("nan"))
        series[slot][n].append(t)
        if num(r["solver_capped"]) == 1 or num(r["optimal"]) == 0:
            capped.append((slot, n, t))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, slot in enumerate(sorted(series, reverse=True)):
        ns = sorted(series[slot])
        med = [sorted(series[slot][n])[len(series[slot][n]) // 2] for n in ns]
        ax.plot(ns, med, color=SLOTS[k], marker="o", markersize=5, label=f"slot {slot:g} s")
        ax.scatter([n for n in ns for _ in series[slot][n]], [t for n in ns for t in series[slot][n]], s=10, color=SLOTS[k], alpha=0.5)
    if capped:
        ax.scatter([c[1] for c in capped], [c[2] for c in capped], s=60, facecolors="none", edgecolors=INK, linewidths=1.2,
                   label="not solved to optimality (cap)")
    ax.set_yscale("log")
    ax.set_xlabel("jobs in the instance (single-worker, 4 nodes)")
    ax.set_ylabel("wall-clock solve time (s, log)")
    ax.set_title("Node-model MILP: solve time against size and grid (median line, instances as dots)")
    ax.legend(loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_dir, "milp_solve_time.png", SIM + "; HiGHS via SciPy; wall-clock on a shared machine, see the manifest")


def fig_s6(res, A, out_dir):
    """S6: high-priority P95 wait against wasted GPU-hours, by checkpoint setting."""
    scen = [("s6-ckpt-none", "no checkpoint"), ("s6-ckpt-short", "checkpoint 600 s"), ("s6-ckpt-long", "checkpoint 3600 s")]
    pols = sorted({p for (s, p, m) in A if s == "s6-ckpt-short" and m == "prio_9_wait_p95"})
    if not pols:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    markers = ["o", "s", "^"]
    for k, p in enumerate(pols):
        for j, (s, name) in enumerate(scen):
            w = A.get((s, p, "prio_9_wait_p95"), (float("nan"),) * 4)[0] / 60
            g = A.get((s, p, "wasted_gpu_hours"), (float("nan"),) * 4)[0]
            ax.scatter([g], [w], s=50, color=SLOTS[k], marker=markers[j], edgecolors=SURFACE, linewidths=1.5,
                       label=short(p) if j == 0 else None)
    for j, (_s, name) in enumerate(scen):
        ax.scatter([], [], marker=markers[j], color=INK2, label=name)
    ax.set_xlabel("wasted GPU-hours in the window (lost work, restart overhead, topology, slack)")
    ax.set_ylabel("P95 wait of priority-9 burst jobs (min)")
    ax.set_title("S6: high-priority wait against waste, with and without preemption")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_dir, "s6_preemption_tradeoff.png", SIM + "; mean over 20 seeds")


NAN4 = (float("nan"),) * 4


def fig_s10(res, A, out_dir):
    """S10: P95 bounded slowdown of the backfill variants, fitted estimate regime against a worse one."""
    panels = [("s10-lognormal", "s10-shift-lognormal", "lognormal: sigma 0.5 -> 1.0"),
              ("s10-user", "s10-shift-user", "user-style: 10% -> 20% underestimates")]
    pols = sorted({p for (s, p, m) in A if s == "s10-lognormal" and m == "bsld_p95"})
    if not pols:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for ax, (base, shifted, title) in zip(axes, panels):
        for k, (s, name) in enumerate([(base, "fitted regime"), (shifted, "shifted regime")]):
            dot_ci(ax, pols, [A.get((s, p, "bsld_p95"), NAN4) for p in pols], SLOTS[k], offset=(k - 0.5) * 0.3, label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("P95 bounded slowdown (log; lower is better)")
    axes[0].set_yticks(range(len(pols)), [short(p) for p in pols])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.suptitle("S10 at load 0.9: backfilling when estimates are worse than at tuning time", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, out_dir, "s10_estimate_shift.png", SIM + "; 95% CI over 20 seeds; parameters tuned on the fitted regime only")


def fig_capacity(res, out_dir):
    """Capacity planning: the smallest cluster that keeps the mean P95 wait under the target, per offered load."""
    need = read_csv(os.path.join(res, "CAP", "capacity_needed.csv"))
    if not need:
        return
    target = num(need[0]["target_p95_wait_s"])
    pols = list(dict.fromkeys(r["policy"] for r in need))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, p in enumerate(pols):
        rows = [r for r in need if r["policy"] == p]
        ax.plot([num(r["load"]) for r in rows], [num(r["min_gpus_meeting_target"]) for r in rows], color=SLOTS[k], marker="o",
                markersize=5, label=short(p))
    ax.set_xlabel("offered load relative to the 128-GPU reference cluster")
    ax.set_ylabel(f"GPUs needed for P95 wait <= {target / 3600:g} h")
    ax.set_title("Capacity planning: GPUs needed against load, per policy")
    ax.legend(loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_dir, "capacity_planning.png", SIM + "; 8-GPU nodes; mean P95 wait over 10 seeds; same traces at every size")


def fmt(m, lo, hi, scale=1.0, digits=2):
    if math.isnan(m):
        return "n/a"
    if math.isnan(lo):
        return f"{m * scale:.{digits}f}"
    return f"{m * scale:.{digits}f} ± {(hi - lo) / 2 * scale:.{digits}f}"


H = 1 / 3600
BASE_COLS = [("J", "J", 1, 2), ("P95 wait (h)", "wait_p95", H, 2), ("P95 bounded slowdown", "bsld_p95", 1, 2)]
TIER2 = [
    ("s10-lognormal", "S10, load 0.90, lognormal estimates (sigma 0.5): fitted regime",
     BASE_COLS + [("broken reservations", "broken_reservations", 1, 0), ("utilization", "utilization", 1, 3)]),
    ("s10-shift-lognormal", "S10, estimate shift: lognormal sigma 1.0, tuning frozen on the fitted regime",
     BASE_COLS + [("broken reservations", "broken_reservations", 1, 0), ("utilization", "utilization", 1, 3)]),
    ("s10-user", "S10, load 0.90, user-style estimates (10% under): fitted regime",
     BASE_COLS + [("broken reservations", "broken_reservations", 1, 0), ("utilization", "utilization", 1, 3)]),
    ("s10-shift-user", "S10, estimate shift: user-style, 20% under, tuning frozen on the fitted regime",
     BASE_COLS + [("broken reservations", "broken_reservations", 1, 0), ("utilization", "utilization", 1, 3)]),
    ("s11-online", "S11, stochastic vs deterministic online MILP (10 seeds)", BASE_COLS + [("utilization", "utilization", 1, 3)]),
    ("s11-shift", "S11, estimate shift: lognormal sigma 1.0 (10 seeds, tuning frozen on s11-online)",
     BASE_COLS + [("utilization", "utilization", 1, 3)]),
    ("s12-hetero-free", "S12, heterogeneous cluster, no class constraints",
     BASE_COLS + [("slack GPU-h", "slack_gpu_hours", 1, 0), ("goodput", "goodput_ratio", 1, 3)]),
    ("s12-hetero-pinned", "S12, heterogeneous cluster, 15% of jobs pinned to each class",
     BASE_COLS + [("slack GPU-h", "slack_gpu_hours", 1, 0), ("goodput", "goodput_ratio", 1, 3)]),
    ("s13-helios-venus", "S13, replayed jobs on a simulated cluster: Helios Venus, 3 days from 2020-08-10, synthetic estimates "
     "(trace: Hu et al., SC'21, CC-BY-4.0; cluster derived from the log's own capacity, an assumption; 10 estimate seeds)",
     BASE_COLS + [("utilization", "utilization", 1, 3), ("broken reservations", "broken_reservations", 1, 0)]),
    ("s14-quota", "S14, quotas: one heavy tenant (t0, 55% of jobs), equal nominal quotas of 25%",
     BASE_COLS + [("quota satisfaction", "quota_satisfaction", 1, 3), ("borrowed GPU-h", "borrowed_gpu_hours", 1, 0),
                  ("Jain (weighted bsld)", "jain_weighted_bsld", 1, 3), ("P95 bsld t1", "tenant_t1_bsld_p95", 1, 2),
                  ("preemptions", "preemptions", 1, 0)]),
    ("s15-fail-ckpt-none", "S15, node failures (MTBF 4 days per node, repair 2 h), no checkpoints",
     BASE_COLS + [("goodput", "goodput_ratio", 1, 3), ("lost GPU-h", "lost_gpu_hours", 1, 0), ("failure kills", "failure_kills", 1, 1)]),
    ("s15-fail-ckpt-short", "S15, node failures, checkpoint every 600 s",
     BASE_COLS + [("goodput", "goodput_ratio", 1, 3), ("lost GPU-h", "lost_gpu_hours", 1, 0), ("failure kills", "failure_kills", 1, 1)]),
    ("s15-fail-ckpt-long", "S15, node failures, checkpoint every 3600 s",
     BASE_COLS + [("goodput", "goodput_ratio", 1, 3), ("lost GPU-h", "lost_gpu_hours", 1, 0), ("failure kills", "failure_kills", 1, 1)]),
]


def tier2_tables(res, A, wins):
    lines = []
    for s, title, cols in TIER2:
        pols = sorted({p for (sc, p, m) in A if sc == s and m == "J"}, key=lambda p: (A[(s, p, "J")][0], p))
        if not pols:
            continue
        lines += [f"**{title}** (mean ± 95% CI half-width; W/T/L on J against `fifo+first_fit+none` where it ran)", "",
                  "| policy | " + " | ".join(c[0] for c in cols) + " | W/T/L on J |", "|---" * (len(cols) + 2) + "|"]
        for p in pols:
            w = wins.get((s, p, "J"), ("-", "-", "-"))
            cells = [fmt(*A.get((s, p, m), NAN4)[:3], scale=sc_, digits=d) for _h, m, sc_, d in cols]
            lines.append(f"| `{p}` | " + " | ".join(cells) + f" | {'/'.join(w)} |")
        lines.append("")
    solver = []
    for g in ("S9", "S11"):
        for r in read_csv(os.path.join(res, g, "wall.csv")):
            if r["policy"].startswith("py:") and r.get("wall_solver_calls"):
                solver.append(r)
    if solver:
        lines += ["**Online MILP solve times** (per scheduling call; mean over seeds of the per-run mean, and the largest single call; "
                  "wall-clock and process CPU time on a shared machine, so indicative only)", "",
                  "| scenario | policy | calls per run | wall mean (ms) | CPU mean (ms) | wall max (s) | capped calls |", "|---|---|---|---|---|---|---|"]
        for key in dict.fromkeys((r["scenario"], r["policy"]) for r in solver):
            rs = [r for r in solver if (r["scenario"], r["policy"]) == key]
            mean = lambda c: sum(num(r.get(c)) for r in rs) / len(rs)  # noqa: E731
            lines.append(f"| {key[0]} | `{key[1]}` | {mean('wall_solver_calls'):.0f} | {mean('wall_solve_mean_s') * 1000:.0f} | "
                         f"{mean('wall_solve_cpu_mean_s') * 1000:.0f} | {max(num(r['wall_solve_max_s']) for r in rs):.2f} | "
                         f"{sum(num(r['wall_solver_capped']) for r in rs):.0f} |")
        lines.append("")
    need = read_csv(os.path.join(res, "CAP", "capacity_needed.csv"))
    if need:
        loads = list(dict.fromkeys(r["load"] for r in need))
        lines += [f"**Capacity planning**: smallest cluster (GPUs, 8 per node) with mean P95 wait <= "
                  f"{num(need[0]['target_p95_wait_s']) / 3600:g} h over 10 seeds; blank = not met within the sweep", "",
                  "| policy | " + " | ".join(f"load {num(x):g}" for x in loads) + " |", "|---" * (len(loads) + 1) + "|"]
        for p in dict.fromkeys(r["policy"] for r in need):
            got = {r["load"]: r["min_gpus_meeting_target"] for r in need if r["policy"] == p}
            lines.append(f"| `{p}` | " + " | ".join(got.get(x, "") or "-" for x in loads) + " |")
        lines.append("")
    dc = read_csv(os.path.join(res, "wall_decision_cost.csv"))
    if dc and "wall_ns_per_op_min" in dc[0]:
        pend = list(dict.fromkeys(r["pending"] for r in dc))
        lines += ["**Decision cost** (one scheduling invocation, Go microbenchmark, minimum of "
                  f"{dc[0]['repetitions']} repetitions, microseconds; wall-clock on a shared machine)", "",
                  "| policy | " + " | ".join(f"{x} pending" for x in pend) + " |", "|---" * (len(pend) + 1) + "|"]
        for p in dict.fromkeys(r["policy"] for r in dc):
            got = {r["pending"]: num(r["wall_ns_per_op_min"]) / 1000 for r in dc if r["policy"] == p}
            lines.append(f"| `{p}` | " + " | ".join(f"{got[x]:.0f}" if x in got else "-" for x in pend) + " |")
        lines.append("")
    return lines


def table(res, A, P):
    """Markdown table: every policy of the main scenarios with J and key metrics."""
    lines = ["<!-- generated by scripts/plot_results.py from aggregate.csv and paired.csv; simulated, assumed parameters -->", ""]
    wins = {(r["scenario"], r["policy"], r["metric"]): (r["win"], r["tie"], r["loss"]) for r in P}
    for s, title in [("s1-load80", "S1, load 0.80, exact estimates"), ("s1-load95", "S1, load 0.95, exact estimates"),
                     ("s2-95-user", "S2, load 0.95, user-style estimates"), ("s3-stream", "S3, fragmentation stream"),
                     ("s4-hetero", "S4, heterogeneous cluster"), ("s5-tenants", "S5, one heavy tenant"),
                     ("s6-ckpt-short", "S6, bursts, checkpoint 600 s"), ("s7-bursty", "S7, bursty arrivals"),
                     ("s9-online", "S9, online MILP (10 seeds)")]:
        pols = sorted({p for (sc, p, m) in A if sc == s and m == "J"}, key=lambda p: (A[(s, p, "J")][0], p))
        if not pols:
            continue
        lines += [f"**{title}** (mean ± 95% CI half-width over seeds; W/T/L = seeds where J beats/ties/loses to `fifo+first_fit+none`)", "",
                  "| policy | J | P95 wait (h) | P95 bounded slowdown | utilization | goodput | W/T/L on J |",
                  "|---|---|---|---|---|---|---|"]
        for p in pols:
            g = lambda m: A.get((s, p, m), (float("nan"),) * 4)  # noqa: E731
            w = wins.get((s, p, "J"), ("-", "-", "-"))
            lines.append(f"| `{p}` | {fmt(*g('J')[:3])} | {fmt(*g('wait_p95')[:3], scale=1 / 3600)} | {fmt(*g('bsld_p95')[:3])} | "
                         f"{fmt(*g('utilization')[:3], digits=3)} | {fmt(*g('goodput_ratio')[:3], digits=3)} | {'/'.join(w)} |")
        lines.append("")
    lines += tier2_tables(res, A, wins)
    for s, title in [("s8-small", "S8 small (50 instances, gap to the optimum)"), ("s8-medium", "S8 medium (50 instances, gap to a lower bound)")]:
        pols = sorted({p for (sc, p, m) in A if sc == s and m == "gap"}, key=lambda p: (A[(s, p, "gap")][0], p))
        if not pols:
            continue
        lines += [f"**{title}**", "", "| policy | mean gap (%) |", "|---|---|"]
        for p in pols:
            lines.append(f"| `{p}` | {fmt(*A[(s, p, 'gap')][:3], scale=100, digits=1)} |")
        lines.append("")
    with open(os.path.join(res, "table.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(res, 'table.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/results/benchmark")
    ap.add_argument("--figures", default="docs/figures")
    a = ap.parse_args()
    os.makedirs(a.figures, exist_ok=True)
    rows, prow = [], []
    for g in sorted(os.listdir(a.results)):
        d = os.path.join(a.results, g)
        if os.path.isdir(d):
            rows += read_csv(os.path.join(d, "aggregate.csv"))
            prow += read_csv(os.path.join(d, "paired.csv"))
    A = agg_index(rows)
    fig_s2(a.results, A, a.figures)
    fig_s1_load(a.results, A, a.figures)
    fig_milp_gap(a.results, A, a.figures)
    fig_solve_time(a.results, a.figures)
    fig_s6(a.results, A, a.figures)
    fig_s10(a.results, A, a.figures)
    fig_capacity(a.results, a.figures)
    table(a.results, A, prow)


if __name__ == "__main__":
    main()
