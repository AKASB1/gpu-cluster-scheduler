"""Framework figure: how a benchmark run flows through the components.

    python scripts/plot_framework.py

Writes docs/figures/framework.png (drawn by this script, not by hand).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
GO, PY, DATA = "#2a78d6", "#eb6834", "#1baf7a"

BOXES = {
    # name: (x, y, w, h, color, title, subtitle)
    "gen": (0.2, 4.6, 2.4, 1.0, GO, "trace sources", "synthetic generator (Go),\nHelios log converter"),
    "traces": (0.2, 3.0, 2.4, 1.0, DATA, "traces + manifests", "schema v1 CSV, SHA-256"),
    "cluster": (0.2, 1.4, 2.4, 1.0, DATA, "cluster config v1", "classes, racks, nodes,\npenalties"),
    "sim": (3.6, 2.6, 2.8, 1.8, GO, "discrete-event simulator", "virtual ms clock, gang placement,\npreemption, checkpoints, node\nfailures, topology, invariants"),
    "gopol": (7.4, 4.3, 2.8, 1.3, GO, "Go policies", "order + placement + backfill\n+ preemption + quotas"),
    "pypol": (7.4, 2.2, 2.8, 1.5, PY, "Python harness", "protocol v1 over stdin/stdout:\nFIFO, EASY (conformance),\nmilp_rolling, scenario_milp"),
    "milp": (7.4, 0.3, 2.8, 1.3, PY, "offline MILP", "pool / node models,\nexact optimum or bound"),
    "metrics": (3.6, 0.3, 2.8, 1.6, GO, "metrics + checks", "window, tails, utilization,\ngoodput, fragmentation,\nfairness, identities"),
    "bench": (11.2, 2.4, 2.6, 2.0, GO, "benchmark", "tuning on tuning seeds,\nfrozen configs, evaluation,\nCI, paired diffs, sensitivity"),
    "out": (11.2, 0.3, 2.6, 1.5, DATA, "results + figures", "runs.csv, aggregates,\nmanifest, PNG, table;\n/metrics in replay mode"),
}

ARROWS = [("gen", "traces", ""), ("traces", "sim", "replay"), ("cluster", "sim", ""), ("sim", "gopol", "view"),
          ("gopol", "sim", "actions"), ("sim", "pypol", "view"), ("pypol", "sim", "actions"), ("sim", "metrics", "records"),
          ("traces", "milp", ""), ("metrics", "bench", ""), ("milp", "bench", "optimum / bound"), ("bench", "out", "")]


def center(name):
    x, y, w, h, *_ = BOXES[name]
    return x + w / 2, y + h / 2


def edge_point(name, toward):
    x, y, w, h, *_ = BOXES[name]
    cx, cy = center(name)
    tx, ty = toward
    dx, dy = tx - cx, ty - cy
    sx = (w / 2) / abs(dx) if dx else float("inf")
    sy = (h / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def main():
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for name, (x, y, w, h, color, title, sub) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.6,
                                    edgecolor=color, facecolor=SURFACE))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.52, sub, ha="center", va="top", fontsize=7.5, color=INK2, linespacing=1.25)
    pairs = {(a, b) for a, b, _ in ARROWS}
    for a, b, label in ARROWS:
        ca, cb = center(a), center(b)
        off = 0.18 if (b, a) in pairs else 0.0
        sign = 1 if a < b else -1
        start = edge_point(a, cb)
        end = edge_point(b, ca)
        start = (start[0], start[1] + sign * off)
        end = (end[0], end[1] + sign * off)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=INK2))
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label, ha="center", va="bottom", fontsize=7, color=INK2)
    for k, (color, text) in enumerate([(GO, "Go"), (PY, "Python harness"), (DATA, "files (committed configs, generated traces, results)")]):
        ax.add_patch(FancyBboxPatch((0.25 + k * 2.6, -0.55), 0.25, 0.22, boxstyle="round,pad=0.01", edgecolor=color, facecolor=SURFACE, linewidth=1.6))
        ax.text(0.6 + k * 2.6, -0.44, text, va="center", fontsize=7.5, color=INK2)
    ax.text(13.8, -0.44, "everything is simulated; assumed parameters", ha="right", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.8, 5.8)
    ax.axis("off")
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    path = "docs/figures/framework.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
