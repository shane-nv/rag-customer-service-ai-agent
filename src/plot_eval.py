# -*- coding: utf-8 -*-
"""Plot best-distance by gold-set group with the calibrated refuse threshold.
The single figure that tells the whole evaluation story: the three query classes
separate on retrieval distance, and the threshold sits in the gap between them.

Reads  data/eval_results.csv  (produced by evaluate.py)
Writes reports/threshold_calibration.png
Run:   python src/plot_eval.py
"""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config

import numpy as np
import matplotlib
matplotlib.use("Agg")               # no display needed
import matplotlib.pyplot as plt

THRESHOLD = float(config.SCORE_THRESHOLD)
CSV = config.REPO_ROOT / "data" / "eval_results.csv"
OUT = config.REPO_ROOT / "reports" / "threshold_calibration.png"

# --- palette (light mode) ---
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
ORDER = ["answerable", "answerable_negative", "out_of_scope"]
COLOR = {"answerable": "#2a78d6", "answerable_negative": "#eb6834", "out_of_scope": "#1baf7a"}
LABEL = {
    "answerable": "Answerable\n(should answer)",
    "answerable_negative": "Answerable-negative\n(grounded “no”)",
    "out_of_scope": "Out-of-scope\n(should refuse)",
}


def load():
    groups = {k: [] for k in ORDER}
    with open(CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d = (r.get("best_distance") or "").strip()
            if d and r.get("label") in groups:
                try:
                    groups[r["label"]].append(float(d))
                except ValueError:
                    pass
    return groups


def main():
    groups = load()
    rng = np.random.default_rng(7)          # fixed jitter -> reproducible figure
    OUT.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # shade the separation gap: max should-pass .. min should-refuse
    should_pass = groups["answerable"] + groups["answerable_negative"]
    pass_max, refuse_min = max(should_pass), min(groups["out_of_scope"])
    ax.axvspan(pass_max, refuse_min, color=GRID, alpha=0.55, lw=0, zorder=0)

    for i, key in enumerate(ORDER):
        xs = groups[key]
        ys = np.full(len(xs), i) + rng.uniform(-0.14, 0.14, len(xs))
        ax.scatter(xs, ys, s=66, color=COLOR[key], alpha=0.85,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)

    ax.axvline(THRESHOLD, color=INK, ls="--", lw=1.6, zorder=4)
    ax.text(THRESHOLD + 0.008, 1.0, f"threshold = {THRESHOLD:g}", color=INK,
            ha="left", va="center", rotation=90, fontsize=9.5, fontweight="bold")
    ax.text(0.185, 2.9, f"gap {pass_max:.3f}–{refuse_min:.3f}", color=MUTED,
            ha="left", va="top", fontsize=8)

    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABEL[k] for k in ORDER], fontsize=9, color=INK)
    ax.set_ylim(-0.9, 2.9)
    ax.invert_yaxis()
    ax.set_xlim(0.15, 0.75)
    ax.set_xlabel("best retrieval distance  (cosine, lower = closer)", color=INK, fontsize=10)
    ax.set_title("Refuse-gate calibration: distance separates answerable from out-of-scope",
                 color=INK, fontsize=11.5, fontweight="bold", pad=10, loc="left")

    ax.tick_params(colors=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
