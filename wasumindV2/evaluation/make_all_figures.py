#!/usr/bin/env python3
"""
Generate ALL publication figures for the paper.

Usage:
  python wasumindV2/evaluation/make_all_figures.py --output-dir experiments/v2/figures
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

COLORS = {"gru": "#4CAF50", "lstm": "#FF9800", "stu": "#2196F3",
           "transformer": "#9C27B0", "stu_t": "#F44336"}
LABELS = {"gru": "GRU", "lstm": "LSTM", "stu": "STU (spectral)",
          "transformer": "Transformer", "stu_t": "STU-T (tensordot)"}
MARKERS = {"gru": "s", "lstm": "^", "stu": "o", "transformer": "D", "stu_t": "x"}


def fig1_tmaze_heatmap(output_dir):
    """Figure 1: TMaze results — architecture × corridor length."""
    # Hardcoded from verified local results
    data = {
        ("stu", 10): 1.0, ("stu", 100): 1.0, ("stu", 200): 1.0,
        ("gru", 10): 1.0, ("gru", 100): 1.0, ("gru", 200): 1.0,
        ("lstm", 10): 1.0, ("lstm", 100): 1.0, ("lstm", 200): 0.05,
        ("stu_t", 10): 0.0, ("stu_t", 100): 0.05,
    }

    archs = ["stu", "gru", "lstm", "stu_t"]
    lengths = [10, 100, 200]
    grid = np.full((len(archs), len(lengths)), np.nan)
    for i, a in enumerate(archs):
        for j, l in enumerate(lengths):
            grid[i, j] = data.get((a, l), np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([f"L={l}" for l in lengths])
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([LABELS[a] for a in archs])
    ax.set_xlabel("Corridor Length")

    for i in range(len(archs)):
        for j in range(len(lengths)):
            val = grid[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12, color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="gray")

    plt.colorbar(im, ax=ax, label="Accuracy (5 demos)")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_tmaze_heatmap.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig2_copytask_curves(output_dir):
    """Figure 2: CopyTask data efficiency curves."""
    # Hardcoded from verified local results (noJEPA)
    budgets = [10, 50, 250, 1000]
    results = {
        "transformer": [1.30, 2.74, 6.19, None],
        "gru": [1.34, 2.22, 4.93, 9.00],
        "lstm": [1.29, 2.16, 4.83, 7.60],
        "stu_t": [1.05, 1.17, 1.19, 1.13],
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in ["transformer", "gru", "lstm", "stu_t"]:
        vals = results[arch]
        valid_b = [b for b, v in zip(budgets, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]
        ax.plot(valid_b, valid_v, marker=MARKERS[arch], color=COLORS[arch],
                label=LABELS[arch], linewidth=2.5, markersize=8)

    ax.axhline(y=10.0, color="gray", linestyle="--", alpha=0.4, label="Teacher (optimal)")
    ax.axhline(y=1.25, color="gray", linestyle=":", alpha=0.4, label="Random")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Demonstrations")
    ax.set_ylabel("Mean Return (max = 10)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 11)

    path = os.path.join(output_dir, "fig2_copytask_efficiency.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig3_jepa_null(output_dir):
    """Figure 3: JEPA lambda sweep — the flat line."""
    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    mse = [0.1053, 0.1053, 0.1053, 0.1053, 0.1052, 0.1052]
    std = [0.033] * 6

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(lambdas, mse, yerr=std, marker="o", color="#2196F3",
                linewidth=2.5, markersize=8, capsize=4)
    ax.set_xlabel("JEPA Weight ($\\lambda$)")
    ax.set_ylabel("MSE (STU-T, LDS $\\rho$=0.99)")
    ax.set_ylim(0, 0.2)
    ax.grid(True, alpha=0.2)

    # Annotate
    ax.text(1.0, 0.16, "Zero effect across\nall $\\lambda$ values",
            ha="center", fontsize=11, style="italic", color="#666")

    path = os.path.join(output_dir, "fig3_jepa_null.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig5_model_comparison(output_dir):
    """Figure 5: Parameter count vs latency vs performance overview."""
    archs = ["gru", "lstm", "transformer", "stu"]
    params = [50886, 67526, 232006, 338118]
    latency = [1.26, 1.33, 3.36, 4.38]
    # Use CopyTask@250 as performance proxy
    perf = [4.93, 4.83, 6.19, None]  # STU not tested on CopyTask

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: params vs latency
    for i, arch in enumerate(archs):
        ax1.scatter(params[i] / 1000, latency[i], s=150, color=COLORS[arch],
                    marker=MARKERS[arch], zorder=5)
        ax1.annotate(LABELS[arch], (params[i] / 1000, latency[i]),
                     textcoords="offset points", xytext=(10, 5), fontsize=10)

    # Add SpectraLDS
    ax1.scatter(238.4, 0.21, s=150, color="#00BCD4", marker="*", zorder=5)
    ax1.annotate("SpectraLDS", (238.4, 0.21), textcoords="offset points",
                 xytext=(10, 5), fontsize=10, color="#00BCD4")

    ax1.set_xlabel("Parameters (K)")
    ax1.set_ylabel("Latency (ms / step)")
    ax1.grid(True, alpha=0.2)

    # Right: params vs CopyTask performance
    for i, arch in enumerate(archs):
        if perf[i] is not None:
            ax2.scatter(params[i] / 1000, perf[i], s=150, color=COLORS[arch],
                        marker=MARKERS[arch], zorder=5)
            ax2.annotate(LABELS[arch], (params[i] / 1000, perf[i]),
                         textcoords="offset points", xytext=(10, 5), fontsize=10)

    ax2.set_xlabel("Parameters (K)")
    ax2.set_ylabel("CopyTask Return (250 demos)")
    ax2.axhline(y=10, color="gray", linestyle="--", alpha=0.3)
    ax2.axhline(y=1.25, color="gray", linestyle=":", alpha=0.3)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig5_model_overview.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig6_lds_replication(output_dir):
    """Figure 6: Flash STU LDS replication results."""
    rhos = [0.9, 0.95, 0.99, 0.999]
    data = {
        "stu": [0.0003, 0.0004, 0.0004, 0.0003],
        "gru": [0.002, 0.002, 0.005, 0.008],
        "lstm": [0.002, 0.002, 0.007, 0.012],
        "transformer": [0.014, 0.023, 0.046, 0.062],
        "stu_t": [0.054, 0.072, 0.104, 0.121],
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in ["stu", "gru", "lstm", "transformer", "stu_t"]:
        ax.plot(rhos, data[arch], marker=MARKERS[arch], color=COLORS[arch],
                label=LABELS[arch], linewidth=2, markersize=7)

    ax.set_xlabel("Spectral Radius ($\\rho$)")
    ax.set_ylabel("MSE ($\\downarrow$)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.2)

    path = os.path.join(output_dir, "fig6_lds_replication.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig4_distill_efficiency(output_dir):
    """Figure 4: THE KEY FIGURE — distillation efficiency vs teacher quality.
    Placeholder: generates template, fill with real data from MuJoCo experiments."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Template with fake data — replace with real data
    teacher_returns = [100, 500, 1500, 3000]
    for arch in ["gru", "lstm", "stu", "transformer"]:
        efficiency = [0.3 + np.random.rand() * 0.2 for _ in teacher_returns]
        efficiency.sort()
        ax.plot(teacher_returns, efficiency, marker=MARKERS[arch], color=COLORS[arch],
                label=LABELS[arch], linewidth=2, markersize=7, linestyle="--", alpha=0.4)

    ax.set_xlabel("Teacher Return")
    ax.set_ylabel("Distillation Efficiency (student / teacher)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_title("PLACEHOLDER — Replace with real MuJoCo data", color="red", fontsize=11)

    path = os.path.join(output_dir, "fig4_distill_efficiency_PLACEHOLDER.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path} (PLACEHOLDER — replace with real data)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/v2/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Generating all paper figures...\n")

    fig1_tmaze_heatmap(args.output_dir)
    fig2_copytask_curves(args.output_dir)
    fig3_jepa_null(args.output_dir)
    fig4_distill_efficiency(args.output_dir)
    fig5_model_comparison(args.output_dir)
    fig6_lds_replication(args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")
    print("Figures 1-3, 5-6 use REAL data.")
    print("Figure 4 is a PLACEHOLDER — regenerate after MuJoCo experiments complete.")


if __name__ == "__main__":
    main()
