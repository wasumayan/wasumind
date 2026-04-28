#!/usr/bin/env python3
"""
Generate publication-quality figures for the conference paper.

Saves PDF figures to writeup/ (configurable via --output-dir).

Usage:
  python wasumindV2/evaluation/generate_conference_figures.py
  python wasumindV2/evaluation/generate_conference_figures.py --base-dir experiments/v2/sweep_adroit --output-dir writeup
"""

import argparse
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
})

COLORS = {
    "gru": "#2ca02c",         # green
    "lstm": "#1f77b4",        # blue
    "transformer": "#ff7f0e", # orange
    "stu": "#9467bd",         # purple
    "mlp": "#7f7f7f",         # gray
    "framestack": "#8c564b",  # brown
}

LABELS = {
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "stu": "STU",
    "mlp": "MLP",
    "framestack": "FrameStack",
}

MARKERS = {
    "gru": "s",
    "lstm": "^",
    "transformer": "D",
    "stu": "o",
    "mlp": "x",
    "framestack": "+",
}

ARCH_ORDER = ["gru", "lstm", "transformer", "stu"]

ENV_PRETTY = {
    "halfcheetah_pomdp": "HalfCheetah-POMDP",
    "ant_pomdp": "Ant-POMDP",
    "walker2d_pomdp": "Walker2d-POMDP",
}


def _load(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"[SKIP] {label}: {path} not found")
        return None
    df = pd.read_csv(path)
    print(f"[OK]   {label}: {len(df)} rows")
    return df


def _save(fig, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# (a) Distillation Efficiency — student return vs teacher return
# ---------------------------------------------------------------------------

def fig_distill_efficiency(df: pd.DataFrame, output_dir: str) -> None:
    """Student return vs teacher return scatter, one panel per env."""
    envs = sorted(df["env"].unique())
    n_envs = len(envs)

    fig, axes = plt.subplots(1, n_envs, figsize=(5.5 * n_envs, 4.5), squeeze=False)
    axes = axes[0]

    for idx, env in enumerate(envs):
        ax = axes[idx]
        edf = df[df["env"] == env]

        # Aggregate: mean/std of student_return per (demo_tag, arch)
        agg = (
            edf.groupby(["demo_tag", "arch"])
            .agg(
                teacher_return=("teacher_return", "first"),
                student_mean=("student_return", "mean"),
                student_std=("student_return", "std"),
            )
            .reset_index()
        )

        for arch in ARCH_ORDER:
            sub = agg[agg["arch"] == arch].sort_values("teacher_return")
            if sub.empty:
                continue
            ax.errorbar(
                sub["teacher_return"],
                sub["student_mean"],
                yerr=sub["student_std"],
                marker=MARKERS[arch],
                color=COLORS[arch],
                label=LABELS[arch],
                linewidth=2,
                markersize=7,
                capsize=3,
                zorder=3,
            )

        # y = x reference line
        all_vals = pd.concat([agg["teacher_return"], agg["student_mean"]])
        lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1, label="$y = x$")
        ax.set_xlim(lo, hi)

        ax.set_xlabel("Teacher Return")
        if idx == 0:
            ax.set_ylabel("Student Return")
        ax.set_title(ENV_PRETTY.get(env, env))
        ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig_distill_efficiency.pdf"))


# ---------------------------------------------------------------------------
# (b) CopyTask — Mean return vs N
# ---------------------------------------------------------------------------

def fig_copytask(output_dir: str) -> None:
    """CopyTask mean return vs sequence length N for each architecture."""
    Ns = [5, 10, 20]
    data = {
        "gru":         [4.00, 9.00, 13.60],
        "lstm":        [4.00, 8.33,  9.78],
        "transformer": [3.87, 8.98, 18.84],
        "stu":         [4.00, 1.09,  2.31],
    }
    stu_label = "STU-T"

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Teacher optimal: y = N
    ax.plot(Ns, Ns, "k--", alpha=0.35, linewidth=1.5, label="Teacher (optimal)", zorder=1)

    for arch in ["gru", "lstm", "transformer", "stu"]:
        label = stu_label if arch == "stu" else LABELS[arch]
        ax.plot(
            Ns,
            data[arch],
            marker=MARKERS[arch],
            color=COLORS[arch],
            label=label,
            linewidth=2.5,
            markersize=8,
            zorder=3,
        )

    ax.set_xlabel("Sequence Length $N$")
    ax.set_ylabel("Mean Return")
    ax.set_xticks(Ns)
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(Ns) + 2)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig_copytask.pdf"))


# ---------------------------------------------------------------------------
# (c) Variance Decomposition — bar chart, "3x" claim
# ---------------------------------------------------------------------------

def fig_variance_decomposition(df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart: teacher quality range vs architecture range per environment."""
    envs = sorted(df["env"].unique())

    # Compute ranges
    agg = (
        df.groupby(["env", "demo_tag", "arch"])["student_return"]
        .mean()
        .reset_index()
    )

    teacher_range_per_env = {}
    arch_range_per_env = {}

    for env in envs:
        edf = agg[agg["env"] == env]

        # Teacher quality range: for each arch, range across demo_tags; then mean
        t_ranges = []
        for arch in ARCH_ORDER:
            sub = edf[edf["arch"] == arch]
            if len(sub) >= 2:
                t_ranges.append(sub["student_return"].max() - sub["student_return"].min())
        teacher_range_per_env[env] = np.mean(t_ranges) if t_ranges else 0

        # Architecture range: for each demo_tag, range across archs; then mean
        a_ranges = []
        for tag in edf["demo_tag"].unique():
            sub = edf[edf["demo_tag"] == tag]
            if len(sub) >= 2:
                a_ranges.append(sub["student_return"].max() - sub["student_return"].min())
        arch_range_per_env[env] = np.mean(a_ranges) if a_ranges else 0

    # Bar chart
    x = np.arange(len(envs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars1 = ax.bar(
        x - width / 2,
        [teacher_range_per_env[e] for e in envs],
        width,
        label="Teacher quality range",
        color="#1f77b4",
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        [arch_range_per_env[e] for e in envs],
        width,
        label="Architecture range",
        color="#ff7f0e",
        edgecolor="white",
    )

    ax.set_ylabel("Return Range")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_PRETTY.get(e, e) for e in envs])
    ax.legend()

    # Annotate ratios
    for i, env in enumerate(envs):
        tr = teacher_range_per_env[env]
        ar = arch_range_per_env[env]
        ratio = tr / ar if ar > 0 else float("inf")
        max_h = max(tr, ar)
        ax.text(i, max_h + max_h * 0.05, f"{ratio:.1f}x", ha="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig_variance_decomposition.pdf"))


# ---------------------------------------------------------------------------
# (d) Baselines — MLP / FrameStack comparison
# ---------------------------------------------------------------------------

def fig_baselines(df_baselines: pd.DataFrame, df_sweep: pd.DataFrame | None, output_dir: str) -> None:
    """Bar chart comparing MLP/FrameStack/GRU/LSTM/Transformer/STU."""
    if df_sweep is not None:
        combined = pd.concat([df_sweep, df_baselines], ignore_index=True)
    else:
        combined = df_baselines

    agg = (
        combined.groupby(["env", "arch"])["student_return"]
        .agg(["mean", "std"])
        .reset_index()
    )

    envs = sorted(combined["env"].unique())
    all_archs = [a for a in ["mlp", "framestack"] + ARCH_ORDER if a in agg["arch"].values]
    n_archs = len(all_archs)
    x = np.arange(len(envs))
    width = 0.8 / n_archs

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(envs)), 5))

    for j, arch in enumerate(all_archs):
        means = []
        stds = []
        for env in envs:
            sub = agg[(agg["env"] == env) & (agg["arch"] == arch)]
            if sub.empty:
                means.append(0)
                stds.append(0)
            else:
                means.append(sub.iloc[0]["mean"])
                stds.append(sub.iloc[0]["std"])

        offset = (j - n_archs / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=LABELS.get(arch, arch),
            color=COLORS.get(arch, "#333333"),
            capsize=2,
            edgecolor="white",
        )

    ax.set_ylabel("Mean Student Return")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_PRETTY.get(e, e) for e in envs])
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig_baselines.pdf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the conference paper."
    )
    parser.add_argument(
        "--base-dir",
        default="experiments/v2/sweep_adroit",
        help="Base directory containing CSV files (default: experiments/v2/sweep_adroit)",
    )
    parser.add_argument(
        "--output-dir",
        default="writeup",
        help="Directory to save PDF figures (default: writeup)",
    )
    args = parser.parse_args()

    base = args.base_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Base directory : {base}")
    print(f"Output directory: {output_dir}\n")

    # Load data
    df_sweep = _load(os.path.join(base, "sweep_results.csv"), "Original sweep")
    df_baselines = _load(os.path.join(base, "baselines_results.csv"), "Baselines")

    print()

    # (a) Distillation efficiency — requires sweep data
    if df_sweep is not None:
        print("Generating fig_distill_efficiency.pdf ...")
        fig_distill_efficiency(df_sweep, output_dir)
    else:
        print("[SKIP] fig_distill_efficiency.pdf — no sweep data")

    # (b) CopyTask — hardcoded data, always generated
    print("Generating fig_copytask.pdf ...")
    fig_copytask(output_dir)

    # (c) Variance decomposition — requires sweep data
    if df_sweep is not None:
        print("Generating fig_variance_decomposition.pdf ...")
        fig_variance_decomposition(df_sweep, output_dir)
    else:
        print("[SKIP] fig_variance_decomposition.pdf — no sweep data")

    # (d) Baselines — requires baselines CSV
    if df_baselines is not None:
        print("Generating fig_baselines.pdf ...")
        fig_baselines(df_baselines, df_sweep, output_dir)
    else:
        print("[SKIP] fig_baselines.pdf — no baselines data")

    print(f"\nDone. Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
