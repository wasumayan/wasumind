#!/usr/bin/env python3
"""
Comprehensive analysis of all experiment CSVs.

Loads each CSV (gracefully skipping missing files), computes summary
statistics, and prints nicely formatted tables to stdout.

Usage:
  python wasumindV2/evaluation/analyze_new_results.py
  python wasumindV2/evaluation/analyze_new_results.py --base-dir experiments/v2/sweep_adroit
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARCH_ORDER = ["gru", "lstm", "transformer", "stu"]
ARCH_LABELS = {
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "stu": "STU",
    "mlp": "MLP",
    "framestack": "FrameStack",
}

SEPARATOR = "-" * 78


def _load(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"[SKIP] {label}: {path} not found")
        return None
    df = pd.read_csv(path)
    print(f"[OK]   {label}: {len(df)} rows from {path}")
    return df


def _table(df: pd.DataFrame, index: bool = True) -> str:
    return df.to_string(index=index)


def _header(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}\n")


# ---------------------------------------------------------------------------
# 1. Original Sweep — variance decomposition ("3x claim")
# ---------------------------------------------------------------------------

def analyze_sweep(df: pd.DataFrame) -> None:
    _header("1. ORIGINAL SWEEP — Variance Decomposition")

    # Aggregate across seeds: mean student_return per (env, demo_tag, arch)
    agg = (
        df.groupby(["env", "demo_tag", "arch"])["student_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for env in sorted(df["env"].unique()):
        print(f"\n--- Environment: {env} ---\n")
        edf = agg[agg["env"] == env]

        # Teacher quality range: for each arch, range of means across demo_tags
        teacher_ranges = {}
        for arch in ARCH_ORDER:
            sub = edf[edf["arch"] == arch]
            if len(sub) < 2:
                continue
            teacher_ranges[arch] = sub["mean"].max() - sub["mean"].min()

        # Architecture range: for each demo_tag, range of means across archs
        arch_ranges = {}
        for tag in sorted(edf["demo_tag"].unique()):
            sub = edf[edf["demo_tag"] == tag]
            if len(sub) < 2:
                continue
            arch_ranges[tag] = sub["mean"].max() - sub["mean"].min()

        mean_teacher_range = np.mean(list(teacher_ranges.values())) if teacher_ranges else 0
        mean_arch_range = np.mean(list(arch_ranges.values())) if arch_ranges else 0

        print("  Teacher quality range per architecture (spread across teacher checkpoints):")
        for arch, rng in teacher_ranges.items():
            print(f"    {ARCH_LABELS.get(arch, arch):15s}: {rng:10.1f}")
        print(f"    {'MEAN':15s}: {mean_teacher_range:10.1f}")

        print()
        print("  Architecture range per teacher level (spread across architectures):")
        for tag, rng in sorted(arch_ranges.items()):
            print(f"    {tag:35s}: {rng:10.1f}")
        print(f"    {'MEAN':35s}: {mean_arch_range:10.1f}")

        ratio = mean_teacher_range / mean_arch_range if mean_arch_range > 0 else float("inf")
        print(f"\n  >>> Teacher range / Arch range = {mean_teacher_range:.1f} / {mean_arch_range:.1f} = {ratio:.2f}x")

    # Overall
    print(f"\n--- Overall (all environments) ---\n")
    all_teacher_ranges = []
    all_arch_ranges = []
    for env in sorted(df["env"].unique()):
        edf = agg[agg["env"] == env]
        for arch in ARCH_ORDER:
            sub = edf[edf["arch"] == arch]
            if len(sub) >= 2:
                all_teacher_ranges.append(sub["mean"].max() - sub["mean"].min())
        for tag in sorted(edf["demo_tag"].unique()):
            sub = edf[edf["demo_tag"] == tag]
            if len(sub) >= 2:
                all_arch_ranges.append(sub["mean"].max() - sub["mean"].min())

    ot = np.mean(all_teacher_ranges) if all_teacher_ranges else 0
    oa = np.mean(all_arch_ranges) if all_arch_ranges else 0
    ratio = ot / oa if oa > 0 else float("inf")
    print(f"  Mean teacher range (across envs & archs): {ot:.1f}")
    print(f"  Mean arch range    (across envs & tags) : {oa:.1f}")
    print(f"  >>> Overall ratio = {ratio:.2f}x")

    # Also print a compact summary table
    print(f"\n{SEPARATOR}")
    print("  Compact Summary — Mean student return by (env, arch) across all teachers:\n")
    pivot = (
        agg.groupby(["env", "arch"])["mean"]
        .mean()
        .unstack("arch")
    )
    pivot = pivot[[c for c in ARCH_ORDER if c in pivot.columns]]
    pivot.columns = [ARCH_LABELS.get(c, c) for c in pivot.columns]
    print(_table(pivot.round(1)))


# ---------------------------------------------------------------------------
# 2. Baselines — MLP / FrameStack
# ---------------------------------------------------------------------------

def analyze_baselines(df_baselines: pd.DataFrame, df_sweep: pd.DataFrame | None) -> None:
    _header("2. BASELINES — MLP / FrameStack vs Temporal Architectures")

    if df_sweep is not None:
        combined = pd.concat([df_sweep, df_baselines], ignore_index=True)
    else:
        combined = df_baselines

    agg = (
        combined.groupby(["env", "arch"])["student_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    all_archs = [a for a in ["mlp", "framestack"] + ARCH_ORDER if a in agg["arch"].values]

    for env in sorted(combined["env"].unique()):
        print(f"\n--- {env} ---\n")
        edf = agg[agg["env"] == env].copy()
        edf["label"] = edf["arch"].map(lambda a: ARCH_LABELS.get(a, a))
        edf = edf.set_index("arch").reindex(all_archs).dropna(subset=["mean"])
        edf["label"] = edf.index.map(lambda a: ARCH_LABELS.get(a, a))
        print(f"  {'Architecture':15s} {'Mean Return':>12s} {'Std':>10s} {'N':>5s}")
        print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*5}")
        for arch in edf.index:
            row = edf.loc[arch]
            lbl = ARCH_LABELS.get(arch, arch)
            marker = " <-- no memory" if arch in ("mlp", "framestack") else ""
            print(f"  {lbl:15s} {row['mean']:12.1f} {row['std']:10.1f} {int(row['count']):5d}{marker}")


# ---------------------------------------------------------------------------
# 3. Parameter-Matched — GRU d=64 vs GRU d=160 vs STU
# ---------------------------------------------------------------------------

def analyze_param_matched(df_pm: pd.DataFrame, df_sweep: pd.DataFrame | None) -> None:
    _header("3. PARAMETER-MATCHED — GRU d=64 (51K) vs GRU d=160 (312K) vs STU (338K)")

    if df_sweep is not None:
        # Combine: original GRU rows + param-matched GRU rows + STU rows
        gru_orig = df_sweep[df_sweep["arch"] == "gru"].copy()
        gru_orig["variant"] = "GRU d=64 (51K)"
        stu_orig = df_sweep[df_sweep["arch"] == "stu"].copy()
        stu_orig["variant"] = "STU (338K)"
        pm = df_pm.copy()
        pm["variant"] = "GRU d=160 (312K)"
        combined = pd.concat([gru_orig, pm, stu_orig], ignore_index=True)
    else:
        combined = df_pm.copy()
        combined["variant"] = combined.apply(
            lambda r: f"{ARCH_LABELS.get(r['arch'], r['arch'])} ({int(r['n_params'])/1000:.0f}K)", axis=1
        )

    agg = (
        combined.groupby(["env", "variant"])["student_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for env in sorted(combined["env"].unique()):
        print(f"\n--- {env} ---\n")
        edf = agg[agg["env"] == env].sort_values("mean", ascending=False)
        print(f"  {'Variant':25s} {'Mean Return':>12s} {'Std':>10s} {'N':>5s}")
        print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*5}")
        for _, row in edf.iterrows():
            print(f"  {row['variant']:25s} {row['mean']:12.1f} {row['std']:10.1f} {int(row['count']):5d}")

        best = edf.iloc[0]
        print(f"\n  >>> Winner: {best['variant']} ({best['mean']:.1f})")


# ---------------------------------------------------------------------------
# 4. Extra Seeds — merge 5 seeds
# ---------------------------------------------------------------------------

def analyze_extra_seeds(df_extra: pd.DataFrame, df_sweep: pd.DataFrame | None) -> None:
    _header("4. EXTRA SEEDS — 5-Seed Averages (42, 123, 456, 789, 1337)")

    if df_sweep is not None:
        combined = pd.concat([df_sweep, df_extra], ignore_index=True)
    else:
        combined = df_extra

    # Verify seed counts
    seeds = sorted(combined["seed"].unique())
    print(f"  Seeds present: {seeds}  (total unique: {len(seeds)})\n")

    agg = (
        combined.groupby(["env", "arch"])["student_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for env in sorted(combined["env"].unique()):
        print(f"\n--- {env} ---\n")
        edf = agg[agg["env"] == env]
        print(f"  {'Architecture':15s} {'Mean Return':>12s} {'Std':>10s} {'Seeds':>6s}")
        print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*6}")
        for arch in ARCH_ORDER:
            sub = edf[edf["arch"] == arch]
            if sub.empty:
                continue
            row = sub.iloc[0]
            lbl = ARCH_LABELS.get(arch, arch)
            print(f"  {lbl:15s} {row['mean']:12.1f} {row['std']:10.1f} {int(row['count']):6d}")

    # Also print updated key numbers: best arch per env
    print(f"\n{SEPARATOR}")
    print("  Updated key numbers (5-seed):\n")
    for env in sorted(combined["env"].unique()):
        edf = agg[agg["env"] == env]
        best_row = edf.loc[edf["mean"].idxmax()]
        print(f"  {env}: best = {ARCH_LABELS.get(best_row['arch'], best_row['arch'])} "
              f"({best_row['mean']:.1f} +/- {best_row['std']:.1f}, n={int(best_row['count'])})")


# ---------------------------------------------------------------------------
# 5. Demo Ablation
# ---------------------------------------------------------------------------

def analyze_demo_ablation(df: pd.DataFrame) -> None:
    _header("5. DEMO ABLATION — Return vs Demo Count")

    # Expect a column like 'demo_tag' or 'n_demos'
    # Derive demo count from demo_tag if needed
    if "n_demos" not in df.columns:
        # Try to parse from demo_tag (format might be like "demos_10" or similar)
        # Or from a dedicated column. Fall back to demo_tag as-is.
        df = df.copy()
        df["demo_label"] = df["demo_tag"]
    else:
        df = df.copy()
        df["demo_label"] = df["n_demos"].astype(str) + " demos"

    agg = (
        df.groupby(["env", "demo_label", "arch"])["student_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for env in sorted(df["env"].unique()):
        print(f"\n--- {env} ---\n")
        edf = agg[agg["env"] == env]
        demos = sorted(edf["demo_label"].unique())
        archs = [a for a in ARCH_ORDER if a in edf["arch"].values]

        # Header
        arch_labels = [ARCH_LABELS.get(a, a) for a in archs]
        hdr = f"  {'Demo Level':30s}" + "".join(f" {l:>12s}" for l in arch_labels)
        print(hdr)
        print(f"  {'-'*30}" + "".join(f" {'-'*12}" for _ in archs))

        for demo in demos:
            vals = []
            for arch in archs:
                sub = edf[(edf["demo_label"] == demo) & (edf["arch"] == arch)]
                if sub.empty:
                    vals.append("     —")
                else:
                    vals.append(f" {sub.iloc[0]['mean']:12.1f}")
            print(f"  {demo:30s}" + "".join(vals))


# ---------------------------------------------------------------------------
# 6 & 7. Walker2d / MetaDrive summary
# ---------------------------------------------------------------------------

def analyze_env_sweep(df: pd.DataFrame, env_name: str) -> None:
    _header(f"{env_name.upper()} — Sweep Summary")

    agg = (
        df.groupby(["env", "demo_tag", "arch"])["student_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for env in sorted(df["env"].unique()):
        print(f"\n--- {env} ---\n")
        edf = agg[agg["env"] == env]
        tags = sorted(edf["demo_tag"].unique())
        archs = [a for a in ARCH_ORDER if a in edf["arch"].values]

        arch_labels = [ARCH_LABELS.get(a, a) for a in archs]
        hdr = f"  {'Teacher Level':35s}" + "".join(f" {l:>12s}" for l in arch_labels)
        print(hdr)
        print(f"  {'-'*35}" + "".join(f" {'-'*12}" for _ in archs))

        for tag in tags:
            vals = []
            for arch in archs:
                sub = edf[(edf["demo_tag"] == tag) & (edf["arch"] == arch)]
                if sub.empty:
                    vals.append("           —")
                else:
                    vals.append(f" {sub.iloc[0]['mean']:12.1f}")
            print(f"  {tag:35s}" + "".join(vals))

        # Overall mean per arch
        print(f"  {'-'*35}" + "".join(f" {'-'*12}" for _ in archs))
        overall = []
        for arch in archs:
            sub = edf[edf["arch"] == arch]
            overall.append(f" {sub['mean'].mean():12.1f}")
        print(f"  {'MEAN':35s}" + "".join(overall))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of all experiment CSVs."
    )
    parser.add_argument(
        "--base-dir",
        default="experiments/v2/sweep_adroit",
        help="Base directory containing CSV files (default: experiments/v2/sweep_adroit)",
    )
    args = parser.parse_args()

    base = args.base_dir

    print(f"Base directory: {base}\n")
    print(SEPARATOR)
    print("  Loading CSV files...")
    print(SEPARATOR)

    df_sweep = _load(os.path.join(base, "sweep_results.csv"), "Original sweep")
    df_baselines = _load(os.path.join(base, "baselines_results.csv"), "Baselines (MLP + FrameStack)")
    df_param = _load(os.path.join(base, "param_matched_results.csv"), "Param-matched GRU d=160")
    df_extra = _load(os.path.join(base, "extra_seeds_results.csv"), "Extra seeds (789, 1337)")
    df_demo = _load(os.path.join(base, "demo_ablation_results.csv"), "Demo ablation")
    df_walker = _load(os.path.join(base, "walker2d_results.csv"), "Walker2d")
    df_meta = _load(os.path.join(base, "metadrive_results.csv"), "MetaDrive")

    print()

    # 1. Original sweep analysis
    if df_sweep is not None:
        analyze_sweep(df_sweep)

    # 2. Baselines
    if df_baselines is not None:
        analyze_baselines(df_baselines, df_sweep)

    # 3. Parameter-matched
    if df_param is not None:
        analyze_param_matched(df_param, df_sweep)

    # 4. Extra seeds
    if df_extra is not None:
        analyze_extra_seeds(df_extra, df_sweep)

    # 5. Demo ablation
    if df_demo is not None:
        analyze_demo_ablation(df_demo)

    # 6. Walker2d
    if df_walker is not None:
        analyze_env_sweep(df_walker, "Walker2d")

    # 7. MetaDrive
    if df_meta is not None:
        analyze_env_sweep(df_meta, "MetaDrive")

    # Final summary
    _header("ANALYSIS COMPLETE")
    loaded = sum(
        x is not None
        for x in [df_sweep, df_baselines, df_param, df_extra, df_demo, df_walker, df_meta]
    )
    print(f"  Loaded {loaded}/7 CSV files from {base}/")
    if loaded == 0:
        print("  No files found. Check --base-dir path.")


if __name__ == "__main__":
    main()
