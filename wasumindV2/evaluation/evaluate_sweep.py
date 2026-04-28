#!/usr/bin/env python3
"""
Evaluate all trained student models and compile results.

Usage:
  python wasumindV2/evaluation/evaluate_sweep.py \
    --results-dir experiments/v2/students \
    --output experiments/v2/sweep_results.csv
"""

import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/v2/students")
    parser.add_argument("--output", default="experiments/v2/sweep_results.csv")
    args = parser.parse_args()

    rows = []
    for root, dirs, files in os.walk(args.results_dir):
        if "result.json" in files:
            with open(os.path.join(root, "result.json")) as f:
                r = json.load(f)
            rows.append({
                "env": r.get("env", ""),
                "demo_tag": r.get("demo_tag", ""),
                "arch": r.get("arch", ""),
                "seed": r.get("seed", ""),
                "n_params": r.get("n_params", ""),
                "noise_sigma": r.get("noise_sigma", 0.0),
                "teacher_return": r.get("teacher_mean_return", ""),
                "student_return": r["eval"].get("mean_return", ""),
                "student_std": r["eval"].get("std_return", ""),
                "distill_efficiency": r.get("distill_efficiency", ""),
                "final_val_loss": r.get("final_val_loss", ""),
                "train_time_s": r.get("train_time_s", ""),
            })

    if not rows:
        print("No results found.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} results to {args.output}")

    # Print summary
    from collections import defaultdict
    by_arch = defaultdict(list)
    for r in rows:
        if r["student_return"]:
            by_arch[r["arch"]].append(float(r["student_return"]))

    print("\nSummary by architecture:")
    for arch, returns in sorted(by_arch.items()):
        import numpy as np
        print(f"  {arch}: {np.mean(returns):.1f} ± {np.std(returns):.1f} (n={len(returns)})")


if __name__ == "__main__":
    main()
