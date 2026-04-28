#!/usr/bin/env python3
"""
Evaluate trained MetaDrive students that were distilled with --skip-eval.

Walks a student directory, finds result.json files with null eval results,
loads the student model, evaluates in MetaDrive, and updates result.json.

Usage:
  python wasumindV2/evaluation/evaluate_metadrive.py \
    --students-dir experiments/v2/students_metadrive \
    --n-episodes 20
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distillation.models import create_student

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from envs.metadrive_wrapper import make_metadrive

METADRIVE_OBS_DIM = 91
METADRIVE_ACT_DIM = 2


def evaluate_student(student, n_episodes=20, seed=5000, device="cpu"):
    env = make_metadrive(num_scenarios=100, seed=seed)
    student.eval()
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_history = [obs]
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 1000:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(
                    np.stack(obs_history)
                ).float().unsqueeze(0).to(device)
                pred = student(obs_tensor)
                action = pred[0, -1].cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            obs_history.append(obs)
            done = terminated or truncated
            steps += 1

        returns.append(total_reward)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--students-dir", required=True)
    parser.add_argument("--n-episodes", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluated = 0

    for root, dirs, files in os.walk(args.students_dir):
        if "result.json" not in files or "student.pt" not in files:
            continue

        result_path = os.path.join(root, "result.json")
        with open(result_path) as f:
            result = json.load(f)

        if result["eval"].get("mean_return") is not None:
            continue

        arch = result["arch"]
        d_model = result.get("d_model", 64)
        n_layers = result.get("n_layers", 2)

        print(f"\nEvaluating: {root}")
        print(f"  arch={arch}, d_model={d_model}, n_layers={n_layers}")

        stu_kwargs = {}
        if arch == "stu":
            weights = torch.load(
                os.path.join(root, "student.pt"), map_location=device, weights_only=True
            )
            for k, v in weights.items():
                if "spectral_filters" in k:
                    stu_kwargs["seq_len"] = v.shape[-1]
                    break
            else:
                stu_kwargs["seq_len"] = 1000
        elif arch == "framestack":
            stu_kwargs["frame_stack"] = 8

        student = create_student(
            arch, obs_dim=METADRIVE_OBS_DIM, act_dim=METADRIVE_ACT_DIM,
            d_model=d_model, n_layers=n_layers, **stu_kwargs
        )
        if arch == "stu":
            student.load_state_dict(weights)
        else:
            student.load_state_dict(torch.load(
                os.path.join(root, "student.pt"), map_location=device, weights_only=True
            ))
        student = student.to(device)

        mean_ret, std_ret = evaluate_student(
            student, n_episodes=args.n_episodes, device=device
        )
        print(f"  Return: {mean_ret:.1f} ± {std_ret:.1f}")

        result["eval"]["mean_return"] = mean_ret
        result["eval"]["std_return"] = std_ret
        if result.get("teacher_mean_return") and result["teacher_mean_return"] != 0:
            result["distill_efficiency"] = mean_ret / result["teacher_mean_return"]

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        evaluated += 1

    print(f"\nEvaluated {evaluated} students total.")


if __name__ == "__main__":
    main()
