#!/usr/bin/env python3
"""
Collect demonstrations from MetaDrive teacher.
Unlike MuJoCo POMDP, the teacher sees the SAME observations as the student
(MetaDrive is inherently partially observable via lidar range limits).

Usage:
  python wasumindV2/teachers/collect_metadrive_demos.py \
    --teacher-path experiments/v2/teachers_metadrive/metadrive_seed42/teacher_1000000 \
    --n-episodes 500 --output-dir experiments/v2/demos_metadrive
"""

import argparse
import json
import os
import sys
import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.metadrive_wrapper import make_metadrive


def collect_demos(teacher, env, n_episodes, noise_sigma=0.0, seed=42):
    all_obs, all_actions, all_returns, episode_lengths = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_seq, act_seq = [], []
        lstm_states = None
        episode_starts = np.array([True])
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 1000:
            action, lstm_states = teacher.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            if noise_sigma > 0:
                action = action + np.random.normal(0, noise_sigma, size=action.shape)
                action = np.clip(action, env.action_space.low, env.action_space.high)

            obs_seq.append(obs.copy())
            act_seq.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            episode_starts = np.array([done])
            steps += 1

        all_obs.append(np.array(obs_seq, dtype=np.float32))
        all_actions.append(np.array(act_seq, dtype=np.float32))
        all_returns.append(total_reward)
        episode_lengths.append(len(obs_seq))

        if (ep + 1) % 50 == 0:
            print(f"  {ep+1}/{n_episodes} episodes, avg return: {np.mean(all_returns[-50:]):.1f}")

    return {
        "obs": all_obs,
        "actions": all_actions,
        "returns": np.array(all_returns, dtype=np.float32),
        "lengths": np.array(episode_lengths, dtype=np.int32),
    }


def pad_to_fixed_length(episodes, max_len):
    obs_dim = episodes[0].shape[-1]
    n = len(episodes)
    padded = np.zeros((n, max_len, obs_dim), dtype=np.float32)
    masks = np.zeros((n, max_len), dtype=np.float32)
    for i, ep in enumerate(episodes):
        T = min(len(ep), max_len)
        padded[i, :T] = ep[:T]
        masks[i, :T] = 1.0
    return padded, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/v2/demos_metadrive")
    args = parser.parse_args()

    print(f"=== Collecting MetaDrive demos, noise={args.noise_sigma} ===")
    teacher = RecurrentPPO.load(args.teacher_path)
    env = make_metadrive(num_scenarios=100, seed=args.seed)

    t0 = time.time()
    demos = collect_demos(teacher, env, args.n_episodes, args.noise_sigma, args.seed)
    env.close()

    max_len = max(demos["lengths"])
    obs_padded, obs_masks = pad_to_fixed_length(demos["obs"], max_len)
    act_padded, _ = pad_to_fixed_length(demos["actions"], max_len)

    noise_tag = f"noise{args.noise_sigma:.1f}".replace(".", "")
    teacher_tag = os.path.basename(args.teacher_path)
    save_dir = os.path.join(args.output_dir, f"{teacher_tag}_{noise_tag}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        "obs": torch.from_numpy(obs_padded),
        "actions": torch.from_numpy(act_padded),
        "masks": torch.from_numpy(obs_masks),
        "returns": torch.from_numpy(demos["returns"]),
        "lengths": torch.from_numpy(demos["lengths"]),
        "metadata": {
            "env": "metadrive",
            "teacher_path": args.teacher_path,
            "noise_sigma": args.noise_sigma,
            "n_episodes": args.n_episodes,
            "obs_dim": int(obs_padded.shape[-1]),
            "act_dim": int(act_padded.shape[-1]),
            "max_len": int(max_len),
            "mean_return": float(demos["returns"].mean()),
            "std_return": float(demos["returns"].std()),
            "seed": args.seed,
        },
    }, os.path.join(save_dir, "demos.pt"))

    print(f"Saved {args.n_episodes} demos. Mean return: {demos['returns'].mean():.1f}")


if __name__ == "__main__":
    main()
