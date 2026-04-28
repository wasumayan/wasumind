#!/usr/bin/env python3
"""
Collect demonstrations from trained teachers for student distillation.

The teacher acts in the FULL-observation environment (it was trained there),
but we record observations as the STUDENT would see them (velocity-masked POMDP).
This simulates: teacher has privileged info, student works with partial obs.

Usage:
  python wasumindV2/teachers/collect_demos.py \
    --teacher-path experiments/v2/teachers/halfcheetahv5_seed42/teacher_2000000 \
    --env halfcheetah_pomdp --n-episodes 500 --noise-sigma 0.0 \
    --output-dir experiments/v2/demos

  # With noisy actions (imperfect teacher simulation):
  python wasumindV2/teachers/collect_demos.py \
    --teacher-path ... --noise-sigma 0.3 ...
"""

import argparse
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.pomdp_wrappers import ENV_CONFIGS

from sb3_contrib import RecurrentPPO


def collect_demos(teacher, full_env, pomdp_env, n_episodes, noise_sigma=0.0, seed=42):
    """Collect demonstrations. Teacher sees full obs, we save POMDP obs."""
    all_obs = []
    all_actions = []
    all_returns = []
    episode_lengths = []

    for ep in range(n_episodes):
        # Reset both envs with same seed for identical initial state
        ep_seed = seed + ep
        full_obs, _ = full_env.reset(seed=ep_seed)
        pomdp_obs, _ = pomdp_env.reset(seed=ep_seed)

        obs_seq = []
        act_seq = []
        lstm_states = None
        episode_starts = np.array([True])
        total_reward = 0
        done = False

        while not done:
            # Teacher predicts from FULL observation
            action, lstm_states = teacher.predict(
                full_obs, state=lstm_states, episode_start=episode_starts,
                deterministic=True
            )

            # Add noise if requested
            if noise_sigma > 0:
                action = action + np.random.normal(0, noise_sigma, size=action.shape)
                action = np.clip(action, full_env.action_space.low, full_env.action_space.high)

            # Save POMDP observation (what the student will see)
            obs_seq.append(pomdp_obs.copy())
            act_seq.append(action.copy())

            # Step both environments with same action
            full_obs, reward, f_term, f_trunc, _ = full_env.step(action)
            pomdp_obs, _, p_term, p_trunc, _ = pomdp_env.step(action)
            total_reward += reward
            done = f_term or f_trunc
            episode_starts = np.array([done])

        all_obs.append(np.array(obs_seq, dtype=np.float32))
        all_actions.append(np.array(act_seq, dtype=np.float32))
        all_returns.append(total_reward)
        episode_lengths.append(len(obs_seq))

        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes, "
                  f"avg return: {np.mean(all_returns[-50:]):.1f}, "
                  f"avg length: {np.mean(episode_lengths[-50:]):.0f}")

    return {
        "obs": all_obs,          # list of (T_i, obs_dim) arrays
        "actions": all_actions,  # list of (T_i, act_dim) arrays
        "returns": np.array(all_returns, dtype=np.float32),
        "lengths": np.array(episode_lengths, dtype=np.int32),
    }


def pad_to_fixed_length(episodes, max_len):
    """Pad variable-length episodes to fixed length for batched training."""
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
    parser.add_argument("--teacher-path", required=True, help="Path to teacher checkpoint")
    parser.add_argument("--env", required=True, choices=list(ENV_CONFIGS.keys()))
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/v2/demos")
    args = parser.parse_args()

    cfg = ENV_CONFIGS[args.env]
    full_env_id = cfg["full_env_id"]

    print(f"=== Collecting demos: {args.env}, noise={args.noise_sigma} ===")
    print(f"Teacher: {args.teacher_path}")

    # Load teacher
    teacher = RecurrentPPO.load(args.teacher_path)

    # Create environments
    full_env = gym.make(full_env_id)
    pomdp_env = cfg["make_fn"]()

    # Collect
    t0 = time.time()
    demos = collect_demos(
        teacher, full_env, pomdp_env,
        n_episodes=args.n_episodes,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
    collect_time = time.time() - t0

    full_env.close()
    pomdp_env.close()

    # Pad episodes to max length
    max_len = max(demos["lengths"])
    obs_padded, obs_masks = pad_to_fixed_length(demos["obs"], max_len)
    act_padded, _ = pad_to_fixed_length(demos["actions"], max_len)

    # Save
    noise_tag = f"noise{args.noise_sigma:.1f}".replace(".", "")
    teacher_tag = os.path.basename(args.teacher_path)
    save_dir = os.path.join(args.output_dir, args.env, f"{teacher_tag}_{noise_tag}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "demos.pt")
    torch.save({
        "obs": torch.from_numpy(obs_padded),        # (N, T, obs_dim)
        "actions": torch.from_numpy(act_padded),     # (N, T, act_dim)
        "masks": torch.from_numpy(obs_masks),        # (N, T)
        "returns": torch.from_numpy(demos["returns"]),
        "lengths": torch.from_numpy(demos["lengths"]),
        "metadata": {
            "env": args.env,
            "full_env_id": full_env_id,
            "teacher_path": args.teacher_path,
            "noise_sigma": args.noise_sigma,
            "n_episodes": args.n_episodes,
            "obs_dim": cfg["obs_dim"],
            "act_dim": cfg["act_dim"],
            "max_len": int(max_len),
            "mean_return": float(demos["returns"].mean()),
            "std_return": float(demos["returns"].std()),
            "seed": args.seed,
        },
    }, save_path)

    print(f"\nSaved {args.n_episodes} demos to {save_path}")
    print(f"  Obs shape: {obs_padded.shape}, Act shape: {act_padded.shape}")
    print(f"  Mean return: {demos['returns'].mean():.1f} ± {demos['returns'].std():.1f}")
    print(f"  Mean length: {demos['lengths'].mean():.0f}")
    print(f"  Collection time: {collect_time:.0f}s")

    # Save metadata as JSON too
    with open(os.path.join(save_dir, "demo_info.json"), "w") as f:
        json.dump({
            "n_episodes": args.n_episodes,
            "mean_return": float(demos["returns"].mean()),
            "std_return": float(demos["returns"].std()),
            "mean_length": float(demos["lengths"].mean()),
            "noise_sigma": args.noise_sigma,
            "teacher_path": args.teacher_path,
            "obs_shape": list(obs_padded.shape),
            "act_shape": list(act_padded.shape),
        }, f, indent=2)


if __name__ == "__main__":
    main()
