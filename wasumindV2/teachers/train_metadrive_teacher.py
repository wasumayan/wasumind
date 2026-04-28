#!/usr/bin/env python3
"""
Train RecurrentPPO teacher on MetaDrive.

MetaDrive is inherently partially observable: lidar has limited range.
The teacher trains with the same observations the student will see.
Teacher quality comes from training duration, not privileged info.

Usage:
  python wasumindV2/teachers/train_metadrive_teacher.py --seed 42 --output-dir experiments/v2/teachers_metadrive
  python wasumindV2/teachers/train_metadrive_teacher.py --seed 42 --quick
"""

import argparse
import json
import os
import sys
import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.metadrive_wrapper import make_metadrive


CHECKPOINT_STEPS = [100_000, 300_000, 500_000, 1_000_000]


def make_env(seed=0):
    def _init():
        return make_metadrive(traffic_density=0.1, num_scenarios=100, max_steps=1000, seed=seed)
    return _init


def evaluate_teacher(model, n_episodes=20, seed=1000):
    env = make_metadrive(num_scenarios=50, seed=seed)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.array([True])
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 1000:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            episode_starts = np.array([done])
            steps += 1
        returns.append(total_reward)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--output-dir", default="experiments/v2/teachers_metadrive")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.total_steps = 10_000
        checkpoints = [5_000, 10_000]
    else:
        checkpoints = [s for s in CHECKPOINT_STEPS if s <= args.total_steps]

    run_dir = os.path.join(args.output_dir, f"metadrive_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"=== Training MetaDrive teacher, seed={args.seed}, steps={args.total_steps} ===")

    env = DummyVecEnv([make_env(seed=args.seed)])

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        seed=args.seed,
        verbose=1,
    )

    results = {"env": "MetaDrive", "seed": args.seed, "checkpoints": {}}
    t0 = time.time()

    # Train all checkpoints first
    trained_so_far = 0
    for ckpt_step in checkpoints:
        steps_to_train = ckpt_step - trained_so_far
        if steps_to_train <= 0:
            continue

        print(f"\n--- Training to {ckpt_step} steps ---")
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        trained_so_far = ckpt_step

        ckpt_path = os.path.join(run_dir, f"teacher_{ckpt_step}")
        model.save(ckpt_path)
        print(f"  Saved: {ckpt_path}")

        results["checkpoints"][str(ckpt_step)] = {
            "path": ckpt_path,
            "wall_time_s": time.time() - t0,
        }

    # Close training env before evaluation (MetaDrive only allows one instance)
    env.close()

    # Now evaluate each checkpoint
    for ckpt_step in checkpoints:
        ckpt_path = os.path.join(run_dir, f"teacher_{ckpt_step}")
        print(f"\n--- Evaluating {ckpt_step} checkpoint ---")
        eval_model = RecurrentPPO.load(ckpt_path)
        mean_ret, std_ret = evaluate_teacher(eval_model, n_episodes=20, seed=args.seed + 10000)
        print(f"  Eval: {mean_ret:.1f} ± {std_ret:.1f}")

        results["checkpoints"][str(ckpt_step)]["mean_return"] = mean_ret
        results["checkpoints"][str(ckpt_step)]["std_return"] = std_ret

        with open(os.path.join(run_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    results["total_time_s"] = time.time() - t0
    with open(os.path.join(run_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Done. Total time: {results['total_time_s']:.0f}s ===")


if __name__ == "__main__":
    main()
