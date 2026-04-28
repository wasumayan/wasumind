#!/usr/bin/env python3
"""
Train RecurrentPPO teachers on FULL-observation MuJoCo environments.
Saves checkpoints at intervals to create "teacher quality levels."

Usage:
  python wasumindV2/teachers/train_teacher.py --env HalfCheetah-v4 --seed 42 --output-dir experiments/v2/teachers
  python wasumindV2/teachers/train_teacher.py --env Ant-v4 --seed 42 --output-dir experiments/v2/teachers
  python wasumindV2/teachers/train_teacher.py --env HalfCheetah-v4 --seed 42 --quick  # 10K steps smoke test
"""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO


CHECKPOINT_STEPS = [200_000, 500_000, 1_000_000, 2_000_000]


def evaluate_teacher(model, env_id, n_episodes=20, seed=1000):
    """Evaluate teacher and return mean/std of episode returns."""
    env = gym.make(env_id)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        lstm_states = None
        episode_starts = np.array([True])
        total_reward = 0
        done = False
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            episode_starts = np.array([done])
        returns.append(total_reward)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Full-obs env ID (e.g. HalfCheetah-v4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--output-dir", default="experiments/v2/teachers")
    parser.add_argument("--quick", action="store_true", help="10K steps smoke test")
    args = parser.parse_args()

    if args.quick:
        args.total_steps = 10_000
        checkpoints = [5_000, 10_000]
    else:
        checkpoints = [s for s in CHECKPOINT_STEPS if s <= args.total_steps]

    env_name = args.env.replace("-", "").replace(".", "").lower()
    run_dir = os.path.join(args.output_dir, f"{env_name}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"=== Training teacher: {args.env}, seed={args.seed}, steps={args.total_steps} ===")
    print(f"Output: {run_dir}")
    print(f"Checkpoints at: {checkpoints}")

    env = gym.make(args.env)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        seed=args.seed,
        verbose=1,
    )

    results = {"env": args.env, "seed": args.seed, "checkpoints": {}}
    t0 = time.time()

    trained_so_far = 0
    for ckpt_step in checkpoints:
        steps_to_train = ckpt_step - trained_so_far
        if steps_to_train <= 0:
            continue

        print(f"\n--- Training to {ckpt_step} steps ({steps_to_train} new) ---")
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        trained_so_far = ckpt_step

        # Save checkpoint
        ckpt_path = os.path.join(run_dir, f"teacher_{ckpt_step}")
        model.save(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Evaluate
        mean_ret, std_ret = evaluate_teacher(model, args.env, n_episodes=20, seed=args.seed + 10000)
        print(f"  Eval: {mean_ret:.1f} ± {std_ret:.1f}")

        results["checkpoints"][str(ckpt_step)] = {
            "path": ckpt_path,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "wall_time_s": time.time() - t0,
        }

        # Save results incrementally
        with open(os.path.join(run_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    env.close()
    total_time = time.time() - t0
    results["total_time_s"] = total_time
    print(f"\n=== Done. Total time: {total_time:.0f}s ===")

    with open(os.path.join(run_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
