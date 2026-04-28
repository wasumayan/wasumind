#!/usr/bin/env python3
"""
Render MuJoCo POMDP demo frames on the cluster.
Saves PNG frame strips and episode stats — no display needed.

Usage:
  python wasumindV2/evaluation/render_mujoco_frames.py \
    --results-dir experiments/v2/students \
    --output-dir experiments/v2/visuals
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distillation.models import create_student
from envs.pomdp_wrappers import ENV_CONFIGS


def rollout_with_frames(student, env_name, n_frames=8, max_steps=500, seed=42, device="cpu"):
    """Roll out student, capturing frames at evenly-spaced intervals."""
    cfg = ENV_CONFIGS[env_name]
    render_env = gym.make(cfg["full_env_id"], render_mode="rgb_array")
    from envs.pomdp_wrappers import VelocityMaskWrapper
    pomdp_env = VelocityMaskWrapper(render_env, cfg["keep_indices"])

    obs, _ = pomdp_env.reset(seed=seed)
    obs_history = [obs]
    all_frames = []
    total_reward = 0
    steps = 0

    student.eval()
    with torch.no_grad():
        while steps < max_steps:
            frame = render_env.render()
            if frame is not None:
                all_frames.append(frame)

            obs_tensor = torch.from_numpy(np.stack(obs_history)).float().unsqueeze(0).to(device)
            pred = student(obs_tensor)
            action = pred[0, -1].cpu().numpy()
            action = np.clip(action, pomdp_env.action_space.low, pomdp_env.action_space.high)

            obs, reward, terminated, truncated, info = pomdp_env.step(action)
            total_reward += reward
            obs_history.append(obs)
            steps += 1

            if terminated or truncated:
                break

    pomdp_env.close()

    # Select evenly-spaced frames
    if len(all_frames) > n_frames:
        indices = np.linspace(0, len(all_frames) - 1, n_frames, dtype=int)
        selected = [all_frames[i] for i in indices]
    else:
        selected = all_frames

    return selected, total_reward, steps


def save_frame_strip(frames, path, title=""):
    """Save frames as a horizontal strip PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frames)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, (ax, frame) in enumerate(zip(axes, frames)):
        ax.imshow(frame)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_comparison(frames_good, ret_good, arch_good,
                    frames_bad, ret_bad, arch_bad,
                    env_name, path):
    """Side-by-side comparison of good vs bad student."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(frames_good), len(frames_bad), 6)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    for i in range(n):
        axes[0, i].imshow(frames_good[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(frames_bad[i])
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel(f"{arch_good.upper()}\nReturn: {ret_good:.0f}", fontsize=11)
    axes[1, 0].set_ylabel(f"{arch_bad.upper()}\nReturn: {ret_bad:.0f}", fontsize=11)
    fig.suptitle(f"{env_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/v2/students")
    parser.add_argument("--output-dir", default="experiments/v2/visuals")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for env_name in ["halfcheetah_pomdp", "ant_pomdp"]:
        cfg = ENV_CONFIGS[env_name]
        print(f"\n=== {env_name} ===")

        # Find the best and worst students from the 1M teacher (good teacher)
        results_by_arch = {}
        teacher_tag = "teacher_1000000_noise00"
        base = os.path.join(args.results_dir, env_name, teacher_tag)
        if not os.path.exists(base):
            print(f"  No results at {base}")
            continue

        for arch in ["gru", "lstm", "stu", "transformer"]:
            result_path = os.path.join(base, arch, "seed42", "result.json")
            student_path = os.path.join(base, arch, "seed42", "student.pt")
            if not os.path.exists(result_path) or not os.path.exists(student_path):
                continue
            with open(result_path) as f:
                r = json.load(f)
            ret = r["eval"]["mean_return"] if r["eval"]["mean_return"] else 0
            results_by_arch[arch] = (ret, student_path)

        if len(results_by_arch) < 2:
            print(f"  Only {len(results_by_arch)} students found, need at least 2")
            continue

        sorted_archs = sorted(results_by_arch.items(), key=lambda x: x[1][0])
        worst_arch, (worst_ret, worst_path) = sorted_archs[0]
        best_arch, (best_ret, best_path) = sorted_archs[-1]

        print(f"  Best: {best_arch} ({best_ret:.0f})")
        print(f"  Worst: {worst_arch} ({worst_ret:.0f})")

        # Render best student
        kwargs = {"seq_len": 1000} if best_arch == "stu" else {}
        best_model = create_student(best_arch, cfg["obs_dim"], cfg["act_dim"], **kwargs)
        best_model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
        frames_best, ret_best, steps_best = rollout_with_frames(best_model, env_name)
        save_frame_strip(frames_best, os.path.join(args.output_dir, f"frames_best_{env_name}.png"),
                         f"{best_arch.upper()} (return: {ret_best:.0f}, {steps_best} steps)")
        print(f"  Rendered best: {best_arch} return={ret_best:.0f}")

        # Render worst student
        kwargs = {"seq_len": 1000} if worst_arch == "stu" else {}
        worst_model = create_student(worst_arch, cfg["obs_dim"], cfg["act_dim"], **kwargs)
        worst_model.load_state_dict(torch.load(worst_path, map_location="cpu", weights_only=True))
        frames_worst, ret_worst, steps_worst = rollout_with_frames(worst_model, env_name)
        save_frame_strip(frames_worst, os.path.join(args.output_dir, f"frames_worst_{env_name}.png"),
                         f"{worst_arch.upper()} (return: {ret_worst:.0f}, {steps_worst} steps)")
        print(f"  Rendered worst: {worst_arch} return={ret_worst:.0f}")

        # Side-by-side comparison
        save_comparison(frames_best, ret_best, best_arch,
                        frames_worst, ret_worst, worst_arch,
                        env_name,
                        os.path.join(args.output_dir, f"comparison_{env_name}.png"))
        print(f"  Saved comparison")

    print(f"\nAll visuals saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
