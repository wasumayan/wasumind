#!/usr/bin/env python3
"""
Distill teacher demonstrations into student policies.

Usage:
  python wasumindV2/distillation/distill.py \
    --demo-path experiments/v2/demos/halfcheetah_pomdp/teacher_2000000_noise00/demos.pt \
    --arch gru --seed 42 --output-dir experiments/v2/students

  # Quick smoke test:
  python wasumindV2/distillation/distill.py --demo-path ... --arch gru --quick
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distillation.models import create_student


class DemoDataset(Dataset):
    """Dataset of (obs_sequence, action_sequence, mask) from teacher demos."""
    def __init__(self, obs, actions, masks, seq_len=256):
        # Chop episodes into fixed-length windows for batching
        self.windows_obs = []
        self.windows_act = []
        self.windows_mask = []

        N, T, _ = obs.shape
        for i in range(N):
            ep_len = int(masks[i].sum())
            for start in range(0, max(1, ep_len - seq_len + 1), seq_len // 2):
                end = min(start + seq_len, ep_len)
                if end - start < 10:
                    continue
                w_obs = torch.zeros(seq_len, obs.shape[-1])
                w_act = torch.zeros(seq_len, actions.shape[-1])
                w_mask = torch.zeros(seq_len)
                chunk_len = end - start
                w_obs[:chunk_len] = obs[i, start:end]
                w_act[:chunk_len] = actions[i, start:end]
                w_mask[:chunk_len] = 1.0
                self.windows_obs.append(w_obs)
                self.windows_act.append(w_act)
                self.windows_mask.append(w_mask)

        self.windows_obs = torch.stack(self.windows_obs)
        self.windows_act = torch.stack(self.windows_act)
        self.windows_mask = torch.stack(self.windows_mask)

    def __len__(self):
        return len(self.windows_obs)

    def __getitem__(self, idx):
        return self.windows_obs[idx], self.windows_act[idx], self.windows_mask[idx]


def evaluate_in_env(student, env_name, n_episodes=20, seed=5000, device="cpu"):
    """Evaluate student by rolling it out in the POMDP environment."""
    import gymnasium as gym
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from envs.pomdp_wrappers import ENV_CONFIGS

    cfg = ENV_CONFIGS[env_name]
    env = cfg["make_fn"]()
    student.eval()

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_history = [obs]
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(np.stack(obs_history)).float().unsqueeze(0).to(device)
                pred_actions = student(obs_tensor)
                action = pred_actions[0, -1].cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            obs_history.append(obs)
            done = terminated or truncated

        returns.append(total_reward)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def train_student(student, train_loader, val_loader, n_epochs, lr, device, patience=10):
    """Train student on demo data. Returns training history."""
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(n_epochs):
        # Train
        student.train()
        train_losses = []
        for obs, act, mask in train_loader:
            obs, act, mask = obs.to(device), act.to(device), mask.to(device)
            pred = student(obs)
            # Masked MSE loss
            diff = (pred - act) ** 2
            loss = (diff * mask.unsqueeze(-1)).sum() / mask.sum() / act.shape[-1]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        student.eval()
        val_losses = []
        with torch.no_grad():
            for obs, act, mask in val_loader:
                obs, act, mask = obs.to(device), act.to(device), mask.to(device)
                pred = student(obs)
                diff = (pred - act) ** 2
                loss = (diff * mask.unsqueeze(-1)).sum() / mask.sum() / act.shape[-1]
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4}/{n_epochs}: train={train_loss:.6f} val={val_loss:.6f} lr={lr_now:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-path", required=True, help="Path to demos.pt")
    parser.add_argument("--arch", required=True, choices=["gru", "lstm", "transformer", "stu", "mlp", "framestack"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--output-dir", default="experiments/v2/students")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--frame-stack", type=int, default=8, help="Frames to stack for framestack arch")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_epochs = 5
        args.batch_size = 32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load demos
    print(f"Loading demos from {args.demo_path}")
    data = torch.load(args.demo_path, map_location="cpu", weights_only=False)
    meta = data["metadata"]
    obs_dim = meta["obs_dim"]
    act_dim = meta["act_dim"]
    env_name = meta["env"]

    print(f"  Env: {env_name}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  {data['obs'].shape[0]} episodes, mean return: {meta['mean_return']:.1f}")

    # Train/val split
    n_total = data["obs"].shape[0]
    n_train = int(0.8 * n_total)
    train_dataset = DemoDataset(data["obs"][:n_train], data["actions"][:n_train],
                                data["masks"][:n_train], seq_len=args.seq_len)
    val_dataset = DemoDataset(data["obs"][n_train:], data["actions"][n_train:],
                              data["masks"][n_train:], seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"  Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")

    # Create student
    stu_kwargs = {}
    if args.arch == "stu":
        stu_kwargs["seq_len"] = meta["max_len"]
    elif args.arch == "framestack":
        stu_kwargs["frame_stack"] = args.frame_stack
    student = create_student(
        args.arch, obs_dim=obs_dim, act_dim=act_dim,
        d_model=args.d_model, n_layers=args.n_layers, **stu_kwargs
    ).to(device)
    n_params = student.count_parameters()
    print(f"  Student: {args.arch}, params={n_params:,}, d_model={args.d_model}, layers={args.n_layers}")

    # Train
    print(f"\nTraining for {args.n_epochs} epochs...")
    t0 = time.time()
    history = train_student(student, train_loader, val_loader, args.n_epochs, args.lr, device)
    train_time = time.time() - t0

    # Evaluate in environment
    eval_result = {"mean_return": None, "std_return": None}
    if not args.skip_eval:
        print(f"\nEvaluating in {env_name}...")
        try:
            mean_ret, std_ret = evaluate_in_env(student, env_name, n_episodes=20, device=device)
            eval_result = {"mean_return": mean_ret, "std_return": std_ret}
            print(f"  Return: {mean_ret:.1f} ± {std_ret:.1f}")
        except Exception as e:
            print(f"  Eval failed: {e}")

    # Save
    demo_tag = os.path.basename(os.path.dirname(args.demo_path))
    save_dir = os.path.join(args.output_dir, env_name, demo_tag, args.arch, f"seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(student.state_dict(), os.path.join(save_dir, "student.pt"))

    result = {
        "arch": args.arch,
        "seed": args.seed,
        "env": env_name,
        "demo_tag": demo_tag,
        "n_params": n_params,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_epochs": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": min(history["val_loss"]),
        "train_time_s": train_time,
        "teacher_mean_return": meta["mean_return"],
        "noise_sigma": meta.get("noise_sigma", 0.0),
        "eval": eval_result,
    }
    if eval_result["mean_return"] is not None and meta["mean_return"] != 0:
        result["distill_efficiency"] = eval_result["mean_return"] / meta["mean_return"]

    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {save_dir}")
    print(f"  Train time: {train_time:.0f}s, Final val loss: {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
