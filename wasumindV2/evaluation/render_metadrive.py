#!/usr/bin/env python3
"""
Render MetaDrive distilled student policies as GIFs and frame strips.
Tries 3D camera rendering first (GPU required), falls back to top-down 2D.

Usage:
  python wasumindV2/evaluation/render_metadrive.py \
    --students-dir experiments/v2/students_metadrive \
    --output-dir experiments/v2/metadrive_visuals
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distillation.models import create_student

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

METADRIVE_OBS_DIM = 91
METADRIVE_ACT_DIM = 2


def load_student(result_dir):
    with open(os.path.join(result_dir, "result.json")) as f:
        r = json.load(f)
    arch = r["arch"]
    d_model = r.get("d_model", 64)
    n_layers = r.get("n_layers", 2)
    stu_kwargs = {}
    if arch == "stu":
        weights = torch.load(os.path.join(result_dir, "student.pt"),
                             map_location="cpu", weights_only=True)
        for k, v in weights.items():
            if "spectral_filters" in k:
                stu_kwargs["seq_len"] = v.shape[-1]
                break
    elif arch == "framestack":
        stu_kwargs["frame_stack"] = 8
    student = create_student(arch, METADRIVE_OBS_DIM, METADRIVE_ACT_DIM,
                             d_model=d_model, n_layers=n_layers, **stu_kwargs)
    if arch == "stu":
        student.load_state_dict(weights)
    else:
        student.load_state_dict(torch.load(
            os.path.join(result_dir, "student.pt"),
            map_location="cpu", weights_only=True))
    return student, r


def try_3d_rollout(student, seed=6000, max_steps=300):
    """3D camera rendering — chase view. Needs GPU with offscreen OpenGL."""
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": True,
        "vehicle_config": {
            "image_source": "rgb_camera",
            "lidar": {"num_lasers": 72, "distance": 50},
        },
        "sensors": {
            "rgb_camera": (RGBCamera, 800, 600),
        },
        "traffic_density": 0.1,
        "num_scenarios": 100,
        "start_seed": seed,
        "horizon": max_steps,
        "interface_panel": [],
        "show_interface": False,
        "show_logo": False,
        "show_fps": False,
    })
    student.eval()

    obs, _ = env.reset()
    # image_observation=True makes obs a dict: {"image": ..., "state": 91-dim}
    state = obs["state"] if isinstance(obs, dict) else obs
    obs_history = [state]
    frames = []
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.stack(obs_history)).float().unsqueeze(0)
            pred = student(obs_tensor)
            action = pred[0, -1].cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        state = obs["state"] if isinstance(obs, dict) else obs
        total_reward += reward
        obs_history.append(state)
        done = terminated or truncated
        step += 1

        # Get 3D camera frame
        try:
            cam = env.engine.get_sensor("rgb_camera")
            frame = cam.perceive(to_float=False)  # uint8 [H,W,3] BGR
            frame = frame[:, :, ::-1].copy()  # BGR -> RGB
            frames.append(frame)
        except Exception as e:
            if step == 1:
                print(f"  3D camera failed: {e}")
                env.close()
                return None

    env.close()
    return {"frames": frames, "return": total_reward, "steps": step}


def topdown_rollout(student, seed=6000, max_steps=300):
    """Top-down 2D rendering — works everywhere including headless."""
    from metadrive.envs.metadrive_env import MetaDriveEnv

    env = MetaDriveEnv({
        "use_render": False,
        "image_observation": False,
        "vehicle_config": {
            "lidar": {"num_lasers": 72, "distance": 50},
        },
        "traffic_density": 0.1,
        "num_scenarios": 100,
        "start_seed": seed,
        "horizon": max_steps,
    })
    student.eval()

    obs, _ = env.reset()
    obs_history = [obs]
    frames = []
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.stack(obs_history)).float().unsqueeze(0)
            pred = student(obs_tensor)
            action = pred[0, -1].cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        obs_history.append(obs)
        done = terminated or truncated
        step += 1

        # Top-down frame (pygame-based, no GPU needed)
        try:
            frame = env.render(mode="topdown",
                               film_size=(800, 800),
                               screen_size=(400, 400),
                               target_vehicle_heading_up=True)
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            if step == 1:
                print(f"  Top-down render failed: {e}")

    env.close()
    return {"frames": frames, "return": total_reward, "steps": step}


def save_gif(frames, path, duration=80):
    from PIL import Image
    if not frames:
        return
    images = []
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        if len(f.shape) == 2:
            f = np.stack([f]*3, axis=-1)
        if f.shape[-1] == 4:
            f = f[:, :, :3]
        images.append(Image.fromarray(f))
    if images:
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=duration, loop=0, optimize=True)
        print(f"  Saved GIF: {path} ({len(images)} frames)")


def save_frame_strip(frames, path, n_frames=6):
    from PIL import Image
    if not frames:
        return
    indices = np.linspace(0, len(frames)-1, n_frames, dtype=int)
    selected = []
    for i in indices:
        f = frames[i]
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        if f.shape[-1] == 4:
            f = f[:, :, :3]
        selected.append(Image.fromarray(f))
    if selected:
        w, h = selected[0].size
        strip = Image.new('RGB', (w * len(selected), h))
        for i, img in enumerate(selected):
            strip.paste(img, (i * w, 0))
        strip.save(path)
        print(f"  Saved strip: {path}")


def save_comparison(all_data, path, n_frames=4):
    from PIL import Image, ImageDraw
    entries = [(a, d) for a, d in sorted(all_data.items()) if d["frames"]]
    if not entries:
        return
    n_frames = min(n_frames, min(len(d["frames"]) for _, d in entries))
    fh, fw = entries[0][1]["frames"][0].shape[:2]
    label_h = 30
    img = Image.new('RGB', (fw * n_frames, (fh + label_h) * len(entries)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for row, (arch, data) in enumerate(entries):
        y = row * (fh + label_h)
        draw.text((5, y + 5), f"{arch.upper()} (ret={data['return']:.0f}, {data['steps']}st)",
                  fill=(0, 0, 0))
        indices = np.linspace(0, len(data["frames"])-1, n_frames, dtype=int)
        for col, idx in enumerate(indices):
            f = data["frames"][idx]
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            if f.shape[-1] == 4:
                f = f[:, :, :3]
            img.paste(Image.fromarray(f), (col * fw, y + label_h))
    img.save(path)
    print(f"  Saved comparison: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--students-dir", default="experiments/v2/students_metadrive")
    parser.add_argument("--output-dir", default="experiments/v2/metadrive_visuals")
    parser.add_argument("--teacher-tag", default="teacher_500000_noise00")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D, use top-down only")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_data = {}
    use_3d = not args.no_3d

    for arch in ["gru", "lstm", "transformer", "stu", "mlp", "framestack"]:
        pattern = os.path.join(args.students_dir, "metadrive",
                               args.teacher_tag, arch, "seed42", "result.json")
        matches = glob.glob(pattern)
        if not matches:
            print(f"No student for {arch} / {args.teacher_tag}")
            continue

        result_dir = os.path.dirname(matches[0])
        print(f"\nRendering {arch}")
        student, result = load_student(result_dir)

        episode = None
        if use_3d:
            print("  Trying 3D camera...")
            try:
                episode = try_3d_rollout(student, max_steps=args.max_steps)
                if episode and episode["frames"]:
                    print(f"  3D OK: {len(episode['frames'])} frames")
                else:
                    print("  3D returned no frames, falling back to top-down")
                    episode = None
            except Exception as e:
                print(f"  3D failed: {e}")
                episode = None
                use_3d = False  # don't retry 3D for subsequent archs

        if episode is None:
            print("  Using top-down renderer...")
            episode = topdown_rollout(student, max_steps=args.max_steps)

        print(f"  Return: {episode['return']:.1f}, Steps: {episode['steps']}, Frames: {len(episode['frames'])}")
        all_data[arch] = episode

        if episode["frames"]:
            save_gif(episode["frames"],
                     os.path.join(args.output_dir, f"VIDEO_metadrive_{arch}.gif"))
            save_frame_strip(episode["frames"],
                             os.path.join(args.output_dir, f"RENDER_metadrive_{arch}.png"))

    if all_data:
        save_comparison(all_data,
                        os.path.join(args.output_dir, "comparison_metadrive_all.png"))

    print(f"\nDone. Visuals in {args.output_dir}")


if __name__ == "__main__":
    main()
