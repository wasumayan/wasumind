"""
MetaDrive environment wrapper for policy distillation experiments.
MetaDrive is inherently partially observable: lidar has limited range,
so the agent cannot see vehicles/obstacles behind it or beyond sensor range.
"""

import gymnasium as gym
import numpy as np

try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

from metadrive.envs.metadrive_env import MetaDriveEnv


def make_metadrive(traffic_density=0.1, num_scenarios=100, max_steps=1000, seed=0):
    """Create a MetaDrive environment for distillation experiments."""
    config = {
        "use_render": False,
        "traffic_density": traffic_density,
        "num_scenarios": num_scenarios,
        "start_seed": seed,
        "horizon": max_steps,
        "vehicle_config": {
            "lidar": {"num_lasers": 72, "distance": 50},
        },
    }
    return MetaDriveEnv(config=config)


def get_metadrive_dims():
    env = make_metadrive(num_scenarios=1, seed=0)
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()
    return obs_dim, act_dim


METADRIVE_CONFIG = {
    "metadrive": {
        "make_fn": make_metadrive,
        "act_dim": 2,
    },
}


if __name__ == "__main__":
    env = make_metadrive(num_scenarios=5)
    obs, _ = env.reset(seed=42)
    print(f"MetaDrive obs: {obs.shape}, act: {env.action_space}")

    total, steps = 0, 0
    while True:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        steps += 1
        if term or trunc:
            break

    print(f"Episode: {steps} steps, reward: {total:.1f}")
    env.close()
    print("MetaDrive wrapper verified.")
