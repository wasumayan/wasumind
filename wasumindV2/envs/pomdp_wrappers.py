"""
POMDP wrappers for MuJoCo environments.
Removes velocity observations to create partially observable tasks
where the agent must infer velocity from position history.
"""

import gymnasium as gym
import numpy as np


class VelocityMaskWrapper(gym.ObservationWrapper):
    """Remove velocity observations from MuJoCo environments."""

    def __init__(self, env, keep_indices):
        super().__init__(env)
        self.keep_indices = np.array(keep_indices)
        low = env.observation_space.low[self.keep_indices]
        high = env.observation_space.high[self.keep_indices]
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def observation(self, obs):
        return obs[self.keep_indices].astype(np.float32)


# HalfCheetah-v4: 17-dim. Positions: 0-7 (8 dims). Velocities: 8-16 (9 dims).
HALFCHEETAH_POSITION_INDICES = np.arange(8)

# Ant-v4: 27-dim. Positions: 0-12 (13 dims). Velocities: 13-26 (14 dims).
ANT_POSITION_INDICES = np.arange(13)

# Walker2d-v4: 17-dim. Positions: 0-7 (8 dims). Velocities: 8-16 (9 dims).
WALKER2D_POSITION_INDICES = np.arange(8)


def make_halfcheetah_pomdp(**kwargs):
    env = gym.make("HalfCheetah-v4", **kwargs)
    return VelocityMaskWrapper(env, keep_indices=HALFCHEETAH_POSITION_INDICES)


def make_ant_pomdp(**kwargs):
    # Ant-v4 has 27-dim obs (well-understood layout). v5 adds contact forces (105-dim).
    env = gym.make("Ant-v4", **kwargs)
    return VelocityMaskWrapper(env, keep_indices=ANT_POSITION_INDICES)


def make_walker2d_pomdp(**kwargs):
    env = gym.make("Walker2d-v4", **kwargs)
    return VelocityMaskWrapper(env, keep_indices=WALKER2D_POSITION_INDICES)


ENV_CONFIGS = {
    "halfcheetah_pomdp": {
        "make_fn": make_halfcheetah_pomdp,
        "full_env_id": "HalfCheetah-v4",
        "obs_dim": 8,
        "act_dim": 6,
        "keep_indices": HALFCHEETAH_POSITION_INDICES,
    },
    "ant_pomdp": {
        "make_fn": make_ant_pomdp,
        "full_env_id": "Ant-v4",
        "obs_dim": 13,
        "act_dim": 8,
        "keep_indices": ANT_POSITION_INDICES,
    },
    "walker2d_pomdp": {
        "make_fn": make_walker2d_pomdp,
        "full_env_id": "Walker2d-v4",
        "obs_dim": 8,
        "act_dim": 6,
        "keep_indices": WALKER2D_POSITION_INDICES,
    },
}


if __name__ == "__main__":
    for name, cfg in ENV_CONFIGS.items():
        print(f"\n=== {name} ===")
        env = cfg["make_fn"]()
        obs, _ = env.reset(seed=42)
        print(f"  POMDP obs shape: {obs.shape} (expected: ({cfg['obs_dim']},))")
        print(f"  Action space: {env.action_space}")
        assert obs.shape == (cfg["obs_dim"],), f"Shape mismatch: {obs.shape}"

        # Step 100 times
        total_reward = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            assert obs.shape == (cfg["obs_dim"],)
            if term or trunc:
                obs, _ = env.reset()

        print(f"  100 steps OK. Total reward: {total_reward:.2f}")
        env.close()

    print("\nAll POMDP wrappers verified.")
