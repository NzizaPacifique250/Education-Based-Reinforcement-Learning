"""EduPath-RL environment package."""

from gymnasium.envs.registration import register

from environment.custom_env import EduPathEnv, ACTION_NAMES, N_CONCEPTS, TARGET_CONCEPT

register(
    id="EduPath-v0",
    entry_point="environment.custom_env:EduPathEnv",
    max_episode_steps=200,
)

__all__ = ["EduPathEnv", "ACTION_NAMES", "N_CONCEPTS", "TARGET_CONCEPT"]
