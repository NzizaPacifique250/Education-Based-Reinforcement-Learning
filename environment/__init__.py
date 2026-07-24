"""SchoolCheckIn-RL environment package."""

from gymnasium.envs.registration import register

from environment.custom_env import (SchoolCheckInEnv, ACTION_NAMES, MAX_STEPS, ROOM_SIZE,
                                     START_POS, SCANNER_A_POS, SCANNER_B_POS,
                                     HYGIENE_POS, OFFICE_POS)

register(
    id="SchoolCheckIn-v0",
    entry_point="environment.custom_env:SchoolCheckInEnv",
    max_episode_steps=MAX_STEPS,
)

__all__ = ["SchoolCheckInEnv", "ACTION_NAMES", "MAX_STEPS", "ROOM_SIZE", "START_POS",
           "SCANNER_A_POS", "SCANNER_B_POS", "HYGIENE_POS", "OFFICE_POS"]
