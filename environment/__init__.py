"""SchoolCheckIn-RL environment package."""

from gymnasium.envs.registration import register

from environment.custom_env import (SchoolCheckInEnv, ACTION_NAMES, MAX_STEPS, ROOM_SIZE,
                                     START_POS, SCANNER_A_POS, SCANNER_B_POS,
                                     HYGIENE_POS, OFFICE_POS)

register(
    id="SchoolCheckIn-v0",
    entry_point="environment.custom_env:SchoolCheckInEnv",
    # No max_episode_steps on purpose: the environment truncates itself at max_steps and
    # applies the lateness penalty when it does. Adding a TimeLimit wrapper here pinned
    # every episode to MAX_STEPS, so gym.make(..., max_steps=400) was cut off at 150 while
    # the observation still reported time remaining out of 400.
)

__all__ = ["SchoolCheckInEnv", "ACTION_NAMES", "MAX_STEPS", "ROOM_SIZE", "START_POS",
           "SCANNER_A_POS", "SCANNER_B_POS", "HYGIENE_POS", "OFFICE_POS"]
