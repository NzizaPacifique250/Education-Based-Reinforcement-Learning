"""Sanity tests for the EduPath-RL environment."""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import environment  # noqa: F401  (registers EduPath-v0)
from environment.custom_env import EduPathEnv, TARGET_CONCEPT, MASTERY_THRESHOLD


def test_sb3_env_checker():
    """SB3's checker validates spaces, reset/step signatures, and dtypes."""
    check_env(EduPathEnv(), warn=True)


def test_reset_obs_in_space():
    env = EduPathEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "current_concept" in info


def test_random_rollout_terminates():
    """A bounded random policy must always terminate or truncate within max_steps."""
    env = gym.make("EduPath-v0")
    for ep in range(20):
        env.reset(seed=ep)
        done = False
        steps = 0
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            assert np.isfinite(r)
            done = term or trunc
            steps += 1
            assert steps <= 200
    env.close()


def test_determinism_with_seed():
    env = EduPathEnv()
    o1, _ = env.reset(seed=123)
    o2, _ = env.reset(seed=123)
    assert np.allclose(o1, o2)


def test_goal_reachable_by_scripted_policy():
    """A sensible scripted tutor should be able to master the target at least once,
    confirming the reward/terminal design is not impossible."""
    env = EduPathEnv(max_steps=400)
    reached = False
    for seed in range(10):
        obs, info = env.reset(seed=seed, options={"aptitude": 1.3})
        for _ in range(400):
            c = info["current_concept"]
            m = obs[c]  # mastery of the current concept (first N obs entries)
            # break if tired; advance when mastered; otherwise drill at ZPD difficulty
            if info["attention"] < 0.25:
                a = 6  # break
            elif info["mastered"][c]:
                a = 5  # advance to next unlockable concept
            elif m < 0.35:
                a = 0  # easy
            elif m < 0.6:
                a = 1  # medium
            else:
                a = 2  # hard
            obs, r, term, trunc, info = env.step(a)
            if info["target_mastered"]:
                reached = True
                break
            if term or trunc:
                break
        if reached:
            break
    assert reached, "target concept was never mastered by the scripted policy"
