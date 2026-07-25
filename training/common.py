"""Shared helpers for training and evaluation across all four algorithms."""

from __future__ import annotations

import os
import csv
import numpy as np
import gymnasium as gym

import environment  # noqa: F401  (registers SchoolCheckIn-v0)

ENV_ID = "SchoolCheckIn-v0"
LOG_ROOT = "logs"
MODEL_ROOT = "models"


def make_env(seed: int | None = None, render_mode: str | None = None):
    """Factory for a single SchoolCheckIn env instance."""
    env = gym.make(ENV_ID, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate_policy(predict_fn, n_episodes: int = 30, seed: int = 10_000,
                    options: dict | None = None):
    """Run a policy (predict_fn: obs -> action) and return summary metrics.

    Uses a disjoint seed range from training so results reflect generalization.
    `options` is forwarded to env.reset (e.g. {"scan_reliability": 0.7}).
    """
    env = gym.make(ENV_ID)
    returns, lengths = [], []
    successes, biometric, manual, tardy, stranded = [], [], [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep, options=options)
        done, total, steps = False, 0.0, 0
        while not done:
            action = predict_fn(obs)
            obs, r, term, trunc, info = env.step(int(action))
            total += r
            steps += 1
            done = term or trunc
        returns.append(total)
        lengths.append(steps)
        checked_in = bool(info.get("checked_in"))
        successes.append(1.0 if checked_in else 0.0)
        biometric.append(1.0 if info.get("checkin_mode") == "biometric" else 0.0)
        manual.append(1.0 if info.get("checkin_mode") == "manual" else 0.0)
        tardy.append(1.0 if checked_in and info.get("tardy") else 0.0)
        stranded.append(1.0 if info.get("stranded") else 0.0)
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        # any check-in, biometric or manual
        "success_rate": float(np.mean(successes)),
        # how the agent resolved the task -- the intended route vs the safe fallback
        "biometric_rate": float(np.mean(biometric)),
        "manual_rate": float(np.mean(manual)),
        "tardy_rate": float(np.mean(tardy)),
        "stranded_rate": float(np.mean(stranded)),
    }


def write_sweep_csv(path: str, rows: list[dict]):
    """Persist a hyperparameter-sweep results table to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] wrote {len(rows)} rows -> {path}")
