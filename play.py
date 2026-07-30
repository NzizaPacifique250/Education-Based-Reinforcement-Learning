"""Run a trained SchoolCheckIn-RL agent in the PyBullet GUI with verbose terminal output.

Usage:
    uv run play.py                      # auto-pick best agent across all sweeps
    uv run play.py --algo ppo --model ppo02
    uv run play.py --episodes 3 --deterministic
"""

from __future__ import annotations

import os
import csv
import glob
import time
import argparse
import numpy as np
import gymnasium as gym

import environment  # noqa: F401
from environment.custom_env import ACTION_NAMES
from training.common import ENV_ID, LOG_ROOT, MODEL_ROOT


def _best_from_sweeps():
    """Scan every sweep_results.csv and return (algo, model_name) with the top mean_return."""
    best = None
    for path in glob.glob(os.path.join(LOG_ROOT, "*", "sweep_results.csv")):
        algo = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for row in csv.DictReader(f):
                ret = float(row.get("mean_return", "-inf"))
                if best is None or ret > best[2]:
                    best = (algo, row["name"], ret)
    return best


def _load(algo: str, model_name: str):
    """Return a predict(obs)->action callable for the given saved model."""
    if algo == "reinforce":
        import torch
        from training.pg_training import PolicyNet, REINFORCE_SWEEP
        cfg = next(c for c in REINFORCE_SWEEP if c["name"] == model_name)
        env = gym.make(ENV_ID)
        net = PolicyNet(env.observation_space.shape[0], env.action_space.n, cfg["hidden"])
        net.load_state_dict(torch.load(os.path.join(MODEL_ROOT, "pg", f"{model_name}.pt")))
        env.close()
        net.eval()

        from torch.distributions import Categorical

        def predict(obs, deterministic=True):
            with torch.no_grad():
                logits = net(torch.as_tensor(obs, dtype=torch.float32))
                if deterministic:
                    return int(torch.argmax(logits).item())
                # honour --stochastic: this branch used to fall through to argmax, so the
                # flag did nothing for REINFORCE models
                return int(Categorical(logits=logits).sample().item())
        return predict

    # SB3 models
    from stable_baselines3 import DQN, PPO, A2C
    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    subdir = "dqn" if algo == "dqn" else "pg"
    model = cls.load(os.path.join(MODEL_ROOT, subdir, model_name))

    def predict(obs, deterministic=True):
        a, _ = model.predict(obs, deterministic=deterministic)
        return int(a)
    return predict


def play(algo, model_name, episodes=3, deterministic=True, sleep=0.25):
    predict = _load(algo, model_name)
    env = gym.make(ENV_ID, render_mode="human")
    print(f"\n=== SchoolCheckIn-RL | agent: {algo}/{model_name} ===\n")

    for ep in range(episodes):
        obs, info = env.reset(seed=5000 + ep)
        done, total, step = False, 0.0, 0
        print(f"--- Episode {ep + 1} (start cleanliness {info['cleanliness']:.2f}) ---")
        while not done:
            a = predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            total += r
            step += 1
            at = ("scanner A" if info["at_scanner_idx"] == 0 else
                  "scanner B" if info["at_scanner_idx"] == 1 else
                  "sanitizer" if info["at_hygiene"] else
                  "office" if info["at_office"] else "-")
            lock = "".join(c for c, l in zip("AB", info["locked"]) if l) or "-"
            print(f"  step {step:3d} | {ACTION_NAMES[a]:<19} | reward {r:+6.2f} | "
                  f"goal-dist {info['distance']:5.2f} | clean {info['cleanliness']:.2f} | "
                  f"at {at:<9} | fails A{info['attempts_per_scanner'][0]}"
                  f"/B{info['attempts_per_scanner'][1]} | locked {lock} | "
                  f"queue A{info['queue'][0]}/B{info['queue'][1]}")
            done = term or trunc
            time.sleep(sleep)
        if info["checked_in"]:
            outcome = f"CHECKED IN via {info['checkin_mode']}"
            outcome += " (TARDY)" if info["tardy"] else " (on time)"
        elif info["stranded"]:
            outcome = "STRANDED (all routes exhausted)"
        else:
            outcome = "LATE (never checked in)"
        print(f"  => episode return {total:+.2f} in {step} steps | outcome: {outcome}\n")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    args = ap.parse_args()

    if args.algo and args.model:
        algo, model = args.algo, args.model
    else:
        found = _best_from_sweeps()
        if found is None:
            raise SystemExit("No trained models found. Run training first (uv run main.py train).")
        algo, model, ret = found
        print(f"[auto] best agent = {algo}/{model} (eval return {ret:.2f})")
    play(algo, model, episodes=args.episodes, deterministic=not args.stochastic)
