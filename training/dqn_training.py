"""DQN (value-based) training + a 10-run hyperparameter sweep on SchoolCheckIn-RL."""

from __future__ import annotations

import os
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from training.common import ENV_ID, LOG_ROOT, MODEL_ROOT, evaluate_policy, write_sweep_csv
import environment  # noqa: F401


# 10 configurations with clearly varying hyperparameters (rubric: >=10 rows / algorithm).
DQN_SWEEP = [
    dict(name="dqn01", learning_rate=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64,
         exploration_fraction=0.20, target_update_interval=500,  net_arch=[64, 64]),
    dict(name="dqn02", learning_rate=5e-4, gamma=0.99, buffer_size=100_000, batch_size=128,
         exploration_fraction=0.30, target_update_interval=1000, net_arch=[128, 128]),
    dict(name="dqn03", learning_rate=1e-4, gamma=0.95, buffer_size=50_000, batch_size=64,
         exploration_fraction=0.10, target_update_interval=500,  net_arch=[64, 64]),
    dict(name="dqn04", learning_rate=1e-3, gamma=0.90, buffer_size=20_000, batch_size=32,
         exploration_fraction=0.40, target_update_interval=250,  net_arch=[64]),
    dict(name="dqn05", learning_rate=3e-4, gamma=0.99, buffer_size=200_000, batch_size=256,
         exploration_fraction=0.25, target_update_interval=2000, net_arch=[256, 256]),
    dict(name="dqn06", learning_rate=5e-4, gamma=0.999, buffer_size=100_000, batch_size=128,
         exploration_fraction=0.15, target_update_interval=1000, net_arch=[128, 128]),
    dict(name="dqn07", learning_rate=1e-3, gamma=0.98, buffer_size=50_000, batch_size=64,
         exploration_fraction=0.50, target_update_interval=500,  net_arch=[128, 64]),
    dict(name="dqn08", learning_rate=7e-4, gamma=0.97, buffer_size=75_000, batch_size=128,
         exploration_fraction=0.20, target_update_interval=750,  net_arch=[128, 128]),
    dict(name="dqn09", learning_rate=2e-4, gamma=0.99, buffer_size=150_000, batch_size=256,
         exploration_fraction=0.30, target_update_interval=1500, net_arch=[256, 128]),
    dict(name="dqn10", learning_rate=1e-3, gamma=0.99, buffer_size=100_000, batch_size=64,
         exploration_fraction=0.10, target_update_interval=1000, net_arch=[64, 64]),
]


def run_dqn(config: dict, total_timesteps: int = 120_000, seed: int = 0, save: bool = True):
    name = config["name"]
    log_dir = os.path.join(LOG_ROOT, "dqn", name)
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(gym.make(ENV_ID), filename=os.path.join(log_dir, "monitor.csv"))

    model = DQN(
        "MlpPolicy", env, verbose=0, seed=seed, tensorboard_log=os.path.join(LOG_ROOT, "dqn_tb"),
        learning_rate=config["learning_rate"], gamma=config["gamma"],
        buffer_size=config["buffer_size"], batch_size=config["batch_size"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=0.05, target_update_interval=config["target_update_interval"],
        learning_starts=1000, train_freq=4,
        policy_kwargs=dict(net_arch=config["net_arch"]),
    )
    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))
    model.learn(total_timesteps=total_timesteps, tb_log_name=name, progress_bar=False)

    def predict(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    metrics = evaluate_policy(predict)
    if save:
        model_dir = os.path.join(MODEL_ROOT, "dqn")
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, name))
    env.close()
    row = {**{k: config[k] for k in config}, **metrics}
    print(f"[DQN {name}] return={metrics['mean_return']:.2f} "
          f"success={metrics['success_rate']:.2f}")
    return model, row


def run_sweep(total_timesteps: int = 120_000):
    rows, best, best_ret = [], None, -1e9
    for cfg in DQN_SWEEP:
        model, row = run_dqn(cfg, total_timesteps=total_timesteps)
        rows.append(row)
        if row["mean_return"] > best_ret:
            best_ret, best = row["mean_return"], cfg["name"]
    write_sweep_csv(os.path.join(LOG_ROOT, "dqn", "sweep_results.csv"), rows)
    print(f"[DQN] best config: {best} (return {best_ret:.2f})")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=120_000)
    ap.add_argument("--quick", action="store_true", help="single short run for smoke testing")
    args = ap.parse_args()
    if args.quick:
        run_dqn(DQN_SWEEP[0], total_timesteps=5_000, save=False)
    else:
        run_sweep(total_timesteps=args.timesteps)
