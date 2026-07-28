"""Policy-gradient training on SchoolCheckIn-RL: PPO, A2C (Stable-Baselines3) and a custom
REINFORCE implementation (SB3 has no REINFORCE), each with a 10-run hyperparameter sweep."""

from __future__ import annotations

import os
import csv
import argparse
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
from torch.distributions import Categorical

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from training.common import ENV_ID, LOG_ROOT, MODEL_ROOT, evaluate_policy, write_sweep_csv
import environment  # noqa: F401


# ======================================================================================
# PPO
# ======================================================================================
PPO_SWEEP = [
    dict(name="ppo01", learning_rate=3e-4, gamma=0.99, n_steps=1024, batch_size=64,
         clip_range=0.2, ent_coef=0.0,   gae_lambda=0.95),
    dict(name="ppo02", learning_rate=1e-3, gamma=0.99, n_steps=2048, batch_size=64,
         clip_range=0.2, ent_coef=0.01,  gae_lambda=0.95),
    dict(name="ppo03", learning_rate=3e-4, gamma=0.95, n_steps=1024, batch_size=128,
         clip_range=0.1, ent_coef=0.0,   gae_lambda=0.90),
    dict(name="ppo04", learning_rate=5e-4, gamma=0.99, n_steps=512,  batch_size=64,
         clip_range=0.3, ent_coef=0.02,  gae_lambda=0.95),
    dict(name="ppo05", learning_rate=1e-4, gamma=0.999, n_steps=2048, batch_size=256,
         clip_range=0.2, ent_coef=0.01,  gae_lambda=0.98),
    dict(name="ppo06", learning_rate=3e-4, gamma=0.98, n_steps=1024, batch_size=64,
         clip_range=0.2, ent_coef=0.05,  gae_lambda=0.95),
    dict(name="ppo07", learning_rate=7e-4, gamma=0.99, n_steps=2048, batch_size=128,
         clip_range=0.15, ent_coef=0.0,  gae_lambda=0.92),
    dict(name="ppo08", learning_rate=2e-4, gamma=0.97, n_steps=512,  batch_size=64,
         clip_range=0.2, ent_coef=0.01,  gae_lambda=0.95),
    dict(name="ppo09", learning_rate=3e-4, gamma=0.99, n_steps=4096, batch_size=256,
         clip_range=0.25, ent_coef=0.0,  gae_lambda=0.95),
    dict(name="ppo10", learning_rate=5e-4, gamma=0.99, n_steps=1024, batch_size=64,
         clip_range=0.2, ent_coef=0.03,  gae_lambda=0.90),
]

A2C_SWEEP = [
    dict(name="a2c01", learning_rate=7e-4, gamma=0.99, n_steps=5,  ent_coef=0.0,   vf_coef=0.5, gae_lambda=1.0),
    dict(name="a2c02", learning_rate=1e-3, gamma=0.99, n_steps=8,  ent_coef=0.01,  vf_coef=0.5, gae_lambda=0.95),
    dict(name="a2c03", learning_rate=3e-4, gamma=0.95, n_steps=5,  ent_coef=0.0,   vf_coef=0.25, gae_lambda=0.90),
    dict(name="a2c04", learning_rate=5e-4, gamma=0.99, n_steps=16, ent_coef=0.02,  vf_coef=0.5, gae_lambda=0.95),
    dict(name="a2c05", learning_rate=1e-3, gamma=0.999, n_steps=32, ent_coef=0.01, vf_coef=0.5, gae_lambda=0.98),
    dict(name="a2c06", learning_rate=7e-4, gamma=0.98, n_steps=5,  ent_coef=0.05,  vf_coef=0.5, gae_lambda=1.0),
    dict(name="a2c07", learning_rate=2e-4, gamma=0.99, n_steps=8,  ent_coef=0.0,   vf_coef=0.75, gae_lambda=0.92),
    dict(name="a2c08", learning_rate=1e-3, gamma=0.97, n_steps=16, ent_coef=0.01,  vf_coef=0.5, gae_lambda=0.95),
    dict(name="a2c09", learning_rate=4e-4, gamma=0.99, n_steps=64, ent_coef=0.0,   vf_coef=0.5, gae_lambda=0.95),
    dict(name="a2c10", learning_rate=7e-4, gamma=0.99, n_steps=5,  ent_coef=0.03,  vf_coef=0.3, gae_lambda=0.90),
]


def _run_sb3(algo_cls, config, total_timesteps, subdir, seed=0, save=True):
    name = config["name"]
    log_dir = os.path.join(LOG_ROOT, subdir, name)
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(gym.make(ENV_ID), filename=os.path.join(log_dir, "monitor.csv"))
    kwargs = {k: v for k, v in config.items() if k != "name"}
    model = algo_cls("MlpPolicy", env, verbose=0, seed=seed,
                     tensorboard_log=os.path.join(LOG_ROOT, f"{subdir}_tb"), **kwargs)
    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))
    model.learn(total_timesteps=total_timesteps, tb_log_name=name, progress_bar=False)

    def predict(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    metrics = evaluate_policy(predict)
    if save:
        model_dir = os.path.join(MODEL_ROOT, "pg")
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, name))
    env.close()
    print(f"[{subdir.upper()} {name}] return={metrics['mean_return']:.2f} "
          f"success={metrics['success_rate']:.2f}")
    return model, {**config, **metrics}


# ======================================================================================
# REINFORCE (custom Monte-Carlo policy gradient)
# ======================================================================================
REINFORCE_SWEEP = [
    dict(name="reinforce01", learning_rate=1e-3, gamma=0.99, hidden=64,  ent_coef=0.0,  baseline=True),
    dict(name="reinforce02", learning_rate=5e-4, gamma=0.99, hidden=128, ent_coef=0.01, baseline=True),
    dict(name="reinforce03", learning_rate=1e-3, gamma=0.95, hidden=64,  ent_coef=0.0,  baseline=False),
    dict(name="reinforce04", learning_rate=3e-4, gamma=0.99, hidden=128, ent_coef=0.02, baseline=True),
    dict(name="reinforce05", learning_rate=1e-3, gamma=0.999, hidden=256, ent_coef=0.01, baseline=True),
    dict(name="reinforce06", learning_rate=7e-4, gamma=0.98, hidden=64,  ent_coef=0.05, baseline=True),
    dict(name="reinforce07", learning_rate=2e-4, gamma=0.99, hidden=128, ent_coef=0.0,  baseline=False),
    dict(name="reinforce08", learning_rate=1e-3, gamma=0.97, hidden=64,  ent_coef=0.01, baseline=True),
    dict(name="reinforce09", learning_rate=5e-4, gamma=0.99, hidden=256, ent_coef=0.0,  baseline=True),
    dict(name="reinforce10", learning_rate=1e-3, gamma=0.99, hidden=128, ent_coef=0.03, baseline=False),
]


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    """State-value baseline b(s) ~ V(s), trained by regression on the observed returns."""

    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ReinforceAgent:
    """REINFORCE (Monte-Carlo policy gradient), optionally with a learned state-value
    baseline -- Sutton & Barto section 13.4.

    The baseline matters a great deal here. With plain normalised returns the gradient
    variance over a 150-step, 9-action episode is high enough that the agent never finds
    the sanitize-then-scan route at all (0% check-in after 400k steps). Subtracting a
    learned V(s) leaves the gradient unbiased but far lower-variance, which is what makes
    the run competitive with the SB3 actor-critics.
    """

    def __init__(self, obs_dim, n_actions, config):
        self.gamma = config["gamma"]
        self.ent_coef = config["ent_coef"]
        self.use_baseline = config["baseline"]
        self.policy = PolicyNet(obs_dim, n_actions, config["hidden"])
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=config["learning_rate"])
        self.value = ValueNet(obs_dim, config["hidden"]) if self.use_baseline else None
        self.vopt = (torch.optim.Adam(self.value.parameters(), lr=1e-3)
                     if self.use_baseline else None)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            logits = self.policy(torch.as_tensor(obs, dtype=torch.float32))
            if deterministic:
                return int(torch.argmax(logits).item())
            return int(Categorical(logits=logits).sample().item())

    def update(self, obs_batch, act_batch, returns):
        obs_t = torch.as_tensor(np.array(obs_batch), dtype=torch.float32)
        act_t = torch.as_tensor(act_batch, dtype=torch.int64)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)

        if self.use_baseline:
            with torch.no_grad():
                adv = ret_t - self.value(obs_t)
        else:
            adv = ret_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # variance reduction

        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_t)
        loss = -(logp * adv).mean() - self.ent_coef * dist.entropy().mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.use_baseline:
            # regress the baseline onto the same Monte-Carlo returns
            vloss = torch.nn.functional.mse_loss(self.value(obs_t), ret_t)
            self.vopt.zero_grad()
            vloss.backward()
            self.vopt.step()

        return float(loss.item()), float(dist.entropy().mean().item())


def run_reinforce(config, total_timesteps=900_000, seed=0, save=True, batch_episodes=4):
    name = config["name"]
    log_dir = os.path.join(LOG_ROOT, "reinforce", name)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    torch.manual_seed(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ReinforceAgent(obs_dim, n_actions, config)

    steps_done, ep_count = 0, 0
    ep_log = open(os.path.join(log_dir, "episodes.csv"), "w", newline="")
    writer = csv.writer(ep_log)
    writer.writerow(["r", "l"])
    upd_log = open(os.path.join(log_dir, "updates.csv"), "w", newline="")
    upd_writer = csv.writer(upd_log)
    upd_writer.writerow(["update", "steps", "loss", "entropy"])
    update_i = 0

    while steps_done < total_timesteps:
        batch_obs, batch_act, batch_ret = [], [], []
        for _ in range(batch_episodes):
            obs, _ = env.reset()
            ep_obs, ep_act, ep_rew = [], [], []
            done = False
            while not done:
                a = agent.act(obs)
                nobs, r, term, trunc, _ = env.step(a)
                ep_obs.append(obs); ep_act.append(a); ep_rew.append(r)
                obs = nobs
                done = term or trunc
            # discounted returns-to-go
            G, returns = 0.0, []
            for rew in reversed(ep_rew):
                G = rew + agent.gamma * G
                returns.append(G)
            returns.reverse()
            batch_obs += ep_obs; batch_act += ep_act; batch_ret += returns
            steps_done += len(ep_rew); ep_count += 1
            writer.writerow([sum(ep_rew), len(ep_rew)])
        loss, ent = agent.update(batch_obs, batch_act, batch_ret)
        upd_writer.writerow([update_i, steps_done, loss, ent])
        update_i += 1
    ep_log.close()
    upd_log.close()

    metrics = evaluate_policy(lambda o: agent.act(o, deterministic=True))
    if save:
        model_dir = os.path.join(MODEL_ROOT, "pg")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(agent.policy.state_dict(), os.path.join(model_dir, f"{name}.pt"))
    env.close()
    print(f"[REINFORCE {name}] return={metrics['mean_return']:.2f} "
          f"success={metrics['success_rate']:.2f}")
    return agent, {**config, **metrics}


# ======================================================================================
# Sweeps / CLI
# ======================================================================================
def run_ppo_sweep(t=150_000):
    rows = [_run_sb3(PPO, c, t, "ppo")[1] for c in PPO_SWEEP]
    write_sweep_csv(os.path.join(LOG_ROOT, "ppo", "sweep_results.csv"), rows)
    return rows


def run_a2c_sweep(t=150_000):
    rows = [_run_sb3(A2C, c, t, "a2c")[1] for c in A2C_SWEEP]
    write_sweep_csv(os.path.join(LOG_ROOT, "a2c", "sweep_results.csv"), rows)
    return rows


def run_reinforce_sweep(t=900_000):
    rows = [run_reinforce(c, total_timesteps=t)[1] for c in REINFORCE_SWEEP]
    write_sweep_csv(os.path.join(LOG_ROOT, "reinforce", "sweep_results.csv"), rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "a2c", "reinforce", "all"], default="all")
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        _run_sb3(PPO, PPO_SWEEP[0], 5_000, "ppo", save=False)
        _run_sb3(A2C, A2C_SWEEP[0], 5_000, "a2c", save=False)
        run_reinforce(REINFORCE_SWEEP[0], total_timesteps=5_000, save=False, batch_episodes=3)
    else:
        if args.algo in ("ppo", "all"):
            run_ppo_sweep(args.timesteps)
        if args.algo in ("a2c", "all"):
            run_a2c_sweep(args.timesteps)
        if args.algo in ("reinforce", "all"):
            # Mirrors main.py: REINFORCE needs *more* samples than the SB3 methods, not
            # fewer. This used to cap the budget at 120k, which silently produced ten runs
            # that all scored 0% no matter what --timesteps the caller asked for.
            run_reinforce_sweep(max(args.timesteps, 900_000))
