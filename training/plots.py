"""Generate the report figures from training logs into assets/.

Produces (rubric criterion 4):
  1. reward_curves.png     cumulative/episode reward per algorithm (subplots)
  2. convergence.png       best reward curve of each algorithm, overlaid + smoothed
  3. dqn_objective.png     DQN training loss (and mean-Q) vs timesteps
  4. pg_entropy.png        policy entropy for REINFORCE / PPO / A2C
  5. generalization.png    check-in rate of each best agent across unseen scanner reliabilities
  6. checkin_modes.png     biometric vs manual vs no check-in, per algorithm
"""

from __future__ import annotations

import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from training.common import LOG_ROOT

ASSETS = "assets"
ALGOS = ["dqn", "ppo", "a2c", "reinforce"]


def _smooth(x, w=20):
    x = np.asarray(x, dtype=float)
    if len(x) < w or w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _best_name(algo: str) -> str | None:
    path = os.path.join(LOG_ROOT, algo, "sweep_results.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return max(rows, key=lambda r: float(r["mean_return"]))["name"]


def _episode_returns(algo: str, name: str):
    """Read per-episode returns for a run (SB3 monitor.csv or REINFORCE episodes.csv)."""
    if algo == "reinforce":
        path = os.path.join(LOG_ROOT, algo, name, "episodes.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)["r"].values
    path = os.path.join(LOG_ROOT, algo, name, "monitor.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, skiprows=1)["r"].values


def plot_reward_curves():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, algo in zip(axes.ravel(), ALGOS):
        name = _best_name(algo)
        r = _episode_returns(algo, name) if name else None
        if r is None:
            ax.set_title(f"{algo.upper()} (no data)")
            continue
        ax.plot(r, alpha=0.25, color="tab:blue")
        ax.plot(np.arange(len(_smooth(r))) + 10, _smooth(r), color="tab:blue", lw=2)
        ax.set_title(f"{algo.upper()}: best config {name}")
        ax.set_xlabel("episode"); ax.set_ylabel("episode return"); ax.grid(alpha=0.3)
    fig.suptitle("Episode reward per algorithm (best hyperparameter configuration)")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "reward_curves.png"), dpi=130)
    plt.close(fig)


def plot_convergence():
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ALGOS:
        name = _best_name(algo)
        r = _episode_returns(algo, name) if name else None
        if r is None:
            continue
        s = _smooth(r)
        ax.plot(np.linspace(0, 1, len(s)), s, lw=2, label=f"{algo.upper()} ({name})")
    ax.set_xlabel("training progress (normalised)"); ax.set_ylabel("smoothed episode return")
    ax.set_title("Convergence comparison (best config per algorithm)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "convergence.png"), dpi=130)
    plt.close(fig)


def plot_dqn_objective():
    name = _best_name("dqn")
    path = os.path.join(LOG_ROOT, "dqn", name, "progress.csv") if name else None
    if not path or not os.path.exists(path):
        return
    df = pd.read_csv(path)
    x = df.get("time/total_timesteps")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    if "train/loss" in df:
        ax1.plot(x, df["train/loss"], color="tab:red", label="TD loss")
        ax1.set_ylabel("TD loss", color="tab:red")
    ax1.set_xlabel("timesteps"); ax1.set_title(f"DQN objective, config {name}"); ax1.grid(alpha=0.3)
    if "rollout/ep_rew_mean" in df:
        ax2 = ax1.twinx()
        ax2.plot(x, df["rollout/ep_rew_mean"], color="tab:blue", label="mean ep reward")
        ax2.set_ylabel("mean episode reward", color="tab:blue")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "dqn_objective.png"), dpi=130)
    plt.close(fig)


def plot_pg_entropy():
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, col in (("ppo", "tab:green"), ("a2c", "tab:orange")):
        name = _best_name(algo)
        path = os.path.join(LOG_ROOT, algo, name, "progress.csv") if name else None
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            if "train/entropy_loss" in df:
                ent = -df["train/entropy_loss"]  # SB3 logs negative entropy
                ax.plot(df.get("time/total_timesteps"), ent, color=col,
                        label=f"{algo.upper()} ({name})")
    name = _best_name("reinforce")
    path = os.path.join(LOG_ROOT, "reinforce", name, "updates.csv") if name else None
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        ax.plot(df["steps"], df["entropy"], color="tab:purple",
                label=f"REINFORCE ({name})")
    ax.set_xlabel("timesteps"); ax.set_ylabel("policy entropy")
    ax.set_title("Policy entropy (exploration) over training")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "pg_entropy.png"), dpi=130)
    plt.close(fig)


def convergence_stats(window: int = 100, frac: float = 0.90, hold: int = 250):
    """Episodes each method needed to reach stable performance.

    Plateau is the mean return over the final 10% of episodes. Convergence is the first
    episode whose `window`-episode moving average reaches `frac` of that plateau and then
    holds it for `hold` consecutive episodes. Requiring it to hold for the entire remainder
    instead is far too strict on a noisy run: A2C and REINFORCE only satisfy that in the
    last few hundred episodes, which says nothing about when they actually learned.
    """
    stats = {}
    for algo in ALGOS:
        name = _best_name(algo)
        r = _episode_returns(algo, name) if name else None
        if r is None or len(r) < window:
            continue
        ma = np.convolve(r, np.ones(window) / window, mode="valid")
        plateau = float(np.mean(r[-max(len(r) // 10, window):]))
        target = frac * plateau
        above = ma >= target
        ep, first = None, None
        for i in range(len(above)):
            if not above[i]:
                continue
            if first is None:
                first = i + window
            if above[i:i + hold].all():
                ep = i + window          # episode index in the original series
                break
        stats[algo] = {
            "name": name, "episodes": len(r), "plateau": plateau, "first_reached": first,
            "converged_at": ep, "moving_avg": ma, "target": target,
        }
    return stats


def plot_episodes_to_converge():
    """Subplots: moving-average curves with the convergence point marked, plus a bar chart
    of how many episodes each method needed to get there."""
    stats = convergence_stats()
    if not stats:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    colours = {"dqn": "tab:blue", "ppo": "tab:green", "a2c": "tab:orange",
               "reinforce": "tab:purple"}
    for algo, s in stats.items():
        c = colours.get(algo, "grey")
        ax1.plot(np.arange(len(s["moving_avg"])) + 100, s["moving_avg"], color=c, lw=1.8,
                 label=f"{algo.upper()} ({s['name']})")
        if s["converged_at"]:
            ax1.axvline(s["converged_at"], color=c, ls=":", alpha=0.7)
            ax1.plot(s["converged_at"], s["target"], "o", color=c, ms=7)
    ax1.set_xlabel("episode"); ax1.set_ylabel("100-episode moving average return")
    ax1.set_title("Learning curves with convergence point marked")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    names = [a.upper() for a in stats]
    eps = [stats[a]["converged_at"] or 0 for a in stats]
    bars = ax2.bar(names, eps, 0.55, color=[colours.get(a, "grey") for a in stats])
    for b, e in zip(bars, eps):
        ax2.text(b.get_x() + b.get_width() / 2, e, f" {e}", ha="center", va="bottom",
                 fontsize=10)
    ax2.set_ylabel("episodes to reach 90% of final performance")
    ax2.set_title("Episodes required to converge")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "episodes_to_converge.png"), dpi=130)
    plt.close(fig)


def _best_agents():
    """Yield (algo, name, predict) for the best config of each algorithm."""
    from play import _load
    for algo in ALGOS:
        name = _best_name(algo)
        if not name:
            continue
        try:
            yield algo, name, _load(algo, name)
        except Exception:
            continue


def plot_generalization():
    """Evaluate each best agent under unseen scanner-reliability conditions.

    The range is deliberately harsher than training (which samples 0.80-1.00 for scanner A
    and 0.45-0.85 for B) so the bars separate instead of all saturating at 1.0.
    """
    from training.common import evaluate_policy
    reliabilities = [0.3, 0.5, 0.7, 0.9]
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    any_data = False
    for i, (algo, name, predict) in enumerate(_best_agents()):
        rates = [evaluate_policy(lambda o: predict(o), n_episodes=20,
                                 options={"scan_reliability": r})["success_rate"]
                 for r in reliabilities]
        ax.bar(np.arange(len(reliabilities)) + i * width, rates, width,
               label=f"{algo.upper()} ({name})")
        any_data = True
    if not any_data:
        plt.close(fig); return
    ax.set_xticks(np.arange(len(reliabilities)) + 1.5 * width)
    ax.set_xticklabels([f"scanner reliability {r}" for r in reliabilities])
    ax.set_ylabel("check-in rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Generalization across unseen scanner-reliability conditions")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "generalization.png"), dpi=130)
    plt.close(fig)


def plot_checkin_modes():
    """How each agent resolves the episode: the intended biometric route, the low-reward
    office fallback, or neither. This is the risk/reward decision the redesign creates."""
    from training.common import evaluate_policy
    labels, bio, man, fail = [], [], [], []
    for algo, name, predict in _best_agents():
        m = evaluate_policy(lambda o: predict(o), n_episodes=40)
        labels.append(f"{algo.upper()}\n({name})")
        bio.append(m["biometric_rate"])
        man.append(m["manual_rate"])
        fail.append(max(0.0, 1.0 - m["biometric_rate"] - m["manual_rate"]))
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    from environment.custom_env import BIOMETRIC_BONUS, MANUAL_BONUS
    ax.bar(x, bio, 0.55, label=f"biometric check-in (+{BIOMETRIC_BONUS:g})", color="tab:green")
    ax.bar(x, man, 0.55, bottom=bio,
           label=f"manual office sign-in (+{MANUAL_BONUS:g})", color="tab:orange")
    ax.bar(x, fail, 0.55, bottom=np.array(bio) + np.array(man),
           label="no check-in (late / stranded)", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("share of episodes"); ax.set_ylim(0, 1.05)
    ax.set_title("Episode outcome by algorithm: intended route vs fallback")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "checkin_modes.png"), dpi=130)
    plt.close(fig)


def generate_all():
    os.makedirs(ASSETS, exist_ok=True)
    plot_reward_curves()
    plot_convergence()
    plot_dqn_objective()
    plot_pg_entropy()
    plot_episodes_to_converge()
    plot_generalization()
    plot_checkin_modes()
    print(f"[plots] figures written to {ASSETS}/")


if __name__ == "__main__":
    generate_all()
