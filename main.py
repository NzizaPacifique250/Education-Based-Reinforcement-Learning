"""SchoolCheckIn-RL entry point.

Examples:
    uv run main.py train --algo all          # run every hyperparameter sweep
    uv run main.py train --algo dqn
    uv run main.py play                       # visualise best agent in the PyBullet GUI
    uv run main.py evaluate                    # print eval metrics for saved models
    uv run main.py plots                       # regenerate report figures into assets/
    uv run main.py demo                        # random-policy GUI demo (no training needed)
"""

from __future__ import annotations

import argparse


def cmd_train(args):
    if args.algo in ("dqn", "all"):
        from training.dqn_training import run_sweep
        run_sweep(total_timesteps=args.timesteps)
    if args.algo in ("ppo", "all"):
        from training.pg_training import run_ppo_sweep
        run_ppo_sweep(args.timesteps)
    if args.algo in ("a2c", "all"):
        from training.pg_training import run_a2c_sweep
        run_a2c_sweep(args.timesteps)
    if args.algo in ("reinforce", "all"):
        # Monte-Carlo policy gradient needs far more samples than the SB3 methods on this
        # task, so it gets its own (larger) budget rather than the shared one.
        from training.pg_training import run_reinforce_sweep
        run_reinforce_sweep(max(args.timesteps, 600_000))


def cmd_play(args):
    from play import play, _best_from_sweeps
    if args.algo and args.model:
        play(args.algo, args.model, episodes=args.episodes)
    else:
        found = _best_from_sweeps()
        if found is None:
            raise SystemExit("No trained models found. Run: uv run main.py train")
        algo, model, ret = found
        print(f"[auto] best agent = {algo}/{model} (return {ret:.2f})")
        play(algo, model, episodes=args.episodes)


def cmd_demo(args):
    """Random policy in the GUI -- verifies rendering without any trained model."""
    import gymnasium as gym
    import time
    import environment  # noqa: F401
    from environment.custom_env import ACTION_NAMES
    env = gym.make("SchoolCheckIn-v0", render_mode="human")
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"{ACTION_NAMES[a]:<19} reward {r:+6.2f} goal-dist {info['distance']:5.2f} "
              f"clean {info['cleanliness']:.2f} "
              f"fails A{info['attempts_per_scanner'][0]}/B{info['attempts_per_scanner'][1]} "
              f"queue A{info['queue'][0]}/B{info['queue'][1]}")
        done = term or trunc
        time.sleep(0.2)
    env.close()


def cmd_evaluate(args):
    import glob, os, csv
    from training.common import LOG_ROOT
    for path in sorted(glob.glob(os.path.join(LOG_ROOT, "*", "sweep_results.csv"))):
        algo = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            rows = list(csv.DictReader(f))
        best = max(rows, key=lambda r: float(r["mean_return"]))
        print(f"{algo:>10}: best={best['name']} "
              f"return={float(best['mean_return']):.2f} success={float(best['success_rate']):.2f}")


def cmd_plots(args):
    from training.plots import generate_all
    generate_all()


def main():
    ap = argparse.ArgumentParser(description="SchoolCheckIn-RL: RL summative (education mission).")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train models / run hyperparameter sweeps")
    p.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce", "all"], default="all")
    p.add_argument("--timesteps", type=int, default=250_000)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("play", help="visualise the best (or a chosen) agent in the GUI")
    p.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("demo", help="random-policy GUI demo (no training required)")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("evaluate", help="print best result per algorithm")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("plots", help="regenerate report figures into assets/")
    p.set_defaults(func=cmd_plots)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
