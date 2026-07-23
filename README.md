# EduPath-RL — Reinforcement Learning for Adaptive Learning Paths

Reinforcement Learning summative (education mission). A custom Gymnasium environment models
an **intelligent tutoring system**: an agent sequences learning activities to bring a
*stochastic* student to mastery of a target concept efficiently — respecting the curriculum's
prerequisite graph, matching exercise difficulty to ability, and avoiding fatigue-driven dropout.

Four RL algorithms are trained and compared on the **same** environment:

| Category | Algorithm | Implementation |
|---|---|---|
| Value-based | **DQN** | Stable-Baselines3 |
| Policy gradient | **REINFORCE** | custom PyTorch (not in SB3) |
| Actor-critic | **PPO** | Stable-Baselines3 |
| Actor-critic | **A2C** | Stable-Baselines3 |

The environment is visualized in 3D with **PyBullet** (a skill-tree the student avatar traverses;
nodes shift red→green with mastery).

## Requirements
This project uses **[uv](https://docs.astral.sh/uv/)** for dependency and environment management.
No manual `pip install` or venv creation is needed.

```bash
uv sync                 # create the venv and install everything
```

## Quick start
```bash
uv run main.py demo                 # random-policy 3D GUI demo (no training needed)
uv run main.py train --algo all     # run all four hyperparameter sweeps (10 runs each)
uv run main.py evaluate             # print the best config per algorithm
uv run main.py plots                # regenerate report figures into assets/
uv run main.py play                 # visualise the best agent in the PyBullet GUI
uv run play.py --algo ppo --model ppo02   # play a specific saved model
```

Faster smoke tests (short training, no save):
```bash
uv run python -m training.dqn_training --quick
uv run python -m training.pg_training --quick
uv run pytest                        # environment sanity tests
```

### JSON API (frontend integration)
```bash
uv run uvicorn api.serve:app --reload
# POST /session, POST /session/{id}/step, POST /session/{id}/act, GET /curriculum
```

## Environment summary
- **Observation** (`Box`, 15-d): per-concept mastery, attention/energy, current-concept one-hot,
  steps remaining, recent-performance streak.
- **Actions** (`Discrete(8)`): easy / medium / hard exercise, hint, review prerequisite,
  advance concept, take break, assessment quiz.
- **Rewards**: mastery gained, success bonus, difficulty-mismatch & prerequisite penalties,
  per-step efficiency cost, large terminal bonus for mastering the target, large penalty on dropout.
- **Start**: near-zero mastery, full attention, at the entry concept (randomised hidden aptitude).
- **Terminal**: target mastered (success), attention → 0 (dropout), or step budget exhausted.

## Project structure
```
├── main.py                  # CLI entry (train / play / demo / evaluate / plots)
├── play.py                  # GUI playback of the best agent, verbose terminal output
├── environment/             # custom_env.py (Gym env) + rendering.py (PyBullet 3D)
├── training/                # dqn_training.py, pg_training.py, plots.py, common.py
├── api/serve.py             # FastAPI JSON service
├── models/{dqn,pg}/         # saved models
├── logs/                    # per-run monitor/progress CSVs + TensorBoard
├── assets/                  # report figures & screenshots
└── tests/                   # env sanity tests
```
