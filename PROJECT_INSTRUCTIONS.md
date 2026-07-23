# RL Summative Assignment — Project Requirements

## Overview
Train a reinforcement learning agent by comparing **Value-Based** (Deep Q-Networks)
and **Policy-Based** methods (REINFORCE, PPO, A2C) to optimize a simulated,
mission-based environment derived from the capstone project.

---

## Tasks

### 1. Custom Environment
Develop a **non-generic** environment based on your capstone project/mission. Implement it
as a custom Gymnasium environment and define:

- **Action Space** — exhaustive, relevant to the use case, and mappable to the real world.
- **Observation Space**
- **Rewards** — associated with states and/or actions.
- **Start State**
- **Terminal Conditions**

### 2. Visualization
- Visualize the environment using an advanced simulation library
  (Three.js, MuJoCo within Gymnasium, OpenGL, etc.).
- Include a visual of the environment in the report.

### 3. Train Four RL Models (Stable-Baselines)
All models must interact with the **same environment** for an objective comparison.

| Category | Algorithm |
|---|---|
| Value-Based | **DQN** |
| Policy Gradient | **REINFORCE** |
| Policy Gradient | **PPO** (Proximal Policy Optimization) |
| Actor-Critic | **A2C** |

### 4. Hyperparameter Tuning
- Tweak hyperparameters and **discuss observed behaviour**.
- Follow the provided template for parameters per algorithm.
- Tuning must be **extensive**: **at least 10 runs** with different hyperparameter
  combinations **per algorithm**.

### 5. Recorded Demo Video (5 minutes)
- Share your **entire screen** with **camera on**.
- In the video:
  - Briefly state the problem.
  - State the agent behaviour.
  - Briefly explain the reward structure.
  - State the objective of the agent.
- Run the simulation with your **best-performing agent**, showing the **GUI and terminal
  verbose output**. Show directory navigation and how you execute the Python files.
  All open windows must be visible.
- File execution must occupy **3/4 of the video duration** (e.g. 3 min of a 5-min video).
- Explain agent performance in the simulation.

### 6. Report
- **7–10 pages** (minimum 7, maximum 10).
- Follow the report template.
- Keep it **focused** — avoid lengthy explanations.
- Saved as a **PDF** and submitted to Canvas.

---

## Repository

Create a GitHub repository named **`student_name_rl_summative`**.

```
project_root/
├── pyproject.toml          # Mandatory
├── uv.lock                 # Recommended
├── README.md
├── main.py
│
├── environment/
│   ├── __init__.py
│   ├── custom_env.py
│   └── rendering.py
│
├── training/
│   ├── __init__.py
│   ├── dqn_training.py
│   └── pg_training.py
│
├── models/
│   ├── dqn/
│   └── pg/
│
├── logs/
├── assets/
└── tests/                  # Recommended
```

---

## Submission Instructions
- Submit a **PDF** to Canvas.
- **`uv` is mandatory** for Python dependency and environment management.
- Repository must include a valid `pyproject.toml` (and `uv.lock` if applicable).
- The project must be runnable after cloning using **only `uv` commands**:
  ```bash
  uv sync
  uv run main.py
  ```
- The marker must **not** need to manually install dependencies or create a virtual environment.
- Failure to provide a project that clones and runs with `uv` results in a **penalty**.
