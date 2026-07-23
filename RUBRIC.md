# Summative Rubric

**Total Points: 40**

| Criterion | Points |
|---|---|
| Environment Validity & Complexity | 10 |
| Hyperparameter Experiments & Analysis (DQN + PG) | 10 |
| System Implementation & Agent Behavior | 5 |
| Discussion & Analysis | 10 |
| Video Demonstration (Agent in Action) | 5 |

---

## 1. Environment Validity & Complexity — 10 pts

| Rating | Range | Description |
|---|---|---|
| **Exemplary** | 10 – 7.5 | Clearly defines a rich environment with a well-structured action space, rewards, and termination conditions. Agent explores all possible actions, including edge cases. Environment can be realistically integrated into a production-level pipeline. Should reflect meaningful complexity and avoid simplified abstractions such as grid worlds unless clearly justified. Encouraged to use advanced tools/libraries to model realistic dynamics. |
| **Good** | 7.5 – 5.0 | Environment works & is logically sound, but low complexity or weak realism. Rewards correct but minimally validated. |
| **Fair** | 5 – 2.5 | Limited state/action interactions. Environment nearly deterministic or overly simplistic. Rewards appear arbitrary. |
| **Needs improvement** | 2.5 – 0 | Ill-defined state/action/reward space. Agent succeeds by trivial behavior or random chance. |

---

## 2. Hyperparameter Experiments & Analysis (DQN + PG) — 10 pts
*Multiple experiments with varied hyperparameters and performance insights.*

| Rating | Range | Description |
|---|---|---|
| **Exemplary** | 10 – 7.5 | All four tables (DQN, REINFORCE, A2C, PPO) fully completed with **10 rows each**, with clearly varying hyperparameters represented as column names. Effects of tuning explained (learning rate, gamma, entropy, buffer size, etc.). Demonstrates understanding of stability, convergence, and exploration. |
| **Good** | 7.5 – 5.0 | One table shows minimal variation in hyperparameter tuning. Basic explanation provided. Or one table contains fewer than ten experiments; explanations are good but require minimal improvement. |
| **Fair** | 5 – 2.5 | Two tables show minimal variation in hyperparameter tuning. Basic explanation provided. Or one table contains fewer than ten experiments; explanations are generic. |
| **Needs improvement** | 2.5 – 0 | No tuning or analysis; identical/default hyperparameters used across runs. Table(s) are missing. |

---

## 3. System Implementation & Agent Behavior — 5 pts
*Agent behavior aligns with goal when running `play.py`.*

| Rating | Range | Description |
|---|---|---|
| **Exemplary** | 5 – 3.5 | High-quality 2D/3D visualization using advanced libraries (OpenGL, Panda3D, Gym, PyBullet). Agent behavior aligns with the environment's goals — moves toward target, reflects learned policy, shows balanced exploration/exploitation. Easy to see how it integrates into web/mobile products. Went the extra step to show serialization to JSON and use as an API to a frontend. |
| **Good** | 3.5 – 2.5 | Basic 2D visualization using Pygame with basic shapes (circles, squares). Agent mostly follows the objective but shows occasional inconsistencies. |
| **Fair** | 2.5 – 1.5 | Minimal visualization, primarily Matplotlib or similar with limited improvements. Agent behaves unpredictably or inconsistently with goal; limited explanation. |
| **Needs improvement** | 1.5 – 0 | Only prints verbose logs to the terminal with no graphical representation. Agent fails to move toward goal; behavior random; no explanation provided. |

---

## 4. Discussion & Analysis — 10 pts

| Rating | Range | Description |
|---|---|---|
| **Exemplary** | 10 – 8.0 | All required visualizations included, well-labelled, and clear: cumulative reward curves (all methods in subplots), DQN objective curves, PG entropy curves, convergence plots, and generalization tests. Descriptions precise. Discussion integrates both qualitative insights and numerical evidence — depth of interpretation of learning behavior, stability, exploration vs exploitation, and convergence. |
| **Good** | 8 – 5.0 | Visualizations included and well-labelled but **not clear**, or **missing one** (cumulative reward, DQN objective, PG entropy, convergence, generalization). Descriptions precise. Discussion could be improved. |
| **Fair** | 5 – 2.5 | Some visualizations missing, or graphs generally not clear with some irrelevant labels/captions/descriptions. Discussion could be stronger or more consistent. Formatting mostly good but may vary between figures. |
| **Needs improvement** | 2.5 – 0 | Few or no graphs included, or visuals unclear, poorly labeled, or misleading. Descriptions missing, inaccurate, or disconnected from visuals. Discussion does not use graphs to support analysis; significant improvement needed in integrating visual data into the overall argument. |

---

## 5. Video Demonstration (Agent in Action) — 5 pts
*Full-screen, camera on, shows simulation, explains behavior/rewards/objective.*

| Rating | Range | Description |
|---|---|---|
| **Exemplary** | 5 – 3.0 | Meets all requirements: camera on, full-screen shared, problem stated, agent behavior explained, reward structure stated, objective stated, simulation shown with GUI + terminal verbose, and agent performance clearly interpreted. |
| **Good** | 3 – 2.0 | Missing one or two required elements but simulation is clear and explained. |
| **Fair** | 2 – 1.0 | Multiple missing components; simulation shown with limited explanation. |
| **No Marks** | 1 – 0 | Video missing or does not demonstrate agent behavior. |
