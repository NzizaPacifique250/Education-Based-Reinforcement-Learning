"""EduPath-RL: a custom Gymnasium environment for adaptive learning-path sequencing.

Mission (education): an intelligent-tutoring policy must sequence learning activities to
bring a *stochastic* student to mastery of a target concept efficiently -- respecting the
curriculum's prerequisite graph, matching exercise difficulty to the student's current
ability, and avoiding fatigue-driven dropout.

The agent observes the student's estimated mastery per concept, remaining attention/energy,
its current position in the curriculum, a recent-performance signal, and the remaining time
budget. It chooses one discrete pedagogical action per step. Student responses are sampled
stochastically, so the same action can succeed or fail depending on hidden ability.

Action space is Discrete(8) so that DQN, REINFORCE, PPO and A2C can all train on the *same*
environment for an objective comparison.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# --- Curriculum definition -------------------------------------------------------------
# A small prerequisite DAG over 6 concepts. concept i unlocks only when every concept in
# PREREQS[i] is mastered. Concept 5 is the target (mastering it implies mastering the chain).
N_CONCEPTS = 6
PREREQS: dict[int, list[int]] = {
    0: [],          # entry concept
    1: [0],
    2: [0],
    3: [1],
    4: [1, 2],
    5: [3, 4],      # target concept
}
TARGET_CONCEPT = 5

# Difficulty of each exercise type (the mastery level it is calibrated for).
DIFFICULTY = {"easy": 0.25, "medium": 0.55, "hard": 0.85}
MASTERY_THRESHOLD = 0.7   # a concept counts as "mastered" at/above this

# --- Action ids ------------------------------------------------------------------------
A_EASY, A_MEDIUM, A_HARD, A_HINT, A_REVIEW, A_ADVANCE, A_BREAK, A_QUIZ = range(8)
ACTION_NAMES = {
    A_EASY: "present_easy_exercise",
    A_MEDIUM: "present_medium_exercise",
    A_HARD: "present_hard_exercise",
    A_HINT: "give_hint",
    A_REVIEW: "review_prerequisite",
    A_ADVANCE: "advance_concept",
    A_BREAK: "take_break",
    A_QUIZ: "assessment_quiz",
}

# --- Reward weights (tunable shaping constants) ----------------------------------------
W_MASTERY = 5.0          # reward per unit of total mastery gained
W_SUCCESS = 0.3          # small bonus for a successful exercise
STEP_COST = 0.05         # per-step cost -> encourages efficient sequencing
MISMATCH_PENALTY = 0.4   # difficulty far from current mastery (boredom / frustration)
PREREQ_PENALTY = 1.0     # attempting a concept whose prerequisites are unmet
ADVANCE_FAIL_PENALTY = 0.6   # trying to advance when nothing is unlockable
QUIZ_FAIL_PENALTY = 0.5      # quizzing a concept that is not yet mastered
GOAL_BONUS = 20.0        # terminal success: target concept mastered
DROPOUT_PENALTY = 10.0   # terminal failure: student disengaged (attention -> 0)


class EduPathEnv(gym.Env):
    """Adaptive learning-path environment (discrete actions, continuous observation)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, max_steps: int = 200):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(8)
        # obs = mastery[N] + attention[1] + current-concept one-hot[N] + steps_left[1] + streak[1]
        obs_dim = 2 * N_CONCEPTS + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # runtime state (initialised in reset)
        self.mastery: np.ndarray | None = None
        self.attention: float = 1.0
        self.current: int = 0
        self.steps: int = 0
        self.streak: float = 0.0
        self._hint_active: bool = False
        self.aptitude: float = 1.0  # hidden per-student learning-rate multiplier

        self._renderer = None  # lazily constructed PyBullet renderer

    # -- helpers ------------------------------------------------------------------------
    def _prereqs_met(self, concept: int) -> bool:
        return all(self.mastery[p] >= MASTERY_THRESHOLD for p in PREREQS[concept])

    def _next_unlockable(self) -> int | None:
        """Lowest-index not-yet-mastered concept whose prerequisites are all met."""
        for c in range(N_CONCEPTS):
            if self.mastery[c] < MASTERY_THRESHOLD and self._prereqs_met(c) and c != self.current:
                return c
        return None

    def _get_obs(self) -> np.ndarray:
        onehot = np.zeros(N_CONCEPTS, dtype=np.float32)
        onehot[self.current] = 1.0
        steps_left = np.float32((self.max_steps - self.steps) / self.max_steps)
        streak = np.float32(np.clip(0.5 * (self.streak + 1.0), 0.0, 1.0))
        return np.concatenate(
            [self.mastery.astype(np.float32),
             np.array([self.attention], dtype=np.float32),
             onehot,
             np.array([steps_left], dtype=np.float32),
             np.array([streak], dtype=np.float32)]
        )

    def _get_info(self) -> dict:
        return {
            "current_concept": self.current,
            "attention": float(self.attention),
            "mastered": [int(self.mastery[c] >= MASTERY_THRESHOLD) for c in range(N_CONCEPTS)],
            "mean_mastery": float(self.mastery.mean()),
            "target_mastered": bool(self.mastery[TARGET_CONCEPT] >= MASTERY_THRESHOLD),
        }

    # -- gym API ------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # a fresh student: low but nonzero mastery, full attention, randomised aptitude
        self.mastery = self.np_random.uniform(0.0, 0.1, size=N_CONCEPTS).astype(np.float32)
        self.attention = 1.0
        self.current = 0
        self.steps = 0
        self.streak = 0.0
        self._hint_active = False
        # hidden aptitude in [0.7, 1.3] -> used for generalization tests across student profiles
        self.aptitude = float(self.np_random.uniform(0.7, 1.3))
        if options and "aptitude" in options:
            self.aptitude = float(options["aptitude"])

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, self._get_info()

    def _attempt_exercise(self, difficulty: float) -> tuple[float, bool]:
        """Return (reward, success) for presenting an exercise of given difficulty."""
        c = self.current
        reward = 0.0

        # prerequisites unmet -> frustrating, low chance, penalised
        prereq_ok = self._prereqs_met(c)
        if not prereq_ok:
            reward -= PREREQ_PENALTY

        m = self.mastery[c]
        # success probability: logistic in (mastery - difficulty), scaled by attention.
        base = 1.0 / (1.0 + np.exp(-6.0 * (m - difficulty)))
        p = base * (0.6 + 0.4 * self.attention)
        if self._hint_active:
            p = min(1.0, p + 0.25)
        if not prereq_ok:
            p *= 0.3
        success = bool(self.np_random.random() < p)

        # zone-of-proximal-development learning gain: largest when difficulty is a touch
        # above current mastery. Only meaningful gain on success.
        zpd = np.exp(-((difficulty - m - 0.1) ** 2) / (2 * 0.18 ** 2))
        gain = (0.25 if success else 0.04) * zpd * self.aptitude
        if not prereq_ok:
            gain *= 0.2
        prev = self.mastery[c]
        self.mastery[c] = float(np.clip(self.mastery[c] + gain, 0.0, 1.0))
        reward += W_MASTERY * (self.mastery[c] - prev)

        # difficulty mismatch: too easy (boredom) or too hard (frustration)
        mismatch = abs(difficulty - m)
        if mismatch > 0.45:
            reward -= MISMATCH_PENALTY * (mismatch - 0.45)

        # attention cost: harder + failures cost more; success is mildly motivating
        cost = 0.05 + 0.10 * difficulty + (0.08 if not success else 0.0)
        self.attention = float(np.clip(self.attention - cost, 0.0, 1.0))

        if success:
            reward += W_SUCCESS
            self.streak = np.clip(self.streak + 1.0, -3.0, 3.0)
        else:
            self.streak = np.clip(self.streak - 1.0, -3.0, 3.0)

        self._hint_active = False  # hint is consumed by the next exercise
        return reward, success

    def step(self, action: int):
        assert self.action_space.contains(action), f"invalid action {action}"
        self.steps += 1
        reward = -STEP_COST
        last_success = None

        if action in (A_EASY, A_MEDIUM, A_HARD):
            diff = {A_EASY: DIFFICULTY["easy"], A_MEDIUM: DIFFICULTY["medium"],
                    A_HARD: DIFFICULTY["hard"]}[action]
            r, last_success = self._attempt_exercise(diff)
            reward += r

        elif action == A_HINT:
            # scaffolding: cheap, boosts next exercise, tiny direct mastery nudge
            self._hint_active = True
            prev = self.mastery[self.current]
            self.mastery[self.current] = float(np.clip(prev + 0.02 * self.aptitude, 0.0, 1.0))
            reward += W_MASTERY * (self.mastery[self.current] - prev)
            self.attention = float(np.clip(self.attention - 0.03, 0.0, 1.0))

        elif action == A_REVIEW:
            prqs = PREREQS[self.current]
            if prqs:
                weakest = min(prqs, key=lambda p: self.mastery[p])
                prev = self.mastery[weakest]
                self.mastery[weakest] = float(np.clip(prev + 0.10 * self.aptitude, 0.0, 1.0))
                reward += W_MASTERY * (self.mastery[weakest] - prev)
                self.attention = float(np.clip(self.attention + 0.02, 0.0, 1.0))  # confidence
            else:
                reward -= 0.1  # nothing to review at the entry concept
            self.attention = float(np.clip(self.attention - 0.04, 0.0, 1.0))

        elif action == A_ADVANCE:
            nxt = self._next_unlockable()
            if nxt is not None:
                self.current = nxt
                reward += 0.2  # correct forward progression
            else:
                reward -= ADVANCE_FAIL_PENALTY

        elif action == A_BREAK:
            self.attention = float(np.clip(self.attention + 0.35, 0.0, 1.0))

        elif action == A_QUIZ:
            if self.mastery[self.current] >= MASTERY_THRESHOLD:
                reward += 0.5  # confirmed mastery
            else:
                reward -= QUIZ_FAIL_PENALTY
            self.attention = float(np.clip(self.attention - 0.06, 0.0, 1.0))

        # -- terminal conditions --------------------------------------------------------
        terminated = False
        truncated = False
        if self.mastery[TARGET_CONCEPT] >= MASTERY_THRESHOLD:
            reward += GOAL_BONUS
            terminated = True
        elif self.attention <= 1e-6:
            reward -= DROPOUT_PENALTY
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True

        info = self._get_info()
        if last_success is not None:
            info["last_success"] = bool(last_success)
        info["action_name"] = ACTION_NAMES[int(action)]

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    # -- rendering ----------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import PyBulletRenderer
            gui = self.render_mode == "human"
            self._renderer = PyBulletRenderer(
                n_concepts=N_CONCEPTS, prereqs=PREREQS, target=TARGET_CONCEPT,
                mastery_threshold=MASTERY_THRESHOLD, gui=gui,
            )
        return self._renderer.draw(
            mastery=self.mastery, current=self.current, attention=self.attention,
            step=self.steps, mode=self.render_mode,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
