"""SchoolCheckIn-RL: a custom Gymnasium environment for biometric school check-in.

Mission (education): a student arrives at the school gate and must register attendance
before the bell. The intended route is biometric -- reach a fingerprint scanner and scan in
-- but the hall is deliberately awkward:

  * The scanner only reads a *clean* finger, and cleanliness can only be restored at a
    physical **hygiene station** on the far side of the hall. So checking in is a two-stage
    plan (detour to sanitize, then cross to a scanner), not a single greedy walk.
  * There are **two scanners**. Scanner A (far corner) is reliable; scanner B (near the
    gate) is quicker to reach but flaky and is sometimes out of order entirely. Whether B
    works is hidden until the student is close enough to read its display.
  * Each scanner keeps its **own** failed-attempt counter. Too many failures lock *that*
    scanner, not the episode, so the student must re-plan to the other scanner, call staff
    to recalibrate, or fall back to the office.
  * Both scanners have a **queue** during the morning rush. Scanning a busy scanner wastes
    the step; waiting for it to clear is the correct action.
  * The **office desk** offers manual sign-in, but only to a student a scanner has actually
    locked out -- the school will not hand-sign anyone who never tried the biometric route.
    Once unlocked it always succeeds, for a far smaller reward, so it is a genuine
    risk/reward alternative (keep retrying the other scanner, or settle?) rather than a
    shortcut. Without this precondition every algorithm simply walks to the office.
  * Bumping the desk or the turnstile **contaminates** the student's hands, so collisions
    have a downstream cost rather than only a flat penalty.
  * A **bell** rings partway through the episode. Checking in before it earns a punctuality
    bonus; after it the student is tardy and the reward is cut.

Action space stays Discrete so that DQN, REINFORCE, PPO and A2C can all train on the *same*
environment for an objective comparison. Cleanliness, scan success, queueing and scanner
health are all stochastic, so the environment is non-deterministic and partially observable.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# --- World geometry --------------------------------------------------------------------
ROOM_SIZE = 10.0                        # square entrance hall [0, ROOM_SIZE]^2
START_POS = np.array([1.0, 1.0])        # the gate
HYGIENE_POS = np.array([1.5, 8.0])      # hand-sanitizer station (only place to clean)
SCANNER_A_POS = np.array([9.0, 9.0])    # primary scanner: reliable, far corner
SCANNER_B_POS = np.array([9.0, 2.0])    # secondary scanner: closer, flaky, may be broken
OFFICE_POS = np.array([5.0, 1.0])       # office desk: manual sign-in fallback

STATION_RADIUS = 0.9                    # interaction radius for every fixed station
INSPECT_RADIUS = 2.5                    # how close before a scanner's display is readable

# Two obstacles force a non-trivial route: the central desk plus a turnstile that blocks
# the direct gate -> hygiene-station corridor.
# Together these form one wall across the middle of the hall, leaving a corridor down the
# west side (x <= 1.5) and another down the east side (x >= 6.5). Routing to the sanitizer
# and then to a scanner therefore means committing to a corridor, not cutting the diagonal.
OBSTACLES = (
    (np.array([5.0, 5.0]), np.array([1.3, 1.3])),   # central desk
    (np.array([3.2, 4.6]), np.array([1.4, 0.5])),   # turnstile
)

STEP = 0.5                              # movement step size
MAX_DIST = float(np.sqrt(2.0) * ROOM_SIZE)  # hall diagonal -> keeps distance obs in [0, 1]

# --- Action ids ------------------------------------------------------------------------
(A_NORTH, A_SOUTH, A_WEST, A_EAST,
 A_SANITIZE, A_SCAN, A_WAIT, A_HELP, A_MANUAL) = range(9)

ACTION_NAMES = {
    A_NORTH: "move_north",
    A_SOUTH: "move_south",
    A_WEST: "move_west",
    A_EAST: "move_east",
    A_SANITIZE: "sanitize_hands",
    A_SCAN: "scan_fingerprint",
    A_WAIT: "wait_in_queue",
    A_HELP: "request_assistance",
    A_MANUAL: "check_in_manually",
}
_MOVES = {
    A_NORTH: np.array([0.0,  STEP]),
    A_SOUTH: np.array([0.0, -STEP]),
    A_WEST:  np.array([-STEP, 0.0]),
    A_EAST:  np.array([STEP,  0.0]),
}

# --- Hygiene / scanner dynamics --------------------------------------------------------
CLEAN_DECAY = 0.012        # cleanliness lost per step of walking
CONTAMINATE = 0.20         # cleanliness lost by bumping an obstacle (dirty hands)
SANITIZE_GAIN = 0.45       # cleanliness restored per sanitize action at the station
FAIL_CONTAMINATE = 0.12    # a rejected scan smudges the sensor further
MAX_ATTEMPTS = 3           # failed scans before *that* scanner locks the student out
HELP_DELAY = 4             # steps before called staff arrive and recalibrate
CLEAN_TARGET = 0.7         # cleanliness the student should reach before scanning

# --- Timing ----------------------------------------------------------------------------
MAX_STEPS = 150
BELL_STEP = 80             # punctuality deadline (soft); MAX_STEPS is the hard cutoff
RUSH_DECAY = 60.0          # queue pressure decays with a ~60-step time constant

# --- Reward weights --------------------------------------------------------------------
# DIST_W and MISUSE_PENALTY are balanced against each other: with 9 actions, 5 of which are
# station interactions, an exploring agent spends most of its early steps "misusing" one. If
# that penalty outweighs the per-step route progress, the shaping signal is buried in noise
# and nothing learns the two-stage route (measured: 13% check-in after 250k steps).
DIST_W = 1.5               # shaping: weight on route progress (see _potential)
STEP_COST = 0.05           # per-step cost -> arrive quickly
BLOCK_PENALTY = 0.25       # bumping a wall / obstacle
MISUSE_PENALTY = 0.2       # interacting with a station the student is not standing at
SCAN_FAIL_PENALTY = 0.5    # scan attempted at a working scanner but rejected
SCAN_BUSY_PENALTY = 0.15   # scanned while someone else was using it
SCAN_LOCKED_PENALTY = 0.6  # scanned a scanner already locked out / out of order
SANITIZE_WASTE_PENALTY = 0.15  # sanitizing hands that are already clean
WAIT_COST = 0.02           # waiting is cheap but not free
HELP_COST = 0.3            # calling staff costs goodwill (and time)
LOCKOUT_PENALTY = 2.0      # a scanner locking out
BIOMETRIC_BONUS = 20.0     # terminal: checked in biometrically (the intended route)
MANUAL_BONUS = 5.0         # terminal: manual sign-in at the office (safe but poor)
PUNCTUAL_BONUS = 5.0       # extra terminal reward, scaled by time left before the bell
TARDY_PENALTY = 4.0        # terminal: checked in after the bell
STRANDED_PENALTY = 8.0     # terminal: every route exhausted
LATE_PENALTY = 6.0         # truncation: never checked in at all


class SchoolCheckInEnv(gym.Env):
    """Biometric school check-in environment (discrete actions, stochastic + partial obs)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    #: order matters -- index 0 is scanner A, index 1 is scanner B
    SCANNERS = (SCANNER_A_POS, SCANNER_B_POS)

    def __init__(self, render_mode: str | None = None, max_steps: int = MAX_STEPS):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.bell_step = min(BELL_STEP, max_steps)

        self.action_space = spaces.Discrete(9)
        # 29-d observation, all components scaled into [-1, 1]:
        #   [0:2]   own position
        #   [2:14]  (dx, dy, distance) to scanner A, scanner B, hygiene station, office
        #   [14:18] at-station flags for the same four targets
        #   [18]    cleanliness
        #   [19:21] failed attempts at scanner A / B
        #   [21:23] lockout flags for scanner A / B
        #   [23]    scanner-B health belief (0 until inspected up close, then +-1)
        #   [24:26] observed queue length at scanner A / B
        #   [26]    time left before the hard cutoff
        #   [27]    time left before the bell (negative once tardy)
        #   [28]    whether the office will accept a manual sign-in yet
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)

        self.pos = START_POS.copy()
        self.cleanliness = 1.0
        self.attempts = np.zeros(2, dtype=int)
        self.locked = np.zeros(2, dtype=bool)
        self.queue = np.zeros(2, dtype=int)
        self.steps = 0
        self.checked_in = False
        self.checkin_mode: str | None = None
        self.stranded = False
        self.b_broken = False
        self.b_inspected = False
        self.help_eta = -1
        self.help_target = -1
        self.scan_reliability = np.ones(2)
        self.rush = 1.0
        self.episode_over = True   # must reset() before stepping
        self._renderer = None

    # -- geometry helpers ---------------------------------------------------------------
    def _dist_to(self, target: np.ndarray) -> float:
        return float(np.linalg.norm(self.pos - target))

    def _at(self, target: np.ndarray) -> bool:
        return self._dist_to(target) <= STATION_RADIUS

    def _at_scanner(self, idx: int) -> bool:
        return self._at(self.SCANNERS[idx])

    def _current_scanner(self) -> int | None:
        """Index of the scanner the student is standing at, if any."""
        for idx in range(2):
            if self._at_scanner(idx):
                return idx
        return None

    def _blocked(self, p: np.ndarray) -> bool:
        if np.any(p < 0.0) or np.any(p > ROOM_SIZE):
            return True
        return any(bool(np.all(np.abs(p - c) <= h)) for c, h in OBSTACLES)

    # -- objective tracking -------------------------------------------------------------
    def _usable_scanners(self) -> list[int]:
        """Scanners that are neither locked out nor known to be out of order."""
        out = []
        for idx in range(2):
            if self.locked[idx]:
                continue
            if idx == 1 and self.b_broken:
                continue
            out.append(idx)
        return out

    def _manual_available(self) -> bool:
        """The office only hand-signs a student a scanner has already locked out."""
        return bool(self.locked.any())

    def _objective(self) -> np.ndarray:
        """The waypoint the student should currently be heading for (HUD / info only).

        Dirty hands -> the hygiene station. Clean hands -> the nearest usable scanner.
        No usable scanner left -> the office desk.
        """
        usable = self._usable_scanners()
        if not usable:
            return OFFICE_POS
        if self.cleanliness < CLEAN_TARGET and not self._at(HYGIENE_POS):
            return HYGIENE_POS
        return min((self.SCANNERS[i] for i in usable), key=self._dist_to)

    def _potential(self) -> float:
        """Potential Phi(s) = -(remaining route length), used for reward shaping.

        Shaping is applied as Phi(s') - Phi(s). Because Phi is a pure function of the
        state, every closed loop in state space sums to zero, so no cyclic behaviour can
        farm it -- the property that matters here. A naive "reward approaching the current
        sub-goal" term does *not* have it: flipping the sub-goal between the sanitizer and
        the scanner pays out on both legs, and an agent learns to shuttle between the two
        forever, scoring well above the check-in bonus while never scanning at all.

        The route length blends the direct leg (pos -> scanner) with the detour leg
        (pos -> sanitizer -> scanner), weighted by how dirty the hands are, so the
        transition is smooth rather than a step change mid-journey.
        """
        usable = self._usable_scanners()
        if not usable:
            # every biometric route is gone; the office is the only thing left
            return -self._dist_to(OFFICE_POS)
        deficit = float(np.clip((CLEAN_TARGET - self.cleanliness) / CLEAN_TARGET, 0.0, 1.0))
        d_hyg = self._dist_to(HYGIENE_POS)
        direct = min(self._dist_to(self.SCANNERS[i]) for i in usable)
        detour = min(d_hyg + float(np.linalg.norm(HYGIENE_POS - self.SCANNERS[i]))
                     for i in usable)
        return -(deficit * detour + (1.0 - deficit) * direct)

    def _queue_pressure(self) -> float:
        """Probability a scanner is occupied -- highest during the morning rush."""
        return 0.45 * self.rush * float(np.exp(-self.steps / RUSH_DECAY))

    def _roll_queues(self):
        for idx in range(2):
            if self.queue[idx] > 0:
                self.queue[idx] -= 1
            elif self.np_random.random() < self._queue_pressure():
                self.queue[idx] = int(self.np_random.integers(1, 4))

    # -- observation / info -------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        parts = [
            2.0 * self.pos[0] / ROOM_SIZE - 1.0,
            2.0 * self.pos[1] / ROOM_SIZE - 1.0,
        ]
        for target in (SCANNER_A_POS, SCANNER_B_POS, HYGIENE_POS, OFFICE_POS):
            d = target - self.pos
            parts += [d[0] / ROOM_SIZE, d[1] / ROOM_SIZE, self._dist_to(target) / MAX_DIST]
        parts += [
            1.0 if self._at_scanner(0) else 0.0,
            1.0 if self._at_scanner(1) else 0.0,
            1.0 if self._at(HYGIENE_POS) else 0.0,
            1.0 if self._at(OFFICE_POS) else 0.0,
            self.cleanliness,
            self.attempts[0] / MAX_ATTEMPTS,
            self.attempts[1] / MAX_ATTEMPTS,
            1.0 if self.locked[0] else 0.0,
            1.0 if self.locked[1] else 0.0,
            # partial observability: B's health reads 0 until inspected from close range
            0.0 if not self.b_inspected else (-1.0 if self.b_broken else 1.0),
            min(self.queue[0], 3) / 3.0,
            min(self.queue[1], 3) / 3.0,
            (self.max_steps - self.steps) / self.max_steps,
            np.clip((self.bell_step - self.steps) / self.bell_step, -1.0, 1.0),
            1.0 if self._manual_available() else 0.0,
        ]
        return np.clip(np.array(parts, dtype=np.float32), -1.0, 1.0)

    def _get_info(self) -> dict:
        scanner = self._current_scanner()
        return {
            "position": self.pos.tolist(),
            "distance": round(self._dist_to(self._objective()), 3),
            "objective": self._objective().tolist(),
            "cleanliness": round(float(self.cleanliness), 3),
            "attempts": int(self.attempts.sum()),
            "attempts_per_scanner": self.attempts.tolist(),
            "locked": self.locked.tolist(),
            "queue": self.queue.tolist(),
            "at_scanner": scanner is not None,
            "at_scanner_idx": scanner,
            "at_hygiene": self._at(HYGIENE_POS),
            "at_office": self._at(OFFICE_POS),
            "manual_available": self._manual_available(),
            "scanner_b_broken": bool(self.b_broken) if self.b_inspected else None,
            "help_eta": int(self.help_eta),
            "tardy": self.steps > self.bell_step,
            "checked_in": bool(self.checked_in),
            "checkin_mode": self.checkin_mode,
            "stranded": bool(self.stranded),
        }

    # -- gym API ------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.pos = START_POS.copy()
        # a fresh student: hands are dirty to varying degrees (came in from outside)
        self.cleanliness = float(self.np_random.uniform(0.15, 0.6))
        self.attempts = np.zeros(2, dtype=int)
        self.locked = np.zeros(2, dtype=bool)
        self.queue = np.zeros(2, dtype=int)
        self.steps = 0
        self.checked_in = False
        self.checkin_mode = None
        self.stranded = False
        self.help_eta = -1
        self.help_target = -1
        # scanner A is dependable; B is flaky and out of order a quarter of the time
        self.scan_reliability = np.array([
            float(self.np_random.uniform(0.80, 1.00)),
            float(self.np_random.uniform(0.45, 0.85)),
        ])
        self.b_broken = bool(self.np_random.random() < 0.25)
        self.b_inspected = False
        self.rush = 1.0
        self.episode_over = False

        if options:
            if "cleanliness" in options:
                self.cleanliness = float(options["cleanliness"])
            if "scan_reliability" in options:
                # scalar scales both scanners; a pair sets them individually
                r = options["scan_reliability"]
                self.scan_reliability = (np.asarray(r, dtype=float) if np.ndim(r)
                                         else np.array([float(r), float(r)]))
            if "scanner_b_broken" in options:
                self.b_broken = bool(options["scanner_b_broken"])
            if "rush" in options:
                # 0.0 disables queueing entirely (useful for isolating other effects)
                self.rush = float(options["rush"])

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, self._get_info()

    def _success_prob(self, idx: int) -> float:
        """Scan acceptance probability: driven by hygiene, capped by scanner quality."""
        hygiene = np.clip((self.cleanliness - 0.45) / 0.45, 0.02, 0.97)
        return float(hygiene * self.scan_reliability[idx])

    def _terminal_reward(self, base: float) -> float:
        """Apply the punctuality bonus or the tardiness penalty to a check-in reward."""
        if self.steps <= self.bell_step:
            return base + PUNCTUAL_BONUS * (self.bell_step - self.steps) / self.bell_step
        return base - TARDY_PENALTY

    def step(self, action: int):
        assert self.action_space.contains(action), f"invalid action {action}"
        if self.episode_over:
            # Without this the episode keeps running past its terminal state and a repeated
            # scan re-awards the check-in bonus every time -- measured +123.50 over five
            # extra scans. Callers must reset() between episodes.
            raise RuntimeError(
                "step() called after the episode ended; call reset() first")
        action = int(action)
        self.steps += 1
        reward = -STEP_COST
        prev_potential = self._potential()
        terminated = False
        truncated = False

        self._roll_queues()

        # staff called earlier may arrive this step and recalibrate the scanner
        if self.help_eta > 0:
            self.help_eta -= 1
            if self.help_eta == 0:
                idx = self.help_target
                self.locked[idx] = False
                self.attempts[idx] = 0
                if idx == 1:
                    self.b_broken = False       # a technician actually fixes B
                    self.b_inspected = True
                self.help_eta = -1
                self.help_target = -1

        # a scanner display becomes readable once the student is close enough
        if self._dist_to(SCANNER_B_POS) <= INSPECT_RADIUS:
            self.b_inspected = True

        if action in _MOVES:
            candidate = self.pos + _MOVES[action]
            if self._blocked(candidate):
                # bumping the desk/turnstile both hurts and dirties the hands
                reward -= BLOCK_PENALTY
                self.cleanliness = float(np.clip(self.cleanliness - CONTAMINATE, 0.0, 1.0))
            else:
                self.pos = candidate
                self.cleanliness = float(np.clip(self.cleanliness - CLEAN_DECAY, 0.0, 1.0))

        elif action == A_SANITIZE:
            if not self._at(HYGIENE_POS):
                reward -= MISUSE_PENALTY            # no sanitizer here
            elif self.cleanliness >= 0.95:
                reward -= SANITIZE_WASTE_PENALTY    # hands are already clean
            else:
                self.cleanliness = float(np.clip(self.cleanliness + SANITIZE_GAIN, 0.0, 1.0))

        elif action == A_SCAN:
            idx = self._current_scanner()
            if idx is None:
                reward -= MISUSE_PENALTY
            elif self.locked[idx] or (idx == 1 and self.b_broken):
                self.b_inspected = True
                reward -= SCAN_LOCKED_PENALTY
            elif self.queue[idx] > 0:
                reward -= SCAN_BUSY_PENALTY         # someone else is at the sensor
            elif self.np_random.random() < self._success_prob(idx):
                self.checked_in = True
                self.checkin_mode = "biometric"
                terminated = True
                reward += self._terminal_reward(BIOMETRIC_BONUS)
            else:
                self.attempts[idx] += 1
                self.cleanliness = float(
                    np.clip(self.cleanliness - FAIL_CONTAMINATE, 0.0, 1.0))
                reward -= SCAN_FAIL_PENALTY
                if self.attempts[idx] >= MAX_ATTEMPTS:
                    self.locked[idx] = True
                    reward -= LOCKOUT_PENALTY

        elif action == A_WAIT:
            reward -= WAIT_COST                     # let the queue ahead clear

        elif action == A_HELP:
            idx = self._current_scanner()
            if idx is None or self.help_eta > 0:
                reward -= MISUSE_PENALTY            # nobody to call, or already called
            else:
                self.help_eta = HELP_DELAY
                self.help_target = idx
                reward -= HELP_COST

        elif action == A_MANUAL:
            if not self._at(OFFICE_POS) or not self._manual_available():
                # either not at the desk, or the office has no grounds to hand-sign yet
                reward -= MISUSE_PENALTY
            else:
                self.checked_in = True
                self.checkin_mode = "manual"
                terminated = True
                reward += self._terminal_reward(MANUAL_BONUS)

        # potential-based shaping: cycle-free, so it cannot be farmed (see _potential)
        if not terminated:
            reward += DIST_W * (self._potential() - prev_potential)

        # every biometric route exhausted and the office is unreachable in time -> stranded
        if not terminated and not self._usable_scanners() and self.help_eta < 0:
            steps_to_office = self._dist_to(OFFICE_POS) / STEP
            if steps_to_office > (self.max_steps - self.steps):
                self.stranded = True
                terminated = True
                reward -= STRANDED_PENALTY

        if not terminated and self.steps >= self.max_steps:
            truncated = True
            reward -= LATE_PENALTY

        self.episode_over = bool(terminated or truncated)
        info = self._get_info()
        info["action_name"] = ACTION_NAMES[action]
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
            self._renderer = PyBulletRenderer(
                room_size=ROOM_SIZE, start_pos=START_POS, scanners=self.SCANNERS,
                hygiene_pos=HYGIENE_POS, office_pos=OFFICE_POS, obstacles=OBSTACLES,
                station_radius=STATION_RADIUS, gui=self.render_mode == "human",
            )
        return self._renderer.draw(
            position=self.pos, cleanliness=self.cleanliness, attempts=self.attempts,
            locked=self.locked, queue=self.queue, b_broken=self.b_broken,
            b_inspected=self.b_inspected, at_scanner_idx=self._current_scanner(),
            checked_in=self.checked_in, checkin_mode=self.checkin_mode,
            help_eta=self.help_eta, step=self.steps, bell_step=self.bell_step,
            mode=self.render_mode,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
