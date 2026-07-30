"""Sanity tests for the SchoolCheckIn-RL environment."""

import numpy as np
import pytest
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import environment  # noqa: F401  (registers SchoolCheckIn-v0)
from environment.custom_env import (
    SchoolCheckInEnv, SCANNER_A_POS, SCANNER_B_POS, HYGIENE_POS, OFFICE_POS,
    MAX_ATTEMPTS, STEP, A_NORTH, A_SOUTH, A_EAST, A_WEST,
    A_SANITIZE, A_SCAN, A_WAIT, A_HELP, A_MANUAL,
)


_DIRS = {A_NORTH: (0.0, STEP), A_SOUTH: (0.0, -STEP), A_WEST: (-STEP, 0.0), A_EAST: (STEP, 0.0)}


def _route(env, start, target):
    """Breadth-first search over the STEP grid, returning a list of move actions.

    The hall has two obstacles and the only gap to the sanitizer is a narrow corridor
    along the west wall, so greedy axis-walking gets stuck. BFS keeps the tests honest
    about *reachability* rather than about navigation cleverness.
    """
    from collections import deque
    key = lambda p: (round(p[0], 2), round(p[1], 2))
    start = np.asarray(start, dtype=float)
    seen, q = {key(start): []}, deque([start])
    while q:
        cur = q.popleft()
        if np.linalg.norm(cur - np.asarray(target)) <= 0.9:
            return seen[key(cur)]
        for a, (dx, dy) in _DIRS.items():
            nxt = cur + np.array([dx, dy])
            if env._blocked(nxt) or key(nxt) in seen:
                continue
            seen[key(nxt)] = seen[key(cur)] + [a]
            q.append(nxt)
    return None


def _walk_to(env, target):
    """Drive the env along a BFS route to `target`; True if it arrives."""
    path = _route(env, env.pos, target)
    if path is None:
        return False
    for a in path:
        env.step(a)
    return bool(np.linalg.norm(env.pos - np.asarray(target)) <= 0.9)


def test_sb3_env_checker():
    """SB3's checker validates spaces, reset/step signatures, and dtypes."""
    check_env(SchoolCheckInEnv(), warn=True)


def test_spaces_are_the_expected_shape():
    env = SchoolCheckInEnv()
    assert env.action_space.n == 9
    assert env.observation_space.shape == (29,)


def test_reset_obs_in_space():
    env = SchoolCheckInEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "cleanliness" in info and "objective" in info


def test_random_rollout_terminates():
    """A bounded random policy must always terminate or truncate within max_steps."""
    env = gym.make("SchoolCheckIn-v0")
    for ep in range(20):
        obs, _ = env.reset(seed=ep)
        done, steps = False, 0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            assert np.isfinite(r)
            assert env.observation_space.contains(obs)
            done = term or trunc
            steps += 1
            assert steps <= 200
    env.close()


def test_determinism_with_seed():
    env = SchoolCheckInEnv()
    o1, _ = env.reset(seed=123)
    o2, _ = env.reset(seed=123)
    assert np.allclose(o1, o2)


def test_station_actions_are_position_gated():
    """Sanitize / scan / manual check-in only work at their own station."""
    env = SchoolCheckInEnv()
    env.reset(seed=0)  # at the gate: not at any station
    for action in (A_SANITIZE, A_SCAN, A_MANUAL, A_HELP):
        env.reset(seed=0)
        _, r, term, _, info = env.step(action)
        assert not info["checked_in"], f"{action} checked in from the gate"
        assert r < 0, f"{action} was not penalised away from its station"


def test_sanitizer_only_cleans_at_the_station():
    """Cleanliness is only restorable at the hygiene station -- the key difficulty."""
    env = SchoolCheckInEnv(max_steps=400)
    env.reset(seed=1, options={"cleanliness": 0.3})
    before = env.cleanliness
    env.step(A_SANITIZE)                      # at the gate -> no effect
    assert env.cleanliness <= before

    assert _walk_to(env, HYGIENE_POS), "could not reach the hygiene station"
    dirty = env.cleanliness
    env.step(A_SANITIZE)
    assert env.cleanliness > dirty, "sanitizing at the station did not clean hands"


def test_bumping_an_obstacle_contaminates():
    """Collisions cost reward *and* cleanliness (downstream consequence)."""
    env = SchoolCheckInEnv()
    env.reset(seed=2, options={"cleanliness": 0.8})
    env.pos = np.array([5.0, 3.5])            # directly south of the central desk
    before_pos, before_clean = env.pos.copy(), env.cleanliness
    _, r, _, _, _ = env.step(A_NORTH)
    assert np.allclose(env.pos, before_pos), "student moved into the obstacle"
    assert env.cleanliness < before_clean, "bumping did not contaminate hands"
    assert r < 0


def test_lockout_is_per_scanner_not_per_episode():
    """Exhausting attempts at one scanner must not end the episode outright."""
    env = SchoolCheckInEnv(max_steps=400)
    env.reset(seed=3, options={"cleanliness": 0.0, "scan_reliability": 0.0,
                               "scanner_b_broken": False, "rush": 0.0})
    env.pos = SCANNER_A_POS.copy()
    terminated = False
    for _ in range(MAX_ATTEMPTS):
        _, _, terminated, _, info = env.step(A_SCAN)
    assert env.locked[0], "scanner A never locked out"
    assert not env.locked[1], "scanner B locked out too"
    assert not terminated, "a single scanner lockout ended the episode"
    assert not info["checked_in"]


def test_request_assistance_clears_a_lockout():
    env = SchoolCheckInEnv(max_steps=400)
    env.reset(seed=4)
    env.pos = SCANNER_A_POS.copy()
    env.locked[0] = True
    env.attempts[0] = MAX_ATTEMPTS
    env.step(A_HELP)
    assert env.help_eta > 0, "calling staff did not schedule a visit"
    for _ in range(env.help_eta + 1):
        env.step(A_WAIT)
    assert not env.locked[0], "staff never recalibrated the scanner"
    assert env.attempts[0] == 0


def test_scanner_b_health_is_hidden_until_inspected():
    """Partial observability: B's status reads 0 from afar, +-1 once close."""
    env = SchoolCheckInEnv(max_steps=400)
    obs, _ = env.reset(seed=5, options={"scanner_b_broken": True})
    assert obs[23] == 0.0, "scanner B health leaked before inspection"
    env.pos = SCANNER_B_POS + np.array([1.0, 0.0])   # inside the inspect radius
    obs, _, _, _, info = env.step(A_WAIT)
    assert obs[23] == -1.0, "broken scanner B was not revealed up close"
    assert info["scanner_b_broken"] is True


def test_manual_checkin_requires_a_lockout_first():
    """The office refuses a student who never tried the biometric route.

    Without this gate the office is a 9-step free win and every algorithm collapses onto
    it, never learning the intended sanitize-then-scan behaviour.
    """
    env = SchoolCheckInEnv(max_steps=400)
    obs, info = env.reset(seed=6)
    env.pos = OFFICE_POS.copy()
    assert not info["manual_available"]
    _, r, term, _, info = env.step(A_MANUAL)
    assert not term and not info["checked_in"], "office signed in an unlocked student"
    assert r < 0


def test_manual_checkin_is_a_valid_but_poorer_terminal():
    """Once a scanner has locked out, the office fallback succeeds -- for less reward."""
    env = SchoolCheckInEnv(max_steps=400)
    env.reset(seed=6)
    env.locked[0] = True                      # scanner A rejected the student three times
    env.pos = OFFICE_POS.copy()
    obs, r, term, _, info = env.step(A_MANUAL)
    assert term and info["checked_in"]
    assert info["checkin_mode"] == "manual"
    assert obs[28] == 1.0, "manual availability was not observable"
    assert 0 < r < 20, f"manual reward {r} should sit below the biometric bonus"


def test_biometric_checkin_outranks_manual():
    env = SchoolCheckInEnv(max_steps=400)
    env.reset(seed=7, options={"cleanliness": 1.0, "scan_reliability": 1.0, "rush": 0.0})
    env.pos = SCANNER_A_POS.copy()
    _, r_bio, term, _, info = env.step(A_SCAN)
    assert term and info["checkin_mode"] == "biometric"

    env.reset(seed=7)
    env.locked[0] = True
    env.pos = OFFICE_POS.copy()
    _, r_manual, _, _, _ = env.step(A_MANUAL)
    assert r_bio > r_manual, "biometric check-in must pay more than the office fallback"


def test_tardiness_reduces_the_terminal_reward():
    env = SchoolCheckInEnv(max_steps=400)

    def scan_at(step):
        env.reset(seed=8, options={"cleanliness": 1.0, "scan_reliability": 1.0, "rush": 0.0})
        env.pos = SCANNER_A_POS.copy()
        env.steps = step
        _, r, _, _, _ = env.step(A_SCAN)
        return r

    assert scan_at(0) > scan_at(env.bell_step + 5), "tardy check-in was not penalised"


def test_shaping_cannot_be_farmed_by_shuttling():
    """Regression: shuttling sanitizer <-> scanner must not out-earn checking in.

    An earlier design rewarded approaching the *current* sub-goal. Because the sub-goal
    flipped as the hands got dirty, walking back and forth paid on every leg and PPO
    learned to loop forever, scoring +40 with a 0% check-in rate.
    """
    env = SchoolCheckInEnv(max_steps=400)
    env.reset(seed=9, options={"cleanliness": 1.0, "rush": 0.0, "scanner_b_broken": False})
    total = 0.0
    for _ in range(4):                        # four full sanitizer <-> scanner round trips
        for target in (HYGIENE_POS, SCANNER_A_POS):
            path = _route(env, env.pos, target)
            assert path is not None
            for a in path:
                _, r, term, trunc, _ = env.step(a)
                total += r
                assert not term
                if trunc:
                    break
        for _ in range(3):                    # top the hands back up at the station
            if env._at(HYGIENE_POS):
                _, r, _, _, _ = env.step(A_SANITIZE)
                total += r
    assert total < 0, f"shuttling earned {total:+.2f}; the shaping loop is farmable"


def test_potential_is_a_pure_state_function():
    """Phi must depend only on state, so any closed loop nets zero shaping."""
    env = SchoolCheckInEnv(max_steps=400)
    env.reset(seed=10, options={"cleanliness": 0.5})
    start_pos, start_clean = env.pos.copy(), env.cleanliness
    phi_start = env._potential()
    env.pos = np.array([4.0, 7.0])            # wander off
    env.cleanliness = 0.9
    assert env._potential() != phi_start
    env.pos, env.cleanliness = start_pos, start_clean   # return to the same state
    assert env._potential() == phi_start


def test_stepping_after_the_episode_ends_is_refused():
    """Regression: a finished episode used to keep paying out.

    Repeating the winning scan re-awarded the check-in bonus every time (+123.50 over five
    extra scans), and the JSON API drove the env directly, so a client could farm it.
    """
    env = SchoolCheckInEnv()
    env.reset(seed=7, options={"cleanliness": 1.0, "scan_reliability": 1.0, "rush": 0.0})
    env.pos = SCANNER_A_POS.copy()
    _, _, term, _, info = env.step(A_SCAN)
    assert term and info["checked_in"]
    assert env.episode_over
    with pytest.raises(RuntimeError):
        env.step(A_SCAN)
    env.reset(seed=7)                      # reset clears it again
    assert not env.episode_over
    env.step(A_WAIT)


def test_max_steps_is_not_overridden_by_a_time_limit_wrapper():
    """Regression: registering with max_episode_steps pinned every episode to 150.

    gym.make(..., max_steps=400) was truncated at 150 while the observation still reported
    time remaining out of 400, and the lateness penalty never fired.
    """
    env = gym.make("SchoolCheckIn-v0", max_steps=220)
    env.reset(seed=0)
    steps, truncated = 0, False
    while True:
        _, _, term, truncated, _ = env.step(A_WAIT)
        steps += 1
        if term or truncated:
            break
    env.close()
    assert truncated and steps == 220, f"episode ended at {steps}, expected 220"


def test_goal_reachable_by_scripted_two_stage_policy():
    """A scripted student that sanitizes first, then scans, should check in --
    confirming the harder reward/terminal design is still solvable."""
    reached = False
    for seed in range(12):
        env = SchoolCheckInEnv(max_steps=400)
        env.reset(seed=seed, options={"scan_reliability": 1.0, "scanner_b_broken": False})
        if not _walk_to(env, HYGIENE_POS):
            continue
        for _ in range(4):
            if env.cleanliness >= 0.95:
                break
            env.step(A_SANITIZE)
        if not _walk_to(env, SCANNER_A_POS):
            continue
        for _ in range(12):
            if env.queue[0] > 0:
                env.step(A_WAIT)
                continue
            _, _, term, trunc, info = env.step(A_SCAN)
            if info["checked_in"]:
                reached = True
                break
            if term or trunc:
                break
        if reached:
            break
    assert reached, "student never checked in under the scripted two-stage policy"
