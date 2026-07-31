"""Machine-readable contract for SchoolCheckIn-v0, shared by the API, the exporter and the report.

Everything a frontend needs in order to render the entrance and drive the agent is defined
here once and serialized to JSON: the observation layout, the action list, the reward table
and the scene geometry. The numbers are read from `environment.custom_env`, so the JSON a
web or mobile client consumes cannot drift away from the environment that was trained.

    from api.schema import manifest, state_dict
"""

from __future__ import annotations

import gymnasium as gym

import environment  # noqa: F401  (registers SchoolCheckIn-v0)
from environment.custom_env import (
    ACTION_NAMES, ROOM_SIZE, START_POS, SCANNER_A_POS, SCANNER_B_POS, HYGIENE_POS,
    OFFICE_POS, OBSTACLES, STATION_RADIUS, INSPECT_RADIUS, MAX_ATTEMPTS, MAX_STEPS,
    BELL_STEP, STEP, CLEAN_TARGET, CLEAN_DECAY, CONTAMINATE, SANITIZE_GAIN,
    FAIL_CONTAMINATE, HELP_DELAY, STEP_COST, DIST_W, BLOCK_PENALTY, MISUSE_PENALTY,
    SCAN_FAIL_PENALTY, SCAN_BUSY_PENALTY, SCAN_LOCKED_PENALTY, SANITIZE_WASTE_PENALTY,
    WAIT_COST, HELP_COST, LOCKOUT_PENALTY, BIOMETRIC_BONUS, MANUAL_BONUS, PUNCTUAL_BONUS,
    TARDY_PENALTY, STRANDED_PENALTY, LATE_PENALTY,
)
from training.common import ENV_ID

SCHEMA_VERSION = "1.0.0"

# --- Observation ------------------------------------------------------------------------
# One entry per group of features, in the order they appear in the 29-d Box. `size` is how
# many floats the group occupies; the index range is derived, so the schema cannot fall out
# of step with the vector. `source` is what would supply the feature on a real school site.
OBSERVATION_GROUPS = [
    dict(key="position", size=2, label="Pupil position (x, y)",
         description="Where the pupil is standing in the courtyard.",
         source="Overhead fisheye camera with a person tracker, or a UWB tag in the student "
                "ID card",
         encoding="float32, 2 values, scaled", range="[-1, 1]"),
    dict(key="to_reader_a", size=3, label="Offset and range to reader A",
         description="Vector and straight line distance to the main door reader.",
         source="Same tracker, combined with the surveyed coordinates of the kiosk from the "
                "site plan",
         encoding="float32, 3 values", range="[-1, 1]"),
    dict(key="to_reader_b", size=3, label="Offset and range to reader B",
         description="Vector and distance to the east side gate reader.",
         source="As above", encoding="float32, 3 values", range="[-1, 1]"),
    dict(key="to_sanitizer", size=3, label="Offset and range to sanitizer",
         description="Vector and distance to the hand sanitizer stand.",
         source="As above", encoding="float32, 3 values", range="[-1, 1]"),
    dict(key="to_reception", size=3, label="Offset and range to reception",
         description="Vector and distance to the reception booth.",
         source="As above", encoding="float32, 3 values", range="[-1, 1]"),
    dict(key="at_station", size=4, label="At station flags (4)",
         description="Whether the pupil is within 0.9 m of each of the four stations.",
         source="Bluetooth Low Energy proximity beacon at each station, or a geofence over "
                "the tracker feed",
         encoding="float32, 4 values, 0 or 1", range="{0, 1}"),
    dict(key="cleanliness", size=1, label="Hand cleanliness",
         description="Estimated cleanliness of the finger that will be presented.",
         source="Dose counter on the sanitizer dispenser plus elapsed time, cross checked "
                "against the reader's own NFIQ image quality score",
         encoding="float32", range="[0, 1]"),
    dict(key="rejections", size=2, label="Rejections at A and B",
         description="Failed scans recorded so far at each reader.",
         source="Attendance and access control REST API, per device event log",
         encoding="float32, 2 values, scaled by 3", range="[0, 1]"),
    dict(key="lockout", size=2, label="Lockout flags A and B",
         description="Whether a reader has locked this pupil out.",
         source="Same access control API, device state field",
         encoding="float32, 2 values, 0 or 1", range="{0, 1}"),
    dict(key="reader_b_health", size=1, label="Reader B health belief",
         description="Whether the side reader is out of service. Reads 0 until the pupil is "
                     "close enough to see the display.",
         source="Device heartbeat endpoint, legible on the kiosk screen within roughly 2.5 m",
         encoding="float32", range="{-1, 0, 1}"),
    dict(key="queue", size=2, label="Queue length at A and B",
         description="How many people are waiting at each reader.",
         source="Overhead camera people counter, or the queue display board feed",
         encoding="float32, 2 values, scaled by 3", range="[0, 1]"),
    dict(key="time_left", size=1, label="Time remaining",
         description="Fraction of the episode budget still left.",
         source="System clock", encoding="float32", range="[0, 1]"),
    dict(key="time_to_bell", size=1, label="Time to the bell",
         description="Fraction of time left before the late bell. Goes negative once late.",
         source="School timetable API plus system clock", encoding="float32",
         range="[-1, 1]"),
    dict(key="manual_available", size=1, label="Manual sign-in available",
         description="Whether reception will accept a signature yet.",
         source="Access control API, lockout status", encoding="float32, 0 or 1",
         range="{0, 1}"),
]

ACTION_DESCRIPTIONS = {
    0: "Walk one step (0.5 m) towards the building.",
    1: "Walk one step away from the building.",
    2: "Walk one step towards the gate side.",
    3: "Walk one step towards the reader side.",
    4: "Take a dose of sanitizer. Only works at the stand.",
    5: "Present a finger to a reader. Only works at a reader.",
    6: "Stand still and let the queue ahead clear.",
    7: "Call a staff member to recalibrate a reader. Takes 4 steps.",
    8: "Sign the register at reception. Only after a lockout.",
}


def observation_schema() -> list[dict]:
    """The 29 observation features as JSON, each group tagged with its index range."""
    out, i = [], 0
    for g in OBSERVATION_GROUPS:
        out.append({"index": [i, i + g["size"]], **g})
        i += g["size"]
    assert i == 29, f"observation schema covers {i} features, expected 29"
    return out


def action_schema() -> list[dict]:
    return [{"id": i, "name": ACTION_NAMES[i], "description": ACTION_DESCRIPTIONS[i]}
            for i in sorted(ACTION_NAMES)]


def reward_schema() -> dict:
    """Every reward constant, so a client can explain a step to a user without guessing."""
    return {
        "step_cost": -STEP_COST,
        "shaping_weight": DIST_W,
        "shaping": "potential based: DIST_W * (Phi(s') - Phi(s)), Phi = -(remaining route "
                   "length), so closed loops sum to zero",
        "blocked_by_obstacle": -BLOCK_PENALTY,
        "station_misuse": -MISUSE_PENALTY,
        "sanitize_when_clean": -SANITIZE_WASTE_PENALTY,
        "scan_rejected": -SCAN_FAIL_PENALTY,
        "scan_busy": -SCAN_BUSY_PENALTY,
        "scan_locked_or_broken": -SCAN_LOCKED_PENALTY,
        "reader_locks_out": -LOCKOUT_PENALTY,
        "wait_in_queue": -WAIT_COST,
        "request_assistance": -HELP_COST,
        "biometric_check_in": BIOMETRIC_BONUS,
        "manual_check_in": MANUAL_BONUS,
        "punctual_bonus_max": PUNCTUAL_BONUS,
        "tardy_penalty": -TARDY_PENALTY,
        "stranded_penalty": -STRANDED_PENALTY,
        "never_checked_in": -LATE_PENALTY,
    }


def dynamics_schema() -> dict:
    return {
        "step_size_m": STEP,
        "clean_decay_per_step": CLEAN_DECAY,
        "clean_lost_on_bump": CONTAMINATE,
        "clean_gained_per_dose": SANITIZE_GAIN,
        "clean_lost_on_rejection": FAIL_CONTAMINATE,
        "clean_target_before_scanning": CLEAN_TARGET,
        "max_attempts_per_reader": MAX_ATTEMPTS,
        "assistance_delay_steps": HELP_DELAY,
        "inspect_radius_m": INSPECT_RADIUS,
        "acceptance_probability": "clip((cleanliness - 0.45) / 0.45, 0.02, 0.97) * "
                                  "reader_reliability",
        "reader_b_broken_probability": 0.25,
    }


def layout() -> dict:
    """Static scene geometry: everything needed to draw the courtyard client side."""
    return {
        "room_size": ROOM_SIZE,
        "start": START_POS.tolist(),
        "scanners": [SCANNER_A_POS.tolist(), SCANNER_B_POS.tolist()],
        "scanner_labels": ["Reader A (main door)", "Reader B (east side gate)"],
        "hygiene_station": HYGIENE_POS.tolist(),
        "office": OFFICE_POS.tolist(),
        "station_radius": STATION_RADIUS,
        "obstacles": [{"center": c.tolist(), "half": h.tolist(),
                       "label": label}
                      for (c, h), label in zip(OBSTACLES, ("Hedge planter", "Queue barriers"))],
        "max_attempts_per_scanner": MAX_ATTEMPTS,
        "bell_step": BELL_STEP,
        "max_steps": MAX_STEPS,
        "actions": ACTION_NAMES,
    }


def manifest() -> dict:
    """The whole environment contract in one JSON document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "env_id": ENV_ID,
        "mission": "Biometric school check-in: a pupil must register attendance at a "
                   "fingerprint reader before the late bell.",
        "observation_space": {
            "type": "Box", "shape": [29], "dtype": "float32", "low": -1.0, "high": 1.0,
            "features": observation_schema(),
        },
        "action_space": {"type": "Discrete", "n": len(ACTION_NAMES),
                         "actions": action_schema()},
        "rewards": reward_schema(),
        "dynamics": dynamics_schema(),
        "episode": {"max_steps": MAX_STEPS, "bell_step": BELL_STEP,
                    "termination": ["biometric check-in", "manual sign-in at reception",
                                    "stranded (every route exhausted)",
                                    "timeout at max_steps"]},
        "layout": layout(),
        "api": {
            "base_url": "http://127.0.0.1:8000",
            "endpoints": [
                {"method": "GET", "path": "/manifest", "returns": "this document"},
                {"method": "GET", "path": "/layout", "returns": "scene geometry only"},
                {"method": "POST", "path": "/session", "returns": "session_id + initial state"},
                {"method": "GET", "path": "/session/{id}", "returns": "current state"},
                {"method": "POST", "path": "/session/{id}/step",
                 "body": {"action": "int 0..8"}, "returns": "state after the action"},
                {"method": "POST", "path": "/session/{id}/act",
                 "returns": "state after the trained agent picks one action"},
                {"method": "DELETE", "path": "/session/{id}", "returns": "ends the session"},
            ],
        },
    }


def state_dict(env: gym.Env, obs=None, reward=None, terminated=False,
               truncated=False) -> dict:
    """Serialize the live environment state to JSON safe primitives.

    The same shape is used by the live API and by the recorded episodes in `api/export.py`,
    so a frontend renders both with one code path.
    """
    u = env.unwrapped
    info = u._get_info()
    state = {
        "position": [round(float(v), 3) for v in u.pos],
        "objective": [round(float(v), 3) for v in info["objective"]],
        "distance_to_objective": info["distance"],
        "cleanliness": info["cleanliness"],
        "attempts_per_scanner": info["attempts_per_scanner"],
        "locked": info["locked"],
        "queue": info["queue"],
        "at_scanner_idx": info["at_scanner_idx"],
        "at_hygiene": info["at_hygiene"],
        "at_office": info["at_office"],
        "manual_available": info["manual_available"],
        # null until the pupil has been close enough to read reader B's display
        "scanner_b_broken": info["scanner_b_broken"],
        "help_eta": info["help_eta"],
        "tardy": info["tardy"],
        "checked_in": info["checked_in"],
        "checkin_mode": info["checkin_mode"],
        "stranded": info["stranded"],
        "step": int(u.steps),
        "steps_left": int(u.max_steps - u.steps),
        "steps_to_bell": int(u.bell_step - u.steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if obs is not None:
        # the raw policy input, so a client can run its own model on the same vector
        state["observation"] = [round(float(v), 4) for v in obs]
    if reward is not None:
        state["reward"] = round(float(reward), 3)
    return state
