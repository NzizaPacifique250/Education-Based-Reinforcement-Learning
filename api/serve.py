"""FastAPI service exposing SchoolCheckIn-RL as a JSON API for a web/mobile frontend.

The environment state is serialized to JSON so a frontend can render the entrance area and
drive the agent. Run with:  uv run uvicorn api.serve:app --reload

Endpoints:
  POST /session                 -> create a session, return initial state
  POST /session/{sid}/step      -> apply an action, return next state
  POST /session/{sid}/act       -> let the best trained agent choose+apply one action
  GET  /session/{sid}           -> current state
  GET  /layout                  -> static scene geometry (for rendering)
"""

from __future__ import annotations

import uuid
import gymnasium as gym
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import environment  # noqa: F401
from environment.custom_env import (ACTION_NAMES, ROOM_SIZE, START_POS, SCANNER_A_POS,
                                     SCANNER_B_POS, HYGIENE_POS, OFFICE_POS, OBSTACLES,
                                     STATION_RADIUS, MAX_ATTEMPTS, BELL_STEP)
from training.common import ENV_ID

app = FastAPI(title="SchoolCheckIn-RL API", version="0.1.0")

_sessions: dict[str, gym.Env] = {}
_last_obs: dict[str, np.ndarray] = {}
_predict = None  # lazily-loaded best-agent policy


class StepRequest(BaseModel):
    action: int


def _state(env: gym.Env, obs, reward=None, terminated=False, truncated=False) -> dict:
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
        # null until the student has been close enough to read scanner B's display
        "scanner_b_broken": info["scanner_b_broken"],
        "help_eta": info["help_eta"],
        "tardy": info["tardy"],
        "checked_in": info["checked_in"],
        "checkin_mode": info["checkin_mode"],
        "stranded": info["stranded"],
        "step": int(u.steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if reward is not None:
        state["reward"] = round(float(reward), 3)
    return state


@app.get("/layout")
def layout():
    return {
        "room_size": ROOM_SIZE,
        "start": START_POS.tolist(),
        "scanners": [SCANNER_A_POS.tolist(), SCANNER_B_POS.tolist()],
        "hygiene_station": HYGIENE_POS.tolist(),
        "office": OFFICE_POS.tolist(),
        "station_radius": STATION_RADIUS,
        "obstacles": [{"center": c.tolist(), "half": h.tolist()} for c, h in OBSTACLES],
        "max_attempts_per_scanner": MAX_ATTEMPTS,
        "bell_step": BELL_STEP,
        "actions": ACTION_NAMES,
    }


@app.post("/session")
def create_session():
    sid = uuid.uuid4().hex[:8]
    env = gym.make(ENV_ID)
    obs, _ = env.reset()
    _sessions[sid] = env
    _last_obs[sid] = obs
    return {"session_id": sid, "state": _state(env, obs)}


@app.get("/session/{sid}")
def get_session(sid: str):
    if sid not in _sessions:
        raise HTTPException(404, "unknown session")
    return {"session_id": sid, "state": _state(_sessions[sid], _last_obs[sid])}


@app.post("/session/{sid}/step")
def step(sid: str, req: StepRequest):
    if sid not in _sessions:
        raise HTTPException(404, "unknown session")
    if req.action not in ACTION_NAMES:
        raise HTTPException(400, f"action must be in [0, {len(ACTION_NAMES) - 1}]")
    env = _sessions[sid]
    obs, r, term, trunc, _ = env.step(req.action)
    _last_obs[sid] = obs
    return {"session_id": sid, "action": ACTION_NAMES[req.action],
            "state": _state(env, obs, r, term, trunc)}


@app.post("/session/{sid}/act")
def agent_act(sid: str):
    """Let the best trained agent pick and apply an action (JSON-driven inference)."""
    global _predict
    if sid not in _sessions:
        raise HTTPException(404, "unknown session")
    if _predict is None:
        from play import _best_from_sweeps, _load
        found = _best_from_sweeps()
        if found is None:
            raise HTTPException(503, "no trained agent available; run training first")
        _predict = _load(found[0], found[1])
    env = _sessions[sid]
    action = int(_predict(_last_obs[sid]))
    obs, r, term, trunc, _ = env.step(action)
    _last_obs[sid] = obs
    return {"session_id": sid, "action": ACTION_NAMES[action],
            "state": _state(env, obs, r, term, trunc)}
