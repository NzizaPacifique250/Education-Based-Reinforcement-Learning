"""FastAPI service exposing EduPath-RL as a JSON API for a web/mobile frontend.

The environment state is serialized to JSON so a frontend can render the skill-tree and
drive the agent. Run with:  uv run uvicorn api.serve:app --reload

Endpoints:
  POST /session                 -> create a session, return initial state
  POST /session/{sid}/step      -> apply an action, return next state
  POST /session/{sid}/act       -> let the best trained agent choose+apply one action
  GET  /session/{sid}           -> current state
  GET  /curriculum              -> static prerequisite graph (for rendering)
"""

from __future__ import annotations

import uuid
import numpy as np
import gymnasium as gym
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import environment  # noqa: F401
from environment.custom_env import (ACTION_NAMES, PREREQS, N_CONCEPTS, TARGET_CONCEPT,
                                     MASTERY_THRESHOLD)
from training.common import ENV_ID

app = FastAPI(title="EduPath-RL API", version="0.1.0")

_sessions: dict[str, gym.Env] = {}
_last_obs: dict[str, np.ndarray] = {}
_predict = None  # lazily-loaded best-agent policy


class StepRequest(BaseModel):
    action: int


def _state(env: gym.Env, obs, reward=None, terminated=False, truncated=False) -> dict:
    info = env.unwrapped._get_info()
    state = {
        "mastery": [round(float(m), 3) for m in env.unwrapped.mastery],
        "attention": round(float(env.unwrapped.attention), 3),
        "current_concept": int(env.unwrapped.current),
        "step": int(env.unwrapped.steps),
        "mean_mastery": round(info["mean_mastery"], 3),
        "target_mastered": info["target_mastered"],
        "mastered": info["mastered"],
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if reward is not None:
        state["reward"] = round(float(reward), 3)
    return state


@app.get("/curriculum")
def curriculum():
    return {
        "n_concepts": N_CONCEPTS,
        "target": TARGET_CONCEPT,
        "mastery_threshold": MASTERY_THRESHOLD,
        "prerequisites": {str(k): v for k, v in PREREQS.items()},
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
    if not (0 <= req.action < 8):
        raise HTTPException(400, "action must be in [0, 7]")
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
