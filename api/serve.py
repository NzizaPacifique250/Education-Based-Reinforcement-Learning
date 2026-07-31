"""FastAPI service exposing SchoolCheckIn-RL as a JSON API for a web/mobile frontend.

The environment state is serialized to JSON so a frontend can render the entrance area and
drive the agent. Run with:  uv run uvicorn api.serve:app --reload

Endpoints:
  GET  /manifest                -> full environment contract (observation, actions, rewards)
  GET  /layout                  -> static scene geometry (for rendering)
  POST /session                 -> create a session, return initial state
  GET  /session/{sid}           -> current state
  POST /session/{sid}/step      -> apply an action, return next state
  POST /session/{sid}/act       -> let the best trained agent choose+apply one action
  DELETE /session/{sid}         -> drop the session
  GET  /app                     -> the bundled demo frontend (web/index.html)

The interactive OpenAPI schema is at /docs, and the machine-readable one at /openapi.json,
so a mobile client can generate its own typed models from this service.
"""

from __future__ import annotations

import os
import uuid
import numpy as np
import gymnasium as gym
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import environment  # noqa: F401
from api.schema import SCHEMA_VERSION, layout as layout_doc, manifest as manifest_doc, state_dict
from environment.custom_env import ACTION_NAMES
from training.common import ENV_ID

app = FastAPI(title="SchoolCheckIn-RL API", version=SCHEMA_VERSION,
              description="JSON interface to the SchoolCheckIn-v0 environment and the best "
                          "trained agent, for web and mobile clients.")

# A browser frontend served from anywhere (Vite dev server, Expo web, a static host) has to
# be able to call this service, so allow cross-origin reads.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MAX_SESSIONS = 64          # simple guard: this is a demo service, not a multi-tenant one

_sessions: dict[str, gym.Env] = {}
_last_obs: dict[str, np.ndarray] = {}
_order: list[str] = []     # creation order, for evicting the oldest session
_predict = None            # lazily-loaded best-agent policy
_agent_id: str | None = None


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=len(ACTION_NAMES) - 1,
                        description="Action id, 0..8. See /manifest for the names.")


class ResetRequest(BaseModel):
    """Optional overrides, mirroring env.reset(options=...) -- handy for a demo UI."""
    seed: int | None = None
    cleanliness: float | None = Field(None, ge=0.0, le=1.0)
    scan_reliability: float | None = Field(None, ge=0.0, le=1.0)
    scanner_b_broken: bool | None = None
    rush: float | None = Field(None, ge=0.0, le=1.0)

    def options(self) -> dict | None:
        opts = {k: v for k, v in self.model_dump(exclude={"seed"}).items() if v is not None}
        return opts or None


@app.get("/health")
def health():
    return {"status": "ok", "env_id": ENV_ID, "schema_version": SCHEMA_VERSION,
            "sessions": len(_sessions), "agent_loaded": _agent_id}


@app.get("/manifest")
def manifest():
    """Observation layout, action list, reward table and geometry, in one document."""
    return manifest_doc()


@app.get("/layout")
def layout():
    return layout_doc()


@app.post("/session")
def create_session(req: ResetRequest | None = None):
    req = req or ResetRequest()
    while len(_order) >= MAX_SESSIONS:
        _drop(_order[0])
    sid = uuid.uuid4().hex[:8]
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=req.seed, options=req.options())
    _sessions[sid] = env
    _last_obs[sid] = obs
    _order.append(sid)
    return {"session_id": sid, "state": state_dict(env, obs)}


@app.get("/session/{sid}")
def get_session(sid: str):
    if sid not in _sessions:
        raise HTTPException(404, "unknown session")
    return {"session_id": sid, "state": state_dict(_sessions[sid], _last_obs[sid])}


@app.delete("/session/{sid}")
def end_session(sid: str):
    if sid not in _sessions:
        raise HTTPException(404, "unknown session")
    _drop(sid)
    return {"session_id": sid, "closed": True}


def _drop(sid: str):
    env = _sessions.pop(sid, None)
    if env is not None:
        env.close()
    _last_obs.pop(sid, None)
    if sid in _order:
        _order.remove(sid)


def _require_live(sid: str):
    """Fetch a session, refusing to drive one whose episode has already finished."""
    if sid not in _sessions:
        raise HTTPException(404, "unknown session")
    env = _sessions[sid]
    if env.unwrapped.episode_over:
        raise HTTPException(409, "episode is over; create a new session")
    return env


@app.post("/session/{sid}/step")
def step(sid: str, req: StepRequest):
    if req.action not in ACTION_NAMES:
        raise HTTPException(400, f"action must be in [0, {len(ACTION_NAMES) - 1}]")
    env = _require_live(sid)
    obs, r, term, trunc, _ = env.step(req.action)
    _last_obs[sid] = obs
    return {"session_id": sid, "action": ACTION_NAMES[req.action], "action_id": req.action,
            "source": "client", "state": state_dict(env, obs, r, term, trunc)}


@app.post("/session/{sid}/act")
def agent_act(sid: str):
    """Let the best trained agent pick and apply an action (JSON-driven inference)."""
    global _predict, _agent_id
    env = _require_live(sid)
    if _predict is None:
        from play import _best_from_sweeps, _load
        found = _best_from_sweeps()
        if found is None:
            raise HTTPException(503, "no trained agent available; run training first")
        _predict = _load(found[0], found[1])
        _agent_id = f"{found[0]}/{found[1]}"
    action = int(_predict(_last_obs[sid]))
    obs, r, term, trunc, _ = env.step(action)
    _last_obs[sid] = obs
    return {"session_id": sid, "action": ACTION_NAMES[action], "action_id": action,
            "source": "agent", "agent": _agent_id,
            "state": state_dict(env, obs, r, term, trunc)}


# --- bundled demo frontend ---------------------------------------------------------------
_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if os.path.isdir(_WEB):
    app.mount("/app", StaticFiles(directory=_WEB, html=True), name="app")

    @app.get("/", include_in_schema=False)
    def _root():
        return RedirectResponse("/app/")
