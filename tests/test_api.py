"""Tests for the JSON contract a web/mobile client consumes.

The point of these is that the serialized schema stays pinned to the environment: if the
observation grows a feature or an action is renamed, the frontend must not keep rendering
a stale document.
"""

import json

import gymnasium as gym
import pytest

import environment  # noqa: F401  (registers SchoolCheckIn-v0)
from api.schema import manifest, observation_schema, state_dict
from api.export import record_episode
from environment.custom_env import ACTION_NAMES
from training.common import ENV_ID


@pytest.fixture
def env():
    e = gym.make(ENV_ID)
    yield e
    e.close()


def test_observation_schema_covers_the_whole_vector(env):
    total = sum(g["size"] for g in observation_schema())
    assert total == env.observation_space.shape[0] == 29
    # index ranges must be contiguous, so a client can slice the vector by feature name
    edges = [g["index"] for g in observation_schema()]
    assert edges[0][0] == 0 and edges[-1][1] == 29
    assert all(a[1] == b[0] for a, b in zip(edges, edges[1:]))


def test_manifest_matches_the_spaces(env):
    m = manifest()
    assert m["env_id"] == ENV_ID
    assert m["action_space"]["n"] == env.action_space.n
    assert [a["name"] for a in m["action_space"]["actions"]] == list(ACTION_NAMES.values())
    assert m["observation_space"]["shape"] == list(env.observation_space.shape)


def test_manifest_is_json_serializable():
    """numpy scalars serialize fine in Python but not in JSON, so this is worth pinning."""
    round_tripped = json.loads(json.dumps(manifest()))
    assert round_tripped["layout"]["scanners"][0] == [9.0, 9.0]


def test_state_dict_hides_scanner_b_health_until_inspected(env):
    obs, _ = env.reset(seed=3)
    state = state_dict(env, obs)
    assert state["scanner_b_broken"] is None          # partial observability preserved
    assert len(state["observation"]) == 29
    assert state["step"] == 0 and state["terminated"] is False


def test_recorded_episode_is_complete_and_serializable():
    ep = record_episode(seed=5000, policy="scripted")
    assert ep["policy"]["kind"] == "scripted_reference"
    frames = ep["frames"]
    assert frames[0]["t"] == 0 and frames[0]["action"] is None
    assert [f["t"] for f in frames] == list(range(len(frames)))
    last = frames[-1]["state"]
    assert last["terminated"] or last["truncated"]
    assert ep["summary"]["steps"] == len(frames) - 1
    json.dumps(ep)                                     # must survive the API boundary


def test_reference_controller_checks_in_biometrically():
    """The bundled demo episode should show the intended route, not the fallback."""
    ep = record_episode(seed=5000, policy="scripted")
    assert ep["summary"]["checkin_mode"] == "biometric"
    assert not ep["summary"]["tardy"]
