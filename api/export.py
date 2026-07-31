"""Serialize the environment contract and a recorded episode to JSON for a frontend.

Two artifacts are written into `web/`, which is what the bundled demo page consumes:

  manifest.json   the environment contract from `api.schema`: observation layout, action
                  list, reward table, dynamics and scene geometry
  episode.json    one full episode, frame by frame: the action taken, the reward, and the
                  same state object the live API returns, plus the raw 29-d observation

The episode is recorded from the best trained agent when saved models are present. When
they are not (a fresh clone, or a phone/browser demo with no Python backend), it falls back
to a scripted reference controller so the frontend still has something real to replay. The
policy that produced the file is recorded inside it, so the two are never confused.

    uv run python -m api.export                 # -> web/manifest.json, web/episode.json
    uv run python -m api.export --seed 7 --policy scripted
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque

import numpy as np
import gymnasium as gym

import environment  # noqa: F401
from api.schema import manifest, state_dict
from environment.custom_env import (ACTION_NAMES, A_SANITIZE, A_SCAN, A_WAIT, A_MANUAL,
                                    HYGIENE_POS, OFFICE_POS, CLEAN_TARGET, STATION_RADIUS,
                                    _MOVES)
from training.common import ENV_ID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")

RETRY_FLOOR = 0.55     # cleanliness below which the reference controller re-sanitizes
SANITIZE_FULL = 0.95   # the env stops paying out for a dose above this, so top up to it


# --- scripted reference controller -------------------------------------------------------
def _plan_move(env, target: np.ndarray) -> int:
    """First action of a shortest walk to `target`, by breadth first search.

    Movement is locked to a 0.5 m lattice and the planter and barriers form a wall across
    the courtyard, so a greedy "step towards it" rule walks into the obstacles. A search
    over the lattice picks the corridor instead.
    """
    u = env.unwrapped
    start = tuple(np.round(u.pos, 3))
    seen = {start}
    queue = deque([(np.asarray(u.pos, dtype=float), None)])
    while queue:
        pos, first = queue.popleft()
        if float(np.linalg.norm(pos - target)) <= STATION_RADIUS and first is not None:
            return first
        for action, delta in _MOVES.items():
            nxt = pos + delta
            key = tuple(np.round(nxt, 3))
            if key in seen or u._blocked(nxt):
                continue
            seen.add(key)
            queue.append((nxt, action if first is None else first))
    return A_WAIT   # boxed in (should not happen on this layout)


def scripted_policy(env) -> int:
    """A hand written controller: sanitize, walk to the nearest usable reader, scan, retry.

    This is not a learned policy. It exists so the exported episode, and therefore the
    frontend demo, works on a machine that has no trained models.
    """
    u = env.unwrapped
    usable = u._usable_scanners()
    if not usable:
        if u._at(OFFICE_POS):
            return A_MANUAL
        return _plan_move(env, OFFICE_POS)
    idx = u._current_scanner()
    if idx is not None and idx in usable and u.cleanliness >= RETRY_FLOOR:
        # already at a working reader: retry here rather than walk the whole detour back
        # for the 0.12 cleanliness a rejection costs
        return A_WAIT if u.queue[idx] > 0 else A_SCAN
    if u._at(HYGIENE_POS) and u.cleanliness < SANITIZE_FULL:
        # top it right up before leaving: walking costs 0.012 a step, so setting out at only
        # just above the scanning threshold means turning round again half way across
        return A_SANITIZE
    if u.cleanliness < CLEAN_TARGET:
        return _plan_move(env, HYGIENE_POS)
    target = min((u.SCANNERS[i] for i in usable), key=lambda s: np.linalg.norm(u.pos - s))
    return _plan_move(env, target)


def _trained_policy():
    """(predict_fn, label) for the best saved agent, or None when nothing is trained yet."""
    try:
        from play import _best_from_sweeps, _load
    except ImportError:
        return None
    found = _best_from_sweeps()
    if found is None:
        return None
    algo, name, ret = found
    try:
        predict = _load(algo, name)
    except (ImportError, FileNotFoundError, OSError):
        # no torch/SB3 in this environment, or the sweep csv lists a model that was not kept
        return None
    return predict, {"kind": "trained", "algo": algo, "model": name,
                     "eval_mean_return": round(float(ret), 2),
                     "note": "best configuration across all four sweeps, greedy actions"}


# --- recording ---------------------------------------------------------------------------
def record_episode(seed: int = 5000, policy: str = "auto") -> dict:
    """Run one episode and return it as a JSON serializable document."""
    chosen = None if policy == "scripted" else _trained_policy()
    if chosen is None:
        predict, meta = scripted_policy, {
            "kind": "scripted_reference",
            "note": "no trained model was available, so a hand written controller was "
                    "recorded; re-run this exporter after training to replace it",
        }
        use_env = True
    else:
        predict, meta = chosen
        use_env = False

    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=seed)
    frames = [{"t": 0, "action_id": None, "action": None, "reward": 0.0,
               "cumulative_reward": 0.0, "state": state_dict(env, obs)}]
    total, done, t = 0.0, False, 0
    while not done:
        action = int(predict(env) if use_env else predict(obs))
        obs, r, term, trunc, _ = env.step(action)
        total += r
        t += 1
        frames.append({
            "t": t, "action_id": action, "action": ACTION_NAMES[action],
            "reward": round(float(r), 3), "cumulative_reward": round(total, 3),
            "state": state_dict(env, obs, r, term, trunc),
        })
        done = term or trunc
    final = frames[-1]["state"]
    env.close()

    return {
        "schema_version": manifest()["schema_version"],
        "env_id": ENV_ID,
        "seed": seed,
        "policy": meta,
        "summary": {
            "steps": t,
            "return": round(total, 2),
            "checked_in": final["checked_in"],
            "checkin_mode": final["checkin_mode"],
            "tardy": final["tardy"],
            "stranded": final["stranded"],
            "rejections_per_reader": final["attempts_per_scanner"],
            "final_cleanliness": final["cleanliness"],
        },
        "frames": frames,
    }


def export(out_dir: str = WEB_DIR, seed: int = 5000, policy: str = "auto") -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, doc in (("manifest.json", manifest()),
                      ("episode.json", record_episode(seed=seed, policy=policy))):
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        written.append(path)
        print(f"[export] wrote {path} ({os.path.getsize(path) / 1024:.1f} kB)")
    return written


def main():
    ap = argparse.ArgumentParser(description="Serialize SchoolCheckIn-RL to JSON.")
    ap.add_argument("--out", default=WEB_DIR, help="output directory (default: web/)")
    ap.add_argument("--seed", type=int, default=5000, help="episode seed to record")
    ap.add_argument("--policy", choices=["auto", "scripted"], default="auto",
                    help="auto uses the best trained agent when one is saved")
    args = ap.parse_args()
    export(args.out, args.seed, args.policy)


if __name__ == "__main__":
    main()
