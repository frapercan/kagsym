"""FastEnv must be indistinguishable from the real engine, turn by turn.

If this fails, everything built on top plays against the wrong dynamics.
Both engines are driven by our own executor (deterministic) on one seat and a
seeded random walker on the other, so every action is reproducible.
"""
from __future__ import annotations

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kagsym.fastenv import FastEnv, _fast_copy  # noqa: E402

kaggle_environments = pytest.importorskip("kaggle_environments")
from kaggle_environments import make  # noqa: E402


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def _snapshot(obs_list):
    o0 = obs_list[0]
    return _plain({
        "day": o0["day"], "hour": o0["hour"], "step": o0["step"],
        "farms": o0["farms"], "market": o0["market"], "town": o0["town"],
        "private": [o["private"] for o in obs_list],
    })


def _walker(seed: int):
    rng = random.Random(seed)
    moves = ["NORTH", "SOUTH", "EAST", "WEST", "PASS"]

    def act(obs):
        return {"farmer": [rng.choice(moves)], "hands": [], "market": []}
    return act


def _executor(steps: int):
    from kagsym.symbolic.executor import Agent
    ag = Agent(episode_steps=steps)
    return lambda obs: ag(obs)


@pytest.mark.parametrize("seed", [1234, 99, 20260919])
def test_equivalence(seed, n_steps=240):
    cfg = {"episodeSteps": n_steps + 2, "seed": seed}
    real = make("kaggriculture", configuration=cfg, debug=False)
    real.reset()
    real_obs = [s.observation for s in real.state]
    fast = FastEnv(configuration={"episodeSteps": n_steps + 2}, seed=seed)
    fast_obs = fast.reset()
    assert real.info.get("seed") == fast.info["seed"]
    assert _snapshot(real_obs) == _snapshot(fast_obs), "initial state differs"

    pol_real = [_executor(n_steps + 2), _walker(11)]
    pol_fast = [_executor(n_steps + 2), _walker(11)]
    for t in range(n_steps):
        acts_real = [pol_real[i](real_obs[i]) for i in range(2)]
        acts_fast = [pol_fast[i](fast_obs[i]) for i in range(2)]
        assert _plain(acts_real) == _plain(acts_fast), f"actions diverge at t={t}"
        real.step([_fast_copy(x) for x in acts_real])
        real_obs = [s.observation for s in real.state]
        fast_obs, _ = fast.step([_fast_copy(x) for x in acts_fast])
        a, b = _snapshot(real_obs), _snapshot(fast_obs)
        if a != b:
            bad = [k for k in a if a[k] != b[k]]
            raise AssertionError(f"state diverges at t={t} in {bad}: "
                                 f"{json.dumps(a[bad[0]])[:300]} vs {json.dumps(b[bad[0]])[:300]}")
