"""The deployed policy, the evaluator and the submission wrapper agree.

These tests need a checkpoint. They use `KAGSYM_TEST_CKPT` or the deployed
`submit_kagsym/model.pt`; without either they are skipped.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.policy import Offset  # noqa: E402


def _ckpt():
    for p in (os.environ.get("KAGSYM_TEST_CKPT"),
              os.path.join(ROOT, "submit_kagsym", "model.pt"),
              os.path.join(ROOT, "submit_kagsym", "modelo.pt")):
        if p and os.path.exists(p):
            return p
    pytest.skip("no checkpoint available")


def test_offset_constant_and_ramp():
    live = [1, 4]
    c = Offset(np.array([0.5, -0.5]), live)
    assert not c.is_ramp
    v = c.vector(0.7, 6)
    assert v.tolist() == [0, 0.5, 0, 0, -0.5, 0]
    r = Offset(np.array([0.5, -0.5, 1.0, 2.0]), live)
    assert r.is_ramp
    v = r.vector(0.5, 6)
    assert np.allclose(v, [0, 1.0, 0, 0, 0.5, 0])
    assert np.allclose(r.vector(0.0, 6), c.vector(0.0, 6))


def test_episode_is_deterministic():
    spec = E.PolicySpec(_ckpt())
    seed = S.RESERVED.seeds(1)[0]
    a = E.play(spec, E.V48, seed, 0)
    b = E.play(spec, E.V48, seed, 0)
    assert a.money == b.money and a.opp_money == b.opp_money


def test_mirror_null_case():
    """A policy against itself: seat 0's money equals the opponent's on seat 1.

    This is the null case that once read 1.000 instead of 0.5, when the
    self-play copy crashed and played PASS.
    """
    path = _ckpt()
    spec = E.PolicySpec(path)
    seed = S.RESERVED.seeds(1)[0]
    a = E.play(spec, E.checkpoint(path), seed, 0)
    b = E.play(spec, E.checkpoint(path), seed, 1)
    assert a.opp_failures == 0 and b.opp_failures == 0
    assert a.money == b.opp_money and a.opp_money == b.money
    assert a.win + b.win == 1.0


def test_submission_matches_evaluator():
    path = _ckpt()
    os.environ["KAGSYM_CKPT"] = os.path.abspath(path)
    import check_submission as C
    agent_fn = C.load_like_kaggle()
    spec = E.PolicySpec(os.path.abspath(path))
    for k, seed in enumerate(S.RESERVED.seeds(2)):
        sub = C.play_submission(agent_fn, E.V48, seed, k % 2)
        ev = E.play(spec, E.V48, seed, k % 2).money
        assert abs(sub - ev) < 1e-6, (seed, sub, ev)
