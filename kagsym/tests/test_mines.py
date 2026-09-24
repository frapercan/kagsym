"""Minesweeper: every silent failure found so far has a test that trips on it.

In this project failures do not crash, they sign: a NaN in a gate, an
opponent that becomes PASS, a global file that redefines the search space, a
vector read against the wrong dial map. Each test here reproduces the shape
of one of those and asserts that the code now refuses instead of signing.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from kagsym import evaluate as E  # noqa: E402
from kagsym.evaluate import Episode  # noqa: E402
from kagsym.policy import Offset  # noqa: E402


# -- offsets never travel without their dial map ------------------------------

def test_offset_length_must_be_exact():
    Offset(np.zeros(3), [1, 2, 3])
    Offset(np.zeros(6), [1, 2, 3], ramp=True)
    with pytest.raises(ValueError):
        Offset(np.zeros(4), [1, 2, 3])            # neither n nor 2n
    with pytest.raises(ValueError):
        Offset(np.zeros(6), [1, 2, 3], ramp=False)  # 2n declared constant
    with pytest.raises(ValueError):
        Offset(np.zeros(3), [1, 1, 3])            # repeated dial


def test_offset_file_carries_map_and_digest(tmp_path):
    off = Offset(np.array([0.1, -0.2, 0.3, 0.4]), [5, 9], ramp=True)
    path = str(tmp_path / "offset.npz")
    off.save(path, "abc123", generation=7)
    back, digest, meta = Offset.load(path)
    assert digest == "abc123" and meta["generation"] == 7
    assert back.live == [5, 9] and back.ramp is True
    assert np.allclose(back.delta, off.delta)


def test_legacy_checkpoint_field_is_read():
    off = Offset.from_checkpoint({"delta_rampa": {"delta": [1, 2, 3, 4], "vivos": [0, 1]}})
    assert off.ramp and off.live == [0, 1]
    assert Offset.from_checkpoint({}) is None


# -- the bake gate refuses NaN, missing boards and blind instruments ----------

def test_bake_gate_refuses_nan_and_degenerate():
    from validate_offset import bake_allowed
    ok, why = bake_allowed({"t": float("nan"), "n": 10, "degenerate": False}, None, 10, 0)
    assert not ok and "finite" in why
    ok, why = bake_allowed({"t": 3.0, "n": 10, "degenerate": True}, None, 10, 0)
    assert not ok and "same policy" in why
    ok, why = bake_allowed({"t": 3.0, "n": 8, "degenerate": False}, None, 10, 0)
    assert not ok and "8 of 10" in why
    ok, why = bake_allowed({"t": 1.5, "n": 10, "degenerate": False}, None, 10, 0)
    assert not ok
    ok, _ = bake_allowed({"t": 3.0, "n": 10, "degenerate": False},
                         {"n": 20, "win_diff": 0.01}, 10, 20)
    assert ok
    ok, why = bake_allowed({"t": 3.0, "n": 10, "degenerate": False},
                           {"n": 20, "win_diff": -0.01}, 10, 20)
    assert not ok and "win" in why


def _ep(opp, seed, seat, money, opp_money, error=None):
    win = 1.0 if money > opp_money else (0.5 if money == opp_money else 0.0)
    return Episode(opp, seed, seat, money, opp_money, win, 0, error)


def test_paired_reports_dropped_boards_and_degeneracy():
    a = [_ep("x", s, 0, 100 + s, 50) for s in range(5)]
    b = [_ep("x", s, 0, 100 + s, 50) for s in range(5)]
    d = E.paired(a, b)
    assert d["degenerate"] and not np.isfinite(d["t"]) and d["n"] == d["n_expected"] == 5
    a[2] = _ep("x", 2, 0, float("nan"), float("nan"), error="ours: boom")
    d = E.paired(a, b)
    assert d["n"] == 4 and d["n_expected"] == 5


def test_search_score_disqualifies_failed_candidates():
    from search import score
    good = [_ep("x", s, 0, 100.0, 40.0) for s in range(4)]
    bad = good[:3] + [_ep("x", 3, 0, float("nan"), float("nan"), error="ours: boom")]
    assert np.isfinite(score(good, "margin")) and score(good, "margin") == 60.0
    assert score(bad, "margin") == -np.inf
    assert score([], "money") == -np.inf


# -- a dial list belongs to one checkpoint --------------------------------------

def test_live_dials_refuse_another_checkpoint(tmp_path):
    from live_dials import load_dials
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"weights-A")
    dials = tmp_path / "model.pt.dials.json"
    dials.write_text(json.dumps({"checkpoint_digest": "not-this-one", "live": [3, 1, 2]}))
    with pytest.raises(SystemExit):
        load_dials(str(ckpt))
    dials.write_text(json.dumps({"checkpoint_digest": E.file_digest(str(ckpt)), "live": [3, 1, 2]}))
    assert load_dials(str(ckpt)) == [1, 2, 3]       # sorted by index, never by effect
    with pytest.raises(SystemExit):
        load_dials(str(tmp_path / "other.pt"))       # no list at all


# -- the summary excludes what did not really play -----------------------------

def test_summary_excludes_broken_and_unloaded_opponents():
    turns = E.HOURS * E.DAYS
    eps = [Episode("real", 1, 0, 50.0, 60.0, 0.0, 0), Episode("real", 1, 1, 70.0, 60.0, 1.0, 0),
           Episode("broken", 1, 0, 90.0, 3000.0, 0.0, turns - 1),
           Episode("broken", 1, 1, 90.0, 3000.0, 0.0, turns - 1),
           Episode("missing", 1, 0, float("nan"), float("nan"), float("nan"), 0,
                   error="opponent-load: no module")]
    s = E.summary(eps)
    assert s["opponents"] == 1 and s["broken_opponents"] == ["broken"]
    assert s["opponents_not_loaded"] == ["missing"] and s["failed"] == 0
    assert s["win_mean"] == 0.5


# -- provenance --------------------------------------------------------------------

def test_fingerprint_refuses_missing_files(monkeypatch):
    from kagsym import version as V
    monkeypatch.setattr(V, "GAME_FILES", ["kagsym/does_not_exist.py"])
    with pytest.raises(FileNotFoundError):
        V.fingerprint()


def test_game_env_lists_only_set_variables(monkeypatch):
    from kagsym import version as V
    for k in V.GAME_ENV:
        monkeypatch.delenv(k, raising=False)
    assert V.game_env() == {}
    monkeypatch.setenv("KAG_VALOR", "heuristica")
    assert V.game_env() == {"KAG_VALOR": "heuristica"}
    prov = V.provenance()
    assert {"commit", "dirty", "game_fingerprint", "model_fingerprint", "game_env", "pid"} <= set(prov)


def test_ledger_entry_carries_provenance(tmp_path, monkeypatch):
    eps = [_ep("v48-fast-routes", 7101, 0, 10.0, 20.0)]
    monkeypatch.setenv("KAGSYM_MLFLOW", "0")
    spec = E.PolicySpec(os.path.join(ROOT, "README.md"))    # any file: only its digest is read
    entry = E.record("test", spec, [E.V48], [7101], eps, path=str(tmp_path / "ledger.jsonl"))
    assert entry["seed_families"] == ["reserved"]
    assert entry["game_fingerprint"] and "dirty" in entry and "game_env" in entry


# -- reproducible draws ----------------------------------------------------------------

def test_network_init_is_seeded():
    import torch
    from kagsym.nets.world import E2EAgent, WorldConfig
    cfg = WorldConfig(device="cpu")
    torch.manual_seed(7)
    a = E2EAgent(cfg).state_dict()
    torch.manual_seed(7)
    b = E2EAgent(cfg).state_dict()
    torch.manual_seed(8)
    c = E2EAgent(cfg).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_tracker_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("KAGSYM_MLFLOW", "0")
    from kagsym.tracking import Tracker, context_tags
    tags = context_tags("evaluation", checkpoint=os.path.join(ROOT, "README.md"))
    assert tags["kind"] == "evaluation" and tags["checkpoint_digest"]
    with Tracker("evaluation", "noop", tags=tags) as t:
        assert not t.enabled
        t.log({"band/win_mean": 0.1})
    with pytest.raises(ValueError):
        Tracker("bogus", "x")


# -- the opponent ladder is complete and loads ---------------------------------

def test_ladder_has_one_cap_per_rung_and_every_agent_loads():
    from kagsym.environment import LADDER, LADDER_CAPS, load_public
    assert len(LADDER) == len(LADDER_CAPS)
    if not os.path.isdir(os.path.join(ROOT, "agents_pub")):
        pytest.skip("agents_pub/ not downloaded")
    for name in {n for n in LADDER if n}:
        assert callable(load_public(name)), name


def test_missing_opponent_raises_instead_of_pass():
    from kagsym.environment import load_public
    with pytest.raises(Exception):
        load_public("this-agent-does-not-exist")


# -- no undefined names anywhere: a NameError in a rarely taken branch signs --

def test_no_undefined_names():
    """pyflakes F821 over the package and the tools. A NameError inside a
    `try: ... except Exception: pass` once switched MLflow off for two runs
    without a word; an undefined name is a mine whatever branch it is in."""
    pyflakes = pytest.importorskip("pyflakes.api")
    from pyflakes.reporter import Reporter
    import io
    out, err = io.StringIO(), io.StringIO()
    n = 0
    for sub in ("kagsym", "tools", "submit_kagsym"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, sub)):
            for f in files:
                if f.endswith(".py"):
                    n += pyflakes.checkPath(os.path.join(dirpath, f), Reporter(out, err))
    undefined = [l for l in out.getvalue().splitlines() if "undefined name" in l]
    assert not undefined, "\n".join(undefined)
