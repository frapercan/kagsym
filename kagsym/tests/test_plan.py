"""The plan is executed literally: what it says is what the executor does.

Each test reproduces one of the leaks found on 2026-09-25 while wiring the
plan interface (docs/experiments/EXP-007-plan-space.md) and asserts the
executor no longer has it.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from kagsym import evaluate as E, spec  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.plan import Plan, active, get_plan, plan_macro, play_plan  # noqa: E402
from kagsym.symbolic.executor import Agent  # noqa: E402

DAYS, HOURS = 8, 24


def _trace(plan: Plan, seed: int = 7101):
    """Per-day (at the last hour) hands, planted tiles, seeds held, money."""
    steps = HOURS * DAYS
    spec.set_turns_per_day(HOURS)
    spec.set_episode_steps(steps)
    rows = {}
    with active(plan):
        env = FastEnv(configuration={"episodeSteps": steps, "turnsPerDay": HOURS, "startingMoney": 3000}, seed=seed)
        obs = env.reset()
        ag = Agent(episode_steps=steps, macro=plan_macro(plan))
        while not env.done:
            ob = obs[0]
            a = ag(ob)
            if ob["hour"] == HOURS - 1:
                f = ob["farms"][0]
                planted = sum(1 for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "PLANT")
                rows[int(ob["day"])] = dict(hands=len(f["hands"]), planted=planted,
                                           seeds=sum(int(v) for v in ob["private"]["seeds"].values()),
                                           quadrants=len(f["unlocked_quadrants"]), money=float(f["money"]))
            obs, _ = env.step([a, dict(E.PASS_ACTION)])
        return rows, float(env.rewards()[0])


def test_plan_is_scoped():
    assert get_plan() is None
    with active(Plan()):
        assert get_plan() is not None
    assert get_plan() is None


def test_schedule_fields_index_by_day_and_repeat_the_last_value():
    p = Plan(hands=(2, 0, 5), tiles=25, crop=("WHEAT", "CARROT"))
    assert [p.hands_on(d) for d in (0, 1, 2, 3, 9)] == [2, 0, 5, 5, 5]
    assert p.tiles_on(7) == 25
    assert p.crop_on(0) == "WHEAT" and p.crop_on(5) == "CARROT"


def test_hands_follow_the_plan_including_the_last_days():
    # MIN_HAND_DAYS used to refuse every hire on the last day; the plan's
    # last-day hands are what harvest and sell the final cycle.
    rows, _ = _trace(Plan("CARROT", 25, (2, 0, 4, 4, 4, 4, 4, 6), 0))
    assert rows[0]["hands"] == 2
    assert rows[1]["hands"] == 0
    assert rows[2]["hands"] == 4
    assert rows[6]["hands"] == 4


def test_tiles_and_land_follow_the_plan():
    rows, _ = _trace(Plan("CARROT", 25, 6, 0))
    assert rows[0]["quadrants"] == 1
    assert 20 <= rows[1]["planted"] <= 25
    rows, _ = _trace(Plan("WHEAT", 50, 7, 1))
    assert rows[0]["quadrants"] == 2
    assert rows[1]["planted"] == 50


def test_no_seed_is_bought_for_a_cycle_that_cannot_finish():
    # `target_crop` was one day looser than `plantable`: 50 wheat seeds
    # ($500) bought on day 4 of 8 and never planted.
    rows, _ = _trace(Plan("WHEAT", 50, 7, 1))
    assert rows[5]["seeds"] <= 1 and rows[6]["seeds"] <= 1


def test_last_day_harvest_is_sold_not_carried():
    # Carrot sown on day 4 ripens on day 7, the last day. Without DROP the
    # units carried it home at the close and the score ignored it: the final
    # cash equalled day 6's.
    rows, final = _trace(Plan(("WHEAT",) * 4 + ("CARROT",) * 4, 50, 7, 1))
    assert final > rows[6]["money"] + 500


def test_a_plan_beats_the_dial_policy_bar():
    # The bar the plan space had to clear (docs/experiments/EXP-007): PPO's
    # best point in this world is 5,776 on seed 7101.
    best = Plan(("CARROT", "WHEAT", "CARROT", "CARROT", "WHEAT", "CARROT", "CARROT", "CARROT"),
                (20, 25, 25, 25, 25, 25, 25, 25), (2, 0, 2, 7, 1, 2, 3, 0), 0, 0,
                (0.05, 0.05, 0.05, 0.05, 0.5, 0.05, 0.05, 0.05))
    assert play_plan(best, 7101) > 5776
