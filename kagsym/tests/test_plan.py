"""The plan is executed literally: what it says is what the executor does.

Each test reproduces one of the leaks found on 2026-09-25 while wiring the
plan interface (docs/experiments/EXP-007-plan-space.md) and asserts the
executor no longer has it.
"""
from __future__ import annotations

import os
import sys

import pytest

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
    # best point in this world is 5,776 on seed 7101; the per-day dial oracle
    # (EXP-006) 6,537 on 20 seeds. Rung 8 of the ladder found this plan.
    best = Plan(crop=({"CARROT": 0.5, "WHEAT": 0.5}, {"CARROT": 0.5, "WHEAT": 0.5}) + ("CARROT",) * 6,
                tiles=(50, 50, 0, 50, 50, 50, 50, 50), hands=(6, 5, 5, 8, 8, 6, 6, 8), land=1,
                selling=0.05, load=(0, 0, 0, 0, 0, 0, 0, 12), water_last=1)
    assert play_plan(best, 7101) > 6537


# -- rung 1 of the ladder: 4 days, one carrot cycle ----------------------------

def _last_day_ledger(plan: Plan, seed: int = 7101, days: int = 4):
    """Units harvested, sold, carried at the close, left on tiles."""
    steps = HOURS * days
    spec.set_turns_per_day(HOURS)
    spec.set_episode_steps(steps)
    with active(plan):
        env = FastEnv(configuration={"episodeSteps": steps, "turnsPerDay": HOURS, "startingMoney": 3000}, seed=seed)
        obs = env.reset()
        ag = Agent(episode_steps=steps, macro=plan_macro(plan))
        sold = 0
        last = None
        while not env.done:
            ob = obs[0]
            a = ag(ob)
            sold += sum(o[2] for o in a.get("market", []) if o[0] == "SELL")
            last = ob
            obs, _ = env.step([a, dict(E.PASS_ACTION)])
        f = last["farms"][0]
        left = sum(t.get("yield_units", 0) for r in f["tiles"] for t in r
                   if isinstance(t, dict) and t.get("kind") == "PLANT")
        carried = sum(sum(i.values()) for i in last["private"]["inventories"])
        return dict(sold=sold, carried=carried, left=left, final=float(env.rewards()[0]))


def test_nothing_is_carried_home_on_the_last_day():
    # The score is cash: a load still in a unit's hands at the close is lost.
    # Before the deadline rule, 6 hands harvested 75 units and carried 51.
    for hands in (3, 6, 8):
        led = _last_day_ledger(Plan("CARROT", 25, hands, 0, 0, 0.05, 12))
        assert led["carried"] == 0, (hands, led)


def test_rung_one_is_near_the_hand_bound():
    # 25 carrot tiles yield 75 units; sold in one day at the day-0 price curve
    # they gross ~2,130 $, so the bound is ~4,600 $. One DROP column for the
    # whole crew gave 3,348; per-unit columns, the four shed-access tiles, the
    # last-day deadline and a load of 12 give 4,379 on five seeds.
    led = _last_day_ledger(Plan("CARROT", 25, 8, 0, 0, 0.05, 12))
    assert led["sold"] >= 66, led
    assert led["final"] >= 4300, led


def test_a_mix_inside_the_day_buys_and_plants_both_crops():
    plan = Plan({"WHEAT": 0.6, "CARROT": 0.4}, 25, 4, 0, 0, 0.05, 12)
    assert plan.crop_targets(0) == {"WHEAT": 15, "CARROT": 10}
    steps = HOURS * 6
    spec.set_turns_per_day(HOURS)
    spec.set_episode_steps(steps)
    with active(plan):
        env = FastEnv(configuration={"episodeSteps": steps, "turnsPerDay": HOURS, "startingMoney": 3000}, seed=7101)
        obs = env.reset()
        ag = Agent(episode_steps=steps, macro=plan_macro(plan))
        while not (obs[0]["day"] == 1 and obs[0]["hour"] == HOURS - 1):
            obs, _ = env.step([ag(obs[0]), dict(E.PASS_ACTION)])
        by = {}
        for r in obs[0]["farms"][0]["tiles"]:
            for t in r:
                if isinstance(t, dict) and t.get("kind") == "PLANT":
                    by[t["crop"]] = by.get(t["crop"], 0) + 1
    assert by == {"WHEAT": 15, "CARROT": 10}, by


def test_rollout_uses_the_games_calendar_not_the_process_state():
    # A forked worker inherits whatever calendar the parent had when the pool
    # was made; the rollout must set it from the game (measured: every
    # candidate played as a 30-day game, 2,540 against a base of 3,714).
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import plan_daysearch as D
    plan = Plan("CARROT", 20, 6, 0, 0, 0.05, 12)
    spec.set_turns_per_day(HOURS)
    spec.set_episode_steps(HOURS * 3)
    env = FastEnv(configuration={"episodeSteps": HOURS * 3, "turnsPerDay": HOURS, "startingMoney": 3000}, seed=7101)
    env.reset()
    with active(plan):
        ag = Agent(episode_steps=HOURS * 3, macro=plan_macro(plan))
    spec.set_episode_steps(720)                       # the stale state of a worker
    money = D._rollout((ag, env, plan))
    assert spec.EPISODE_STEPS == HOURS * 3
    assert money == play_plan(plan, 7101, days=3)
