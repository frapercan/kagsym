"""Day-by-day plan search with exact rollouts: the oracle over the plan space.

    python tools/plan_daysearch.py --days 8 --seeds 5 --procs 8

At every day boundary, with the real state in hand, every single-field
variation of that day's plan is played to the end of the game (the rest of
the schedule is the continuation); the best is kept, a second round starts
from it, and the day is committed. What the ladder's coordinate descent does
blind over the whole schedule, this does with the state in front of it.

Every decision is recorded (state at the day boundary, the day's plan, the
rollout's money): the dataset the plan head learns from (EXP-008).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kagsym import spec  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.plan import Plan, active, plan_macro, set_plan  # noqa: E402
from kagsym.seeds import RESERVED  # noqa: E402
from kagsym.symbolic.executor import Agent  # noqa: E402

CASH = 3000
FIELDS = ("hands", "load", "water_last", "tiles", "crop", "selling", "animals")


def _values(days: int, land: int) -> dict:
    singles = [c for c in spec.CROP_LIST if spec.CROPS[c]["first_yield_day"] < days]
    crops = list(singles)
    crops += [{a: 0.5, b: 0.5} for i, a in enumerate(singles) for b in singles[i + 1:]]
    # PORTFOLIOS: every product sells into its own price curve, so spreading
    # the tiles over three or all viable crops is a different plan from any
    # pair (measured at 30 days: v5 sells seven products; a 25-tile
    # monoculture-per-day search reached 50,861 against v5's 104,307).
    if len(singles) >= 3:
        crops += [{a: 1 / 3, b: 1 / 3, c: 1 / 3} for i, a in enumerate(singles)
                  for j, b in enumerate(singles[i + 1:], i + 1) for c in singles[j + 1:]]
    if len(singles) >= 4:
        crops.append({c: 1.0 / len(singles) for c in singles})
    max_tiles = 25 * (1 + land)
    return {"hands": list(range(0, 9)), "load": [0, 3, 6, 9, 12, 15], "water_last": [0, 1],
            "tiles": [t for t in (0, 10, 15, 20, 25, 30, 40, 50, 75) if t <= max_tiles],
            "crop": crops, "selling": [0.05, 0.25, 0.5, 0.75, 0.95],
            "animals": [0, 1, 2, 3, 4, 6, 8]}


def _sched(plan: Plan, f: str, days: int) -> list:
    v = getattr(plan, f)
    v = list(v) if isinstance(v, (tuple, list)) else [v]
    return v + [v[-1]] * (days - len(v)) if len(v) < days else v[:days]


def with_value(plan: Plan, f: str, day: int, val, days: int) -> Plan:
    s = _sched(plan, f, days)
    s[day] = val
    return replace(plan, **{f: tuple(s)})


def candidates(plan: Plan, day: int, days: int, rng=None, pairs: int = 12) -> list:
    """Every single-field variation of the day, plus `pairs` random two-field
    variations (a single-field fixed point is not a joint one: measured, the
    ladder's coordinate-descent optimum survived every single move)."""
    out = []
    # Land on ANY day: after a melon harvest the cash is there and a quadrant
    # plus a short cycle is the reinvestment (only day 0 was offered before).
    cur_land = int(_sched(plan, "land", days)[day])
    for land in (0, 1, 2, 3):
        if land != cur_land and (day == 0 or land > cur_land):
            out.append(with_value(plan, "land", day, land, days))
    vals = _values(days - day, cur_land)
    for f in FIELDS:
        cur = _sched(plan, f, days)[day]
        for v in vals[f]:
            if v != cur:
                out.append(with_value(plan, f, day, v, days))
    if rng is not None and pairs > 0:
        live = [f for f in FIELDS if vals[f]]          # no crop is viable on the last days
        for _ in range(pairs if len(live) >= 2 else 0):
            f1, f2 = rng.sample(live, 2)
            c = with_value(plan, f1, day, rng.choice(vals[f1]), days)
            c = with_value(c, f2, day, rng.choice(vals[f2]), days)
            if c != plan:
                out.append(c)
    return out


OPPONENT = None       # a recorded rival ("replay:<file>") inside the rollouts: state-free, so a rollout can branch mid-game


def _rival_for(seed: int):
    from kagsym.plan import _TAPES
    from kagsym.evaluate import PASS_ACTION
    import json
    if not OPPONENT:
        return lambda ob: dict(PASS_ACTION)
    tape = _TAPES.get(OPPONENT)
    if tape is None:
        tape = _TAPES[OPPONENT] = json.load(open(OPPONENT[len("replay:"):]))
    acts = tape[str(seed)]
    return lambda ob, _a=acts: _a[int(ob["step"])] if int(ob["step"]) < len(_a) else dict(PASS_ACTION)


def _rollout(task):
    ag, env, plan = task
    # THE CALENDAR IS PROCESS STATE. A forked worker carries whatever
    # spec.EPISODE_STEPS the parent had when the pool was made; when the
    # grid was cached nothing had set it, and every candidate was played as
    # a 30-day game (2,540 against a correct base of 3,714 at 3 days) so no
    # candidate ever "improved". Set it here, from the game itself.
    spec.set_turns_per_day(int(env.configuration["turnsPerDay"]))
    spec.set_episode_steps(int(env.configuration["episodeSteps"]))
    with active(plan):
        env2 = env.clone()
        ag2 = copy.deepcopy(ag)
        rival = _rival_for(int(env.info.get("seed", 0)) if hasattr(env, "info") else env.seed)
        while not env2.done:
            obs2 = env2.observations()
            env2.step([ag2(obs2[0]), rival(obs2[1])])
        return float(env2.rewards()[0])


def _state(ob) -> dict:
    f = ob["farms"][0]
    by = {}
    for r in f["tiles"]:
        for t in r:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                k = f"{t['crop']}@{int(ob['day']) - t['planted_day']}"
                by[k] = by.get(k, 0) + 1
    # The demand (the town's shops) and the rival's visible farm: what a
    # fixed plan cannot see and the plan head must (EXP-007 parts 6-8).
    other = ob["farms"][1 - int(ob["player"])]
    rival_planted = {}
    for r in other["tiles"]:
        for t in r:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                rival_planted[t["crop"]] = rival_planted.get(t["crop"], 0) + 1
    rival_animals = {}
    for r in other["tiles"]:
        for t in r:
            if isinstance(t, dict) and t.get("animal"):
                rival_animals[t["animal"]] = rival_animals.get(t["animal"], 0) + 1
    animals = {}
    for r in f["tiles"]:
        for t in r:
            if isinstance(t, dict) and t.get("animal"):
                animals[t["animal"]] = animals.get(t["animal"], 0) + 1
    return dict(day=int(ob["day"]), cash=round(float(f["money"])), hands=len(f["hands"]),
                quadrants=len(f["unlocked_quadrants"]), planted=by, animals=animals,
                seeds={k: int(v) for k, v in ob["private"]["seeds"].items() if v},
                shed={k: int(v) for k, v in ob["private"]["shed"].items() if v},
                prices={k: ob["market"]["prices"][k] for k in spec.PRODUCTS},
                inventory={k: int(ob["market"]["inventory"][k]) - 10000 for k in spec.PRODUCTS},
                shops=list(ob["town"].get("unlocked_shops", [])),
                rival=dict(money=round(float(other["money"])), hands=len(other["hands"]),
                           quadrants=len(other["unlocked_quadrants"]), planted=rival_planted, animals=rival_animals))


def search_game(seed: int, days: int, starts: list, pool, rounds: int = 2, hours: int = 24, log=None) -> tuple:
    steps = hours * days
    spec.set_turns_per_day(hours)
    spec.set_episode_steps(steps)
    cfg = {"episodeSteps": steps, "turnsPerDay": hours, "startingMoney": CASH}
    env = FastEnv(configuration=cfg, seed=seed)
    obs = env.reset()
    ag = Agent(episode_steps=steps, macro=plan_macro(starts[0]))
    rival = _rival_for(seed)
    # Day 0: the whole-schedule alternatives compete first.
    vals = pool.map(_rollout, [(ag, env, p) for p in starts], chunksize=1)
    plan = starts[max(range(len(vals)), key=lambda i: vals[i])]
    set_plan(plan)
    records = []
    import random
    rng = random.Random(seed)
    try:
        while not env.done:
            ob = obs[0]
            if ob["hour"] == 0:
                day = int(ob["day"])
                base = _rollout((ag, env, plan))
                best, best_plan = base, plan
                for _ in range(rounds):
                    cands = candidates(best_plan, day, days, rng)
                    vals = pool.map(_rollout, [(ag, env, c) for c in cands], chunksize=1)
                    j = max(range(len(vals)), key=lambda i: vals[i])
                    if vals[j] > best + 1e-6:
                        best, best_plan = vals[j], cands[j]
                    else:
                        break
                plan = best_plan
                set_plan(plan)
                records.append(dict(seed=seed, days=days, state=_state(ob), base=base, best=best,
                                    plan={f: _sched(plan, f, days)[day] for f in FIELDS} | {"land": _sched(plan, "land", days)[day]}))
                if log:
                    print(f"  seed {seed} day {day}: {base:,.0f} -> {best:,.0f}  {records[-1]['plan']}", file=log, flush=True)
            a = ag(ob)
            obs, _ = env.step([a, rival(obs[1])])
        return float(env.rewards()[0]), plan, records
    finally:
        set_plan(None)


def _as_plan(d: dict) -> Plan:
    return Plan(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def grid_plans(days: int, seeds, pool, top: int = 5) -> list:
    """The constant-plan grid of this horizon (cached in runs/ladder/plan_<d>d.json).

    Local moves cannot reach a plan three coordinated changes away: measured
    at 11 days, melon (crop + 20 tiles + no land) is worth 26,463 while the
    10-day plan gives 9,449, and each intermediate step is worse (6,170 and
    3,175). The grid is where such plans are found; the day search refines."""
    path = f"runs/ladder/plan_{days}d.json"
    if not os.path.exists(path):
        from plan_search import grid, _one
        rows = pool.map(_one, [(pl, list(seeds), days, 24) for pl in grid(days)], chunksize=4)
        rows.sort(key=lambda r: -r[1])
        json.dump([dict(plan=pl, mean=m, values=v) for pl, m, v in rows], open(path, "w"), indent=1)
    rows = json.load(open(path))
    return [_as_plan(r["plan"]) for r in rows[:top]]


def start_plans(days: int, seeds, pool, extra: list | None = None) -> list:
    """Day-0 alternatives: the ladder's schedule for this horizon if there is
    one, the grid's top plans, the nearest lower rung's schedule padded, and
    any `extra` plans (the protocol passes the previous rung's schedule:
    without it, rung 8 searched to 6,733 while the 7-day plan padded gave
    6,756)."""
    out = list(extra or [])
    for suffix in ("_mix_schedule", "_schedule"):
        p = f"runs/ladder/plan_{days}d{suffix}.json"
        if os.path.exists(p):
            out.append(_as_plan(json.load(open(p))[0]["plan"]))
    out += grid_plans(days, seeds, pool)
    for d in range(days - 1, 0, -1):
        p = f"runs/ladder/plan_{d}d_schedule.json"
        if os.path.exists(p):
            out.append(_as_plan(json.load(open(p))[0]["plan"]))
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start-json", default=None, help="a plan (dict) to include among the day-0 alternatives")
    ap.add_argument("--opponent", default=None, help="replay:<tape.json> (tools/record_rival.py); passive if omitted")
    a = ap.parse_args()
    global OPPONENT
    OPPONENT = a.opponent
    extra = [_as_plan(json.load(open(a.start_json)))] if a.start_json else []
    out = a.out or f"runs/ladder/daysearch_{a.days}d.jsonl"
    seeds = RESERVED.seeds(a.seeds, a.offset)
    t0 = time.time()
    finals = []
    with get_context("fork").Pool(a.procs) as pool, open(out, "a") as fo:
        starts = start_plans(a.days, seeds, pool, extra)
        print(f"{a.days} days, seeds {seeds[0]}..{seeds[-1]}, {len(starts)} day-0 alternatives "
              f"(grid {time.time() - t0:.0f}s); best start {starts[0].to_dict()}", flush=True)
        for s in seeds:
            final, plan, recs = search_game(s, a.days, starts, pool, rounds=a.rounds, log=sys.stdout)
            finals.append(final)
            for r in recs:
                fo.write(json.dumps(r) + "\n")
            print(f"seed {s}: {final:,.0f}   ({time.time() - t0:.0f}s)", flush=True)
    print(f"mean {sum(finals) / len(finals):,.0f} over {len(finals)} seeds; {time.time() - t0:.0f}s; records -> {out}")


if __name__ == "__main__":
    main()
