"""Exhaustive search over explicit plans on the exact engine (reduced solitaire).

    python tools/plan_search.py --days 8 --seeds 5 --procs 11

Every plan in the grid is played deterministically on the same reserved seeds
and the mean money is reported. This is the executor's true optimum over
structured plans: the number the learner has to reach, and the teacher for
expert iteration (kagsym/plan.py).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym.plan import Plan, play_plan          # noqa: E402
from kagsym.seeds import RESERVED               # noqa: E402


def _one(args):
    plan_d, seeds, days, hours = args
    plan = Plan(**plan_d)
    vals = [play_plan(plan, s, days=days, hours=hours) for s in seeds]
    return plan_d, sum(vals) / len(vals), vals


def grid(days: int):
    """The constant-plan grid: the GLOBAL stage of the search.

    Coarse on purpose, but it has to span the levers that a local search
    cannot reach in one or two moves. Measured on the first two climbs: a
    grid of single crops without animals found 50,861 at 30 days while a
    5-crop portfolio with 4 animals and 7 hands, written by hand, gave
    56,380; the day-by-day refinement never got there from any grid plan.
    """
    from kagsym import spec
    singles = [c for c in spec.CROP_LIST if spec.CROPS[c]["first_yield_day"] < days]
    if not singles:                        # nothing yields in time: idle plans only
        for h in range(0, 9):
            yield dict(crop="CARROT", tiles=0, hands=h, land=0, load=12, animals=0)
        return
    crops = list(singles)
    crops += [{a: 0.5, b: 0.5} for i, a in enumerate(singles) for b in singles[i + 1:]]
    if len(singles) >= 3:
        crops.append({c: 1.0 / len(singles) for c in singles})
    animals = [0, 2, 4, 6] if days >= 5 else [0]     # a goose yields from day 4
    hands = [1, 3, 5, 7] if days >= 5 else [0, 1, 2, 3, 5, 7]
    for c, t, h, l, a in itertools.product(crops, (10, 20, 25, 50), hands, (0, 1), animals):
        if t > 25 * (1 + l) or (l == 1 and t < 50):
            continue
        yield dict(crop=c, tiles=t, hands=h, land=l, load=12, animals=a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--out", default="runs/plan_search.json")
    ap.add_argument("--schedule", action="store_true",
                    help="after the grid, coordinate descent over per-day hands and tiles from the best plan")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--grid-json", default=None, help="reuse a finished grid instead of replaying it")
    a = ap.parse_args()
    seeds = RESERVED.seeds(a.seeds)
    if a.grid_json:                       # reuse a finished grid
        rows = [(r["plan"], r["mean"], r["values"]) for r in json.load(open(a.grid_json))]
    else:
        plans = list(grid(a.days))
        print(f"{len(plans)} plans x {len(seeds)} seeds, {a.days}d x {a.hours}h", flush=True)
        t0 = time.time()
        with Pool(a.procs) as pool:
            rows = pool.map(_one, [(p, seeds, a.days, a.hours) for p in plans], chunksize=4)
        rows.sort(key=lambda r: -r[1])
        print(f"{time.time() - t0:.0f}s", flush=True)
    print("top 15:")
    for p, m, vals in rows[:15]:
        print(f"  {m:8,.0f}  {p}  {[round(v) for v in vals]}")
    print("bottom 3:")
    for p, m, vals in rows[-3:]:
        print(f"  {m:8,.0f}  {p}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump([dict(plan=p, mean=m, values=vals) for p, m, vals in rows], open(a.out, "w"), indent=1)
    if a.schedule:
        fields = ("hands", "load", "water_last", "tiles", "crop", "selling")
        days = a.days
        starts = [dict(rows[0][0], selling=0.05, load=0)]
        if days >= 6:
            starts += [dict(crop="WHEAT", tiles=50, hands=7, land=1, selling=0.05, load=0),
                       dict(crop="CARROT", tiles=50, hands=7, land=1, selling=0.05, load=0)]
        found = []
        for st in starts:
            cur, best = schedule_search(st, seeds, a.days, a.hours, a.procs, rounds=a.rounds, fields=fields)
            print(f"schedule optimum {best:,.0f}  {cur}", flush=True)
            found.append(dict(plan=cur, mean=best))
        found.sort(key=lambda r: -r["mean"])
        json.dump(found, open(a.out.replace(".json", "_schedule.json"), "w"), indent=1)
        print(f"BEST {found[0]['mean']:,.0f}  {found[0]['plan']}")



# ---------------------------------------------------------------- schedules

def _eval_plan(plan_d, seeds, days, hours):
    return _one((plan_d, seeds, days, hours))[1]


def schedule_search(start: dict, seeds, days: int, hours: int, procs: int, rounds: int = 3,
                    fields=("hands", "tiles"), values=None):
    """Coordinate descent over per-day schedules: for every (field, day) try
    every value with the rest fixed, keep the best, repeat until no gain."""
    from kagsym import spec
    crops = [c for c in spec.CROP_LIST if spec.CROPS[c]["first_yield_day"] < days]
    # A mix inside the day: every pair of viable crops at 50/50 (the executor
    # buys and plants both; the shares are of `tiles`).
    crops = crops + [{a: 0.5, b: 0.5} for i, a in enumerate(crops) for b in crops[i + 1:]]
    max_tiles = 25 * (1 + int(start.get("land", 0)))
    values = values or {"hands": list(range(0, 9)),
                        "tiles": [t for t in (0, 5, 10, 15, 20, 25, 30, 40, 50, 75) if t <= max_tiles],
                        "crop": crops, "selling": [0.05, 0.25, 0.5, 0.75, 0.95],
                        "load": [0, 3, 6, 9, 12, 15], "water_last": [0, 1]}
    cur = dict(start)
    for f in fields:
        v = cur.get(f, getattr(Plan(), f))     # a field the start did not set: the Plan default
        cur[f] = tuple(v) if isinstance(v, (list, tuple)) else tuple([v] * days)
    best = _eval_plan(cur, seeds, days, hours)
    print(f"start {best:,.0f}  {cur}", flush=True)
    with Pool(procs) as pool:
        for r in range(rounds):
            improved = False
            for f in fields:
                for d in range(days):
                    cands = []
                    for val in values[f]:
                        if val == cur[f][d]:
                            continue
                        c = dict(cur)
                        sched = list(cur[f]); sched[d] = val; c[f] = tuple(sched)
                        cands.append(c)
                    if not cands:                 # a field with one legal value
                        continue
                    res = pool.map(_one, [(c, seeds, days, hours) for c in cands], chunksize=1)
                    top = max(res, key=lambda x: x[1])
                    if top[1] > best + 1e-6:
                        best, cur, improved = top[1], top[0], True
                        print(f"  round {r} {f}[{d}] -> {cur[f][d]}: {best:,.0f}", flush=True)
            print(f"round {r} done: {best:,.0f}  {cur}", flush=True)
            if not improved:
                break
    return cur, best


if __name__ == "__main__":
    main()
