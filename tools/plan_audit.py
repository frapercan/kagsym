"""Audit the execution of a plan: where the money the plan could make goes.

    python tools/plan_audit.py runs/ladder/plan_8d_schedule.json --days 8 --seed 7101

For the plan's own sowings it reports, per day and in total: units grown on
the tiles, harvested, sold, discarded (carried at the close, the shed's
overflow, weeds), and how the crew spent its turns (work, moves, pass). The
gross at the day-0 price curve for everything sold is the price bound; the
difference with the cash actually taken is the price paid for selling fast.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, spec  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.plan import Plan, active, plan_macro  # noqa: E402
from kagsym.symbolic.executor import Agent  # noqa: E402
from kagsym.symbolic.market_ops import marginal_prices  # noqa: E402

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def load_plan(path: str) -> Plan:
    d = json.load(open(path))
    d = d[0]["plan"] if isinstance(d, list) else d.get("plan", d)
    return Plan(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def audit(plan: Plan, days: int, seed: int, hours: int = 24, cash: int = 3000) -> dict:
    steps = hours * days
    spec.set_turns_per_day(hours)
    spec.set_episode_steps(steps)
    cfg = {"episodeSteps": steps, "turnsPerDay": hours, "startingMoney": cash}
    day_rows = []
    tot = collections.Counter()
    sold_units = collections.Counter()
    spent = collections.Counter()
    verbs = collections.Counter()
    with active(plan):
        env = FastEnv(configuration=cfg, seed=seed)
        obs = env.reset()
        ag = Agent(episode_steps=steps, macro=plan_macro(plan))
        cur = collections.Counter()
        ob0 = obs[0]
        last = None
        while not env.done:
            ob = obs[0]
            a = ag(ob)
            f = ob["farms"][0]
            for o in a.get("market", []):
                if o[0] == "SELL":
                    sold_units[o[1]] += int(o[2])
                    cur["sold"] += int(o[2])
                elif o[0] == "BUY_SEED":
                    spent["seed"] += int(o[2]) * spec.CROPS[o[1]]["seed"]
                elif o[0] == "HIRE":
                    cur["hires"] += 1
                elif o[0] == "BUY_LAND":
                    spent["land"] += 1
            positions = [tuple(f["farmer"])] + [tuple(p) for p in f["hands"]]
            if ob["hour"] == 0:
                walk = {}                         # unit -> (steps since last work, tile of last work)
            for i, v in enumerate([a.get("farmer", ["PASS"])[0]] + [h[0] for h in a.get("hands", [])]):
                verbs["move" if v in MOVES else ("pass" if v == "PASS" else "work")] += 1
                cur["harvests"] += v == "HARVEST"
                cur["drops"] += v == "DROP"
                # STEPS OF EXCESS: between two work actions a unit needs the
                # Manhattan distance; anything above is a detour or a change
                # of target on the way. Steps with no work after them before
                # the day ends are orphans.
                steps_, last_tile = walk.get(i, (0, None))
                if v in MOVES:
                    walk[i] = (steps_ + 1, last_tile)
                elif v != "PASS":
                    here = positions[i]
                    need = abs(here[0] - last_tile[0]) + abs(here[1] - last_tile[1]) if last_tile else steps_
                    tot["steps_needed"] += need
                    tot["steps_excess"] += max(0, steps_ - need)
                    walk[i] = (0, here)
            if ob["hour"] == hours - 1:
                for steps_, _ in walk.values():
                    tot["steps_orphan"] += steps_
            if ob["hour"] == hours - 1:
                planted = sum(1 for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "PLANT")
                weeds = sum(1 for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "WEED")
                on_tiles = sum(t.get("yield_units", 0) for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "PLANT")
                carried = sum(sum(i.values()) for i in ob["private"]["inventories"])
                shed = sum(ob["private"]["shed"].get(k, 0) for k in spec.PRODUCTS)
                day_rows.append(dict(day=int(ob["day"]), cash=round(float(f["money"])), hands=len(f["hands"]),
                                     planted=planted, weeds=weeds, on_tiles=on_tiles, carried=carried, shed=shed,
                                     harvests=cur["harvests"], drops=cur["drops"], sold=cur["sold"]))
                overflow = max(0, carried + shed - spec.DEFAULT_CONFIG["shedCapacity"])
                tot["overflow"] += overflow
                cur = collections.Counter()
            last = ob
            obs, _ = env.step([a, dict(E.PASS_ACTION)])
        f = last["farms"][0]
        if not day_rows or day_rows[-1]["day"] != int(last["day"]):   # the last day's row
            day_rows.append(dict(day=int(last["day"]), cash=round(float(f["money"])), hands=len(f["hands"]),
                                 planted=sum(1 for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "PLANT"),
                                 weeds=sum(1 for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "WEED"),
                                 on_tiles=sum(t.get("yield_units", 0) for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "PLANT"),
                                 carried=sum(sum(i.values()) for i in last["private"]["inventories"]),
                                 shed=sum(last["private"]["shed"].get(k, 0) for k in spec.PRODUCTS),
                                 harvests=cur["harvests"], drops=cur["drops"], sold=cur["sold"]))
        left = sum(t.get("yield_units", 0) for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "PLANT")
        carried = sum(sum(i.values()) for i in last["private"]["inventories"])
        shed = sum(last["private"]["shed"].get(k, 0) for k in spec.PRODUCTS)
        final = float(env.rewards()[0])
    price_bound = sum(sum(marginal_prices(ob0, k, n)) for k, n in sold_units.items())
    hands_cost = sum(sum(_fib(n) for n in range(d["hands"])) for d in day_rows)
    sales = final - cash + spent["seed"] + sum(spec.LAND_PRICES[i] for i in range(spent["land"])) + hands_cost
    return dict(final=final, days=day_rows, sold=dict(sold_units), seed=spent["seed"], land=spent["land"],
                left_on_tiles=left, carried_at_close=carried, shed_at_close=shed, overflow_discarded=tot["overflow"],
                turns=dict(verbs), price_bound=round(price_bound), sales_taken=round(sales),
                steps=dict(needed=tot["steps_needed"], excess=tot["steps_excess"], orphan=tot["steps_orphan"]))


def _fib(n: int) -> int:
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    return int(K._fib(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--days", type=int, required=True)
    ap.add_argument("--seed", type=int, default=7101)
    a = ap.parse_args()
    r = audit(load_plan(a.plan), a.days, a.seed)
    print(f"final {r['final']:,.0f}   sold {r['sold']}   seed {r['seed']}  land {r['land']}")
    print(f"lost: on tiles {r['left_on_tiles']}, carried at close {r['carried_at_close']}, in shed {r['shed_at_close']}, "
          f"shed overflow {r['overflow_discarded']}")
    t = r["turns"]; n = sum(t.values())
    print(f"crew turns: work {t.get('work',0)/n:.0%}  move {t.get('move',0)/n:.0%}  pass {t.get('pass',0)/n:.0%}  ({n} unit-turns)")
    st = r["steps"]; tm = t.get("move", 0)
    print(f"moves: {tm} = needed {st['needed']} ({st['needed']/max(1,tm):.0%}) + excess {st['excess']} ({st['excess']/max(1,tm):.0%}) "
          f"+ orphan {st['orphan']} ({st['orphan']/max(1,tm):.0%})")
    print(f"price: taken {r['sales_taken']:,} vs day-0 curve for the same units {r['price_bound']:,}")
    print("day  cash  hands planted weeds on_tiles carried shed harv drops sold")
    for d in r["days"]:
        print(f"{d['day']:3d} {d['cash']:6d} {d['hands']:5d} {d['planted']:7d} {d['weeds']:5d} {d['on_tiles']:8d} {d['carried']:7d} {d['shed']:4d} {d['harvests']:4d} {d['drops']:5d} {d['sold']:4d}")


if __name__ == "__main__":
    main()
