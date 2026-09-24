#!/usr/bin/env python
"""Audit one solitaire game: what the policy does each day, and what it wastes.

Per day: cash at the end of the day, hands, tiles under crop, animals,
market orders by type, and the unit-turns spent on PASS or moving. At the
end: what was left on the table (unsold produce in shed and hands, crops
still standing, seed never planted). The point is to judge the plays, not
only the money: an optimal 8-day solitaire ends with nothing unsold, no
seed in hand, no hands hired for work that never came, and few idle turns.

    python tools/audit_play.py runs/model.pt --days 8 [--seed 7101] [--opponent passive]
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, spec  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.policy import Policy  # noqa: E402
from kagsym.potential import _lot_value  # noqa: E402

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--days", type=int, default=8)
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--seed", type=int, default=7101)
    p.add_argument("--opponent", default="passive")
    a = p.parse_args()
    cfg = {"episodeSteps": a.hours * a.days, "turnsPerDay": a.hours, "startingMoney": 3000}
    pol = Policy.from_checkpoint(a.checkpoint)
    pol.configure(cfg)
    pol.reset()
    rival = E._opponent_callable(E.passive() if a.opponent == "passive" else E.public(a.opponent))
    env = FastEnv(configuration=cfg, seed=a.seed)
    obs = env.reset()
    day_rows = []
    cur = collections.Counter()
    unit_turns = collections.Counter()
    last_day = 0
    hands_seen = 0                      # hands is 0 at hour 0: read it at hour 23
    tile_days = 0                       # tile-days under crop
    idle_cash = 0.0                     # cash at end of day while tiles were free
    harvest_waits = []                  # units ready but not harvested, per day
    weeds_seen = 0
    digs = 0
    plant_cycles = collections.Counter()   # tile -> plantings
    free_tiles_seen = 0
    print(f"[audit] {os.path.basename(a.checkpoint)}  {a.hours}h x {a.days}d  seed {a.seed}  vs {a.opponent}\n")
    print(f"{'day':>3s} {'cash':>7s} {'hands':>5s} {'crops':>5s} {'anim':>4s} {'unit-turns: pass':>16s} {'move':>5s} {'work':>5s}   orders")
    while not env.done:
        ob = obs[0]
        a_ = pol.act(ob)
        acts = [a_.get("farmer") or ["PASS"]] + list(a_.get("hands") or [])
        for act in acts:
            op = act[0] if act else "PASS"
            unit_turns["pass" if op == "PASS" else ("move" if op in MOVES else "work")] += 1
            if op == "DIG":
                digs += 1
        if ob["hour"] == 23:
            hands_seen = max(hands_seen, len(ob["farms"][0]["hands"]))
        for o in a_.get("market", []):
            cur[o[0]] += int(o[2]) if len(o) > 2 and isinstance(o[2], (int, float)) else 1
        prev_tiles = {(x, y): t for y, r in enumerate(ob["farms"][0]["tiles"]) for x, t in enumerate(r)}
        obs, _ = env.step([a_, rival(obs[1])])
        nb = obs[0]
        for (x, y), t in prev_tiles.items():
            n = nb["farms"][0]["tiles"][y][x]
            if isinstance(n, dict) and n.get("kind") == "PLANT" and not (isinstance(t, dict) and t.get("kind") == "PLANT"):
                plant_cycles[(x, y)] += 1
            if isinstance(n, dict) and n.get("kind") == "WEED" and not (isinstance(t, dict) and t.get("kind") == "WEED"):
                weeds_seen += 1
        if nb["hour"] == 0 or env.done:
            f = nb["farms"][0]
            all_tiles = [t for r in f["tiles"] for t in r]
            tiles = [t for t in all_tiles if isinstance(t, dict)]
            crops = sum(1 for t in tiles if t.get("kind") == "PLANT")
            anim = sum(1 for t in tiles if t.get("animal"))
            free = sum(1 for t in all_tiles if t is None)
            ready = sum(int(t.get("yield_units", 0)) for t in tiles if t.get("kind") == "PLANT")
            tile_days += crops
            free_tiles_seen += free
            if free > 0:
                idle_cash += float(f["money"])
            harvest_waits.append(ready)
            day_rows.append((last_day, f["money"], hands_seen, crops, anim, dict(unit_turns), dict(cur)))
            print(f"{last_day:3d} {f['money']:7,.0f} {hands_seen:5d} {crops:5d} {anim:4d} "
                  f"{unit_turns['pass']:16d} {unit_turns['move']:5d} {unit_turns['work']:5d}   "
                  f"free {free:2d} ready {ready:3d}  {dict(cur)}")
            cur.clear()
            unit_turns.clear()
            hands_seen = 0
            last_day = nb["day"]
    ob = obs[0]
    f = ob["farms"][0]
    priv = ob["private"]
    shed = {k: v for k, v in (priv.get("shed") or {}).items() if v}
    carried = collections.Counter()
    for inv in priv.get("inventories") or []:
        for k, v in inv.items():
            if v:
                carried[k] += v
    seeds_left = {k: v for k, v in (priv.get("seeds") or {}).items() if v}
    standing = [(t["crop"], int(t.get("yield_units", 0))) for r in f["tiles"] for t in r
                if isinstance(t, dict) and t.get("kind") == "PLANT"]
    unsold = sum(_lot_value(ob, k, v) for k, v in shed.items() if k in spec.PRODUCTS)
    unsold += sum(_lot_value(ob, k, v) for k, v in carried.items() if k in spec.PRODUCTS)
    n_tiles = sum(1 for r in f["tiles"] for t in r)
    print(f"\nfinal cash {f['money']:,.0f} $   left on the table: unsold produce ~{unsold:,.0f} $ "
          f"(shed {shed}, carried {dict(carried)}), standing crops {len(standing)} {standing[:6]}, "
          f"seed unplanted {seeds_left}")
    print(f"space : {len(plant_cycles)} tiles ever planted of {n_tiles}; plantings per tile "
          f"{collections.Counter(plant_cycles.values())}; occupancy {tile_days}/{n_tiles * a.days} tile-days "
          f"({100 * tile_days / (n_tiles * a.days):.0f}%)")
    print(f"time  : units ready and waiting at day ends {harvest_waits}")
    print(f"labour: unit-turns pass/move/work over the game "
          f"{sum(r[5].get('pass', 0) for r in day_rows)}/{sum(r[5].get('move', 0) for r in day_rows)}/"
          f"{sum(r[5].get('work', 0) for r in day_rows)}")
    print(f"capital: cash idle at day ends while tiles were free: {idle_cash:,.0f} $-days")
    print(f"events: weeds appeared {weeds_seen}, DIG actions {digs}")


if __name__ == "__main__":
    main()
