"""Portfolio search: the 30-day game as a schedule of blocks, played exactly.

    python tools/plan_blocks.py --gens 6 --pop 160 --seeds 3 --procs 8

A portfolio is ~16 numbers that generate a 30-day Plan: the opening (animals
by kind, tiles and crop mix, hands), the days each quadrant is bought, the
final animal targets by kind, tiles and mix after the expansion, hands
after the expansion, and the selling pace. The executor buys what the cash
allows towards the targets, so a rising target is "grow as the cash comes".
Every candidate is played exactly on the search seeds; a small evolution
(elite + mutation) over generations, seeded with v48's opening; the best
is then refined day by day (tools/plan_daysearch.py) and reported on
unseen seeds with the mechanics audit.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from multiprocessing import get_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from kagsym import spec  # noqa: E402
from kagsym.plan import Plan, play_plan  # noqa: E402
from kagsym.seeds import RESERVED  # noqa: E402

DAYS = 30
MIXES = {
    "wheat": "WHEAT", "carrot": "CARROT", "melon": "MELON", "strawberry": "STRAWBERRY", "tomato": "TOMATO",
    "straw+melon": {"STRAWBERRY": 0.5, "MELON": 0.5},
    "straw+wheat": {"STRAWBERRY": 0.6, "WHEAT": 0.4},
    "straw+melon+wheat": {"STRAWBERRY": 0.4, "MELON": 0.3, "WHEAT": 0.3},
    "melon+wheat": {"MELON": 0.5, "WHEAT": 0.5},
    "all": {c: 0.2 for c in spec.CROP_LIST},
    "demand": "DEMAND",          # the mix follows the shops observed at runtime
}
NEVER = 99
SPACE = {
    "g0": [0, 1, 2, 3, 4], "c0": [0, 1, 2, 3], "s0": [0, 1, 2, 3],          # opening animals by kind
    "t0": [0, 6, 10, 12, 15, 20], "m0": ["wheat", "carrot", "straw+wheat", "melon+wheat", "demand"],
    "h0": [0, 1, 2, 3, 4, 5],
    "land2": [3, 5, 6, 8, 10, 12, NEVER], "land3": [8, 10, 12, 14, 16, NEVER], "land4": [12, 15, 18, NEVER],
    "g1": [0, 2, 4, 6, 8], "c1": [0, 2, 4, 6, 7], "s1": [0, 2, 4, 6, 7],      # final animal targets
    "d1": [4, 6, 8, 10, 12],                                                # day the final targets apply
    "t1": [25, 40, 50, 60, 75, 100], "m1": list(MIXES), "dcrop": [6, 8, 10, 12, 14],
    "h1": [4, 6, 8, 10, 12, 14, 15], "dh": [4, 6, 8, 10, 12],
    "sell": [0.05, 0.25, 0.5, 0.75],
    "discount": [0.5, 0.6, 0.7, 0.82],   # the executor's distance discount (Plan.discount)
    "zones": [0.0, 0.3, 0.6],            # each unit owns a quadrant (Plan.zones)
}
TEMPLATES = [
    # v48-like: 4 animals, 12 tiles, 2 hands on day 0; land on days 6 and 10; strawberry-heavy; 12 hands
    dict(g0=2, c0=1, s0=1, t0=12, m0="wheat", h0=2, land2=6, land3=10, land4=NEVER, g1=4, c1=6, s1=5, d1=8,
         t1=60, m1="straw+melon+wheat", dcrop=8, h1=12, dh=10, sell=0.25),
    # v5-like: 6 animals and 5 hands on day 0, nothing sown; land days 6 and 9; strawberry and melon from day 9
    dict(g0=2, c0=2, s0=2, t0=0, m0="wheat", h0=5, land2=6, land3=9, land4=NEVER, g1=2, c1=7, s1=3, d1=6,
         t1=50, m1="straw+melon", dcrop=9, h1=10, dh=8, sell=0.5),
    # the sheep block, then land and strawberries with the wool money
    dict(g0=0, c0=0, s0=3, t0=0, m0="wheat", h0=2, land2=8, land3=12, land4=NEVER, g1=0, c1=4, s1=7, d1=4,
         t1=50, m1="strawberry", dcrop=10, h1=8, dh=8, sell=0.25),
]


def to_plan(p: dict) -> Plan:
    quadrants = [1 + (d >= p["land2"]) + (d >= p["land3"]) + (d >= p["land4"]) for d in range(DAYS)]
    land = tuple(q - 1 for q in quadrants)
    animals, tiles, crops, hands = [], [], [], []
    for d in range(DAYS):
        a = ({"GOOSE": p["g1"], "COW": p["c1"], "SHEEP": p["s1"]} if d >= p["d1"]
             else {"GOOSE": p["g0"], "COW": p["c0"], "SHEEP": p["s0"]})
        a = {k: v for k, v in a.items() if v > 0}
        n_an = sum(a.values())
        room = 25 * quadrants[d] - n_an                      # a coop or pasture takes a tile
        t = p["t1"] if d >= p["dcrop"] else p["t0"]
        tiles.append(max(0, min(t, room)))
        crops.append(MIXES[p["m1"] if d >= p["dcrop"] else p["m0"]])
        animals.append(a if a else 0)
        hands.append(p["h1"] if d >= p["dh"] else p["h0"])
    return Plan(crop=tuple(crops), tiles=tuple(tiles), hands=tuple(hands), land=land, animals=tuple(animals),
                selling=p["sell"], load=12, water_last=1, discount=p.get("discount"), zones=p.get("zones", 0.0))


def sample(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SPACE.items()}


def mutate(p: dict, rng: random.Random, k: int = 2) -> dict:
    q = dict(p)
    for key in rng.sample(list(SPACE), k):
        q[key] = rng.choice(SPACE[key])
    return q


OPPONENT = None       # set from --opponent: the market is shared, fitness is money against a real rival


def _eval(task):
    p, seeds = task
    plan = to_plan(p)
    return st.mean(play_plan(plan, s, days=DAYS, opponent=OPPONENT) for s in seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--pop", type=int, default=160)
    ap.add_argument("--elite", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/blocks/portfolio.json")
    ap.add_argument("--opponent", default=None, help="a public agent as the rival (default: passive)")
    ap.add_argument("--init", default=None, help="a consolidated portfolio json to seed the population with")
    ap.add_argument("--race", action="store_true", help="successive halving over the seeds (about 3x fewer games)")
    a = ap.parse_args()
    global OPPONENT
    OPPONENT = a.opponent
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(a.seed)
    search_seeds = RESERVED.seeds(a.seeds)
    unseen = RESERVED.seeds(5, 5)
    seeds_pop = list(TEMPLATES)
    if a.init:
        seeds_pop = [r["params"] for r in json.load(open(a.init))[:10]] + seeds_pop
    pop = seeds_pop + [mutate(rng.choice(seeds_pop), rng, k=rng.choice((1, 2, 3))) for _ in range(a.pop // 2 - len(seeds_pop))]
    pop += [sample(rng) for _ in range(a.pop - len(pop))]
    seen: dict = {}
    t0 = time.time()
    with get_context("fork").Pool(a.procs) as pool:
        for g in range(a.gens):
            todo = [p for p in pop if json.dumps(p, sort_keys=True) not in seen]
            if a.race and len(search_seeds) >= 3:
                # RACING (successive halving): everyone on one seed, the top
                # 40 % on three, the top 10 % on all; the rest keep their
                # partial mean. About a third of the games for the same elite.
                s1 = pool.map(_eval, [(p, search_seeds[:1]) for p in todo], chunksize=2)
                order = sorted(range(len(todo)), key=lambda i: -s1[i])
                keep2 = order[: max(4, int(0.4 * len(todo)))]
                s2 = dict(zip(keep2, pool.map(_eval, [(todo[i], search_seeds[1:3]) for i in keep2], chunksize=1)))
                m2 = {i: (s1[i] + 2 * s2[i]) / 3 for i in keep2}
                keep3 = sorted(keep2, key=lambda i: -m2[i])[: max(2, int(0.25 * len(keep2)))]
                s3 = dict(zip(keep3, pool.map(_eval, [(todo[i], search_seeds[3:]) for i in keep3], chunksize=1)))
                n3 = len(search_seeds) - 3
                vals = [((s1[i] + 2 * s2[i] + n3 * s3[i]) / (3 + n3)) if i in s3 else (m2[i] if i in m2 else s1[i] - 1e6)
                        for i in range(len(todo))]      # a one-seed score never beats a full one
            else:
                vals = pool.map(_eval, [(p, search_seeds) for p in todo], chunksize=2)
            for p, v in zip(todo, vals):
                seen[json.dumps(p, sort_keys=True)] = v
            ranked = sorted(pop, key=lambda p: -seen[json.dumps(p, sort_keys=True)])
            best = ranked[0]
            print(f"gen {g}: best {seen[json.dumps(best, sort_keys=True)]:,.0f}  median {st.median(seen[json.dumps(p, sort_keys=True)] for p in pop):,.0f}  "
                  f"({len(seen)} played, {time.time() - t0:.0f}s)  {best}", flush=True)
            elite = ranked[:a.elite]
            children = []
            while len(children) < a.pop - a.elite:
                parent = rng.choice(elite)
                children.append(mutate(parent, rng, k=rng.choice((1, 2, 3))))
            pop = elite + children
        ranked = sorted(seen.items(), key=lambda kv: -kv[1])
        best = json.loads(ranked[0][0])
        plan = to_plan(best)
        u = pool.starmap(play_plan, [(plan, s, DAYS, 24, 3000, OPPONENT) for s in unseen])
    print(f"best portfolio on search seeds {ranked[0][1]:,.0f}; on unseen seeds {st.mean(u):,.0f} {[round(x) for x in u]}")
    json.dump(dict(params=best, search=ranked[0][1], unseen=st.mean(u), plan=plan.to_dict(),
                   top=[(json.loads(k), v) for k, v in ranked[:10]]), open(a.out, "w"), indent=1, default=str)
    print("saved", a.out)


if __name__ == "__main__":
    main()
