"""The same audit for any actor: a public agent, the deployed policy, a plan.

    python tools/audit_agent.py --days 30 --seed 7106 v48-fast-routes v5 plan:runs/ladder/ladder.jsonl:30

Per actor: money, expansion milestones (quadrants, animals, tiles by day),
the crew's unit-turns split into work / needed moves / excess / orphan /
pass, moves per work action, and sales by product. What "precise like a
clock" means in numbers, side by side.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kagsym import evaluate as E, spec  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.plan import Plan, active, plan_macro  # noqa: E402

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
WORK = ("PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "FEED", "CARE", "COLLECT_FERTILIZER", "PLACE",
        "BUILD_COOP", "BUILD_PASTURE", "PICKUP", "DROP")


def actor_for(name: str, days: int):
    hours = 24
    cfg = {"episodeSteps": hours * days, "turnsPerDay": hours, "startingMoney": 3000}
    if name == "v5":
        from kagsym.policy import Policy
        pol = Policy.from_checkpoint("runs/partida_v5.pt"); pol.configure(cfg); pol.reset()
        return pol.act, None
    if name.startswith("plan:"):
        _, path, d = name.split(":")
        rows = [json.loads(l) for l in open(path)]
        pd = next(r["plan"] for r in rows if r["days"] == int(d))
        plan = Plan(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pd.items()})
        from kagsym.symbolic.executor import Agent
        ag = Agent(episode_steps=hours * days, macro=plan_macro(plan))
        return ag, plan
    if name.startswith("block:"):          # block:SHEEP:7:2  -> kind, count, hands
        _, kind, n, hands = name.split(":")
        plan = Plan("WHEAT", 0, int(hands), 0, {kind: int(n)}, 0.05, 12)
        from kagsym.symbolic.executor import Agent
        ag = Agent(episode_steps=hours * days, macro=plan_macro(plan))
        return ag, plan
    fn = E._opponent_callable(E.public(name))
    return fn, None


def audit(name: str, days: int, seed: int) -> dict:
    hours = 24
    spec.set_turns_per_day(hours)
    spec.set_episode_steps(hours * days)
    actor, plan = actor_for(name, days)
    cfg = {"episodeSteps": hours * days, "turnsPerDay": hours, "startingMoney": 3000}
    env = FastEnv(configuration=cfg, seed=seed)
    obs = env.reset()
    verbs = collections.Counter(); sold = collections.Counter(); steps = collections.Counter()
    walk = {}; milestones = {}; per_day = []
    ctx = active(plan) if plan is not None else None
    if ctx: ctx.__enter__()
    try:
        while not env.done:
            ob = obs[0]
            a = actor(ob)
            f = ob["farms"][0]
            positions = [tuple(f["farmer"])] + [tuple(p) for p in f["hands"]]
            if ob["hour"] == 0:
                walk = {}
            for o in a.get("market", []) or []:
                if o[0] == "SELL":
                    sold[o[1]] += int(o[2])
            acts = [a.get("farmer", ["PASS"])[0]] + [h[0] for h in (a.get("hands", []) or [])]
            for i, v in enumerate(acts):
                kind = "move" if v in MOVES else ("pass" if v == "PASS" else "work")
                verbs[kind] += 1
                st_, last = walk.get(i, (0, None))
                if v in MOVES:
                    walk[i] = (st_ + 1, last)
                elif v != "PASS" and i < len(positions):
                    here = positions[i]
                    need = abs(here[0] - last[0]) + abs(here[1] - last[1]) if last else st_
                    steps["needed"] += need; steps["excess"] += max(0, st_ - need)
                    walk[i] = (0, here)
            if ob["hour"] == hours - 1:
                for st_, _ in walk.values():
                    steps["orphan"] += st_
                animals = sum(1 for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("animal"))
                planted = sum(1 for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "PLANT")
                q = len(f["unlocked_quadrants"])
                per_day.append(dict(day=int(ob["day"]), cash=round(float(f["money"])), hands=len(f["hands"]), quadrants=q,
                                    animals=animals, planted=planted))
                for key, cond in (("q2", q >= 2), ("q3", q >= 3), ("q4", q >= 4), ("an6", animals >= 6), ("an12", animals >= 12),
                                  ("t25", planted >= 25), ("t50", planted >= 50)):
                    if cond and key not in milestones:
                        milestones[key] = int(ob["day"])
            obs, _ = env.step([a, dict(E.PASS_ACTION)])
        final = float(env.rewards()[0])
    finally:
        if ctx: ctx.__exit__(None, None, None)
    n = max(1, sum(verbs.values()))
    return dict(name=name, final=final, verbs=dict(verbs), share={k: v / n for k, v in verbs.items()},
                moves_per_work=verbs["move"] / max(1, verbs["work"]), steps=dict(steps), sold=dict(sold),
                milestones=milestones, last=per_day[-1] if per_day else {}, days=per_day)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actors", nargs="+")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7106)
    a = ap.parse_args()
    rows = [audit(x, a.days, a.seed) for x in a.actors]
    print(f"{'actor':34s} {'money':>8s}  work  move  pass  mv/work  excess orphan   q2  q3  q4 an6 an12 t25 t50   hands(last)")
    for r in rows:
        m = r["milestones"]; s = r["share"]; st_ = r["steps"]; tm = max(1, r["verbs"].get("move", 0))
        ms = "  ".join(f"{m.get(k, '-'):>2}" for k in ("q2", "q3", "q4", "an6", "an12", "t25", "t50"))
        print(f"{r['name'][:34]:34s} {r['final']:8,.0f}  {s.get('work',0):4.0%}  {s.get('move',0):4.0%}  {s.get('pass',0):4.0%}   {r['moves_per_work']:5.2f}   "
              f"{st_.get('excess',0)/tm:4.0%}  {st_.get('orphan',0)/tm:4.0%}   {ms}   {r['last'].get('hands','-')}")
    for r in rows:
        print(f"  {r['name'][:30]}: sold {dict(sorted(r['sold'].items(), key=lambda kv: -kv[1]))}")
    for r in rows:
        print(f"  {r['name'][:30]} by day (cash/hands/quadrants/animals/planted):", " ".join(
            f"d{d['day']}:{d['cash']//1000}k/{d['hands']}/{d['quadrants']}/{d['animals']}/{d['planted']}" for d in r["days"] if d["day"] in (0, 2, 4, 6, 9, 12, 15, 20, 25, 29)))


if __name__ == "__main__":
    main()
