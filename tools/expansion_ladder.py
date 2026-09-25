"""The expansion ladder: blocks of growing size under a plan, solo, audited on 5 seeds; one MLflow run per block.

    python tools/expansion_ladder.py [--discount 0.65]
"""
"""The expansion ladder: blocks of growing size under a plan, solo, audited."""
import sys, os, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kagsym import spec, evaluate as E
from kagsym.fastenv import FastEnv
from kagsym.plan import Plan, active, plan_macro
from kagsym.symbolic.executor import Agent

def audit(plan, seed=7106, days=30):
    spec.set_turns_per_day(24); spec.set_episode_steps(24 * days)
    with active(plan):
        env = FastEnv(configuration={"episodeSteps": 24 * days, "turnsPerDay": 24, "startingMoney": 3000}, seed=seed); obs = env.reset()
        ag = Agent(episode_steps=24 * days, macro=plan_macro(plan)); sold = collections.Counter(); verbs = collections.Counter()
        cash_min = 1e9; unplaced_days = 0; escaped = 0; prev_alive = 0; feed = 0; rows = []
        while not env.done:
            ob = obs[0]; a = ag(ob)
            for o in a.get("market", []):
                if o[0] == "SELL": sold[o[1]] += o[2]
                if o[0] == "BUY_PRODUCT": feed += o[2]
            for v in [a.get("farmer", ["PASS"])[0]] + [h[0] for h in a.get("hands", [])]:
                verbs["move" if v in ("NORTH", "SOUTH", "EAST", "WEST") else ("pass" if v == "PASS" else "work")] += 1
            f = ob["farms"][0]; cash_min = min(cash_min, f["money"])
            if ob["hour"] == 23:
                alive = sum(1 for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("animal"))
                unplaced = sum(int(ob["private"]["shed"].get(k, 0)) for k in spec.ANIMALS)
                unplaced_days += unplaced > 0; escaped += max(0, prev_alive - alive); prev_alive = alive
                planted = sum(1 for r in f["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "PLANT")
                if ob["day"] in (3, 6, 9, 12, 18, 24, 29):
                    rows.append(f"d{ob['day']:2d} cash {f['money']:6.0f} hands {len(f['hands']):2d} quad {len(f['unlocked_quadrants'])} planted {planted:2d} animals {alive:2d} unplaced {unplaced}")
            obs, _ = env.step([a, dict(E.PASS_ACTION)])
        n = sum(verbs.values())
        return dict(final=float(env.rewards()[0]), cash_min=cash_min, unplaced_days=unplaced_days, escaped=escaped, feed=feed,
                    idle=verbs["pass"] / n, move=verbs["move"] / n, sold=dict(sorted(sold.items(), key=lambda kv: -kv[1])), rows=rows)

blocks = {
  "A: 25 tiles DEMAND + 4 animals (2g 1c 1s), 3 hands":
      Plan("DEMAND", 25, 3, 0, {"GOOSE": 2, "COW": 1, "SHEEP": 1}, 0.25, 12),
  "B: land d6, 50 tiles + 8 animals (2g 3c 3s), 6 hands from d6":
      Plan("DEMAND", (25,)*6 + (50,)*24, (3,)*6 + (6,)*24, (0,)*6 + (1,)*24, ({"GOOSE": 2, "COW": 1, "SHEEP": 1},)*6 + ({"GOOSE": 2, "COW": 3, "SHEEP": 3},)*24, 0.25, 12),
  "C: land d6/d10, 60 tiles + 15 animals (4g 6c 5s), 12 hands from d10":
      Plan("DEMAND", (25,)*6 + (40,)*4 + (60,)*20, (3,)*6 + (8,)*4 + (12,)*20, (0,)*6 + (1,)*4 + (2,)*20,
           ({"GOOSE": 2, "COW": 1, "SHEEP": 1},)*6 + ({"GOOSE": 3, "COW": 3, "SHEEP": 3},)*4 + ({"GOOSE": 4, "COW": 6, "SHEEP": 5},)*20, 0.25, 12),
}
if __name__ == '__main__':
    import argparse, statistics as st
    from dataclasses import replace
    from kagsym.tracking import Tracker
    ap = argparse.ArgumentParser(); ap.add_argument("--discount", type=float, default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    SEEDS = [7106, 7107, 7108, 7109, 7110]
    for name, pl in blocks.items():
        if a.discount is not None:
            pl = replace(pl, discount=a.discount)
        rs = [audit(pl, s) for s in SEEDS]
        tot = collections.Counter()
        for r in rs:
            tot.update(r["sold"])
        m = st.mean(r["final"] for r in rs)
        print(f"== {name}\n   final {m:,.0f} {[round(r['final']) for r in rs]} | escaped {sum(r['escaped'] for r in rs)} | unplaced-days {sum(r['unplaced_days'] for r in rs)} | feed bought {sum(r['feed'] for r in rs)} | idle {st.mean(r['idle'] for r in rs):.0%} move {st.mean(r['move'] for r in rs):.0%}")
        print("   sold (5 seeds):", dict(sorted(tot.items(), key=lambda kv: -kv[1])))
        with Tracker("evaluation", f"block/{name.split(':')[0]}", params=dict(block=name, seeds=5, discount=a.discount, tag=a.tag),
                     tags=dict(tool="expansion_ladder", yardstick="solo money, 5 seeds, plan mode"),
                     description="An expansion block of the planner's ladder, played solo under a literal plan.") as tr:
            tr.log({"block/money": m, "block/escaped": sum(r["escaped"] for r in rs), "block/unplaced_days": sum(r["unplaced_days"] for r in rs),
                    "block/feed_bought": sum(r["feed"] for r in rs), "block/idle": st.mean(r["idle"] for r in rs), "block/move": st.mean(r["move"] for r in rs)})
            tr.log({f"sold/{k}": v / 5 for k, v in tot.items()})
