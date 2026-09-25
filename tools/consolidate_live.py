"""Consolidate a portfolio search's top ten LIVE against a rival on 20 seeds:
money, the rival's money and WINS (the ladder's score). Logged to MLflow.

    python tools/consolidate_live.py runs/blocks/P8.json runs/blocks/P8_consolidated.json 8 testkaggriculture-hamburger
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from multiprocessing import get_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import plan_blocks as B  # noqa: E402
from kagsym import evaluate as E, spec  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.plan import active, plan_macro  # noqa: E402
from kagsym.seeds import RESERVED  # noqa: E402
from kagsym.symbolic.executor import Agent  # noqa: E402
from kagsym.tracking import Tracker  # noqa: E402

RIVAL = "v48-fast-routes"


def duel(plan, seed):
    spec.set_turns_per_day(24)
    spec.set_episode_steps(720)
    rival = E._opponent_callable(E.public(RIVAL))
    env = FastEnv(configuration={"episodeSteps": 720, "turnsPerDay": 24, "startingMoney": 3000}, seed=seed)
    obs = env.reset()
    with active(plan):
        ag = Agent(episode_steps=720, macro=plan_macro(plan))
        while not env.done:
            obs, _ = env.step([ag(obs[0]), rival(obs[1])])
    r = env.rewards()
    return float(r[0]), float(r[1])


def main():
    global RIVAL
    src, out, procs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    RIVAL = sys.argv[4] if len(sys.argv) > 4 else RIVAL
    run_id = os.path.splitext(os.path.basename(src))[0]
    res = json.load(open(src))
    tops = [t[0] for t in res["top"]]
    seeds = RESERVED.seeds(20)
    tr = Tracker("evaluation", f"consolidate/{run_id}", params=dict(source=src, rival=RIVAL, seeds=20, seat=0, candidates=len(tops)),
                 tags=dict(tool="consolidate_live", yardstick="live20: wins and money vs the rival, seeds 7101-7120, seat 0"),
                 description="The top ten of a portfolio search played live against a public rival on 20 seeds.").start()
    with get_context("fork").Pool(procs) as pool:
        rows = []
        for i, p in enumerate(tops):
            pairs = pool.starmap(duel, [(B.to_plan(p), s) for s in seeds])
            vals = [a for a, b in pairs]
            wins = sum(a > b for a, b in pairs)
            rows.append((wins, st.mean(vals), st.stdev(vals) / 20 ** .5, i, p, vals, st.mean(b for a, b in pairs)))
            print(f"  #{i}: wins {wins}/20  money {rows[-1][1]:,.0f} (se {rows[-1][2]:,.0f})  rival {rows[-1][6]:,.0f}", flush=True)
            tr.log({"candidate/wins": wins, "candidate/money": rows[-1][1], "candidate/rival_money": rows[-1][6]}, step=i)
    rows.sort(key=lambda r: (-r[0], -r[1]))
    json.dump([dict(params=p, wins20=w, mean20=m, se=se, rival_mean=rm, values=vals) for w, m, se, i, p, vals, rm in rows],
              open(out, "w"), indent=1)
    w, m, se, i, p, vals, rm = rows[0]
    print(f"best live vs {RIVAL}: wins {w}/20 money {m:,.0f}", p)
    tr.log({"best/wins": w, "best/money": m, "best/money_se": se, "best/rival_money": rm})
    tr.log_params({f"best/{k}": v for k, v in p.items()})
    tr.log_json(rows[0][4], "best_params.json")
    tr.end()


if __name__ == "__main__":
    main()
