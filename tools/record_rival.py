"""Record a public rival's actions per turn on given seeds, to replay it as an
opponent at zero cost (a public agent costs 2 s a game, our executor 0.7).

    python tools/record_rival.py v48-fast-routes --seeds 20 --out runs/rivals/v48.json

The replay is an approximation: the rival reacted to the prices of THAT
game. Measure the error against the live rival before trusting it."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from kagsym import evaluate as E, spec  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.plan import Plan, active, plan_macro  # noqa: E402
from kagsym.seeds import RESERVED  # noqa: E402
from kagsym.symbolic.executor import Agent  # noqa: E402


def record(name: str, seed: int, plan: Plan, days: int = 30, hours: int = 24) -> list:
    rival = E._opponent_callable(E.public(name))
    steps = hours * days
    spec.set_turns_per_day(hours)
    spec.set_episode_steps(steps)
    acts = []
    with active(plan):
        env = FastEnv(configuration={"episodeSteps": steps, "turnsPerDay": hours, "startingMoney": 3000}, seed=seed)
        obs = env.reset()
        ag = Agent(episode_steps=steps, macro=plan_macro(plan))
        while not env.done:
            a1 = rival(obs[1])
            acts.append(a1)
            obs, _ = env.step([ag(obs[0]), a1])
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--plan", default="runs/blocks/portfolio_v1_best_plan.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"runs/rivals/{a.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd = json.load(open(a.plan))
    plan = Plan(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pd.items()})
    tapes = {}
    for s in RESERVED.seeds(a.seeds):
        tapes[str(s)] = record(a.name, s, plan)
        print(f"seed {s}: {len(tapes[str(s)])} turns", flush=True)
    json.dump(tapes, open(out, "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
