"""The wins panel: a plan against several public agents, both seats, 20 seeds. Logged to MLflow.

    python tools/panel.py runs/blocks/P9_best_plan.json --id P9
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from multiprocessing import get_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kagsym import evaluate as E, spec  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.plan import Plan, active, plan_macro  # noqa: E402
from kagsym.seeds import RESERVED  # noqa: E402
from kagsym.symbolic.executor import Agent  # noqa: E402
from kagsym.tracking import Tracker  # noqa: E402

PANEL_PREFIXES = ("testkaggriculture-hamburger", "kaggriculture-finding-condit", "kaggriculture-v53", "v48-fast")
_PLAN = None


def duel(rival_name: str, seed: int, seat: int):
    spec.set_turns_per_day(24)
    spec.set_episode_steps(720)
    rival = E._opponent_callable(E.public(rival_name))
    other = 1 - seat
    env = FastEnv(configuration={"episodeSteps": 720, "turnsPerDay": 24, "startingMoney": 3000}, seed=seed)
    obs = env.reset()
    with active(_PLAN):
        ag = Agent(episode_steps=720, macro=plan_macro(_PLAN))
        while not env.done:
            acts = [None, None]
            acts[seat] = ag(obs[seat])
            acts[other] = rival(obs[other])
            obs, _ = env.step(acts)
    r = env.rewards()
    return float(r[seat]), float(r[other])


def main():
    global _PLAN
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--id", default=None)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    pid = a.id or os.path.splitext(os.path.basename(a.plan))[0]
    _PLAN = Plan(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(a.plan)).items()})
    names = {f[:-3] for f in os.listdir(os.path.join(ROOT, "agents_pub")) if f.endswith(".py")}
    panel = [next(n for n in sorted(names) if n.startswith(p)) for p in PANEL_PREFIXES if any(n.startswith(p) for n in names)]
    seeds = RESERVED.seeds(a.seeds)
    print(f"{'rival':32s} {'seat':>4s} | {'wins':>5s} {'ours':>8s} {'theirs':>8s}")
    with Tracker("evaluation", f"panel/{pid}", params=dict(plan=a.plan, seeds=a.seeds, rivals=",".join(panel)),
                 tags=dict(tool="panel", yardstick="wins vs live public agents, both seats")) as tr, get_context("fork").Pool(a.procs) as pool:
        for rn in panel:
            for seat in (0, 1):
                res = pool.starmap(duel, [(rn, s, seat) for s in seeds])
                w = sum(x > y for x, y in res)
                print(f"{rn[:32]:32s} {seat:>4d} | {w:>2d}/{a.seeds} {st.mean(x for x, y in res):8,.0f} {st.mean(y for x, y in res):8,.0f}", flush=True)
                key = rn.split("-")[0][:12]
                tr.log({f"panel/wins_{key}_seat{seat}": w, f"panel/money_{key}_seat{seat}": st.mean(x for x, y in res),
                        f"panel/rival_money_{key}_seat{seat}": st.mean(y for x, y in res)})


if __name__ == "__main__":
    main()
