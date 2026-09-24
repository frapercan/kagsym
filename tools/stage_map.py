#!/usr/bin/env python
"""The stage-consistency map: day x dial x direction -> dollars, on the exact engine.

Along the policy's own solitaire trajectory (passive opponent, so a rollout
is exactly what would happen), at every day boundary and for every macro
dial, the dial is pushed by +delta and by -delta for that day only and the
game is rolled out to the end. The gain over the policy's own choice says,
to the dollar, which decision is inconsistent with the stage and in which
direction. The trajectory itself is never changed: the map is about what
the policy actually plays.

    python tools/stage_map.py runs/model.pt --days 8 [--delta 0.15] [--seed 7401] [--procs 8]
"""
import argparse
import copy
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.macro import Macro, N_MACRO  # noqa: E402
from kagsym.policy import Policy  # noqa: E402

CASH = 3000
_G = {}


def _rollout(pol, env, first_macro):
    env2, pol2 = env.clone(), copy.deepcopy(pol)
    forced = first_macro
    while not env2.done:
        a = pol2.act(env2.observations()[0], macro_override=forced)
        forced = None
        env2.step([a, dict(E.PASS_ACTION)])
    return float(env2.rewards()[0])


def _probe(task):
    day, j, sign, delta = task
    pol, env, mean = _G["pol"], _G["env"], _G["mean"]
    v = mean.copy()
    v[j] = float(np.clip(v[j] + sign * delta, 0.0, 1.0))
    return day, j, sign, _rollout(pol, env, v) - _G["base"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--seed", type=int, default=7401)
    p.add_argument("--delta", type=float, default=0.15)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    import multiprocessing as mp
    names = [f.name for f in Macro.__dataclass_fields__.values()][:N_MACRO]
    cfg = {"episodeSteps": 24 * a.days, "turnsPerDay": 24, "startingMoney": CASH}
    pol = Policy.from_checkpoint(a.checkpoint)
    pol.configure(cfg)
    pol.reset()
    env = FastEnv(configuration=cfg, seed=a.seed)
    obs = env.reset()
    rows = []
    while not env.done:
        ob = obs[0]
        if ob["hour"] == 0:
            mean = pol.macro_candidates(ob, 1)[0]
            _G.update(pol=pol, env=env, mean=mean)
            _G["base"] = _rollout(pol, env, mean)
            tasks = [(int(ob["day"]), j, s, a.delta) for j in range(N_MACRO) for s in (+1, -1)]
            with mp.get_context("fork").Pool(a.procs) as pool:
                res = pool.map(_probe, tasks, chunksize=4)
            rows += [{"day": d, "dial": names[j], "index": j, "direction": s, "gain": g} for d, j, s, g in res]
            best = max(res, key=lambda r: r[3])
            print(f"  day {best[0]:2d}  base {_G['base']:8,.0f} $   best single change "
                  f"{names[best[1]]}{'+' if best[2] > 0 else '-'}{a.delta}: {best[3]:+,.0f} $", flush=True)
        a_ = pol.act(ob)
        obs, _ = env.step([a_, dict(E.PASS_ACTION)])
    final = float(env.rewards()[0])
    out = a.out or os.path.join("runs", "ablation", f"stage_map_{a.days}d_{a.seed}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    gains = sorted(rows, key=lambda r: -r["gain"])[:15]
    print(f"\n[stage-map] {a.days}d seed {a.seed}: policy final {final:,.0f} $; strongest single-day changes:")
    for r in gains:
        print(f"  day {r['day']:2d}  {r['dial']:20s} {'+' if r['direction'] > 0 else '-'}{a.delta}   {r['gain']:+,.0f} $")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
