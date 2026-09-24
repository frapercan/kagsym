#!/usr/bin/env python
"""One dial, one day, one direction, at full scale against a real opponent, paired.

The stage map (tools/stage_map.py) finds, in a solitaire, which dial on which
day is worth dollars. This checks the same change in the full game against a
public agent on paired reserved seeds, which is the only number that counts.

    python tools/probe_dial.py runs/model.pt --dial f_manure_credit --day 0 --delta 0.15 --n 60
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.macro import Macro, N_MACRO  # noqa: E402
from kagsym.policy import Policy  # noqa: E402

NAMES = [f.name for f in Macro.__dataclass_fields__.values()][:N_MACRO]
_G = {}


def _game(task):
    seed, j, delta, day = task
    import torch
    torch.set_num_threads(1)
    if "pol" not in _G:
        _G["pol"] = Policy.from_checkpoint(_G["ckpt"])
        _G["rival"] = E._opponent_callable(E.V48)
    pol, rival = _G["pol"], _G["rival"]
    cfg = {"episodeSteps": 720, "turnsPerDay": 24, "startingMoney": 3000}
    pol.configure(cfg)
    pol.reset()
    env = FastEnv(configuration=cfg, seed=seed)
    obs = env.reset()
    while not env.done:
        ob = obs[0]
        override = None
        if j is not None and ob["hour"] == 0 and int(ob["day"]) == day:
            v = pol.macro_candidates(ob, 1)[0]
            v[j] = float(np.clip(v[j] + delta, 0.0, 1.0))
            override = v
        a = pol.act(ob, macro_override=override)
        try:
            b = rival(obs[1])
        except Exception:
            b = dict(E.PASS_ACTION)
        obs, _ = env.step([a, b])
    return seed, float(env.rewards()[0]), float(env.rewards()[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--dial", required=True)
    p.add_argument("--day", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.15)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--procs", type=int, default=11)
    a = p.parse_args()
    import multiprocessing as mp
    j = NAMES.index(a.dial)
    _G["ckpt"] = os.path.abspath(a.checkpoint)
    seeds = S.RESERVED.seeds(a.n)
    with mp.get_context("fork").Pool(a.procs) as pool:
        base = dict((s, m) for s, m, _ in pool.map(_game, [(s, None, 0.0, 0) for s in seeds], chunksize=1))
        new = dict((s, m) for s, m, _ in pool.map(_game, [(s, j, a.delta, a.day) for s in seeds], chunksize=1))
    d = np.array([new[s] - base[s] for s in seeds])
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"[probe] {a.dial} {a.delta:+.2f} on day {a.day}, vs v48, {len(d)} reserved seeds paired: "
          f"{np.mean(list(base.values())):,.0f} -> {np.mean(list(new.values())):,.0f}   "
          f"{d.mean():+,.0f} +- {se:,.0f}  t {d.mean() / se:+.2f}  better on {100 * (d > 0).mean():.0f}%")


if __name__ == "__main__":
    main()
