#!/usr/bin/env python
"""Which macro dials actually change the game.

Measured before: 18 of the 67 dials produced identical episodes at both
extremes, because the network's value map replaced the heuristic values they
used to scale and nobody retired them. A dead dimension in a search is not
neutral, it is noise that dilutes the live ones.

Method: for each dial, play the same boards with the dial pushed to both
extremes (an offset of -/+8 in logit space, i.e. sigmoid ~0.0003 and 0.9997)
on top of what the checkpoint emits. If our money and the opponent's are
identical to the cent on every board, the dial does not exist.

    python tools/live_dials.py runs/model.pt [--n 3] [--out runs/live_dials.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.macro import Macro, N_MACRO  # noqa: E402
from kagsym.policy import Offset, load_network  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTREMES = (-8.0, 8.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out", default=os.path.join(ROOT, "runs", "live_dials.json"))
    p.add_argument("--procs", type=int, default=None)
    a = p.parse_args()
    ckpt = os.path.abspath(a.checkpoint)
    _, ck = load_network(ckpt)
    stored = Offset.from_checkpoint(ck)
    base_a = stored.vector(0.0, N_MACRO) if stored else np.zeros(N_MACRO)
    base_b = (stored.vector(1.0, N_MACRO) - base_a) if stored else np.zeros(N_MACRO)
    all_dials = list(range(N_MACRO))
    seeds = S.DIALS.seeds(a.n)
    todo = []
    for d in all_dials:
        for v in EXTREMES:
            aa, bb = base_a.copy(), base_b.copy()
            aa[d] += v
            spec = E.PolicySpec(ckpt, offset=(np.concatenate([aa, bb]).tolist(), all_dials))
            todo += [(spec, E.V48, s, 0) for s in seeds]
    print(f"[live] {ckpt}: {N_MACRO} dials x 2 extremes x {a.n} seeds = {len(todo)} episodes",
          flush=True)
    res = {}
    for task, ep in E.run_tasks(todo, procs=a.procs, progress=True):
        off = task[0].offset[0]
        d = int(np.argmax(np.abs(np.asarray(off[:N_MACRO]) - base_a)))
        v = float(off[d] - base_a[d])
        res[(d, v > 0, ep.seed)] = (ep.money, ep.opp_money)
    names = [f.name for f in Macro.__dataclass_fields__.values()][:N_MACRO]
    live, dead = [], []
    for d in all_dials:
        diffs = []
        same = 0
        for s in seeds:
            lo, hi = res.get((d, False, s)), res.get((d, True, s))
            if lo is None or hi is None or np.isnan(lo[0]) or np.isnan(hi[0]):
                continue
            same += (lo == hi)
            diffs.append(abs(hi[0] - lo[0]))
        (dead if same == len(seeds) else live).append([d, float(np.mean(diffs)) if diffs else 0.0])
    live.sort(key=lambda x: -x[1])
    print(f"\n  LIVE {len(live)}/{N_MACRO}   DEAD {len(dead)}: "
          f"{[names[d] if d < len(names) else d for d, _ in dead]}")
    print("\n  strongest (|money(+8) - money(-8)|, mean per seed):")
    for d, diff in live[:12]:
        print(f"    {d:2d} {names[d] if d < len(names) else '':18s} {diff:10,.0f}")
    with open(a.out, "w") as f:
        json.dump({"checkpoint": os.path.relpath(ckpt, ROOT), "seeds": seeds,
                   "commit": E.git_commit(), "live": [d for d, _ in live],
                   "effect": {str(d): x for d, x in live}, "dead": [d for d, _ in dead]},
                  f, indent=1)
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
