#!/usr/bin/env python
"""Which macro dials actually change the game, for ONE checkpoint.

Measured before: 18 of the 67 dials produced identical episodes at both
extremes, because the network's value map replaced the heuristic values they
used to scale and nobody retired them. A dead dimension in a search is not
neutral, it is noise that dilutes the live ones.

Method: for each dial, play the same boards with the dial pushed to both
extremes (an offset of -/+8 in logit space, i.e. sigmoid ~0.0003 and 0.9997)
on top of what the checkpoint emits. If our money and the opponent's are
identical to the cent on every board, the dial does not exist.

The result is written NEXT TO THE CHECKPOINT (`<checkpoint>.dials.json`) with
the checkpoint's digest, and the search refuses a dial list whose digest is
not the checkpoint it is searching. A shared global file once let one probe
silently redefine the search space of every later search.

    python tools/live_dials.py runs/model.pt [--n 3]
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
from kagsym.version import provenance  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTREMES = (-8.0, 8.0)


def dials_path(checkpoint: str) -> str:
    return os.path.abspath(checkpoint) + ".dials.json"


def load_dials(checkpoint: str, path: str | None = None) -> list[int]:
    """The live dials of `checkpoint`, refusing a file made for another one."""
    path = path or dials_path(checkpoint)
    if not os.path.exists(path):
        raise SystemExit(f"no live-dial list for {checkpoint}: run tools/live_dials.py first "
                         f"(expected {path})")
    with open(path) as f:
        d = json.load(f)
    digest = E.file_digest(checkpoint)
    if d.get("checkpoint_digest") != digest:
        raise SystemExit(f"{path} was made for checkpoint digest {d.get('checkpoint_digest')}, "
                         f"but {checkpoint} has digest {digest}: re-run tools/live_dials.py")
    return sorted(int(i) for i in d["live"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out", default=None, help="default: <checkpoint>.dials.json")
    p.add_argument("--procs", type=int, default=None)
    a = p.parse_args()
    ckpt = os.path.abspath(a.checkpoint)
    out = a.out or dials_path(ckpt)
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
            spec = E.PolicySpec(ckpt, offset=(tuple(float(x) for x in np.concatenate([aa, bb])),
                                              tuple(all_dials), True))
            todo += [(spec, E.V48, s, 0, (d, v > 0)) for s in seeds]
    print(f"[live] {ckpt}: {N_MACRO} dials x 2 extremes x {a.n} seeds = {len(todo)} episodes",
          flush=True)
    res = {}
    failed = 0
    for task, ep in E.run_tasks(todo, procs=a.procs, progress=True):
        if ep.error:
            failed += 1
            continue
        d, hi = task[4]
        res[(d, hi, ep.seed)] = (ep.money, ep.opp_money)
    if failed:
        raise SystemExit(f"{failed} episodes failed; a dial list from a partial probe is not a dial list")
    names = [f.name for f in Macro.__dataclass_fields__.values()][:N_MACRO]
    live, dead, effect = [], [], {}
    for d in all_dials:
        same = sum(res[(d, False, s)] == res[(d, True, s)] for s in seeds)
        diffs = [abs(res[(d, True, s)][0] - res[(d, False, s)][0]) for s in seeds]
        if same == len(seeds):
            dead.append(d)
        else:
            live.append(d)
            effect[str(d)] = float(np.mean(diffs))
    print(f"\n  LIVE {len(live)}/{N_MACRO}   DEAD {len(dead)}: "
          f"{[names[d] if d < len(names) else d for d in dead]}")
    print("\n  strongest (|money(+8) - money(-8)|, mean per seed):")
    for d, diff in sorted(effect.items(), key=lambda kv: -kv[1])[:12]:
        d = int(d)
        print(f"    {d:2d} {names[d] if d < len(names) else '':18s} {diff:10,.0f}")
    with open(out, "w") as f:
        json.dump({"checkpoint": os.path.relpath(ckpt, ROOT), "checkpoint_digest": E.file_digest(ckpt),
                   "seeds": seeds, "provenance": provenance(),
                   "live": live, "dead": dead, "effect": effect}, f, indent=1)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
