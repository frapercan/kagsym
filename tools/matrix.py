#!/usr/bin/env python
"""Head-to-head matrix between our own checkpoints.

Why our own and not the public band: the band is all above us, so it cannot
tell two of our checkpoints apart. Our snapshots are at our level by
construction. Two things an aggregate hides:

  * cycles: A beats B, B beats C, C beats A make any scalar ranking (Elo
    included) a fiction; measured among strong public agents by others;
  * who teaches: the useful training opponent is the one you are near 50%
    against, p(1-p) maximal.

Deterministic, both seats, paired seeds from the MATRIX family.

    python tools/matrix.py runs/A.pt runs/B.pt runs/C.pt [--n 4]
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--n", type=int, default=4, help="seeds per pair (x2 seats)")
    p.add_argument("--procs", type=int, default=None)
    a = p.parse_args()
    paths = [os.path.abspath(c) for c in a.checkpoints]
    seeds = S.MATRIX.seeds(a.n)
    todo = []
    for i, j in itertools.combinations(range(len(paths)), 2):
        spec = E.PolicySpec(paths[i])
        todo += [(spec, E.checkpoint(paths[j]), s, seat, (i, j))
                 for s in seeds for seat in (0, 1)]
    res = {}
    for task, ep in E.run_tasks(todo, procs=a.procs, progress=True):
        res.setdefault(task[4], []).append(ep)
    names = [os.path.basename(c) for c in paths]
    n = len(paths)
    win = np.full((n, n), np.nan)
    money = np.full((n, n), np.nan)
    for (i, j), eps in res.items():
        eps = [e for e in eps if not e.error]
        w = float(np.mean([e.win for e in eps]))
        win[i, j], win[j, i] = w, 1.0 - w
        money[i, j] = float(np.mean([e.money - e.opp_money for e in eps]))
        money[j, i] = -money[i, j]
    print(f"\nwin rate of ROW against COLUMN ({a.n} seeds x 2 seats)\n")
    print(" " * 22 + "".join(f"{nm[:10]:>11s}" for nm in names))
    for i in range(n):
        print(f"{names[i][:20]:20s}  " + "".join(
            "      -    " if i == j else f"{win[i, j]:11.2f}" for j in range(n)))
    print("\nmean margin of ROW over COLUMN ($)\n")
    for i in range(n):
        print(f"{names[i][:20]:20s}  " + "".join(
            "      -    " if i == j else f"{money[i, j]:11,.0f}" for j in range(n)))
    cycles = [(names[i], names[j], names[k]) for i, j, k in itertools.permutations(range(n), 3)
              if i < j and i < k and win[i, j] > 0.5 and win[j, k] > 0.5 and win[k, i] > 0.5]
    if cycles:
        print("\nnon-transitive triples:", cycles)


if __name__ == "__main__":
    main()
