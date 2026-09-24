#!/usr/bin/env python
"""Deterministic money against ONE opponent, paired across checkpoints.

A diagnostic, not the criterion (that is `tools/band.py`). Useful because a
single strong opponent on 200 paired seeds resolves a few thousand dollars,
which the band cannot with six seeds per opponent.

    python tools/evaluate.py runs/A.pt [runs/B.pt ...] [--n 200] [--seeds reserved]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seeds", default="reserved")
    p.add_argument("--offset-from", type=int, default=0)
    p.add_argument("--opponent", default=E.V48.name)
    p.add_argument("--cap", type=int, default=None)
    p.add_argument("--seats", default="0", help="'0', '1' or '0,1'")
    p.add_argument("--procs", type=int, default=None)
    p.add_argument("--no-offset", action="store_true")
    a = p.parse_args()
    seeds = S.family(a.seeds).seeds(a.n, a.offset_from)
    seats = tuple(int(x) for x in a.seats.split(","))
    opp = E.public(a.opponent, a.cap)
    offset = None if a.no_offset else "checkpoint"
    results = []
    for path in a.checkpoints:
        spec = E.PolicySpec(os.path.abspath(path), offset=offset)
        eps = E.run(spec, [opp], seeds, seats=seats, procs=a.procs, progress=True)
        E.record("evaluate", spec, [opp], seeds, eps)
        s = E.summary(eps)
        print(f"\n  {spec.label:40s} vs {opp.label}: {s['money_mean']:10,.0f} +- {s['money_se']:,.0f}"
              f"   win {s['win_mean']:.3f}   n {s['episodes']}   opp failures {s['opp_failures']}")
        results.append((spec, eps))
    for spec, eps in results[1:]:
        d = E.paired(eps, results[0][1])
        print(f"  PAIRED {spec.label} - {results[0][0].label}: {d['money_diff']:+,.0f} "
              f"+- {d['money_se']:,.0f}  t {d['t']:+.2f}  win {d['win_diff']:+.3f}  "
              f"better on {100 * d['boards_better']:.0f}% of boards")


if __name__ == "__main__":
    main()
