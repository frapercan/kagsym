#!/usr/bin/env python
"""The criterion: a checkpoint against the whole band of public agents.

The ladder scores a Bradley-Terry fit over WINS against the other submissions,
not mean money. Measured: 25% more money against v48 bought two points of win
rate. So the number that decides is the mean win rate over the band and how
many opponents we beat; money is a diagnostic column.

Both seats are played on every seed, so board luck and seat cancel in paired
comparisons. Opponents that throw are played as PASS and COUNTED: a broken
opponent is a weak opponent, and hiding that inflates the win rate.

    python tools/band.py runs/model.pt [--n 6] [--seeds reserved] [--no-offset]
    python tools/band.py A.pt --against B.pt      # paired difference A - B
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402


def evaluate(path, seeds, procs, offset, limit, save):
    spec = E.PolicySpec(os.path.abspath(path), offset=offset)
    opponents = E.band(limit)
    eps = E.run(spec, opponents, seeds, seats=(0, 1), procs=procs, progress=True)
    if save:
        E.save_episodes(eps, save)
    E.record("band", spec, opponents, seeds, eps)
    return spec, opponents, eps


def report(spec, eps):
    per = E.by_opponent(eps)
    rows = sorted(per.items(), key=lambda kv: -kv[1]["win"])
    print(f"\n{spec.label}   {len(rows)} opponents\n")
    print(f"{'opponent':46s} {'win':>5s} {'ours':>9s} {'theirs':>9s} {'fail':>5s}")
    for name, r in rows:
        print(f"  {name[:44]:44s} {r['win']:5.2f} {r['money']:9,.0f} "
              f"{r['opp_money']:9,.0f} {r['opp_failures']:5d}")
    s = E.summary(eps)
    print(f"\n  WIN MEAN {s['win_mean']:.3f}   beaten (>0.5): {s['beaten']} of "
          f"{s['opponents']}   contested (0.35-0.65): {s['contested']}")
    print(f"  money {s['money_mean']:,.0f} +- {s['money_se']:,.0f}   "
          f"opponent failures {s['opp_failures']}   our failures {s['failed']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--against", help="second checkpoint for a paired comparison")
    p.add_argument("--n", type=int, default=6, help="seeds per opponent (x2 seats)")
    p.add_argument("--seeds", default="reserved", help="seed family")
    p.add_argument("--offset-from", type=int, default=0, help="offset inside the family")
    p.add_argument("--limit", type=int, default=0, help="first N opponents only")
    p.add_argument("--procs", type=int, default=None)
    p.add_argument("--no-offset", action="store_true",
                   help="ignore the macro offset stored in the checkpoint")
    p.add_argument("--save", help="write per-episode records to this JSON")
    a = p.parse_args()
    seeds = S.family(a.seeds).seeds(a.n, a.offset_from)
    offset = None if a.no_offset else "checkpoint"
    spec, opps, eps = evaluate(a.checkpoint, seeds, a.procs, offset, a.limit, a.save)
    report(spec, eps)
    if a.against:
        spec_b, _, eps_b = evaluate(a.against, seeds, a.procs, offset, a.limit,
                                    a.save and a.save + ".against")
        report(spec_b, eps_b)
        d = E.paired(eps, eps_b)
        print(f"\n  PAIRED {spec.label} - {spec_b.label}:  win {d['win_diff']:+.3f}   "
              f"money {d['money_diff']:+,.0f} +- {d['money_se']:,.0f}  t {d['t']:+.2f}   "
              f"n {d['n']}   better on {100 * d['boards_better']:.0f}% of boards")


if __name__ == "__main__":
    main()
