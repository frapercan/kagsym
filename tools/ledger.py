#!/usr/bin/env python
"""Print the experiment ledger: every evaluation any tool has recorded.

    python tools/ledger.py [--last 30] [--kind band]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kagsym.evaluate import LEDGER  # noqa: E402
from kagsym.version import fingerprint  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--last", type=int, default=30)
    p.add_argument("--kind", default=None)
    p.add_argument("--path", default=LEDGER)
    a = p.parse_args()
    if not os.path.exists(a.path):
        print("no ledger yet")
        return
    with open(a.path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.kind:
        rows = [r for r in rows if r["kind"] == a.kind]
    current = fingerprint()
    print(f"{'time':19s} {'kind':14s} {'commit':8s} {'game':6s} {'checkpoint':28s} {'offset':10s} "
          f"{'seeds':16s} {'win':>6s} {'beaten':>6s} {'money':>9s} {'fail':>4s}")
    print("  game column: 'same' = same game code as now; a different fingerprint is another game; "
          "'*' = dirty tree, 'env' = KAG_* variables were set")
    for r in rows[-a.last:]:
        s = r["summary"]
        seeds = f"{'/'.join(r['seed_families'])}:{r['seeds'][2]}"
        fp = r.get("game_fingerprint")
        game = "same" if fp == current else (fp[:6] if fp else "?")
        flags = ("*" if r.get("dirty") else "") + ("env" if r.get("game_env") else "")
        print(f"{r['time']:19s} {r['kind'][:14]:14s} {r['commit'][:7] + flags:8s} {game:6s} "
              f"{r['checkpoint'][-28:]:28s} {str(r['offset'])[:10]:10s} {seeds:16s} "
              f"{s['win_mean']:6.3f} {s['beaten']:3d}/{s['opponents']:<3d}"
              f"{s['money_mean']:9,.0f} {s['opp_failures'] + s['failed']:4d}")


if __name__ == "__main__":
    main()
