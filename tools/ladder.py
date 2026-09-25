"""The ladder protocol: one horizon at a time, searched, audited, compared, written.

    python tools/ladder.py --from 1 --to 30 --procs 8

For every horizon d (24 hours a day, passive opponent, 3,000 $):
  1. search   the staged search of tools/plan_daysearch.py on SEARCH-side
              reserved seeds 7101-7105: constant-plan grid, then day-by-day
              refinement with exact rollouts; records -> runs/ladder/daysearch_<d>d.jsonl
  2. check    the searched schedule of seed 7101 replayed on 5 UNSEEN seeds (7106-7110)
  3. audit    tools/plan_audit.py on seed 7101: produced, sold, lost, crew turns
  4. compare  the previous rung's schedule padded by one day, on the same seeds:
              what the extra day buys, and which plan fields changed
  5. write    one section per rung in runs/ladder/report.md and a row in
              runs/ladder/ladder.jsonl
The code is frozen for the whole climb (the report carries the git commit):
if the search or the executor changes, the ladder starts again.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from multiprocessing import get_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from kagsym.plan import Plan, play_plan  # noqa: E402
from kagsym.seeds import RESERVED  # noqa: E402
import plan_daysearch as D  # noqa: E402
from plan_audit import audit  # noqa: E402

SEARCH_SEEDS = RESERVED.seeds(5)          # 7101-7105: what the search sees
UNSEEN_SEEDS = RESERVED.seeds(5, 5)       # 7106-7110: the report's check
FIELDS = ("crop", "tiles", "hands", "load", "water_last", "selling", "land")


def schedule_of(records, seed: int, days: int) -> Plan:
    rows = sorted((r for r in records if r["seed"] == seed), key=lambda r: r["state"]["day"])
    assert len(rows) == days, (len(rows), days)
    return Plan(crop=tuple(r["plan"]["crop"] for r in rows), tiles=tuple(r["plan"]["tiles"] for r in rows),
                hands=tuple(r["plan"]["hands"] for r in rows), land=tuple(r["plan"]["land"] for r in rows),
                animals=0, selling=tuple(r["plan"]["selling"] for r in rows),
                load=tuple(r["plan"]["load"] for r in rows), water_last=tuple(r["plan"]["water_last"] for r in rows))


def padded(plan: Plan, days: int) -> Plan:
    def pad(v):
        v = list(v) if isinstance(v, (tuple, list)) else [v]
        return tuple(v + [v[-1]] * (days - len(v)))
    return Plan(crop=pad(plan.crop), tiles=pad(plan.tiles), hands=pad(plan.hands), land=pad(plan.land),
                animals=0, selling=pad(plan.selling), load=pad(plan.load), water_last=pad(plan.water_last))


def _mean(plan: Plan, seeds, days: int, pool) -> float:
    vals = pool.starmap(play_plan, [(plan, s, days) for s in seeds])
    return float(st.mean(vals))


def describe(plan: Plan, days: int) -> str:
    lines = []
    for d in range(days):
        c = plan.crop_on(d)
        c = "+".join(f"{k}:{v:.0%}" for k, v in c.items()) if isinstance(c, dict) else c
        lines.append(f"  day {d:2d}: {c:<22s} tiles {plan.tiles_on(d):3d} hands {plan.hands_on(d)} load {plan.load_on(d):2d} "
                     f"water_last {plan.water_last_on(d)} selling {plan.selling_on(d):.2f} land {plan.land_on(d)}")
    return "\n".join(lines)


def changed_fields(prev: Plan | None, cur: Plan, days: int) -> list:
    if prev is None:
        return list(FIELDS)
    out = []
    p = padded(prev, days)
    for f in FIELDS:
        a = [getattr(p, f + "_on")(d) for d in range(days)]
        b = [getattr(cur, f + "_on")(d) for d in range(days)]
        if a != b:
            out.append(f)
    return out


def run_rung(days: int, procs: int, prev: Plan | None, commit: str) -> Plan:
    t0 = time.time()
    out = f"runs/ladder/daysearch_{days}d.jsonl"
    if os.path.exists(out):
        os.remove(out)
    cmd = [sys.executable, "tools/plan_daysearch.py", "--days", str(days), "--seeds", "5", "--procs", str(procs), "--out", out]
    if prev is not None:                       # the previous rung's schedule competes on day 0
        json.dump(padded(prev, days).to_dict(), open("runs/ladder/prev_plan.json", "w"))
        cmd += ["--start-json", "runs/ladder/prev_plan.json"]
    log = open(f"runs/ladder/daysearch_{days}d.log", "w")
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    records = [json.loads(l) for l in open(out)]
    search_mean = st.mean(max(r["best"] for r in records if r["seed"] == s) for s in SEARCH_SEEDS)
    plan = schedule_of(records, SEARCH_SEEDS[0], days)
    grid_best = json.load(open(f"runs/ladder/plan_{days}d.json"))[0]
    with get_context("fork").Pool(procs) as pool:
        unseen = _mean(plan, UNSEEN_SEEDS, days, pool)
        prev_unseen = _mean(padded(prev, days), UNSEEN_SEEDS, days, pool) if prev is not None else None
        grid_unseen = _mean(D._as_plan(grid_best["plan"]), UNSEEN_SEEDS, days, pool)
    a = audit(plan, days, SEARCH_SEEDS[0])
    produced = sum(a["sold"].values()) + a["left_on_tiles"] + a["carried_at_close"] + a["shed_at_close"] + a["overflow_discarded"]
    turns = a["turns"]; n = max(1, sum(turns.values()))
    row = dict(days=days, commit=commit, search_mean=search_mean, unseen=unseen, grid_unseen=grid_unseen,
               prev_padded_unseen=prev_unseen, plan=plan.to_dict(), changed=changed_fields(prev, plan, days),
               sold=sum(a["sold"].values()), produced=produced, lost=dict(tiles=a["left_on_tiles"], carried=a["carried_at_close"],
               shed=a["shed_at_close"], overflow=a["overflow_discarded"]), turns=turns, steps=a["steps"],
               seconds=round(time.time() - t0))
    with open("runs/ladder/ladder.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    gain = "" if prev_unseen is None else f"; the extra day buys {unseen - prev_unseen:+,.0f} over the previous rung padded ({prev_unseen:,.0f})"
    with open("runs/ladder/report.md", "a") as f:
        f.write(f"\n## {days} days  ({time.strftime('%Y-%m-%d %H:%M')}, {row['seconds']} s, commit {commit})\n\n")
        f.write(f"Search (seeds 7101-7105): **{search_mean:,.0f} $**. Unseen seeds 7106-7110: **{unseen:,.0f} $** "
                f"(best constant plan {grid_unseen:,.0f}){gain}.\n\n")
        f.write(f"Fields that changed against the previous rung: {', '.join(row['changed']) or 'none'}.\n\n")
        f.write(f"Audit on seed 7101: produced {produced} units, sold {row['sold']}; lost on tiles {a['left_on_tiles']}, "
                f"carried at close {a['carried_at_close']}, shed overflow {a['overflow_discarded']}. Crew turns: work "
                f"{turns.get('work', 0) / n:.0%}, move {turns.get('move', 0) / n:.0%} (needed {a['steps']['needed']}, excess "
                f"{a['steps']['excess']}, orphan {a['steps']['orphan']}), pass {turns.get('pass', 0) / n:.0%}.\n\n")
        f.write("```\n" + describe(plan, days) + "\n```\n")
    print(f"[{days:2d} d] search {search_mean:,.0f}  unseen {unseen:,.0f}  grid {grid_unseen:,.0f}"
          + ("" if prev_unseen is None else f"  prev+1 {prev_unseen:,.0f}") + f"  changed {row['changed']}  {row['seconds']}s", flush=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="lo", type=int, default=1)
    ap.add_argument("--to", dest="hi", type=int, default=30)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    os.makedirs("runs/ladder", exist_ok=True)
    if a.lo == 1:
        for f in ("runs/ladder/report.md", "runs/ladder/ladder.jsonl"):
            if os.path.exists(f):
                os.remove(f)
        with open("runs/ladder/report.md", "w") as f:
            f.write(f"# The ladder, one day at a time\n\nCode frozen at commit {commit}; 24 hours a day, passive opponent, "
                    f"3,000 $. Search seeds 7101-7105, unseen 7106-7110.\n")
    prev = None
    if a.lo > 1:
        rows = [json.loads(l) for l in open("runs/ladder/ladder.jsonl")]
        prev = D._as_plan(next(r["plan"] for r in rows if r["days"] == a.lo - 1))
    for d in range(a.lo, a.hi + 1):
        prev = run_rung(d, a.procs, prev, commit)
    print("LADDER DONE", flush=True)


if __name__ == "__main__":
    main()
