#!/usr/bin/env python
"""The gate of one rung of the ladder of universes.

A rung is the first `k` days of a 30-day game. This plays a searched offset
and the checkpoint's own offset on the SAME reserved boards, against the
rung's opponents, both seats, cutting at day `k` while the policy values a
30-day game, and reads the position value at the cut (exact liquidation with
the full horizon), paired. The rung's yardstick is the opponents' position at
the same day: they are schedules, so it is fixed.

    python tools/rung.py runs/search/exp002_opening8 runs/model.pt --days 8 [--n 30]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.policy import Offset  # noqa: E402

DEFAULT_OPPONENTS = "v48-fast-routes,the-2945-farm-96-vs-the-top-10-public-bots"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("search_dir")
    p.add_argument("checkpoint")
    p.add_argument("--days", type=int, required=True, help="the rung: days played")
    p.add_argument("--horizon", type=int, default=30, help="days the policy values")
    p.add_argument("--n", type=int, default=30, help="reserved seeds")
    p.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    p.add_argument("--which", default="offset.npz", choices=["offset.npz", "best.npz"])
    p.add_argument("--gain-min", type=float, default=3000.0)
    p.add_argument("--t-min", type=float, default=3.0)
    p.add_argument("--procs", type=int, default=None)
    a = p.parse_args()

    ckpt = os.path.abspath(a.checkpoint)
    off, digest, meta = Offset.load(os.path.join(a.search_dir, a.which))
    if digest != E.file_digest(ckpt):
        raise SystemExit(f"{a.which} was searched on digest {digest}; {ckpt} has {E.file_digest(ckpt)}")
    base = E.PolicySpec(ckpt)
    new = E.PolicySpec(ckpt, offset=(tuple(float(x) for x in off.delta), tuple(off.live), off.ramp, off.until_day, off.from_day))
    world = {"hours": 24, "days": a.days, "agent_horizon_days": a.horizon, "value_horizon_days": a.horizon}
    opps = [E.public(n.strip()) for n in a.opponents.split(",")]
    seeds = S.RESERVED.seeds(a.n)
    todo = [(sp, o, s, seat, tag, world) for tag, sp in (("base", base), ("new", new))
            for o in opps for s in seeds for seat in (0, 1)]
    res = {"base": {}, "new": {}}
    for task, ep in E.run_tasks(todo, procs=a.procs, progress=True):
        if not ep.error:
            res[task[4]][(ep.opponent, ep.seed, ep.seat)] = ep
    keys = sorted(set(res["base"]) & set(res["new"]))
    print(f"\n[rung {a.days}d] {os.path.relpath(a.search_dir)} on {os.path.basename(ckpt)}: "
          f"{len(keys)} paired boards, {len(opps)} opponents\n")
    rows = []
    for name, opp in [(o.label, o.label) for o in opps] + [("all", None)]:
        ks = [k for k in keys if opp is None or k[0] == opp]
        if not ks:
            continue
        gain = np.array([(res["new"][k].value - res["new"][k].opp_value)
                         - (res["base"][k].value - res["base"][k].opp_value) for k in ks])
        ours0 = np.mean([res["base"][k].value for k in ks])
        ours1 = np.mean([res["new"][k].value for k in ks])
        theirs = np.mean([res["new"][k].opp_value for k in ks])
        se = gain.std(ddof=1) / np.sqrt(len(gain)) if len(gain) > 1 else float("nan")
        t = gain.mean() / se if se and se > 0 else float("nan")
        rows.append((name, len(ks), ours0, ours1, theirs, gain.mean(), se, t, (gain > 0).mean()))
        print(f"  {name[:28]:28s} n {len(ks):3d}  ours {ours0:7,.0f} -> {ours1:7,.0f}  theirs {theirs:7,.0f}  "
              f"ratio {ours0 / theirs:.2f} -> {ours1 / theirs:.2f}  gain {gain.mean():+6,.0f} +- {se:,.0f}  "
              f"t {t:+.1f}  better {100 * (gain > 0).mean():.0f}%")
    name, n, o0, o1, th, g, se, t, better = rows[-1]
    ok = np.isfinite(t) and g >= a.gain_min and t >= a.t_min and n == len(seeds) * len(opps) * 2
    print(f"\n  rung gate: {'PASSED' if ok else 'REFUSED'} (gain {g:+,.0f} >= {a.gain_min:,.0f}, "
          f"t {t:+.1f} >= {a.t_min}, boards {n}/{len(seeds) * len(opps) * 2})")
    E.record(f"rung-{a.days}d", new, opps, seeds,
             [res["new"][k] for k in keys],
             extra={"rung": {"days": a.days, "horizon": a.horizon, "ours_before": o0, "ours_after": o1,
                             "theirs": th, "gain": g, "se": se, "t": t, "better": better, "passed": bool(ok)}})
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
