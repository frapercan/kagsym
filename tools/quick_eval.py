#!/usr/bin/env python
"""Quick deterministic money check of one checkpoint, for the trainer's log.

Prints one machine-readable line, `EVAL {json}`, that `kagsym/cli/train.py`
parses when `--eval-cada` is set. Twelve reserved seeds resolve nothing finer
than ~5,000 $ and short evaluations are anticorrelated with the truth
(-0.46 measured), so this is a log line, never a selection criterion.

    python tools/quick_eval.py runs/model.pt.ultimo --n 12
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--opponent", default=E.V48.name)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    t0 = time.time()
    spec = E.PolicySpec(os.path.abspath(a.checkpoint))
    opp = E.public(a.opponent)
    eps = E.run(spec, [opp], S.RESERVED.seeds(a.n), seats=(0,), procs=a.procs)
    s = E.summary(eps)
    d = {"money": s["money_mean"], "se": s["money_se"], "win": s["win_mean"],
         "opponent": float(sum(e.opp_money for e in eps if not e.error) / max(1, s["episodes"])),
         "n": s["episodes"], "opp_failures": s["opp_failures"], "seconds": round(time.time() - t0, 1)}
    d["margin_pct"] = 100.0 * (d["money"] - d["opponent"]) / d["opponent"] if d["opponent"] else float("nan")
    print("EVAL " + json.dumps(d), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(d, f)


if __name__ == "__main__":
    main()
