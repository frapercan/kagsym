#!/usr/bin/env python
"""Validate a searched offset on reserved seeds, paired, and bake it if it holds.

The optimum over the seeds a search saw is a hypothesis: the best of 32
candidates on 16 seeds is inflated by selection (measured once at +$10,708).
Here the candidate offset and the checkpoint's current offset play the SAME
reserved boards and the per-board difference is what is read.

Two instruments must agree in sign before baking: money against v48 on the
reserved family, and the band win rate (the criterion) on a smaller sample.
Opposite signs mean an artefact, and that has happened (+8,436 vs -636, the
cause was a mutilated measurement loop).

Baking: a constant offset is added to the bias of `macro_mu` (zero cost at
inference); a ramp travels in the checkpoint as `offset` and the policy
applies it once a day.

    python tools/validate_offset.py runs/search/name_mu.npy runs/model.pt [--n 200]
    python tools/validate_offset.py ... --bake runs/new.pt
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.policy import Offset, load_network  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("offset", help="*_mu.npy from tools/search.py")
    p.add_argument("checkpoint")
    p.add_argument("--live", default=None, help="live dials json (default: next to the offset history)")
    p.add_argument("--n", type=int, default=200, help="reserved seeds for the money instrument")
    p.add_argument("--band-n", type=int, default=2, help="seeds per opponent for the band instrument (0 = skip)")
    p.add_argument("--procs", type=int, default=None)
    p.add_argument("--bake", help="write the validated checkpoint here")
    p.add_argument("--t-min", type=float, default=2.0)
    a = p.parse_args()

    ckpt = os.path.abspath(a.checkpoint)
    hist_path = a.offset.replace("_mu.npy", "_history.json").replace("_best.npy", "_history.json")
    live_path = a.live
    if live_path is None and os.path.exists(hist_path):
        with open(hist_path) as f:
            live = [int(i) for i in json.load(f)["live"]]
    else:
        with open(live_path or os.path.join(ROOT, "runs", "live_dials.json")) as f:
            live = [int(i) for i in json.load(f)["live"]]
    delta = np.load(a.offset)
    cand = Offset(np.asarray(delta, dtype=np.float32), live)
    base = E.PolicySpec(ckpt, offset="checkpoint")
    new = E.PolicySpec(ckpt, offset=(delta.tolist(), live))
    print(f"[validate] {a.offset} ({'ramp' if cand.is_ramp else 'constant'}, {len(live)} dials) on {ckpt}")

    seeds = S.RESERVED.seeds(a.n)
    eps_b = E.run(base, [E.V48], seeds, seats=(0,), procs=a.procs)
    eps_n = E.run(new, [E.V48], seeds, seats=(0,), procs=a.procs)
    d = E.paired(eps_n, eps_b)
    print(f"  money vs v48, {d['n']} reserved boards: {d['money_diff']:+,.0f} +- {d['money_se']:,.0f}"
          f"   t {d['t']:+.2f}   better on {100 * d['boards_better']:.0f}%")
    E.record("validate-money", new, [E.V48], seeds, eps_n, extra={"paired_vs_checkpoint": d})

    win_ok = True
    if a.band_n > 0:
        bseeds = S.RESERVED.seeds(a.band_n, offset=a.n if a.n + a.band_n <= S.RESERVED.size else 0)
        band = E.band()
        eb = E.run(base, band, bseeds, procs=a.procs)
        en = E.run(new, band, bseeds, procs=a.procs)
        w = E.paired(en, eb)
        sb, sn = E.summary(eb), E.summary(en)
        print(f"  band, {a.band_n} seeds x 2 seats x {len(band)} opponents: win {sb['win_mean']:.3f} -> "
              f"{sn['win_mean']:.3f} ({w['win_diff']:+.3f})   beaten {sb['beaten']} -> {sn['beaten']}")
        E.record("validate-band", new, band, bseeds, en, extra={"paired_vs_checkpoint": w})
        win_ok = w["win_diff"] >= 0

    if a.bake:
        if d["t"] < a.t_min or not win_ok:
            print(f"\n  NOT baked: t {d['t']:+.2f} (need >= {a.t_min}) and band sign "
                  f"{'ok' if win_ok else 'NEGATIVE'}")
            sys.exit(1)
        import torch
        _, ck = load_network(ckpt)
        ck.pop("delta_rampa", None)
        if cand.is_ramp:
            ck["offset"] = cand.to_checkpoint()
        else:
            bias = ck["sd"]["macro_mu.bias"]
            ck["sd"]["macro_mu.bias"] = bias + torch.from_numpy(cand.vector(0.0, bias.shape[0]))
            ck["offset"] = None
        ck["offset_provenance"] = {"from": os.path.relpath(ckpt, ROOT), "offset": a.offset,
                                   "money_paired": d, "commit": E.git_commit()}
        for k in ("opt", "opt_nombres"):        # a baked checkpoint is not resumed
            ck.pop(k, None)
        torch.save(ck, a.bake)
        print(f"\n  baked -> {a.bake}")


if __name__ == "__main__":
    main()
