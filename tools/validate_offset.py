#!/usr/bin/env python
"""Validate a searched offset on reserved seeds, paired, and bake it if it holds.

The optimum over the seeds a search saw is a hypothesis: the best of 32
candidates on 16 seeds is inflated by selection (measured once at +$10,708).
Here the candidate offset and the checkpoint's current offset play the SAME
boards and the per-board difference is what is read.

Two instruments must agree before baking: money against v48 on a seed family
(paired), and the band win rate (the criterion) on reserved seeds (paired).

The gate is positive: it requires a finite t, every expected board played,
and a non-degenerate comparison. `nan < 2.0` is False in Python, and a gate
written as "reject if t < 2" once let a blind measurement (+0 +- 0, sd 0)
bake a checkpoint.

The offset comes from a search directory (`offset.npz`, self-describing:
delta, live dials, ramp, checkpoint digest) and the directory must carry
`done.json`, the mark that the search completed.

    python tools/validate_offset.py runs/search/name runs/model.pt [--family clean --n 200]
    python tools/validate_offset.py ... --bake runs/new.pt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.policy import Offset, load_network  # noqa: E402
from kagsym.version import provenance  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bake_allowed(money: dict, band: dict | None, n_money: int, n_band: int,
                 t_min: float = 2.0, win_min: float = 0.0) -> tuple[bool, str]:
    """The gate, written positively so that NaN and missing boards refuse."""
    t = money.get("t", float("nan"))
    if not np.isfinite(t):
        return False, f"money t is not finite ({t}): blind or degenerate instrument"
    if money.get("degenerate"):
        return False, "every board gives zero difference: the two sides are the same policy"
    if money["n"] != n_money:
        return False, f"money instrument played {money['n']} of {n_money} boards"
    if t < t_min:
        return False, f"money t {t:+.2f} < {t_min}"
    if band is not None:
        if band["n"] != n_band:
            return False, f"band instrument played {band['n']} of {n_band} boards"
        wd = band.get("win_diff", float("nan"))
        if not np.isfinite(wd):
            return False, "band win difference is not finite"
        if wd < win_min:
            return False, f"band win difference {wd:+.3f} < {win_min:+.3f}"
    return True, "passed"


def load_search(search_dir: str, checkpoint: str, allow_unfinished: bool, which: str):
    off, digest, meta = Offset.load(os.path.join(search_dir, which))
    if not allow_unfinished and not os.path.exists(os.path.join(search_dir, "done.json")):
        raise SystemExit(f"{search_dir} has no done.json: the search did not finish "
                         f"(--allow-unfinished to validate an intermediate centre)")
    actual = E.file_digest(checkpoint)
    if digest != actual:
        raise SystemExit(f"{which} was searched on checkpoint digest {digest}; "
                         f"{checkpoint} has digest {actual}")
    return off, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("search_dir", help="directory written by tools/search.py")
    p.add_argument("checkpoint")
    p.add_argument("--which", default="offset.npz", choices=["offset.npz", "best.npz"])
    p.add_argument("--allow-unfinished", action="store_true")
    p.add_argument("--family", default="reserved", help="seed family for the money instrument")
    p.add_argument("--n", type=int, default=200, help="seeds for the money instrument")
    p.add_argument("--band-n", type=int, default=6,
                   help="reserved seeds per opponent for the band instrument (0 = skip)")
    p.add_argument("--procs", type=int, default=None)
    p.add_argument("--bake", help="write the validated checkpoint here")
    p.add_argument("--t-min", type=float, default=2.0)
    p.add_argument("--win-min", type=float, default=0.0)
    a = p.parse_args()

    ckpt = os.path.abspath(a.checkpoint)
    cand, meta = load_search(os.path.abspath(a.search_dir), ckpt, a.allow_unfinished, a.which)
    base = E.PolicySpec(ckpt, offset="checkpoint")
    new = E.PolicySpec(ckpt, offset=(tuple(float(x) for x in cand.delta), tuple(cand.live), cand.ramp,
                                     cand.until_day, cand.from_day))
    print(f"[validate] {a.search_dir}/{a.which} ({'ramp' if cand.ramp else 'constant'}, "
          f"{len(cand.live)} dials{f', opening day < {cand.until_day}' if cand.until_day else ''}, "
          f"generation {meta.get('generation')}) on {ckpt}; validation plays the FULL game")

    fam = S.family(a.family)
    seeds = fam.seeds(a.n)
    eps_b = E.run(base, [E.V48], seeds, seats=(0,), procs=a.procs)
    eps_n = E.run(new, [E.V48], seeds, seats=(0,), procs=a.procs)
    d = E.paired(eps_n, eps_b)
    print(f"  money vs v48, {d['n']}/{d['n_expected']} {fam.name} boards: {d['money_diff']:+,.0f} "
          f"+- {d['money_se']:,.0f}   t {d['t']:+.2f}   better on {100 * d['boards_better']:.0f}%"
          f"{'   DEGENERATE' if d['degenerate'] else ''}")
    E.record("validate-money", new, [E.V48], seeds, eps_n, extra={"paired_vs_checkpoint": d})

    w, n_band = None, 0
    if a.band_n > 0:
        bseeds = S.RESERVED.seeds(a.band_n)
        band = E.band()
        eb = E.run(base, band, bseeds, procs=a.procs)
        en = E.run(new, band, bseeds, procs=a.procs)
        w = E.paired(en, eb)
        n_band = w["n_expected"]
        sb, sn = E.summary(eb), E.summary(en)
        print(f"  band, {a.band_n} seeds x 2 seats x {len(band)} opponents: win {sb['win_mean']:.3f} -> "
              f"{sn['win_mean']:.3f} ({w['win_diff']:+.3f})   beaten {sb['beaten']} -> {sn['beaten']}"
              f"   boards {w['n']}/{w['n_expected']}")
        E.record("validate-band", new, band, bseeds, en, extra={"paired_vs_checkpoint": w})

    ok, why = bake_allowed(d, w, d["n_expected"], n_band, a.t_min, a.win_min)
    print(f"\n  gate: {'PASSED' if ok else 'REFUSED'} ({why})")
    if a.bake:
        if not ok:
            sys.exit(1)
        import torch
        _, ck = load_network(ckpt)
        ck.pop("delta_rampa", None)
        if cand.until_day is not None or cand.from_day is not None:
            # A windowed offset is stored as a schedule next to the stored one:
            # the policy applies the window's offset inside it, the stored one
            # outside. (Policy.from_checkpoint reads `offset`; `offset_windows`
            # is applied by tools that know about phases; until the policy
            # reads schedules, refuse to bake a windowed offset.)
            raise SystemExit("baking a windowed offset is not supported yet: the deployed policy "
                             "reads a single stored offset (see docs/DEBT.md)")
        if cand.ramp:
            ck["offset"] = cand.to_checkpoint()
        else:
            bias = ck["sd"]["macro_mu.bias"]
            ck["sd"]["macro_mu.bias"] = bias + torch.from_numpy(cand.vector(0.0, bias.shape[0]))
            ck["offset"] = None
        prov = provenance()
        ck["offset_provenance"] = {"from": os.path.relpath(ckpt, ROOT),
                                   "search": os.path.relpath(a.search_dir, ROOT), "which": a.which,
                                   "money_paired": d, "band_paired": w, "provenance": prov}
        for k in ("opt", "opt_nombres"):        # a baked checkpoint is not resumed
            ck.pop(k, None)
        ck["fingerprint"] = prov["game_fingerprint"]
        tmp = a.bake + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, a.bake)
        print(f"  baked -> {a.bake}")


if __name__ == "__main__":
    main()
