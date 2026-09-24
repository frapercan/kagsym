#!/usr/bin/env python
"""Cross-entropy search over the macro offset of a checkpoint.

Why search and not gradient. Measured: PPO does not improve any checkpoint
(-$6,787 from the best one, +$904 from one $10,000 worse), and the macro head
moves at a KL of 5e-4 per update. What the gradient cannot move, a search can:
the offsets found this way produced every dollar of the last two days.

What is searched. An additive offset in logit space on the LIVE dials of the
macro (`runs/live_dials.json`, from `tools/live_dials.py`), either constant
(`a`) or a ramp (`a + b*progress`, `--ramp`). The search starts from the
offset the checkpoint already carries, so generation 0 reproduces it exactly.

How it reads. Every candidate of a generation plays the SAME seeds (paired),
the seeds rotate each generation, and the centre `mu` is itself a candidate:
the mean of the population is `mu` plus noise and reads far below it where
perturbing is expensive. "best" is the expected maximum of ~30 noisy draws and
is never a result. The number that matters is CENTRE minus BASE, and even that
is a hypothesis until `tools/validate_offset.py` confirms it on reserved seeds.

Objectives:
  money   mean money against one opponent (default v48), seat 0
  margin  mean (ours - theirs) against a sample of the band, both seats.
          The surrogate for the criterion while every board is lost: a win
          rate of 0.00 everywhere has no gradient, a margin does, and it
          rewards sinking their market as much as growing ours.
  win     mean win rate against a sample of the band (the ladder criterion)

    python tools/search.py runs/model.pt --out runs/search/name [--ramp] [--gens 40]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.macro import N_MACRO  # noqa: E402
from kagsym.policy import Offset, load_network  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def live_dials(path):
    with open(path) as f:
        d = json.load(f)
    return [int(i) for i in d["live"]]


def initial_offset(ckpt, live, ramp):
    """The offset the checkpoint carries, expressed on `live` (padded for a ramp).

    Dials the stored offset touches are kept live even if the probe called
    them dead, so generation 0 reproduces the checkpoint exactly.
    """
    _, ck = load_network(ckpt)
    stored = Offset.from_checkpoint(ck)
    if stored is not None:
        extra = [i for i in stored.live if i not in live]
        if extra:
            print(f"[search] stored offset touches dials the probe called dead: "
                  f"{extra}; kept live", flush=True)
            live = sorted(set(live) | set(extra))
    n = len(live)
    mu = np.zeros(2 * n if ramp else n)
    if stored is not None:
        full_a = stored.vector(0.0, N_MACRO)
        full_b = stored.vector(1.0, N_MACRO) - full_a
        mu[:n] = full_a[live]
        if ramp:
            mu[n:] = full_b[live]
        elif np.abs(full_b).max() > 0:
            raise SystemExit("checkpoint carries a ramp; search it with --ramp")
    return mu, live


def score(episodes, objective):
    """money: our cash. margin: ours minus theirs (continuous, sabotage-aware,
    the surrogate for wins while every board is lost). win: the criterion."""
    if objective == "money":
        return float(np.nanmean([e.money for e in episodes]))
    if objective == "margin":
        return float(np.nanmean([e.money - e.opp_money for e in episodes]))
    per = E.by_opponent(episodes)
    return float(np.mean([v["win"] for v in per.values()])) if per else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True, help="prefix for mu/best/history files")
    p.add_argument("--objective", choices=["money", "margin", "win"], default="margin")
    p.add_argument("--opponent", default=E.V48.name, help="money objective opponent")
    p.add_argument("--band-sample", type=int, default=8, help="win objective: opponents per generation")
    p.add_argument("--live", default=os.path.join(ROOT, "runs", "live_dials.json"))
    p.add_argument("--ramp", action="store_true")
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--pop", type=int, default=32)
    p.add_argument("--elite", type=int, default=8)
    p.add_argument("--seeds", type=int, default=16, help="common seeds per generation")
    p.add_argument("--sigma", type=float, default=0.6)
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--procs", type=int, default=None)
    p.add_argument("--rng", type=int, default=20260924)
    a = p.parse_args()

    ckpt = os.path.abspath(a.checkpoint)
    mu, live = initial_offset(ckpt, live_dials(a.live), a.ramp)
    D = len(live) * (2 if a.ramp else 1)
    base = mu.copy()
    sd = np.full(D, a.sigma)
    rng = np.random.default_rng(a.rng)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    names = E.public_names()
    print(f"[search] {ckpt}  {len(live)} live dials  {'ramp' if a.ramp else 'constant'} "
          f"({D} dims)  objective {a.objective}  pop {a.pop}  elite {a.elite}  "
          f"{a.seeds} seeds/gen  {a.gens} gens", flush=True)
    history = []
    best, best_val = mu.copy(), -np.inf
    for g in range(a.gens):
        t0 = time.time()
        seeds = S.SEARCH.seeds(a.seeds, offset=(g * a.seeds) % (S.SEARCH.size - a.seeds))
        if a.objective == "money":
            opps, seats = [E.public(a.opponent)], (0,)
        else:
            pick = rng.choice(len(names), size=min(a.band_sample, len(names)), replace=False)
            opps, seats = [E.public(names[i]) for i in sorted(pick)], (0, 1)
        cand = [base.copy(), mu.copy()] + [rng.normal(mu, sd) for _ in range(a.pop - 2)]
        specs = [E.PolicySpec(ckpt, offset=(tuple(float(x) for x in c), tuple(live)))
                 for c in cand]
        todo = [(specs[i], o, s, seat, i) for i in range(len(cand))
                for o in opps for s in seeds for seat in seats]
        by_cand = {i: [] for i in range(len(cand))}
        for task, ep in E.run_tasks(todo, procs=a.procs):
            by_cand[task[4]].append(ep)
        scores = np.array([score(by_cand[i], a.objective) for i in range(len(cand))])
        order = np.argsort(-scores)
        elite = [cand[i] for i in order[:a.elite]]
        mu = np.mean(elite, axis=0)
        sd = np.std(elite, axis=0) + a.sigma_floor
        if scores[order[0]] > best_val:
            best_val, best = scores[order[0]], cand[order[0]].copy()
        row = {"gen": g, "base": float(scores[0]), "centre": float(scores[1]),
               "best": float(scores[order[0]]), "mean": float(scores.mean()),
               "sd": float(sd.mean()), "seconds": time.time() - t0,
               "opponents": [o.label for o in opps]}
        history.append(row)
        np.save(a.out + "_mu.npy", mu)
        np.save(a.out + "_best.npy", best)
        with open(a.out + "_history.json", "w") as f:
            json.dump({"checkpoint": os.path.relpath(ckpt, ROOT), "live": live,
                       "ramp": a.ramp, "objective": a.objective, "args": vars(a),
                       "commit": E.git_commit(), "history": history}, f, indent=1)
        fmt = "{:9.3f}" if a.objective == "win" else "{:9,.0f}"
        print(f"  gen {g:3d}  base " + fmt.format(row["base"]) + "  CENTRE "
              + fmt.format(row["centre"]) + f" ({row['centre'] - row['base']:+.3g})  best "
              + fmt.format(row["best"]) + "  mean " + fmt.format(row["mean"])
              + f"  |sd| {row['sd']:.3f}  {row['seconds']:.0f}s", flush=True)
    print(f"\n  -> {a.out}_mu.npy   validate on RESERVED seeds with tools/validate_offset.py;"
          f" nothing here is a result", flush=True)


if __name__ == "__main__":
    main()
