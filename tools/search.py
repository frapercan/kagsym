#!/usr/bin/env python
"""Cross-entropy search over the macro offset of a checkpoint.

Why search and not gradient. Measured: PPO does not improve any checkpoint
(-$6,787 from the best one, +$904 from one $10,000 worse), and the macro head
moves at a KL of 5e-4 per update. What the gradient cannot move, a search can.

What is searched. An additive offset in logit space on the LIVE dials of the
macro (`<checkpoint>.dials.json`, from `tools/live_dials.py`, whose digest
must match the checkpoint), either constant (`a`) or a ramp
(`a + b*progress`, `--ramp`). The search starts from the offset the
checkpoint already carries, so generation 0 reproduces it exactly.

How it reads. Every candidate of a generation plays the SAME seeds (paired),
the seeds rotate each generation, and the centre `mu` is itself a candidate:
the mean of the population is `mu` plus noise and reads far below it where
perturbing is expensive. "best" is the expected maximum of ~30 noisy draws and
is never a result. The number that matters is CENTRE minus BASE, and even that
is a hypothesis until `tools/validate_offset.py` confirms it on reserved seeds.

A candidate with ANY failed episode is disqualified, not averaged over the
boards that worked: `nanmean` once pointed the selection pressure at the
candidate that broke on the hard boards.

Outputs, in a directory that must not exist beforehand (`--out`):
  meta.json      written first: checkpoint digest, live dials, ramp, gens,
                 provenance (commit, dirty tree, fingerprints, KAG_* env), pid
  checkpoint.pt  a frozen copy of the checkpoint the search plays
  offset.npz     the centre, self-describing (delta, live, ramp, digest);
                 rewritten atomically every generation
  best.npz       the best single candidate seen (never a result)
  history.json   one row per generation
  done.json      written only when the run completes

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
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.macro import N_MACRO  # noqa: E402
from kagsym.policy import Offset, load_network  # noqa: E402
from kagsym.version import provenance  # noqa: E402
from live_dials import load_dials  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def initial_offset(ckpt, live, ramp):
    """The offset the checkpoint carries, on `live` (padded for a ramp).

    Dials the stored offset touches are kept live even if the probe called
    them dead, so generation 0 reproduces the checkpoint exactly.
    """
    _, ck = load_network(ckpt)
    stored = Offset.from_checkpoint(ck)
    if stored is not None:
        extra = [i for i in stored.live if i not in live]
        if extra:
            print(f"[search] stored offset touches dials the probe called dead: {extra}; kept live",
                  flush=True)
            live = sorted(set(live) | set(extra))
        if stored.ramp and not ramp:
            raise SystemExit("checkpoint carries a ramp; search it with --ramp")
    n = len(live)
    mu = np.zeros(2 * n if ramp else n)
    if stored is not None:
        full_a = stored.vector(0.0, N_MACRO)
        full_b = stored.vector(1.0, N_MACRO) - full_a
        mu[:n] = full_a[live]
        if ramp:
            mu[n:] = full_b[live]
    return mu, live


def score(episodes, objective):
    """money: our cash. margin: ours minus theirs. value: our final position
    valued with the full horizon, minus theirs (the opening objective).
    win: the criterion. Any failed episode disqualifies the candidate."""
    if not episodes or any(e.error for e in episodes):
        return -np.inf
    if objective == "money":
        return float(np.mean([e.money for e in episodes]))
    if objective == "margin":
        return float(np.mean([e.money - e.opp_money for e in episodes]))
    if objective == "value":
        return float(np.mean([e.value - e.opp_value for e in episodes]))
    per = E.by_opponent(episodes)
    return float(np.mean([v["win"] for v in per.values()])) if per else -np.inf


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True, help="output DIRECTORY; must not exist (or --force)")
    p.add_argument("--force", action="store_true", help="remove an existing --out directory first")
    p.add_argument("--objective", choices=["money", "margin", "win", "value"], default="margin")
    p.add_argument("--days", type=int, default=30, help="days played per episode (a universe of the ladder)")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--agent-horizon", type=int, default=30,
                   help="days our policy believes the game lasts (the first k days of a 30-day game)")
    p.add_argument("--until-day", type=int, default=None,
                   help="the offset applies only while day < until_day (an opening offset)")
    p.add_argument("--from-day", type=int, default=None,
                   help="the offset applies only from this day on (a closing offset)")
    p.add_argument("--opponents", default=None,
                   help="comma-separated public agents (or 'passive' for solitaire) for value/margin; "
                        "default: a band sample per generation")
    p.add_argument("--opponent", default=E.V48.name, help="money objective opponent")
    p.add_argument("--band-sample", type=int, default=6, help="margin/win: opponents per generation")
    p.add_argument("--dials", default=None, help="live-dial json (default: <checkpoint>.dials.json)")
    p.add_argument("--ramp", action="store_true")
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--pop", type=int, default=32)
    p.add_argument("--elite", type=int, default=8)
    p.add_argument("--seeds", type=int, default=16, help="common seeds per generation")
    p.add_argument("--sigma", type=float, default=0.6)
    p.add_argument("--sigma-floor", type=float, default=0.05)
    p.add_argument("--procs", type=int, default=None)
    p.add_argument("--rng", type=int, default=20260924)
    p.add_argument("--experiment", default=None, help="preregistration id, e.g. EXP-001")
    a = p.parse_args()

    out = os.path.abspath(a.out)
    if os.path.exists(out):
        if not a.force:
            raise SystemExit(f"{out} exists; a search never overwrites another's files (use --force)")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=False)
    # The checkpoint is frozen inside the run directory: nothing can rewrite
    # it under the search's feet, and every artefact carries the copy's digest.
    ckpt_src = os.path.abspath(a.checkpoint)
    ckpt = os.path.join(out, "checkpoint.pt")
    shutil.copyfile(ckpt_src, ckpt)
    digest = E.file_digest(ckpt)
    live = load_dials(ckpt_src, a.dials)
    mu, live = initial_offset(ckpt, live, a.ramp)
    D = len(live) * (2 if a.ramp else 1)
    base = mu.copy()
    sd = np.full(D, a.sigma)
    rng = np.random.default_rng(a.rng)
    names = E.public_names()
    world = {"hours": a.hours, "days": a.days, "agent_horizon_days": a.agent_horizon,
             "value_horizon_days": a.agent_horizon if a.objective == "value" else None}
    fixed_opps = ([E.passive() if n.strip() == "passive" else E.public(n.strip())
                   for n in a.opponents.split(",")] if a.opponents else None)
    meta = {"checkpoint": os.path.relpath(ckpt_src, ROOT), "checkpoint_digest": digest,
            "live": live, "ramp": bool(a.ramp), "until_day": a.until_day, "from_day": a.from_day, "world": world,
            "dims": D, "objective": a.objective,
            "args": vars(a), "provenance": provenance(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write_json(os.path.join(out, "meta.json"), meta)
    print(f"[search] {ckpt_src} ({digest})  {len(live)} live dials  "
          f"{'ramp' if a.ramp else 'constant'} ({D} dims)  objective {a.objective}  "
          f"pop {a.pop}  elite {a.elite}  {a.seeds} seeds/gen  {a.gens} gens  -> {out}", flush=True)
    if meta["provenance"]["game_env"]:
        print(f"[search] WARNING game environment variables set: {meta['provenance']['game_env']}",
              flush=True)

    from kagsym.tracking import Tracker, context_tags, describe
    tracker = Tracker(
        "search", os.path.relpath(out, os.path.join(ROOT, "runs")),   # e.g. search/exp002 or pipeline/smoke/search
        params={**vars(a), "live_dials": len(live), "dims": D},
        tags=context_tags("search", checkpoint=ckpt_src, objective=a.objective,
                          seed_family="search", experiment=a.experiment,
                          universe=f"{a.hours}hx{a.days}d/{a.agent_horizon}d",
                          opponent=(a.opponent if a.objective == "money"
                                    else f"band-sample({a.band_sample})")),
        description=describe([
            f"CEM over the macro offset of {os.path.basename(ckpt_src)}: "
            f"{'ramp a+b*progress' if a.ramp else 'constant'}, {len(live)} live dials, {D} dims"
            f"{f', opening only (day < {a.until_day})' if a.until_day else ''}.",
            f"Universe: {a.hours}h x {a.days}d played, the agent values a {a.agent_horizon}-day game.",
            f"Objective `{a.objective}`; pop {a.pop}, elite {a.elite}, {a.seeds} common seeds per generation.",
            "Read search/centre_minus_base: the centre and the base play the same boards. "
            "search/best is the expected maximum of noisy draws and is never a result.",
            f"Artefacts in {os.path.relpath(out, ROOT)}; validated by tools/validate_offset.py.",
        ])).start()

    history = []
    best, best_val = mu.copy(), -np.inf
    for g in range(a.gens):
        t0 = time.time()
        seeds = S.SEARCH.seeds(a.seeds, offset=(g * a.seeds) % (S.SEARCH.size - a.seeds))
        if fixed_opps is not None:
            opps, seats = fixed_opps, (0, 1)
        elif a.objective == "money":
            opps, seats = [E.public(a.opponent)], (0,)
        else:
            pick = rng.choice(len(names), size=min(a.band_sample, len(names)), replace=False)
            opps, seats = [E.public(names[i]) for i in sorted(pick)], (0, 1)
        cand = [base.copy(), mu.copy()] + [rng.normal(mu, sd) for _ in range(a.pop - 2)]
        specs = [E.PolicySpec(ckpt, offset=(tuple(float(x) for x in c), tuple(live), bool(a.ramp), a.until_day, a.from_day))
                 for c in cand]
        todo = [(specs[i], o, s, seat, i, world) for i in range(len(cand))
                for o in opps for s in seeds for seat in seats]
        by_cand = {i: [] for i in range(len(cand))}
        for task, ep in E.run_tasks(todo, procs=a.procs):
            by_cand[task[4]].append(ep)
        scores = np.array([score(by_cand[i], a.objective) for i in range(len(cand))])
        disqualified = int(np.sum(~np.isfinite(scores)))
        if not np.isfinite(scores[0]) or not np.isfinite(scores[1]):
            raise SystemExit(f"generation {g}: the base or the centre failed an episode; "
                             f"the search cannot continue on a broken loop")
        order = np.argsort(-scores)
        elite = [cand[i] for i in order[:a.elite]]
        mu = np.mean(elite, axis=0)
        sd = np.std(elite, axis=0) + a.sigma_floor
        if scores[order[0]] > best_val:
            best_val, best = scores[order[0]], cand[order[0]].copy()
        finite = scores[np.isfinite(scores)]
        row = {"gen": g, "base": float(scores[0]), "centre": float(scores[1]),
               "best": float(scores[order[0]]), "mean": float(finite.mean()),
               "sd": float(sd.mean()), "disqualified": disqualified,
               "seconds": time.time() - t0, "seeds": [int(seeds[0]), int(seeds[-1])],
               "opponents": [o.label for o in opps]}
        history.append(row)
        tracker.log({"search/base": row["base"], "search/centre": row["centre"],
                     "search/centre_minus_base": row["centre"] - row["base"],
                     "search/best": row["best"], "search/population_mean": row["mean"],
                     "search/sigma": row["sd"], "search/disqualified": disqualified,
                     "search/seconds": row["seconds"]}, step=g)
        Offset(mu, live, bool(a.ramp), a.until_day, a.from_day).save(os.path.join(out, "offset.npz"), digest, generation=g, world=world)
        Offset(best, live, bool(a.ramp), a.until_day, a.from_day).save(os.path.join(out, "best.npz"), digest, generation=g, world=world)
        _write_json(os.path.join(out, "history.json"), history)
        fmt = "{:9.3f}" if a.objective == "win" else "{:9,.0f}"
        print(f"  gen {g:3d}  base " + fmt.format(row["base"]) + "  CENTRE "
              + fmt.format(row["centre"]) + f" ({row['centre'] - row['base']:+.3g})  best "
              + fmt.format(row["best"]) + "  mean " + fmt.format(row["mean"])
              + f"  |sd| {row['sd']:.3f}  dq {disqualified}  {row['seconds']:.0f}s", flush=True)
    _write_json(os.path.join(out, "done.json"),
                {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "generations": len(history)})
    for name in ("offset.npz", "history.json", "meta.json"):
        tracker.log_artifact(os.path.join(out, name))
    tracker.end()
    print(f"\n  -> {out}/offset.npz   validate on reserved seeds with tools/validate_offset.py;"
          f" nothing here is a result", flush=True)


if __name__ == "__main__":
    main()
