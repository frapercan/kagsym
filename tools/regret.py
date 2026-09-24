#!/usr/bin/env python
"""Solitaire regret: the policy against a KNOWN optimum, to the dollar.

No opponent, no reference agent: a passive rival, the exact engine, and a
scenario whose optimal outcome is known or computable. Two numbers per
scenario:

  regret       optimum - what the policy made
  consistency  potential at day t - cash realised at the end (0 = the
               potential predicted the executor's own conversion exactly)

Scenarios:
  idle-<k>     24h x k days with k so short that nothing pays: the optimum is
               exactly the starting cash. Any spend is a decision inconsistent
               with the stage. Measured 2026-09-24 on partida_v5: 3, 9 and 22
               hands hired in 2, 3 and 5-day games with nothing for them to do.

    python tools/regret.py runs/model.pt [--scenarios idle-2,idle-3,idle-5] [--seeds 3]
"""
import argparse
import collections
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.policy import Policy  # noqa: E402

CASH = 3000


def play_solitaire(policy_path: str, days: int, seed: int, hours: int = 24):
    """One solitaire game; returns final cash, per-day potentials and the
    market orders issued, for the regret and consistency readings."""
    from kagsym import spec
    from kagsym.potential import liquidation
    cfg = {"episodeSteps": hours * days, "turnsPerDay": hours, "startingMoney": CASH}
    pol = Policy.from_checkpoint(policy_path)
    pol.configure(cfg)
    pol.reset()
    env = FastEnv(configuration=cfg, seed=seed)
    obs = env.reset()
    orders = collections.Counter()
    potentials = {}
    while not env.done:
        ob = obs[0]
        if ob["hour"] == 0:
            potentials[int(ob["day"])] = float(liquidation(ob, 0, ob.get("private")))
        a = pol.act(ob)
        for o in a.get("market", []):
            orders[o[0]] += 1
        if a.get("farmer", ["PASS"])[0] != "PASS":
            orders["farmer:" + a["farmer"][0]] += 1
        obs, _ = env.step([a, dict(E.PASS_ACTION)])
    return float(env.rewards()[0]), potentials, dict(orders)


def scenario_idle(policy_path: str, days: int, seeds):
    rows = []
    for s in seeds:
        final, pots, orders = play_solitaire(policy_path, days, s)
        rows.append({"seed": s, "final": final, "optimum": float(CASH), "regret": CASH - final,
                     "consistency": {d: p - final for d, p in pots.items()}, "orders": orders})
    return rows


def _rollout_from(pol, env, first_macro):
    """Play a cloned game to the end: today with `first_macro`, then the
    policy as it is. The policy is deep-copied WITH its executor state
    (destinations, turns-per-tile): a fresh executor once made the null case
    miss by $389 at day 7 and every seed collapse to the same number."""
    import copy
    env2 = env.clone()
    pol2 = copy.deepcopy(pol)
    forced = first_macro
    while not env2.done:
        ob = env2.observations()[0]
        a = pol2.act(ob, macro_override=forced)
        forced = None
        env2.step([a, dict(E.PASS_ACTION)])
    return float(env2.rewards()[0])


def _rollout_task(task):
    pol, env, cand = task
    return _rollout_from(pol, env, cand)


def oracle_solitaire(policy_path: str, days: int, seed: int, k: int = 16, hours: int = 24,
                     procs: int = 1):
    """The policy improved by exact rollout search over its own macro
    Gaussian at every day boundary (k candidates, index 0 = the mean). With
    the exact engine and a passive opponent a rollout is not a prediction.
    Returns (oracle money, plain money, days on which the search won,
    null-case error). The null case is the mean's rollout on day 0, which
    must equal the plain policy's final money exactly."""
    import torch
    cfg = {"episodeSteps": hours * days, "turnsPerDay": hours, "startingMoney": CASH}
    plain, _, _ = play_solitaire(policy_path, days, seed, hours)
    pol = Policy.from_checkpoint(policy_path)
    pol.configure(cfg)
    pol.reset()
    env = FastEnv(configuration=cfg, seed=seed)
    obs = env.reset()
    gen = torch.Generator().manual_seed(seed)
    wins, null_err, forced = [], None, None
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(procs) if procs > 1 else None
    while not env.done:
        ob = obs[0]
        if ob["hour"] == 0:
            cands = pol.macro_candidates(ob, k, gen)
            if pool is not None:
                vals = pool.map(_rollout_task, [(pol, env, c) for c in cands], chunksize=1)
            else:
                vals = [_rollout_from(pol, env, c) for c in cands]
            if ob["day"] == 0:
                null_err = vals[0] - plain
            i = max(range(len(vals)), key=lambda j: vals[j])
            if i != 0:
                wins.append((int(ob["day"]), vals[i] - vals[0]))
            forced = cands[i]
        a = pol.act(ob, macro_override=forced)
        forced = None
        obs, _ = env.step([a, dict(E.PASS_ACTION)])
    if pool is not None:
        pool.close()
        pool.join()
    return float(env.rewards()[0]), plain, wins, null_err


def oracle_global_solitaire(policy_path: str, days: int, seed: int, pop: int = 32, gens: int = 3,
                            sigma0: float = 0.8, hours: int = 24, procs: int = 1):
    """A GLOBAL per-day oracle: at every day boundary a small CEM over the
    macro (pop x gens candidates around the policy's mean in logit space,
    wide sigma, exact rollouts to the end) chooses the day's vector. Unlike
    the local oracle it can leave the policy's neighbourhood: it measures
    what the executor's interface can express with near-perfect daily
    decisions, not what the policy almost does."""
    cfg = {"episodeSteps": hours * days, "turnsPerDay": hours, "startingMoney": CASH}
    plain, _, _ = play_solitaire(policy_path, days, seed, hours)
    pol = Policy.from_checkpoint(policy_path)
    pol.configure(cfg)
    pol.reset()
    env = FastEnv(configuration=cfg, seed=seed)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(procs) if procs > 1 else None
    wins, forced, null_err = [], None, None
    while not env.done:
        ob = obs[0]
        if ob["hour"] == 0:
            mean_vec = pol.macro_candidates(ob, 1)[0]
            mu = np.log(np.clip(mean_vec, 1e-4, 1 - 1e-4)) - np.log1p(-np.clip(mean_vec, 1e-4, 1 - 1e-4))
            sd = np.full(len(mean_vec), sigma0)
            best_vec, best_val, base_val = mean_vec, -np.inf, None
            for g in range(gens):
                cands = [mean_vec] if g == 0 else []
                cands += [1.0 / (1.0 + np.exp(-(mu + sd * rng.standard_normal(len(mean_vec))))) for _ in range(pop - len(cands))]
                tasks = [(pol, env, c.astype(np.float32)) for c in cands]
                vals = pool.map(_rollout_task, tasks, chunksize=1) if pool else [_rollout_task(t) for t in tasks]
                if g == 0:
                    base_val = vals[0]
                    if ob["day"] == 0:
                        null_err = vals[0] - plain
                order = np.argsort(vals)[::-1]
                elite = [cands[i] for i in order[:max(2, pop // 4)]]
                if vals[order[0]] > best_val:
                    best_val, best_vec = vals[order[0]], cands[order[0]]
                el = np.array([np.log(np.clip(e, 1e-4, 1 - 1e-4)) - np.log1p(-np.clip(e, 1e-4, 1 - 1e-4)) for e in elite])
                mu, sd = el.mean(0), el.std(0) + 0.05
            if best_val > base_val:
                wins.append((int(ob["day"]), best_val - base_val))
            forced = best_vec.astype(np.float32)
        a = pol.act(ob, macro_override=forced)
        forced = None
        obs, _ = env.step([a, dict(E.PASS_ACTION)])
    if pool is not None:
        pool.close()
        pool.join()
    return float(env.rewards()[0]), plain, wins, null_err


def scenario_oracle_global(policy_path: str, days: int, seeds, k: int = 32, procs: int = 1):
    rows = []
    for s in seeds:
        oracle, plain, wins, null_err = oracle_global_solitaire(policy_path, days, s, pop=k, procs=procs)
        rows.append({"seed": s, "final": plain, "optimum": oracle, "regret": oracle - plain,
                     "consistency": {}, "orders": {"search_won_on_days": [d for d, _ in wins],
                                                   "gain_by_day": [round(g) for _, g in wins], "null_error": null_err}})
    return rows


def scenario_oracle(policy_path: str, days: int, seeds, k: int = 16, procs: int = 1):
    rows = []
    for s in seeds:
        oracle, plain, wins, null_err = oracle_solitaire(policy_path, days, s, k, procs=procs)
        rows.append({"seed": s, "final": plain, "optimum": oracle, "regret": oracle - plain,
                     "consistency": {}, "orders": {"search_won_on_days": [d for d, _ in wins],
                                                   "null_error": null_err}})
    return rows


# Idle worlds: nothing can pay. Carrot needs 3 days and wheat 4 (plant, grow,
# harvest, sell), the fastest animal produces on day 4; with k <= 3 the
# optimum is the starting cash. From k = 5 on, a plan pays and the yardstick
# must be a planner's, not 3,000 $.
SCENARIOS = {f"idle-{k}": (scenario_idle, k) for k in (1, 2, 3)}
# Worlds where a plan pays: the yardstick is the policy improved by rollout
# search on the exact engine (a lower bound on the optimum, and the same
# search that was worth +37% in the full game before the time budget).
SCENARIOS.update({f"oracle-{k}": (scenario_oracle, k) for k in (5, 8, 14, 30)})
SCENARIOS.update({f"global-{k}": (scenario_oracle_global, k) for k in (5, 8, 14, 30)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--scenarios", default="idle-1,idle-2,idle-3")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--k", type=int, default=16, help="oracle candidates per day")
    p.add_argument("--procs", type=int, default=1)
    a = p.parse_args()
    seeds = S.DIALS.seeds(a.seeds)
    worst = 0.0
    for name in a.scenarios.split(","):
        fn, arg = SCENARIOS[name]
        rows = (fn(os.path.abspath(a.checkpoint), arg, seeds, a.k, a.procs)
                if fn in (scenario_oracle, scenario_oracle_global)
                else fn(os.path.abspath(a.checkpoint), arg, seeds))
        reg = [r["regret"] for r in rows]
        worst = max(worst, max(reg))
        print(f"[{name}] policy {sum(r['final'] for r in rows) / len(rows):,.0f} $  optimum "
              f"{sum(r['optimum'] for r in rows) / len(rows):,.0f} $  regret mean {sum(reg) / len(reg):,.0f} $  "
              f"max {max(reg):,.0f} $   {rows[0]['orders']}", flush=True)
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
