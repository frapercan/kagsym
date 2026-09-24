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


def oracle_solitaire(policy_path: str, days: int, seed: int, k: int = 16, hours: int = 24):
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
    while not env.done:
        ob = obs[0]
        if ob["hour"] == 0:
            cands = pol.macro_candidates(ob, k, gen)
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
    return float(env.rewards()[0]), plain, wins, null_err


def scenario_oracle(policy_path: str, days: int, seeds, k: int = 16):
    rows = []
    for s in seeds:
        oracle, plain, wins, null_err = oracle_solitaire(policy_path, days, s, k)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--scenarios", default="idle-1,idle-2,idle-3")
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    seeds = S.DIALS.seeds(a.seeds)
    worst = 0.0
    for name in a.scenarios.split(","):
        fn, arg = SCENARIOS[name]
        rows = fn(os.path.abspath(a.checkpoint), arg, seeds)
        reg = [r["regret"] for r in rows]
        worst = max(worst, max(reg))
        print(f"[{name}] policy {sum(r['final'] for r in rows) / len(rows):,.0f} $  optimum "
              f"{sum(r['optimum'] for r in rows) / len(rows):,.0f} $  regret mean {sum(reg) / len(reg):,.0f} $  "
              f"max {max(reg):,.0f} $   {rows[0]['orders']}", flush=True)
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
