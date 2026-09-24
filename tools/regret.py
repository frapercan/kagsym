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


# Idle worlds: nothing can pay. Carrot needs 3 days and wheat 4 (plant, grow,
# harvest, sell), the fastest animal produces on day 4; with k <= 3 the
# optimum is the starting cash. From k = 5 on, a plan pays and the yardstick
# must be a planner's, not 3,000 $.
SCENARIOS = {f"idle-{k}": (scenario_idle, k) for k in (1, 2, 3)}


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
        print(f"[{name}] regret mean {sum(reg) / len(reg):,.0f} $  max {max(reg):,.0f} $   "
              f"orders {rows[0]['orders']}   potential-final at day 0: {rows[0]['consistency'].get(0, 0):+,.0f}")
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
