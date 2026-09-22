"""Shaping potential: EXACT liquidation value, mine minus the opponent's.

    r'_t  =  r_t  +  gamma * Phi(s_{t+1})  -  Phi(s_t)

With Phi(s_T) equal to the true terminal value, shaping is *policy-invariant*
(Ng, Harada & Russell 1999): it does not change which policy is optimal, it
only moves credit earlier. That condition is what failed before, and it has to
hold to the dollar.

## What failed with net worth

PPO with a net-worth reward converged to $2,794, below the $3,000 you get by
not playing at all. The precise diagnosis is not that net worth is wrong: it is
that **Phi(s_T) != cash**. The shed, standing crops and animals are worth ZERO
when the episode closes, so a potential that counts them lies at exactly the
instant that scores. Measured: +611 to +917 of mean bias, up to $4,131.

Here only what HAS TIME to turn into cash before turn 720 is counted. At T
there is no time left for anything, so Phi(s_T) = cash exactly and the bias
vanishes by construction.

## Why the DIFFERENCE and not my absolute level

The scoreboard is `sign(my_money - their_money)`: zero-sum in the sign. So
destroying the opponent's value is worth exactly as much as creating your own.
Without the negative term the reward cannot see the most profitable play in the
game: dumping product into a thin market crashes its price by 98% (MILK 169 ->
5), and with it the opponent's entire livestock output.

## Information asymmetry, accepted

On my side everything is visible. Of the opponent only the public part: their
money and their board. Their shed, their seeds and whatever their units carry
are hidden -they resolve exactly one turn later, but the potential is evaluated
NOW-. What is observable is used, which is a well-defined function of the state
and therefore a valid potential; it simply ignores part of their value.
"""
from __future__ import annotations

from . import spec
from .symbolic.market_ops import marginal_prices

# spec.TURNS_PER_DAY is read at call time (see spec.set_turns_per_day): as a
# module alias it froze at import and stopped tracking `turnsPerDay`, silently
# desynchronising the executor from the engine.


def _lot_value(obs, product: str, n: int) -> float:
    """What n units would fetch if sold now, at the engine's MARGINAL price.

    Nominal price overestimates: each unit sold lowers the price of the next.
    """
    n = int(n)
    if n <= 0:
        return 0.0
    return float(sum(marginal_prices(obs, product, min(n, 200))))


def liquidation(obs, pid: int, private=None) -> float:
    """Cash plus everything that has time to become cash before the close."""
    farm = obs["farms"][pid]
    total = float(farm["money"])
    days = max(0, (spec.EPISODE_STEPS - 1 - int(obs["step"])) // spec.TURNS_PER_DAY)

    if private is not None:
        for item, n in (private.get("shed") or {}).items():
            if item in spec.PRODUCTS and n:
                total += _lot_value(obs, item, n)
        for inv in (private.get("inventories") or []):
            for item, n in inv.items():
                if item in spec.PRODUCTS and n:
                    total += _lot_value(obs, item, n)

    day = int(obs["day"])
    for row in farm["tiles"]:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                cd = spec.CROPS[t["crop"]]
                age = day - int(t["planted_day"])
                # counts only if it will yield AND there is time to sell it
                if age + days < cd["first_yield_day"]:
                    continue
                units = int(t.get("yield_units", 0))
                if cd["ongoing"]:
                    remaining = max(0, min(cd["max_yield"] - units,
                                           days // max(1, cd["interval"])))
                    units += remaining
                total += _lot_value(obs, t["crop"], units)
            elif t.get("animal"):
                a = spec.ANIMALS[t["animal"]]
                units = int(t.get("yield_units", 0))
                to_come = max(0, days - a["first_yield_day"]) // max(1, a["interval"])
                units = min(a["max_held"] + to_come, units + to_come)
                total += _lot_value(obs, a["product"], units)
    return total


def phi(obs, me: int = 0, private=None, with_rival: bool = False) -> float:
    """Relative potential: mine minus the opponent's.

    Only the opponent's public state is counted (money and board); their shed
    is not observable. That systematically underestimates their position, but
    it does so CONSISTENTLY, which is what keeps the telescoping difference
    valid.
    """
    mine = liquidation(obs, me, private)
    if not with_rival:
        # OFF BY DEFAULT. Measured: the opponent's term accounts for 99.1% of
        # the variance of the daily reward (sd 1.93 of 1.94 total), because
        # their liquidation jumps when they harvest and sell and our state
        # cannot anticipate it. With it, the critic gets R2 = -2.535 (worse
        # than predicting the mean); without it, R2 = +0.736. No critic means
        # no advantage, and without advantage PPO is noise centred on zero.
        #
        # The idea of rewarding the destruction of their value is NOT lost: it
        # lives in the terminal term (+-1 for win/loss), which is a sign and
        # therefore already counts their money as much as ours. The opponent
        # belongs in the OBJECTIVE, not in the shaping: shaping can only carry
        # what the state predicts.
        return mine
    return mine - liquidation(obs, 1 - me, None)
