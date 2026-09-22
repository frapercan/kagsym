"""The agent: joins farm and market within the one-second-per-turn budget.

The world model is OPTIONAL on purpose. The submission environment may not have
torch, and an agent that fails for that reason is worth zero. Without a model
the market layer assumes zero opponent flow -conservative: it underestimates
the future price drop and therefore pushes towards selling earlier- and
everything else works the same.
"""
from __future__ import annotations

import os
import time

from .. import spec
from . import market_ops, tasks

# spec.TURNS_PER_DAY is read at call time (see spec.set_turns_per_day): as a
# module alias it froze at import and stopped following `turnsPerDay`,
# silently desynchronising the executor from the engine.


# Use the opponent flow estimated from their board when valuing a sale. The
# input already existed (`sell_orders(opp_flow=...)`) and defaulted to ZERO,
# which the code itself describes as "conservative: it underestimates the
# future drop and therefore pushes towards selling earlier".
USE_RIVAL_FLOW = bool(int(os.environ.get("KAG_FLUJO_MERCADO", "0")))

TURNS_PER_TILE_INIT = 3.0   # learned (`f_turns_init`)
TURNS_PER_TILE_MIN = 2.0    # learned (`f_turns_min`)
# How the per-tile cost ADAPTS. Learned (`f_cost_rise`, `f_cost_decay`): the
# 0.5 capped the rise and the 0.93 the decay, and neither had ever entered a
# search.
COST_RISE = 0.5
COST_DECAY = 0.93


def sustainable_tiles(obs, n_units: int, turns_per_tile: float) -> int:
    """How many tiles this workforce can keep alive.

    `turns_per_tile` is NOT a constant: the agent measures it during the
    episode (see `Agent._recalibrate`). An a priori formula does not know how
    much time units lose moving, and overcommitting tiles is fatal: two days
    without water and the plant turns into a weed, losing the seed, the tile
    and the work invested.
    """
    return max(1, int(n_units * spec.TURNS_PER_DAY / max(1.0, turns_per_tile)))


class Agent:
    """Keeps state across turns: the world model and its confidence gate."""

    def __init__(self, model_path: str | None = None, horizon: int = 12,
                 use_model: bool = False, episode_steps: int | None = None,
                 macro=None, rival=None, micro=None):
        spec.set_episode_steps(episode_steps)
        # None = the scripted layer as-is. A `Macro` = the policy decides.
        self.macro = macro
        self.horizon = horizon
        # Injection point for an opponent model. None = the exact layer only.
        self.rival = rival
        # Micro: a function obs -> 10x10 value map (plus verb logits in ops).
        # None = the heuristic valuation as-is.
        self.micro = micro
        # Destination assigned to each unit last turn, for stickiness.
        self._destinations = {}
        self.prev = None          # (obs, action) from the previous turn
        self.t_total = 0.0
        self.n_turns = 0
        # Unit-turns it costs to keep one tile alive. Self-recalibrating: it
        # rises when there are unwatered plants (we overcommitted) and falls
        # when there are none (we can take on more).
        self.turns_per_tile = TURNS_PER_TILE_INIT
        self._last_day = -1

    # -- opponent prediction (injection point) -------------------------------
    def _opponent_flow(self, obs, provisional_action):
        """Expected opponent supply, or None when no model is plugged in.

        `rival` is an optional object with `flow(obs, action) -> np.ndarray`
        and `update(prev_obs, prev_action, obs)`. It is injected, not loaded
        from disk: that way the executor does not depend on any checkpoint
        existing and the exact layer stays exact without it.

        Measured before separating it: the model gets the LEVEL of the
        opponent's supply right (+19% on their money, +35% on their spending)
        but is WORSE than not correcting turn by turn (-26%). Whatever plugs in
        here must predict a cumulative level, never the specific turn.
        """
        if self.rival is None:
            if not USE_RIVAL_FLOW:
                return None
            # From the opponent BOARD, which is fully observable. It converts
            # the cumulative LEVEL -units they will have ready within the
            # horizon- into the PER-TURN RATE that `future_price` expects,
            # since it computes `opp_flow * horizon`.
            #
            # The conversion is not a detail: the docstring above warns that
            # predicting the specific turn came out WORSE than not correcting
            # (-26%) and that only the cumulative level is usable. Plugging the
            # level in as if it were a rate would multiply it by the horizon.
            try:
                from .. import obs as _O
                import numpy as _np
                days = max(1.0, self.horizon / float(spec.TURNS_PER_DAY))
                f = _O.rival_flow(obs).reshape(len(_O.RIVAL_WINDOWS), -1)
                # the closest window covering the horizon
                k = min(range(len(_O.RIVAL_WINDOWS)),
                        key=lambda i: abs(_O.RIVAL_WINDOWS[i] - days))
                return _np.asarray(f[k], dtype=_np.float64) / max(1.0, self.horizon)
            except Exception:
                return None
        try:
            return self.rival.flow(obs, provisional_action)
        except Exception:
            return None

    def _update_gate(self, obs) -> None:
        if self.rival is None or self.prev is None:
            return
        try:
            self.rival.update(self.prev[0], self.prev[1], obs)
        except Exception:
            pass

    def _recalibrate(self, obs) -> None:
        """Adjust the watering capacity from what actually happened today."""
        if obs["day"] == self._last_day or obs["hour"] != spec.TURNS_PER_DAY - 1:
            return
        self._last_day = obs["day"]
        mi = obs["farms"][int(obs["player"])]
        dry = planted = 0
        for row in mi["tiles"]:
            for t in row:
                if isinstance(t, dict) and t.get("kind") == "PLANT":
                    planted += 1
                    if not t["watered_today"]:
                        dry += 1
        if planted == 0:
            return
        if dry > 0:
            # We overcommitted: each tile costs more than we thought.
            self.turns_per_tile *= 1.0 + min(COST_RISE, dry / planted)
        else:
            self.turns_per_tile = max(TURNS_PER_TILE_MIN,
                                          self.turns_per_tile * COST_DECAY)

    # -- turn ----------------------------------------------------------------
    def __call__(self, obs) -> dict:
        t0 = time.perf_counter()
        self._last_action = None
        me = int(obs["player"])
        mi = obs["farms"][me]

        self._update_gate(obs)
        self._recalibrate(obs)
        # The 30 parameters that used to be hand-set now live in the macro
        # vector, and three different modules read them from functions that do
        # not receive it. They are written here, once per turn, instead of
        # threading the macro through six more signatures.
        if self.macro is not None:
            from ..macro import apply_params
            apply_params(self.macro)

        # PLANNED units, not the ones present right now. Third time the same
        # trap bites: `len(mine["hands"])` is ALWAYS 0 at hour 0, which is
        # exactly when the decision is made. With that, `sustainable` came out
        # at 9-12 while 17-18 tiles were planted, i.e. `free = 0` and the
        # executor NEVER allowed planting again. That is why raising the tile
        # target only bought seed that was never planted: 178 seeds for 3.5
        # live crops, ~$4,000 wasted per episode and no cash for livestock.
        n_units = 1 + len(mi["hands"])
        if self.macro is not None:
            from ..macro import target_hands
            n_units = max(n_units, 1 + target_hands(obs, self.macro))
        sustainable = sustainable_tiles(obs, n_units, self.turns_per_tile)
        planted = sum(1 for row in mi["tiles"] for t in row
                        if isinstance(t, dict) and t.get("kind") == "PLANT")
        free = max(0, sustainable - planted)

        # The micro head returns TWO maps: a 10x10 value map and N_OPS x 10 x
        # 10 verb logits. An agent without a network returns neither.
        _mv, _mo = None, None
        if self.micro:
            _out = self.micro(obs)
            if isinstance(_out, tuple):
                _mv, _mo = _out
            else:
                _mv = _out
        units = tasks.assign_units(
            obs, free,
            value_map=_mv, verb_map=_mo,
            previous=self._destinations,
            macro=self.macro)
        provisional = {"farmer": units[0] if units else ["PASS"],
                       "hands": units[1:], "market": []}

        flow = self._opponent_flow(obs, provisional)
        # ORDER matters: the engine processes orders in sequence, so selling
        # first funds hiring, and hiring comes before buying seed because
        # without hands to water it the seed is lost.
        orders = []
        mac = self.macro
        # ORDERED BY VALUE AT STAKE, not by the order the calls were written
        # in. The engine accepts only `maxMarketOrdersPerTurn` (10) per turn
        # and the rest are silently dropped. Measured: `land_orders` went last
        # and BUY_LAND can only be issued at hour 0, which is exactly when
        # `hire_orders` issues 8 hires. The agent emitted the buy-quadrant
        # order from day 15 onwards and it NEVER executed: it stayed on one
        # quadrant cramming 17 animals into 25 tiles.
        #
        # The criterion is what cannot wait and what it is worth:
        #   land     $1000 for 25 tiles (~$14,000 of return), hour 0 only
        #   feed     without it an animal escapes in two days (~$1700 each)
        #   animals  the highest return per tile
        #   sales    price-sensitive, but repeatable next turn
        #   seeds    repeatable
        #   hands    the n-th of the day costs fib(n); they can be issued in
        #            hours 0-3, so they tolerate waiting best
        # The ORDER is decided by the policy, not by a list I wrote. Measured:
        # lack of cash blocks buying land on 31% of turns, feed on 19% and
        # animals on 15%, so whoever goes first decides who goes without. With
        # uniform priorities the previous order is recovered.
        from ..macro import category_order
        by_category = {
            "land": lambda: market_ops.land_orders(obs, macro=mac),
            "feed": lambda: market_ops.feed_orders(obs),
            "animal": lambda: market_ops.animal_orders(obs, macro=mac),
            "sell": lambda: market_ops.sell_orders(obs, opp_flow=flow,
                                                   horizon=self.horizon, macro=mac),
            # `tile_target` is overridden by `target_tiles(obs, macro)` inside
            # `seed_orders` whenever a macro is present, which is always in
            # training and in play. The value passed here is the fallback
            # for a macro-less agent.
            "seed": lambda: market_ops.seed_orders(obs, tile_target=free,
                                                   macro=mac),
            "hand": lambda: market_ops.hire_orders(obs, macro=mac),
        }
        # The keys MUST match `macro.CATEGORIES`; a mismatch is a KeyError, not
        # a silent fallback, which is what we want after the `turn_weights`
        # lesson.
        cats = category_order(mac) if mac is not None else list(by_category)
        for c in cats:
            orders += by_category[c]()
        orders = orders[: spec.DEFAULT_CONFIG["maxMarketOrdersPerTurn"]]

        action = dict(provisional, market=orders)
        self._last_action = action
        # The copy is consumed only by `_update_gate`, which gives up at once
        # when there is no opponent model. Copying the WHOLE observation -two
        # 100-tile farms, market, town, inventories- every turn when nobody
        # will read it was 14.4% of profiled episode time, plus its share of
        # `isinstance` and `dict.get`. No training or evaluation path passes
        # `rival=`.
        if self.rival is not None:
            from ..fastenv import _fast_copy
            self.prev = (_fast_copy(obs), _fast_copy(action))
        self.t_total += time.perf_counter() - t0
        self.n_turns += 1
        return action

    @property
    def ms_per_turn(self) -> float:
        return 1000 * self.t_total / max(1, self.n_turns)


_AGENT = None


def agent(obs, config=None) -> dict:
    """Entry point expected by kaggle-environments."""
    global _AGENT
    if _AGENT is None:
        _AGENT = Agent()
    if config is not None:
        spec.set_episode_steps(
            config.get("episodeSteps") if isinstance(config, dict)
            else getattr(config, "episodeSteps", None))
    return _AGENT(obs)
