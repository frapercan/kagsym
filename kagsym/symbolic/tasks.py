"""Farm management: valuing tasks in dollars and assigning units.

The principle is the one that governs the rest of the project: **nothing
guessed**. Each priority comes from what that action is genuinely worth
according to the engine, not from a hand-picked number.

  - Watering inside the bonus window is worth **+1 unit** of that crop, so its
    value is the market price of one unit.
  - Watering a plant with `consecutive_unwatered >= 1` saves the whole plant:
    it is worth its entire future yield, because tomorrow it turns into a weed.
  - Harvesting realises the accumulated yield and frees the tile.
  - Planting starts a cycle whose value is the crop's net profit.

Since units move one tile per turn, value is discounted by distance: a $40 task
four steps away returns less than a $20 one right next to you.

IN PLAY, MOST OF THIS IS DEAD. The network emits both the value and the verb
per tile, so these valuations only survive as the fallback for an agent with no
network. Measured: multiplying `DIG_VALUE` by a thousand does not move a single
dollar. What stays live for everyone is the LEGALITY in `tile_options`, which
is derived from the engine.
"""
from __future__ import annotations

import math

from .. import spec

# spec.TURNS_PER_DAY is read at call time (see spec.set_turns_per_day): as a
# module alias it froze at import and stopped following `turnsPerDay`, silently
# desynchronising the executor from the engine.
from .assignment import max_assignment

BOARD = spec.BOARD
DIRS = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def quadrant_of(x: int, y: int) -> str:
    h = BOARD // 2
    return ("N" if y < h else "S") + ("W" if x < h else "E")


def unlocked(farm, x: int, y: int) -> bool:
    return quadrant_of(x, y) in farm["unlocked_quadrants"]


def step_toward(pos, target) -> str:
    """One greedy step. There are no obstacles: every tile is walkable."""
    x, y = pos
    tx, ty = target
    if x != tx:
        return "EAST" if x < tx else "WEST"
    if y != ty:
        return "SOUTH" if y < ty else "NORTH"
    return "PASS"


def dist(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- value of a crop --------------------------------------------------------
def cycle_yield(crop: str) -> int:
    """Units a full cycle yields, per the engine's exact mechanics."""
    cd = spec.CROPS[crop]
    if cd["ongoing"]:
        return cd["max_yield"]
    w0 = (cd["max_yield_day"] + 1) // 2
    return min(cd["max_yield"], 1 + (cd["max_yield_day"] - w0 + 1))


def cycle_days(crop: str) -> int:
    cd = spec.CROPS[crop]
    if cd["ongoing"]:
        return cd["first_yield_day"] + cd["interval"] * (cd["max_yield"] - 1)
    return cd["max_yield_day"]


def unit_price(obs, item: str) -> float:
    return float(obs["market"]["prices"].get(item, 1))


def cycle_profit(obs, crop: str) -> float:
    """Net profit of one cycle at the CURRENT market price."""
    return cycle_yield(crop) * unit_price(obs, crop) - spec.CROPS[crop]["seed"]


def days_left(obs) -> int:
    total = spec.EPISODE_STEPS // spec.TURNS_PER_DAY
    return total - obs["day"]


def plantable(obs, crop: str) -> bool:
    """There is no point planting what there will be no time to harvest."""
    return cycle_days(crop) < days_left(obs)


def growth_factor(obs, crop: str) -> float:
    """Factor by which capital multiplies per day when reinvested.

    A cycle turns `seed` dollars into `units x price` dollars in
    `cycle_days` days, so the daily factor is (return/cost)^(1/days).
    With wheat: 10 -> 100 in 4 days = x1.78 a day. With melon: 80 -> 1500 in
    12 days = x1.28. Melon wins per tile-day and loses per capital-day, and
    when capital is the constraint, the second is what matters.
    """
    cost = max(1, spec.CROPS[crop]["seed"])
    ret = cycle_yield(crop) * unit_price(obs, crop)
    d = max(1, cycle_days(crop))
    return (ret / cost) ** (1.0 / d)


def capital_is_binding(obs, free_tiles: int) -> bool:
    """Is there too little money to fill the tiles that could be tended?"""
    if free_tiles <= 0:
        return False
    money = float(obs["farms"][int(obs["player"])]["money"])
    cheapest = min(spec.CROPS[c]["seed"] for c in spec.CROP_LIST if plantable(obs, c)) \
        if any(plantable(obs, c) for c in spec.CROP_LIST) else 1
    return money < free_tiles * cheapest


def best_crop(obs, seeds_only: bool = False, free_tiles: int = 0) -> str | None:
    """The best crop GIVEN WHICH RESOURCE IS BINDING.

    With tiles to spare and little money the constraint is capital and the
    winner is whatever multiplies it fastest. With money to spare and few
    tiles, the constraint is land and the winner is whatever yields most per
    tile-day.

    Confusing the two costs the game: optimising tile-day from turn 0 leads to
    planting melon, which pays nothing for 12 days, running out of cash and
    not even being able to hire hands.
    """
    seeds = obs["private"].get("seeds", {})
    capital_bound = capital_is_binding(obs, free_tiles)
    best, value = None, 0.0
    for c in spec.CROP_LIST:
        if not plantable(obs, c):
            continue
        if seeds_only and seeds.get(c, 0) <= 0:
            continue
        v = growth_factor(obs, c) if capital_bound else cycle_profit(obs, c) / max(1, cycle_days(c))
        if v > value:
            best, value = c, v
    return best


# --- tasks ------------------------------------------------------------------
def _shed_access_set():
    """Shed access tiles, computed ONCE.

    The set was rebuilt on every call, and it is called 44,275 times per
    episode (once per tile per turn). It is a board constant.
    """
    global _SHED_ACCESS
    if _SHED_ACCESS is None:
        import kaggle_environments.envs.kaggriculture.kaggriculture as K
        _SHED_ACCESS = frozenset(tuple(p) for p in K._shed_access_tiles(BOARD))
    return _SHED_ACCESS


_SHED_ACCESS = None


def _is_shed_access(x: int, y: int) -> bool:
    return (x, y) in _shed_access_set()


def _shed_task(obs, farm, ctx=None, macro=None):
    """What to take out of the shed. Without this the animal chain never
    closes: the animal is bought, lands in the shed and stays there forever.

    Wheat matters just as much: FEED consumes 1 wheat FROM THE UNIT'S
    INVENTORY, so an animal nobody brings wheat to escapes after two days.
    """
    priv = obs["private"]
    shed = priv["shed"]
    # 1) an animal to place, if there is or could be room
    if ctx is None:
        ctx = TurnContext(obs, farm)
    # Only the animal actually in the shed is valued.
    in_shed = [a for a in spec.ANIMALS if int(shed.get(a, 0)) > 0]
    for a in sorted(in_shed, key=lambda a: -animal_value(ctx, a, macro)):
        if ctx.free_slots.get(spec.ANIMALS[a]["structure"], 0) > 0:
            return (max(1.0, animal_value(ctx, a, macro) / ctx.days),
                    ["PICKUP", a, 1])
    # 1b) DROP. The harvest stays in the unit's inventory until the close of
    # the day; with DROP it reaches the shed NOW and can be sold the same day,
    # besides avoiding the nightly flush overflowing the 100-unit shed and
    # discarding the excess. The opponent uses it 417 times per episode and it
    # was simply missing from our repertoire.
    #
    # MARGINAL VALUE. It is NOT worth what the unit carries: the engine
    # flushes inventories to the shed only at the close of the day, so dropping
    # early does not change that the goods end up there. All it adds is being
    # able to SELL IT TODAY, before the price falls. Valuing it gross made
    # every unit run to the shed: measured, $39,558 -> $13,250.
    #
    # This is the third place where gross value was confused with marginal
    # (before: the animal's nominal price and the fertiliser bonus).
    from .market_ops import future_price, marginal_prices
    priv_inv = priv.get("inventories") or []
    best = 0.0
    for inv in priv_inv:
        v_ = 0.0
        for item, n_ in (inv or {}).items():
            if item in spec.PRODUCTS and n_:
                now = float(sum(marginal_prices(obs, item, int(n_))))
                later = float(future_price(obs, item, spec.TURNS_PER_DAY)) * int(n_)
                v_ += max(0.0, now - later)      # only the drop avoided
        best = max(best, v_)
    if best > 0:
        return (best, ["DROP"])

    # 2) fertiliser, if there are plants to fertilise and it is in the shed
    fert = int(shed.get("FERTILIZER", 0))
    if fert > 0:
        fertilizable = sum(1 for row in farm["tiles"] for t in row
                            if isinstance(t, dict) and t.get("kind") == "PLANT"
                            and t.get("fertilized_until_day", -1) < obs["day"])
        if fertilizable > 0:
            n = min(fert, fertilizable, int(FERT_PER_TRIP))
            return (FERT_TRIP_VALUE * unit_price(obs, "FERTILIZER"),
                    ["PICKUP", "FERTILIZER", n])

    # 3) wheat to feed the animals that have not eaten yet
    hungry = sum(1 for row in farm["tiles"] for t in row
                      if isinstance(t, dict) and t.get("animal") and not t.get("fed_today"))
    if hungry > 0 and int(shed.get("WHEAT", 0)) > 0:
        n = min(hungry, int(shed["WHEAT"]))
        return (WHEAT_TRIP_VALUE * unit_price(obs, "WHEAT"), ["PICKUP", "WHEAT", n])
    return None


# What a unit must be CARRYING for the operation not to be a no-op.
REQUIRES = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}


def _can_drop(inv) -> bool:
    """DROP only makes sense while carrying something."""
    return any(int(v) > 0 for v in (inv or {}).values())


def _animals_waiting(obs) -> list:
    """Animals bought but not yet placed (in the shed or in hand)."""
    priv = obs["private"]
    waiting = []
    for a in spec.ANIMALS:
        n = int(priv["shed"].get(a, 0))
        n += sum(int(inv.get(a, 0)) for inv in priv.get("inventories", []))
        waiting += [a] * n
    return waiting


def _free_structures(farm) -> dict:
    free_slots = {"COOP": 0, "PASTURE": 0}
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") in free_slots and not t.get("animal"):
                free_slots[t["kind"]] += 1
    return free_slots


def _watering_rate(farm) -> float:
    """Fraction of plants ALREADY watered today.

    It is the empirical probability that future watering happens, and
    therefore the factor by which the fertiliser bonus must be discounted: if
    only 39% of plants get watered, fertilising returns 39% of what it
    promises. It comes out of the state itself, not out of a constant.
    """
    total = watered = 0
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                total += 1
                watered += 1 if t.get("watered_today") else 0
    return (watered / total) if total else 0.0


class TurnContext:
    """Facts shared by all 100 tiles of the turn, computed LAZILY.

    Two measurements, both with cProfile:
      - computing them inside `tile_task`: 10,547 board scans and 15,215
        animal revaluations per episode -> 2,172 steps/s.
      - computing them all on entering the turn: WORSE, 1,853 steps/s, because
        most turns have no pending animal and the scan was paid anyway.
    Lazy captures the best of both: zero cost when not needed, and a single
    scan when it is. (The engine alone does 38,567 steps/s: the bottleneck was
    always here, not in the simulation.)
    """

    __slots__ = ("obs", "farm", "_pend", "_free_struct", "_val", "_cycle_days")

    def __init__(self, obs, farm):
        self.obs, self.farm = obs, farm
        self._pend = self._free_struct = self._val = self._cycle_days = None

    @property
    def pending_animals(self):
        if self._pend is None:
            self._pend = _animals_waiting(self.obs)
        return self._pend

    @property
    def free_slots(self):
        if self._free_struct is None:
            self._free_struct = _free_structures(self.farm)
        return self._free_struct

    @property
    def days(self):
        if self._cycle_days is None:
            self._cycle_days = max(1, days_left(self.obs))
        return self._cycle_days

    def value(self, animal):
        if self._val is None:
            self._val = {}
        if animal not in self._val:
            from .market_ops import animal_net_value
            self._val[animal] = animal_net_value(self.obs, animal)
        return self._val[animal]


def animal_value(ctx, a, macro):
    """Animal value WEIGHTED by the learned factor of its product.

    `ctx.value` is `animal_net_value`, a hand-written formula. Deciding with
    it WHICH animal is picked up, which is placed and which shed is built is
    the same family of hard-wired decisions as `best_crop` and `seed_orders`,
    and it is where the other measured hole lives: v48 makes $60,179 from MILK
    and we make $13,536. The per-product factor (EGG, MILK, WOOL) already
    lives in the macro vector, so weighting by it hands the preference to the
    network. Neutral = 1.0: with no macro, the order is the one of always.

    It exists as a function rather than repeated at each site because that
    already went wrong once: ONE site was weighted -PICKUP in `tile_options`-
    and the other four stayed unweighted for a whole session without anyone
    noticing.
    """
    v = ctx.value(a)
    if macro is None:
        return v
    try:
        from ..macro import market_factors
        return v * market_factors(macro).get(spec.ANIMALS[a]["product"], 1.0)
    except Exception:
        return v


def tile_task(obs, farm, x: int, y: int, free_capacity: int, ctx=None, macro=None):
    """(value in $, operation) of the best possible action on that tile."""
    tile = farm["tiles"][y][x]
    day = obs["day"]
    seeds = obs["private"].get("seeds", {})
    if ctx is None:
        ctx = TurnContext(obs, farm)

    if _is_shed_access(x, y):
        t = _shed_task(obs, farm, ctx, macro)
        if t is not None:
            return t

    if tile is None:
        # An animal waiting in the shed is worth nothing until there is
        # somewhere to put it. Building unlocks ~$1,700 of income already
        # paid for.
        for kind in ("COOP", "PASTURE"):
            animals = [a for a in ctx.pending_animals if spec.ANIMALS[a]["structure"] == kind]
            if animals and ctx.free_slots.get(kind, 0) <= 0:
                v = max(animal_value(ctx, a, macro) for a in animals)
                return (max(1.0, v / ctx.days), ["BUILD_" + kind])
        if free_capacity <= 0:
            return None
        crop = best_crop(obs, seeds_only=True, free_tiles=free_capacity)
        if crop is None:
            return None
        # Prorated: planting today captures the whole cycle, but the value is
        # compared against actions that pay now, so it is measured per day.
        return (cycle_profit(obs, crop) / max(1, cycle_days(crop)), ["PLANT", crop])

    if not isinstance(tile, dict):
        return None
    kind = tile.get("kind")

    if kind == "WEED":
        crop = best_crop(obs, seeds_only=True, free_tiles=free_capacity)
        v = cycle_profit(obs, crop) / max(1, cycle_days(crop)) if crop else 1.0
        return (v * DIG_VALUE, ["DIG"])    # exposed fraction, not fixed

    if kind == "PLANT":
        cd = spec.CROPS[tile["crop"]]
        age = day - tile["planted_day"]
        price = unit_price(obs, tile["crop"])
        ripe = age >= cd["first_yield_day"] and tile["yield_units"] > 0

        if not tile["watered_today"]:
            # `consecutive_unwatered` IS BORN AT 1 on planting: a plant not
            # watered the same day turns into weed that night. That is why >= 1
            # is already an emergency, not a warning. Verified against the
            # engine.
            if tile["consecutive_unwatered"] >= 1:
                remaining = max(1, cycle_yield(tile["crop"]) - tile["yield_units"])
                return (remaining * price, ["WATER"])
            if not cd["ongoing"]:
                w0 = (cd["max_yield_day"] + 1) // 2
                if w0 <= age <= cd["max_yield_day"] and tile["yield_units"] < cd["max_yield"]:
                    bonus = 2 if tile["fertilized_until_day"] >= day else 1
                    return (bonus * price, ["WATER"])
            else:
                if age >= cd["first_yield_day"] - 1:
                    return (price, ["WATER"])
            return (WATER_IDLE_VALUE * price, ["WATER"])

        # FERTILISE: only where the NETWORK provides the value.
        #
        # The capability was needed -fertiliser was picked up 550 times per
        # game and never used, and the engine gives `bonus = 2 if fertilized`-
        # but there is no hand-written way to value it. Three attempts by eye,
        # all three worse:
        #   gross value (days x price)      -> always beats watering: $22,768
        #   marginal value x watering rate  -> still steals waterings: $25,770
        #   moved after watering            -> $11,086
        # PAIRED comparison over 8 seeds: -$21,998 +- 4,563 (4.8 sigma), with
        # two games sunk to $568 and $33.
        #
        # So the operation is left AVAILABLE for the network to price, and out
        # of the hand-written heuristic. It is
        # exactly the case that motivates the learned micro head: an action
        # whose value depends on the state in a way I cannot write down.
        # Fertilise dial: 0 turns it off, and the search decides.
        from ..macro import fertilize_weight
        _pf = fertilize_weight(macro) if macro is not None else 0.0
        if _pf > 0.0:
            # FERTILISE, AFTER watering. It used to sit BEFORE and `return`,
            # so a plant needing both got fertilised and the watering was never
            # evaluated: it died that night. The symptom was 563 plantings
            # against 386 waterings -more plantings than waterings is a death
            # sentence-.
            if (not cd["ongoing"] and tile.get("fertilized_until_day", -1) < day):
                w0 = (cd["max_yield_day"] + 1) // 2
                if w0 <= age + 1 <= cd["max_yield_day"] and tile["yield_units"] < cd["max_yield"]:
                    useful_days = min(FERTILIZER_HORIZON,
                                      cd["max_yield_day"] - age, days_left(obs))
                    if useful_days > 0:
                        # MARGINAL VALUE, not gross. The bonus adds +1 unit
                        # per day WATERED, not per day. Pricing it at
                        # `days * price` made it always beat watering
                        # (1-2 x price): measured, watering fell from 1,791 to
                        # 904, harvests from 768 to 360 and money from $36,566
                        # to $22,768.
                        return (_pf * useful_days * price * _watering_rate(farm), ["FERTILIZE"])
            elif cd["ongoing"] and tile.get("fertilized_until_day", -1) < day:
                if age >= cd["first_yield_day"] - 1 and days_left(obs) > 1:
                    return (_pf * min(FERTILIZER_HORIZON, days_left(obs))
                            * price * _watering_rate(farm), ["FERTILIZE"])

        if ripe and (cd["ongoing"] or age >= cd["max_yield_day"]):
            return (tile["yield_units"] * price, ["HARVEST"])
        if ripe and tile["yield_units"] >= cd["max_yield"]:
            return (tile["yield_units"] * price, ["HARVEST"])
        return None

    if kind in ("COOP", "PASTURE"):
        animal = tile.get("animal")
        if animal is None:
            candidates = [a for a in ctx.pending_animals
                          if spec.ANIMALS[a]["structure"] == kind]
            if candidates:
                best = max(candidates,
                            key=lambda a: animal_value(ctx, a, macro))
                return (max(1.0, animal_value(ctx, best, macro) / ctx.days),
                        ["PLACE", best, 1])
            return None
        prod = spec.ANIMALS[animal]["product"]
        price = unit_price(obs, prod)
        if not tile.get("fed_today"):
            return (FEED_VALUE * price, ["FEED"])   # two days unfed and it escapes
        if tile.get("yield_units", 0) > 0:
            return (tile["yield_units"] * price, ["HARVEST"])
        if tile.get("fertilizer_available"):
            return (unit_price(obs, "FERTILIZER"), ["COLLECT_FERTILIZER"])
        if not tile.get("cared_today"):
            return (CARE_VALUE * price, ["CARE"])
    return None


STEP_DISCOUNT = 0.82        # a step costs a turn: the value is discounted
# Two more that were hand-set. Exposing 12 similar constants raised the cell
# ceiling from $939 to $1,201 (+27.9%), so none is taken on trust without
# having entered a search.
DIG_VALUE = 0.9             # clearing, as a fraction of the planting value
# HOW THE NETWORK'S MAP ENTERS THE VALUE. `expm1` is exponential, so GAIN
# decides whether the map SUGGESTS or IMPOSES, and CAP decides where it is
# clipped. Nobody ever searched them. This may explain why the map saturated at
# 4 parameters: perhaps 4 were not enough, the transform was limiting their
# effect.
MAP_GAIN = 1.0              # how much the network's emission weighs
MAP_CAP = 20.0              # clip inside expm1
FERTILIZER_HORIZON = 3      # days counted towards the fertiliser bonus
FERT_PER_TRIP = 4.0         # fertiliser picked up in one trip. Learned.
# WHAT EACH OPERATION IS WORTH, where the engine does not say it. A trip to the
# shed for wheat or fertiliser, a watering outside the yield window, feeding an
# animal and caring for it were five multipliers over the item price written
# straight into `tile_task`, which is why the constant audits that swept module
# globals never saw them. They are preferences -how much a feeding is worth
# against a harvest- so they are learned like the rest; the defaults below
# reproduce the numbers they had.
FERT_TRIP_VALUE = 2.0       # a fertiliser pickup, as a multiple of its price
WHEAT_TRIP_VALUE = 2.0      # a wheat pickup, likewise
WATER_IDLE_VALUE = 0.4      # watering with no yield in sight
FEED_VALUE = 2.0            # feeding: two days unfed and the animal escapes
CARE_VALUE = 0.5            # caring for an animal


# ---------------------------------------------------------------------------
# LEGAL ENUMERATION. The opposite of `tile_task`.
#
# `tile_task` is a cascade of `return`s: it mixes LEGALITY (what the engine
# allows) with PREFERENCE (what someone preferred). The order of those returns
# and the value formulas were eleven hand-written constants, and measured at
# the time their optimum was NOT TO PLANT: sweeping the tile target paired over
# 8 seeds, planting cost $41-50k (8-11 sigma) because crops displaced the
# animal economy, which yields 831 units of product against 316.
#
# Only legality lives here, which is derivable from the engine. The verb is
# chosen by the network.
#
# ONE VERB PER CROP. There used to be a single "PLANT" and WHAT got planted was
# decided by `best_crop`, a hand-written function that the comment above called
# "mechanics, not preference". But it is preference, and it is the one that
# decides 84% of the gap: v48 makes $71,170 from STRAWBERRIES we never touch
# and $60,179 from MILK where we do a fifth. With a single verb the network
# could choose WHETHER to plant, never WHAT, and no downstream head could fix
# it: the market head cannot sell what is not produced -measured, moving its
# strawberry factor does not change a single dollar-.
OPS_VOCAB = [
    *("PLANT_" + c for c in spec.CROP_LIST),
    "WATER", "HARVEST", "FERTILIZE", "DIG",
    "FEED", "CARE", "COLLECT_FERTILIZER", "PLACE",
    "BUILD_COOP", "BUILD_PASTURE",
    "PICKUP_ANIMAL", "PICKUP_WHEAT", "PICKUP_FERTILIZER", "DROP",
]
OPS_IX = {v: i for i, v in enumerate(OPS_VOCAB)}
N_OPS = len(OPS_VOCAB)

# MASK OF THE DIMENSIONS THAT REALLY DECIDE.
#
# The micro log-prob is a SUM over 1+N_OPS channels x 100 tiles = 1,614
# Gaussian dimensions, but only the tiles with legal operations decide
# anything, and a verb's logit only matters if there WAS something to compare
# it against. Measured: unmasked, 99.4% of PPO's importance ratios saturate at
# the +-10 clip after ONE gradient step -the ratio is a product over ~1,570
# dimensions of pure noise-. Including them is not a design choice: it is a
# bug.
MASK_ACC = None
FILTER_BY_INVENTORY = False

# The verb is thresholded with argmax over the logits of the LEGAL options.
# Exploration does not come from sampling here: the micro map arrives already
# perturbed by the policy's Gaussian, so the argmax of a perturbed map is
# already a stochastic decision with a log-probability PPO can use.
import numpy as np

# THRESHOLD for the EXTRA tiles. `macro.apply_params` sets it once per turn
# from the learned `f_extra_threshold` parameter; the value below is only the
# starting point and amounts to "no extra tile enters".
#
# What the extra tiles are. Measured at championship scale: the heuristic
# offers 8.94 tasks per turn for 10.88 units and leaves out another 5.81 the
# engine DOES consider legal. Turns are passed 52.8% of the time -v48 passes
# 5.3%- and 44.5 of those points are from having no task to assign, not from
# choosing badly. On 22.7% of turns the heuristic offers nothing while legal
# plays exist. Wasted verbs: FERTILIZE 3,010, HARVEST 837, DROP 327,
# PICKUP 164 -and we harvest 111 times per episode against v48's 420-.
#
# Why the network could not fix it: the map MULTIPLIES the value of existing
# tasks, but a tile where `tile_task` returns None never enters the dictionary
# and is invisible to learning.
#
# What decides here: LEGALITY comes from the engine, the VERB is chosen by the
# network among the legal options and the VALUE is emitted by the network.
# There is no hand-written preference order or value: that would put a
# heuristic back exactly where one is being removed.
EXTRA_TILE_THRESHOLD = 2500.0


def enable_mask(n_keys=0):
    import numpy as np
    global MASK_ACC
    MASK_ACC = np.zeros((1 + N_OPS + 2 * int(n_keys), BOARD, BOARD),
                        dtype=np.float32)


def collect_mask():
    global MASK_ACC
    m = MASK_ACC
    MASK_ACC = None
    return m


# WHY A UNIT PASSES. Measured against an uncapped v48, 11% of our actions are
# PASS against their 5%: about 480 wasted unit-turns per episode, which is
# roughly the whole watering deficit that kills half our plants. But "PASS"
# has two causes that need opposite fixes -no legal task was offered, or one
# was offered and the emitted value was not positive- and the number alone
# does not separate them. Off by default; costs nothing when disabled.
PASS_ACC = None


def enable_pass_stats():
    global PASS_ACC
    PASS_ACC = {"units": 0, "pass": 0, "no_task_at_all": 0,
                "all_blocked_by_inventory": 0, "all_value_nonpositive": 0,
                "taken_by_another": 0, "offers": 0, "blocked": 0}


def collect_pass_stats():
    global PASS_ACC
    d = PASS_ACC
    PASS_ACC = None
    return d


def tile_options(obs, farm, x: int, y: int, free_capacity: int, ctx=None,
                 macro=None) -> list:
    """All LEGAL operations on that tile, unvalued and unordered.

    Returns [(index in OPS_VOCAB, full operation), ...].
    """
    tile = farm["tiles"][y][x]
    day = obs["day"]
    if ctx is None:
        ctx = TurnContext(obs, farm)
    out = []

    if _is_shed_access(x, y):
        shed = obs["private"]["shed"]
        in_shed = [a for a in spec.ANIMALS if int(shed.get(a, 0)) > 0]
        if in_shed:
            # WHICH animal to take from the shed was another hand-written
            # formula, the fourth of the same family as `best_crop` and
            # `seed_orders`. It is weighted by the LEARNED factor of the
            # product the animal gives, which already lives in the macro
            # vector (see `animal_value`).
            best = max(in_shed, key=lambda a: animal_value(ctx, a, macro))
            out.append((OPS_IX["PICKUP_ANIMAL"], ["PICKUP", best, 1]))
        # QUANTITY, not just the verb. `OPS_VOCAB` has `PICKUP_WHEAT` as a
        # verb with no argument, so this used to emit ALWAYS 1 while the
        # heuristic takes `min(hungry, wheat in shed)`. It was measured by
        # comparing the two task tables over THE SAME state: 96.8% identical,
        # 0% of tiles lost, 0% of values different, and the remaining 3.2% were
        # exactly this -`PICKUP WHEAT 2` against `PICKUP WHEAT 1`-. With one
        # unit bringing a single wheat per trip, the livestock feeding chain
        # runs at half capacity, and livestock is the entire economy at this
        # scale.
        #
        # The quantity is NOT a strategic decision: it is fixed by how many
        # animals are hungry and how much is in the shed. It belongs to the
        # symbolic side, like legality and routing. The network still chooses
        # the VERB.
        wheat = int(shed.get("WHEAT", 0))
        if wheat > 0:
            hungry = sum(1 for row in farm["tiles"] for t in row
                              if isinstance(t, dict) and t.get("animal")
                              and not t.get("fed_today"))
            out.append((OPS_IX["PICKUP_WHEAT"],
                        ["PICKUP", "WHEAT", max(1, min(hungry, wheat))]))
        fert = int(shed.get("FERTILIZER", 0))
        if fert > 0:
            fertilizable = sum(1 for row in farm["tiles"] for t in row
                                if isinstance(t, dict) and t.get("kind") == "PLANT"
                                and t.get("fertilized_until_day", -1) < obs["day"])
            out.append((OPS_IX["PICKUP_FERTILIZER"],
                        ["PICKUP", "FERTILIZER", max(1, min(fert, fertilizable, int(FERT_PER_TRIP)))]))
        out.append((OPS_IX["DROP"], ["DROP"]))

    if tile is None:
        for kind in ("COOP", "PASTURE"):
            if any(spec.ANIMALS[a]["structure"] == kind for a in ctx.pending_animals):
                out.append((OPS_IX["BUILD_" + kind], ["BUILD_" + kind]))
        # NO SUSTAINABILITY CAP. There used to be `if free_capacity > 0`, and
        # `free_capacity` comes from `sustainable_tiles`, MY estimate of how
        # many tiles can be tended. That is not engine legality, it is an
        # opinion, and it blocked planting on 17.9% of the empty tiles that had
        # seed available (2,904 of 16,241 measured). If overplanting is costly
        # the return will say so and the network will stop; if it is not, it
        # can now do it.
        #
        # What IS kept is `plantable`: a crop that does not have time to mature
        # yields ZERO by engine mechanics, exactly like not being able to plant
        # on LOCKED. That is a fact, not an opinion.
        # ONE OPTION PER CROP with seed available and time to mature.
        # Legality and viability still come from the engine; the PREFERENCE
        # goes to the network.
        _seeds = obs["private"].get("seeds", {})
        for _c in spec.CROP_LIST:
            if int(_seeds.get(_c, 0)) <= 0:
                continue
            if not plantable(obs, _c):
                continue
            out.append((OPS_IX["PLANT_" + _c], ["PLANT", _c]))
        return out

    if not isinstance(tile, dict):
        return out
    kind = tile.get("kind")

    if kind == "WEED":
        out.append((OPS_IX["DIG"], ["DIG"]))
        return out

    if kind == "PLANT":
        cd = spec.CROPS[tile["crop"]]
        age = day - tile["planted_day"]
        if not tile["watered_today"]:
            out.append((OPS_IX["WATER"], ["WATER"]))
        if tile.get("fertilized_until_day", -1) < day:
            out.append((OPS_IX["FERTILIZE"], ["FERTILIZE"]))
        if tile.get("yield_units", 0) > 0 and age >= cd["first_yield_day"]:
            out.append((OPS_IX["HARVEST"], ["HARVEST"]))
        return out

    if kind in ("COOP", "PASTURE"):
        animal = tile.get("animal")
        if animal is None:
            cands = [a for a in ctx.pending_animals if spec.ANIMALS[a]["structure"] == kind]
            if cands:
                best = max(cands, key=lambda a: animal_value(ctx, a, macro))
                out.append((OPS_IX["PLACE"], ["PLACE", best, 1]))
            return out
        if not tile.get("fed_today"):
            out.append((OPS_IX["FEED"], ["FEED"]))
        if tile.get("yield_units", 0) > 0:
            out.append((OPS_IX["HARVEST"], ["HARVEST"]))
        if tile.get("fertilizer_available"):
            out.append((OPS_IX["COLLECT_FERTILIZER"], ["COLLECT_FERTILIZER"]))
        if not tile.get("cared_today"):
            out.append((OPS_IX["CARE"], ["CARE"]))
    return out


def board_tasks(obs, farm, free_capacity: int, value_map=None, macro=None,
                       verb_map=None) -> dict:
    """(x,y) -> (value in $, operation) for each tile that offers something."""
    tasks = {}
    _deferred = []          # invisible tiles, resolved at the end
    free = free_capacity
    ctx = TurnContext(obs, farm)
    _turn_invs = (obs["private"].get("inventories") or []
                   )
    _turn_invs = [iv for iv in _turn_invs if isinstance(iv, dict)] or [{}]
    for y in range(BOARD):
        for x in range(BOARD):
            if not unlocked(farm, x, y):
                continue
            if verb_map is not None:
                # END TO END: LEGALITY from the engine, the VERB from the
                # network. This branch is only entered WITH a verb map: an
                # agent WITHOUT a network -a league opponent, a public agent-
                # used to fall in here and pick `options[0]`, the first legal
                # one, which is at random. Measured: the 8-day expert made
                # $2,340 as an opponent when it is worth $4,861, with 1 unit
                # instead of 9, and we won 1.000 against an opponent
                # lobotomised by OUR own training setup. With no map the
                # heuristic is used, and nothing of `tile_task` enters here
                # -neither its preference order nor its value formulas-.
                import math
                options = tile_options(obs, farm, x, y, free, ctx, macro)
                # Filtering by what the unit CARRIES is legality, not an
                # assignment concern: the engine IGNORES the action if the unit
                # is not carrying what it consumes. With the heuristic it made
                # no difference -it offered something else- but committing to
                # ONE verb per tile makes it decide. Measured without the
                # filter: the network chose PLACE on 95% of tiles, 66.9% of the
                # tasks could be executed by nobody, 86% PASS and $3,036.
                #
                # REVERTED after measuring it: filtering here trades one block
                # for another. Without the filter, 10.7 tasks/turn and $6,727;
                # with it, 0% unexecutable tasks but 3.8 tasks/turn and $4,484,
                # because a tile asking for FEED DISAPPEARS when nobody carries
                # wheat and then nobody goes to fetch it. The aggregate
                # `carried` is in the observation, so avoiding it is learnable;
                # teaching it by CLONING is not, because the expert is never in
                # that situation.
                if FILTER_BY_INVENTORY and _turn_invs:
                    options = [o_ for o_ in options
                                if any(_can_do(iv, o_[1]) for iv in _turn_invs)]
                if not options:
                    continue
                k, op = max(options,
                            key=lambda o: float(verb_map[o[0]][y][x]))
                # Only the LEARNER brings a verb map; the opponent does not, so
                # this says who accumulates without passing flags down the
                # pipe.
                if MASK_ACC is not None:
                    MASK_ACC[0, y, x] = 1.0        # the value decided here
                    if len(options) > 1:
                        for _k, _ in options:
                            MASK_ACC[1 + _k, y, x] = 1.0   # there was a choice
                r = float(value_map[y][x]) if value_map is not None else 0.0
                v = math.copysign(
                    math.expm1(abs(min(MAP_CAP, MAP_GAIN * r))), r)
                tasks[(x, y)] = (v, op)
                if op[0] == "PLANT":
                    free -= 1
                continue
            t = tile_task(obs, farm, x, y, free, ctx, macro)
            if t is None and verb_map is not None and value_map is not None:
                # SECOND PASS, and the order matters. Resolved here, an extra
                # tile with PLANT consumed the planting budget and strangled
                # the plantings the heuristic would have proposed: measured,
                # 8.62 tasks/turn fell to 5.78 -the mechanism was TAKING AWAY
                # instead of adding-. Noted down and resolved at the end,
                # the heuristic decides first at full capacity and the extra
                # tiles fill whatever is left. That makes it purely additive
                # and the learned threshold the only dial.
                _deferred.append((x, y))
                continue
            if t is not None:
                if value_map is not None:
                    import math
                    # THE NETWORK EMITS THE VALUE. This branch is only
                    # reached for tiles where `tile_options` offered nothing
                    # but `tile_task` did. Measured over 77,150 tile queries:
                    # the vocabulary offered nothing on 37,010 and in NONE of
                    # them did `tile_task` have anything to propose, so the
                    # heuristic fallback that used to live here never fired.
                    # The network emits in symlog space (where it was fitted);
                    # it is undone to get back to dollars.
                    r = float(value_map[y][x])
                    t = (math.copysign(math.expm1(
                        abs(min(MAP_CAP, MAP_GAIN * r))), r), t[1])
                tasks[(x, y)] = t
                if t[1][0] == "PLANT":
                    free -= 1
    # SECOND PASS: the tiles the heuristic declared empty and the engine
    # considers legal. LEGALITY from the engine, the VERB chosen by the network
    # among the legal options and the VALUE emitted by the network; the
    # THRESHOLD deciding whether it deserves a unit is the learned
    # `f_extra_threshold`. There is no hand-written preference order or value.
    #
    # It comes after the main sweep on purpose: resolved inline, an extra tile
    # with PLANT consumed the budget and strangled the heuristic's plantings
    # -measured, 8.62 tasks/turn fell to 5.78-. Here the heuristic has already
    # decided at full capacity and this only fills what is left.
    for _x, _y in _deferred:
        _ex = tile_options(obs, farm, _x, _y, free, ctx, macro)
        if not _ex:
            continue
        import math
        _lg = np.array([float(verb_map[o[0]][_y][_x]) for o in _ex],
                       dtype=np.float64)
        _sel = int(np.argmax(_lg))
        _k2, _o2 = _ex[_sel]
        _r2 = float(value_map[_y][_x])
        _v2 = math.copysign(
            math.expm1(abs(min(MAP_CAP, MAP_GAIN * _r2))), _r2)
        _v2 -= EXTRA_TILE_THRESHOLD
        if _v2 <= 0.0:
            continue
        if _o2[0] == "PLANT":
            if free <= 0:
                continue
            free -= 1
        tasks[(_x, _y)] = (_v2, _o2)
        if MASK_ACC is not None:
            MASK_ACC[0, _y, _x] = 1.0
            if len(_ex) > 1:
                for _k3, _ in _ex:
                    MASK_ACC[1 + _k3, _y, _x] = 1.0

    return tasks


def _action_for(pos, tile, tasks) -> list:
    """Execute the task if already on the tile; otherwise step towards it."""
    if tile == pos:
        return tasks[tile][1]
    return [step_toward(pos, tile)]


def _can_do(inv, op) -> bool:
    """If the unit is NOT carrying what the operation consumes, the engine ignores it.

    FEED spends 1 wheat from THE UNIT's inventory, FERTILIZE 1 fertiliser, and
    PLACE needs the animal in hand. Assigning those tasks to someone carrying
    nothing gives the turn away: a silent no-op. The pair's value is zeroed so
    the assignment prefers anything else, including the dummy column.
    """
    if op[0] == "DROP":
        return _can_drop(inv)
    req = REQUIRES.get(op[0])
    if req is not None:
        return int(inv.get(req, 0)) > 0
    if op[0] == "PLACE" and len(op) > 1:
        return int(inv.get(op[1], 0)) > 0
    return True


# NOT IMPLEMENTED, written down so it is not rediscovered. There used to be a
# constant STICKINESS = 0.0 here -a bonus for keeping the previous turn's
# destination- declared and never read: tuning it did absolutely nothing. The
# measurement that motivated it still stands and is still interesting: 23.8%
# of the destination decisions are a change made ALREADY EN ROUTE, because the
# Hungarian reassigns from scratch every turn. A change is not waste -something
# urgent may have appeared- so if it is ever implemented, the weight must be
# LEARNED like the others, not set by hand.


def _assign_hungarian(units, tasks, invs=None, previous=None, stickiness=0.0,
                      key_map=None, query_map=None) -> list:
    """Minimum-cost assignment: maximises the TOTAL discounted value.

    The greedy fails in a specific and frequent way: the first unit takes a
    task that was equally close to another unit, and leaves that other one with
    nothing nearby. Since the order of units means nothing, that loss is pure
    arbitrariness.

    The `n` dummy columns of value 0 do two things at once: they guarantee
    there are at least as many columns as rows, and they ensure no unit accepts
    a negative-value task -a free dummy always remains, which is better-.
    `best_crop` can return a crop of negative profit when capital dominates, so
    the case does happen.
    """
    tiles = list(tasks)
    n = len(units)
    m = len(tiles) + n
    value = [[0.0] * m for _ in range(n)]
    invs = invs or [{}] * len(units)
    if PASS_ACC is not None:
        PASS_ACC["units"] += len(units)
    for i, pos in enumerate(units):
        row = value[i]
        inv = invs[i] if i < len(invs) and isinstance(invs[i], dict) else {}
        for j, tile in enumerate(tiles):
            v, op = tasks[tile]
            if not _can_do(inv, op):
                if PASS_ACC is not None:
                    PASS_ACC["blocked"] += 1
                continue                  # stays 0: loses to the dummy
            if PASS_ACC is not None:
                PASS_ACC["offers"] += 1
            row[j] = v * (STEP_DISCOUNT ** dist(pos, tile))
            # PER-UNIT PREFERENCE. The value is the same for everybody, so
            # without this term the only thing telling two units apart is
            # distance, and when they want the same tile the loser takes the
            # dummy column: measured, that is 94% of all PASSes.
            #
            # MULTIPLICATIVE, like the market factors, so the entry keeps its
            # dollar meaning and needs no scale constant: the dot product of
            # the unit's query with the tile's key starts at exactly 0 -the
            # head is zero-initialised- and exp(0) = 1 leaves the matrix
            # identical to the one before this existed.
            if key_map is not None and query_map is not None:
                _z = 0.0
                for _c in range(len(key_map)):
                    _z += float(query_map[_c][pos[1]][pos[0]]) * \
                        float(key_map[_c][tile[1]][tile[0]])
                row[j] *= math.exp(max(-13.8, min(13.8, _z)))
            if stickiness and previous is not None and previous.get(i) == tile:
                row[j] *= (1.0 + stickiness)

    # WHICH KEY DIMS DECIDED. Same rule as the verbs: a dimension only
    # counts if there was a COMPARISON. A tile wanted by a single unit, or a
    # unit with a single positive option, is settled without the key term, so
    # its dims cannot change the action and must contribute exactly zero to
    # the importance ratio. Marking every task tile instead took the ratio
    # saturation from 3% to 15-23%, which is the same bug that once saturated
    # 99.4% of it.
    if (MASK_ACC is not None and key_map is not None
            and MASK_ACC.shape[0] > 1 + N_OPS):
        _k = len(key_map)
        _per_tile = [sum(1 for i in range(n) if value[i][j] > 0.0)
                     for j in range(len(tiles))]
        for j, tile in enumerate(tiles):
            if _per_tile[j] > 1:
                for _c in range(_k):
                    MASK_ACC[1 + N_OPS + _c, tile[1], tile[0]] = 1.0
        for i, pos in enumerate(units):
            if sum(1 for j in range(len(tiles)) if value[i][j] > 0.0) > 1:
                for _c in range(_k):
                    MASK_ACC[1 + N_OPS + _k + _c, pos[1], pos[0]] = 1.0

    actions = []
    _assigned = set()
    if PASS_ACC is not None:
        _assigned = {j for j in max_assignment(value) if j < len(tiles)}
    for i, j in enumerate(max_assignment(value)):
        if j >= len(tiles) or value[i][j] <= 0.0:
            if PASS_ACC is not None:
                PASS_ACC["pass"] += 1
                _row = value[i]
                _pos = [k for k in range(len(tiles)) if _row[k] > 0.0]
                if not tiles:
                    PASS_ACC["no_task_at_all"] += 1
                elif not _pos:
                    # the row is all zeros: either nothing was legal for this
                    # unit's inventory, or every task was valued at <= 0
                    _legal = sum(1 for k, t in enumerate(tiles)
                                 if _can_do(invs[i] if i < len(invs)
                                            and isinstance(invs[i], dict) else {},
                                            tasks[t][1]))
                    if _legal == 0:
                        PASS_ACC["all_blocked_by_inventory"] += 1
                    else:
                        PASS_ACC["all_value_nonpositive"] += 1
                else:
                    # there WAS something positive for this unit; another unit
                    # took it, which is the Hungarian doing its job
                    PASS_ACC["taken_by_another"] += 1
            actions.append(["PASS"])
            if previous is not None:
                previous[i] = None
        else:
            actions.append(_action_for(units[i], tiles[j], tasks))
            if previous is not None:
                previous[i] = tiles[j]
    return actions


def assign_units(obs, free_capacity: int,
                 value_map=None, previous=None, macro=None, verb_map=None,
                 key_map=None, query_map=None) -> list:
    """One action per unit, maximising the discounted value of the set.

    The budget is not the problem it was feared to be: 16 units x 116 columns
    solve exactly in 0.23 ms, against the ~83 ms average allowed by the 60 s
    given for 720 turns (see `tests/test_assign.py`).

    ONE METHOD ONLY. The greedy and the route assigner were kept to be measured
    against; they are measured and the Hungarian wins. Unused paths are just
    surface for a bug to hide in.
    """
    me = int(obs["player"])
    farm = obs["farms"][me]
    units = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]

    tasks = board_tasks(obs, farm, free_capacity, value_map, macro, verb_map)
    if not tasks:
        return [["PASS"] for _ in units]
    invs = obs["private"].get("inventories", [])
    adh = 0.0
    if macro is not None:
        from ..macro import assignment_stickiness
        adh = assignment_stickiness(macro)
    return _assign_hungarian(units, tasks, invs, previous, adh,
                             key_map, query_map)
