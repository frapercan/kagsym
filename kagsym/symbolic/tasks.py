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
import os as _os_coord

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


_PLANTABLE_MEMO: dict = {}


def plantable(obs, crop: str) -> bool:
    key = (days_left(obs), crop, _under_plan())
    hit = _PLANTABLE_MEMO.get(key)
    if hit is None:
        hit = _PLANTABLE_MEMO[key] = _plantable(obs, crop)
        if len(_PLANTABLE_MEMO) > 4096:
            _PLANTABLE_MEMO.clear()
    return hit


def _plantable(obs, crop: str) -> bool:
    """There is no point planting what there will be no time to harvest.

    THE FULL CYCLE, AND IT IS NOT A MISREADING -- it was measured. An audit
    pointed out that the engine starts yielding at `first_yield_day` and keeps
    accumulating until `max_yield_day`, so asking for the whole cycle to fit
    refuses a melon sown with eleven days left even though it would yield on
    day ten. Correct about the engine, wrong about the game: on 200 paired
    seeds against v48, relaxing the test to `first_yield_day` LOSES $1,813
    (se 301, t -6.0, better on only 63 of 200).

    Why: what binds is unit-turns, not tiles. A crop that yields once and never
    fills still occupies a tile and eats the waterings that a crop which
    completes would have used. The test was never a yield guard, it is a
    LABOUR guard -- the comment above it had the mechanism wrong and the
    outcome right.
    """
    from ..plan import get_plan
    if get_plan() is not None:
        # UNDER A PLAN the guard is the engine's, not the labour heuristic:
        # a crop can be sown if it yields at all before the game ends
        # (first yield on `first_yield_day`, harvested and sold that day).
        # Whether a green cycle pays is the plan's decision to make and the
        # search's to measure (rung 3 of the ladder: 3 days, carrot).
        return spec.CROPS[crop]["first_yield_day"] < days_left(obs)
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


def _under_plan() -> bool:
    from ..plan import get_plan
    return get_plan() is not None


def _zone_of(tile) -> str:
    h = BOARD // 2
    return ("N" if tile[1] < h else "S") + ("W" if tile[0] < h else "E")


def _plan_zones(obs) -> float:
    from ..plan import get_plan
    pl = get_plan()
    return float(getattr(pl, "zones", 0.0)) if pl is not None else 0.0


def _home_zones(obs, units) -> list:
    """Each unit's home quadrant: the unlocked quadrants dealt round-robin in
    unit order (the farmer first), so a quadrant with more tiles gets the
    same share as the others; good enough to stop the criss-crossing."""
    farm = obs["farms"][int(obs["player"])]
    qs = [q for q in ("NW", "NE", "SW", "SE") if q in farm["unlocked_quadrants"]] or ["NW"]
    return [qs[i % len(qs)] for i in range(len(units))]


def _plan_load(obs) -> int:
    from ..plan import get_plan
    pl = get_plan()
    return 0 if pl is None else pl.load_on(int(obs["day"]))


def _shed_dist(tile) -> int:
    return min(dist(tile, a) for a in _shed_access_set())


_CARRIED_MEMO: dict = {}


def _carried_values(obs) -> list:
    """Per turn and content, computed once: it was recomputed for every
    unit and every shed tile of the turn (12 % of a game)."""
    priv = obs["private"]
    key = (int(obs["step"]), int(obs["player"]),
           tuple(tuple(sorted((k, int(v)) for k, v in (inv or {}).items())) for inv in (priv.get("inventories") or [])),
           tuple(sorted((k, int(v)) for k, v in (priv.get("shed", {}) or {}).items() if v)),
           tuple(sorted((k, float(v)) for k, v in obs["market"]["prices"].items())))
    hit = _CARRIED_MEMO.get(key)
    if hit is None:
        if len(_CARRIED_MEMO) > 256:
            _CARRIED_MEMO.clear()
        hit = _CARRIED_MEMO[key] = _carried_values_raw(obs)
    return list(hit)


def _carried_values_raw(obs) -> list:
    """What each unit's load is worth if dropped now: the price drop it avoids
    plus what the nightly flush would discard (shed overflow) or the game's
    end would forfeit (last day). Same margins as `_shed_task`, per unit."""
    from .market_ops import future_price, marginal_prices
    priv = obs["private"]
    shed = priv.get("shed", {}) or {}
    priv_inv = priv.get("inventories") or []
    shed_room = max(0, int(spec.DEFAULT_CONFIG["shedCapacity"]) - int(sum(shed.values())))
    carried_total = sum(int(n_) for inv in priv_inv for item, n_ in (inv or {}).items()
                        if item in spec.PRODUCTS)
    overflow_share = 0.0 if carried_total <= 0 else max(0.0, carried_total - shed_room) / carried_total
    lost = 1.0 if days_left(obs) <= 1 else overflow_share
    out = []
    for inv in priv_inv:
        v_ = 0.0
        for item, n_ in (inv or {}).items():
            if item in spec.PRODUCTS and n_:
                now = float(sum(marginal_prices(obs, item, int(n_))))
                later = float(future_price(obs, item, spec.TURNS_PER_DAY)) * int(n_)
                v_ += max(0.0, now - later) + lost * now
        out.append(v_)
    return out


def _shed_tasks(obs, farm, ctx=None, macro=None) -> list:
    """What to take out of the shed, RANKED by value: one task per shed-access
    tile. Without this the animal chain never closes: the animal is bought,
    lands in the shed and stays there forever. Wheat matters just as much:
    FEED consumes 1 wheat FROM THE UNIT'S INVENTORY, so an animal nobody
    brings wheat to escapes after two days. The shed used to offer ONE task
    for all four access tiles, and with an animal waiting inside it was
    always the pickup of the animal: the wheat never came out (block B of
    the expansion ladder: 3 starving animals, 22 wheat in the shed, zero
    FEED on day 16).
    """
    priv = obs["private"]
    shed = priv["shed"]
    if ctx is None:
        ctx = TurnContext(obs, farm)
    out = []
    in_shed = [a for a in spec.ANIMALS if int(shed.get(a, 0)) > 0]
    for a in sorted(in_shed, key=lambda a: -animal_value(ctx, a, macro)):
        if ctx.free_slots.get(spec.ANIMALS[a]["structure"], 0) > 0:
            out.append((max(1.0, animal_value(ctx, a, macro) / ctx.days), ["PICKUP", a, 1]))
            break
    best = max(_carried_values(obs), default=0.0)
    if best > 0:
        out.append((best, ["DROP"]))
    fert = int(shed.get("FERTILIZER", 0))
    if fert > 0:
        fertilizable = sum(1 for row in farm["tiles"] for t in row
                            if isinstance(t, dict) and t.get("kind") == "PLANT"
                            and t.get("fertilized_until_day", -1) < obs["day"])
        if fertilizable > 0:
            n = min(fert, fertilizable, int(FERT_PER_TRIP))
            out.append((FERT_TRIP_VALUE * unit_price(obs, "FERTILIZER"), ["PICKUP", "FERTILIZER", n]))
    hungry_tiles = [t for row in farm["tiles"] for t in row
                    if isinstance(t, dict) and t.get("animal") and not t.get("fed_today")]
    hungry = len(hungry_tiles)
    if hungry > 0 and int(shed.get("WHEAT", 0)) > 0:
        n = min(hungry, int(shed["WHEAT"]))
        if _under_plan():
            # FEEDING IS A CHAIN: pick the wheat up, then feed. Chains are off
            # (CHAIN_VALUE 0), so only a unit already carrying wheat can be
            # given a FEED task; the trip is worth the animals it feeds.
            v = 0.0
            for t in hungry_tiles[:n]:
                av = _plan_animal_value(obs, t["animal"])
                v += av if int(t.get("consecutive_unfed", 0)) >= 1 else 2.0 * av / max(1, days_left(obs))
            out.append((max(WHEAT_TRIP_VALUE * unit_price(obs, "WHEAT"), v), ["PICKUP", "WHEAT", n]))
        else:
            out.append((WHEAT_TRIP_VALUE * unit_price(obs, "WHEAT"), ["PICKUP", "WHEAT", n]))
    out.sort(key=lambda t: -t[0])
    return out


def _shed_task(obs, farm, ctx=None, macro=None, rank: int = 0):
    """The rank-th shed task by value (the top one past the end, so every
    access tile keeps a task)."""
    ranked = _shed_tasks(obs, farm, ctx, macro)
    if not ranked:
        return None
    return ranked[rank] if rank < len(ranked) else ranked[0]


def _shed_rank(x: int, y: int) -> int:
    return sorted(_shed_access_set()).index((x, y))


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
        self.obs = obs
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
    if _under_plan():
        # UNDER A PLAN an animal already bought is dead capital until it is
        # placed, and a target animal is worth its remaining daily product.
        # `animal_net_value` books manure as a watering credit and comes out
        # <= 0, so BUILD, PICKUP and PLACE were valued 1 $ and lost every
        # tile and every turn to watering: measured, 15 animals bought for
        # 9,600 $ and 15 in the shed unplaced on day 24 (EXP-007 part 6).
        return _plan_animal_value(ctx.obs, a)
    v = ctx.value(a)
    if macro is None:
        return v
    try:
        from ..macro import market_factors
        return v * market_factors(macro).get(spec.ANIMALS[a]["product"], 1.0)
    except Exception:
        return v


def _plan_animal_value(obs, a: str) -> float:
    """What an animal returns over the rest of the game: its product at the
    current price plus one fertiliser a day (sold, so at half price to allow
    for the fall), minus feed; never below its cost while a day of income
    remains, because unplaced it returns nothing at all."""
    d = spec.ANIMALS[a]
    left = max(0, days_left(obs) - 1)
    prices = obs["market"]["prices"]
    prod_days = max(0, left - d["first_yield_day"])
    units = int(prod_days / max(1, d["interval"]))
    income = units * float(prices.get(d["product"], 0)) + left * 0.5 * float(prices.get("FERTILIZER", 0))
    feed = left * float(prices.get("WHEAT", 0))
    return max(float(d["cost"]) if prod_days > 0 else 0.0, income - feed)


def tile_task(obs, farm, x: int, y: int, free_capacity: int, ctx=None, macro=None):
    """(value in $, operation) of the best possible action on that tile."""
    tile = farm["tiles"][y][x]
    day = obs["day"]
    seeds = obs["private"].get("seeds", {})
    if ctx is None:
        ctx = TurnContext(obs, farm)

    if _is_shed_access(x, y):
        t = _shed_task(obs, farm, ctx, macro, rank=_shed_rank(x, y))
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
                if _under_plan():
                    # UNDER A PLAN the pen is worth the animal waiting for it:
                    # in the shed it earns nothing (block C: five sheep ten
                    # days in the shed, coops free, no pasture built, the
                    # build worth ~100 $ a turn against a 250 $ watering).
                    return (v, ["BUILD_" + kind])
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

        if ripe and days_left(obs) <= 1 and _under_plan():
            # LAST DAY under a plan: whatever is on the tile is all it will
            # ever yield. The plan says whether to water it first (+1 unit
            # for one action) or take it now; labour decides which pays and
            # the search measures it. Before FERTILIZE, which returned a
            # 5e-5 $ task ahead of a 70 $ harvest.
            from ..plan import get_plan
            if tile["watered_today"] or get_plan().water_last_on(int(day)) == 0:
                return (tile["yield_units"] * price, ["HARVEST"])

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

        if ripe and _under_plan() and (cd["ongoing"] or age >= cd["max_yield_day"]):
            # UNDER A PLAN, HARVEST BEFORE FERTILISE (and after the watering:
            # a watering in the yield window adds a unit first). The
            # fertilise branch below fires every day on an ongoing crop past
            # its first yield and was the tile's only task; it needs
            # fertiliser in hand, chains are off, so nobody could take it and
            # the fruit sat on the plant: 45 strawberry tiles, 66 units ripe,
            # the price at 339, zero harvests in a game (EXP-007 part 10).
            return (tile["yield_units"] * price, ["HARVEST"])

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
                if _under_plan():
                    return (animal_value(ctx, best, macro), ["PLACE", best, 1])
                return (max(1.0, animal_value(ctx, best, macro) / ctx.days),
                        ["PLACE", best, 1])
            return None
        prod = spec.ANIMALS[animal]["product"]
        price = unit_price(obs, prod)
        if not tile.get("fed_today"):
            if _under_plan():
                # UNDER A PLAN feeding is worth the animal, not two units of
                # its product: unfed for two days it escapes with everything
                # it would still yield, the same loss structure as a plant
                # that dies tonight (valued at its remaining yield above).
                # Measured before: feeding a goose was 100 $ against 250-1,500
                # for watering a melon, the crews watered, and the expansion
                # blocks lost 28-45 animals to hunger with wheat in the shed.
                v = _plan_animal_value(obs, animal)
                if int(tile.get("consecutive_unfed", 0)) >= 1:
                    return (v, ["FEED"])                     # tonight it escapes
                return (max(FEED_VALUE * price, v / max(1, days_left(obs))) * 2.0, ["FEED"])
            return (FEED_VALUE * price, ["FEED"])   # two days unfed and it escapes
        if tile.get("yield_units", 0) > 0:
            return (tile["yield_units"] * price, ["HARVEST"])
        if tile.get("fertilizer_available"):
            return (unit_price(obs, "FERTILIZER"), ["COLLECT_FERTILIZER"])
        if not tile.get("cared_today"):
            if _under_plan() and tile.get("fed_today"):
                # CARE on a fed day banks a bonus unit for the next production
                # day (engine: pending_care_bonus): it is worth one unit of
                # the product, not half. Measured in the v48-shape block:
                # 6-9 of 15 animals cared for, milk 128 against v48's 335.
                return (price, ["CARE"])
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
# CLIP INSIDE expm1, ON BOTH SIDES. It used to read
# `abs(min(MAP_CAP, gain * r))`, which caps only the positive tail: for a very
# negative `r` the min passes it straight through and the abs then makes it
# large, so expm1 can overflow exactly the way `priorities()` did before it was
# fixed. `min(MAP_CAP, abs(gain * r))` caps both. At the operating point
# measured this changes nothing; it is insurance against a worker dying.
import os as _os_v
_VALOR = _os_v.environ.get("KAG_VALOR", "mapa")   # mapa | heuristica

MAP_CAP = 20.0
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
# MULTI-TURN COMMITMENT, as a fraction of the best alternative. >= 1 means a
# unit in flight never keeps its destination, which is how the assignment
# behaved before this existed. Learned (`f_commit`).
COMMIT_FRACTION = 2.0
# What a CHAINED errand is worth against a task the unit can do right now.
# 0 drops the pair, which is how the matrix behaved before chains existed.
# Learned (`f_chain`).
CHAIN_VALUE = 0.0


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


def swap_mask(arr):
    """Intercambia el acumulador de mascara y devuelve el anterior.

    Con DOS ASIENTOS los dos agentes deciden en el mismo proceso y el
    acumulador es un global del modulo: sin esto las dimensiones que decidio
    el asiento 0 se mezclarian con las del 1 y la mascara de PPO ignoraria
    ratios que si importan.
    """
    global MASK_ACC
    prev = MASK_ACC
    MASK_ACC = arr
    return prev


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
# "multiplicativa" (historico) o "aditiva": ver la nota en _assign_hungarian
_COORD = _os_coord.environ.get("KAG_COORD", "multiplicativa")
_ESCALA_TURNO = [0.0]


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
            # TAKING AN ANIMAL WITH NOWHERE TO PUT IT IS NOT WASTE.
            # Measured on one episode: 110 cows out of the shed for 14
            # placements, and six of ten units carrying something at any
            # moment. Restricting the pickup to when an empty structure of the
            # right kind exists LOSES $4,644 on 200 paired seeds against v48
            # (se 767, t -6.1). Carrying the animal is PRE-POSITIONING: the
            # trip to the shed is paid in advance so the placement is instant
            # when a structure frees up.
            #
            # It is the fourth of its kind. Withdrawing the green harvest loses
            # on 48 of 48 seeds; relaxing `plantable` loses $1,813; forcing the
            # destination commitment loses $6,890. What looks like waste in
            # this layer has been tuned into a configuration where it carries
            # weight, and measuring is the only way to tell.
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
            # NOT IMPLEMENTED, written down so the same wrong calculation is
            # not derived again. Harvesting before `max_yield_day` looks
            # strictly dominated: a non-ongoing crop keeps accumulating, so
            # picking it early throws away `max_yield - yield_units` units.
            # Measured on the 8h x 5d checkpoint, 8 episodes: 80 harvests land
            # on a planted tile, 64 of them in green (80%), holding 2.30 units
            # of a possible 4.2 -- 17 units an episode, $425 at base price
            # against a net of $275.
            #
            # Withdrawing the option was measured on 48 paired seeds and it
            # LOSES: 8h x 5d, $3,276 -> $3,229, -$46 with se 2, worse on 48 of
            # 48; at 24h x 30d it is null, +$277 +- 284. The gross-yield
            # arithmetic ignores what collecting those extra units costs: the
            # tile has to be watered every day until it fills, and unit-turns
            # are the binding resource, not fruit.
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
                # The engine resolves DROP/PICKUP before the LOCKED guard and
                # lets units stand on locked tiles: the three locked
                # shed-access tiles are legal standing positions for the shed.
                # Only the unlocked one was offered, so every unit queued on
                # the same corner.
                # Only WITHOUT a value map: with one, every other column is
                # priced by the network and a heuristic dollar figure here
                # dominates the matrix (measured: v5 vs passive on seed
                # 7102, 89,117 -> 74,176 alone, 43,754 with per-unit columns).
                if _is_shed_access(x, y) and value_map is None:
                    t = _shed_task(obs, farm, ctx, macro, rank=_shed_rank(x, y))
                    if t is not None:
                        tasks[(x, y)] = t
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
                    math.expm1(min(MAP_CAP, abs(MAP_GAIN * r))), r)
                # WHOSE DOLLARS DECIDE. The map REPLACES whatever the symbolic
                # layer computed, so `tile_task` only ever supplies the verb.
                # Instrumented on a real episode: on the 251 turns with a plant
                # that dies tonight, the rescue enters the matrix at 0.3 --
                # against the $472 the heuristic computes for it -- with the
                # nearest unit 1.02 steps away. Not capacity, not distance, not
                # the discount: the valuation says it is worth nothing.
                # And handing them back LOSES $11,420 on 200 paired seeds
                # against v48 (se 740, t -15.4, better on 27 of 200). So the
                # 0.3 is not a bug: relative to everything else on the board,
                # the network is saying that saving that plant is not worth a
                # unit-turn -- and its relative scale beats the heuristic's
                # dollars by eleven thousand. The heuristic's figures are not
                # commensurable across task types; the map's are.
                # KAG_VALOR=heuristica keeps the A/B available.
                if _VALOR == "heuristica":
                    _th = tile_task(obs, farm, x, y, free, ctx, macro)
                    if _th is not None:
                        v = _th[0]
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
                    if _VALOR != "heuristica":
                        r = float(value_map[y][x])
                        t = (math.copysign(math.expm1(
                            min(MAP_CAP, abs(MAP_GAIN * r))), r), t[1])
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
            math.expm1(min(MAP_CAP, abs(MAP_GAIN * _r2))), _r2)
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


def _chain_for(pos, tile, op, inv, ctx):
    """(shed tile, item, quantity, total distance) to fetch what `op` needs.

    None when the operation needs nothing this unit lacks, or when the shed
    does not hold it -- in which case the pair is dropped as before.
    """
    if not ctx or not ctx.get("access"):
        return None
    item = REQUIRES.get(op[0])
    if item is None and op[0] == "PLACE" and len(op) > 1:
        item = op[1]
    if item is None or int(inv.get(item, 0)) > 0:
        return None
    if int((ctx.get("shed") or {}).get(item, 0)) <= 0:
        return None
    # The nearest shed access tile, by the total path: going to the far side
    # of the shed to save one step on the way out is a real trade-off and the
    # engine settles it, not a preference.
    best = None
    for acc in ctx["access"]:
        d = dist(pos, acc) + dist(acc, tile)
        if best is None or d < best[1]:
            best = (acc, d)
    if best is None:
        return None
    return best[0], item, 1, best[1]


def _assign_hungarian(units, tasks, invs=None, previous=None, stickiness=0.0,
                      key_map=None, query_map=None, chain_ctx=None) -> list:
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
    invs = invs or [{}] * len(units)
    # ONE DROP COLUMN PER CARRYING UNIT. The shed has one access tile per
    # unlocked quadrant, so with one quadrant there was ONE column for the
    # whole crew: the Hungarian sent one unit per turn to the shed, the rest
    # kept harvesting or passed, and the load died in their hands. Measured
    # on the 4-day solitaire, 25 carrot tiles, 6 hands: 75 units harvested,
    # 24 sold, 51 carried at the close. Every carrying unit now has its own
    # column on the same tile, worth what IT carries (the column's value was
    # the crew's maximum, so the unit with 3 units looked like the one with
    # 12). In map mode the network's value is kept and scaled by that share.
    _deadline = None if chain_ctx is None else chain_ctx.get("deadline")
    _zones = 0.0 if chain_ctx is None else float(chain_ctx.get("zones") or 0.0)
    _home = (chain_ctx.get("home") or []) if chain_ctx is not None else []
    if _zones and len(_home) < len(units):
        _zones = 0.0
    _load = 0 if chain_ctx is None else int(chain_ctx.get("load") or 0)
    _carried_n = [sum(int(n_) for it, n_ in (inv or {}).items() if it in spec.PRODUCTS)
                  if isinstance(inv, dict) else 0 for inv in invs]
    _drop_tiles = [t for t in tiles if tasks[t][1][0] == "DROP"]
    _carried = None
    _share = None
    if _drop_tiles and chain_ctx is not None and chain_ctx.get("carried") is not None:
        _carried = list(chain_ctx["carried"])
        _cmax = max(_carried) if _carried else 0.0
        _share = [(c / _cmax if _cmax > 0 else 0.0) for c in _carried]
        _n_carry = sum(1 for c in _carried if c > 0)
        for _t in _drop_tiles:
            tiles.extend([_t] * max(0, _n_carry - 1))
    m = len(tiles) + n
    value = [[0.0] * m for _ in range(n)]
    if PASS_ACC is not None:
        PASS_ACC["units"] += len(units)
    # THE KEY TERM, VECTORISED. One dot product per (unit, tile) pair in pure
    # Python is ~123,000 lookups per episode per environment, in the hottest
    # loop of the executor -- which is 94% of the cost of a turn. Measured, it
    # took the trainer from 7.5 to 3.0 updates a minute. One matmul per turn
    # instead: (n_units, K) x (K, n_tiles).
    _chain = {}
    _EZ = None
    if key_map is not None and query_map is not None and tiles and units:
        _ks = np.asarray(key_map, dtype=np.float64)
        _qs = np.asarray(query_map, dtype=np.float64)
        _tx = np.fromiter((t[0] for t in tiles), int, len(tiles))
        _ty = np.fromiter((t[1] for t in tiles), int, len(tiles))
        _ux = np.fromiter((p[0] for p in units), int, len(units))
        _uy = np.fromiter((p[1] for p in units), int, len(units))
        _EZ = np.exp(np.clip(_qs[:, _uy, _ux].T @ _ks[:, _ty, _tx], -13.8, 13.8))
    # mediana de los valores de tarea del turno: robusta a la cola larga que
    # produce el expm1, y comun a todas las celdas
    if _COORD == "aditiva" and tasks:
        _vs = np.array([abs(tasks[t][0]) for t in tiles], dtype=float)
        _ESCALA_TURNO[0] = float(np.median(_vs)) if len(_vs) else 0.0
    for i, pos in enumerate(units):
        row = value[i]
        inv = invs[i] if i < len(invs) and isinstance(invs[i], dict) else {}
        for j, tile in enumerate(tiles):
            v, op = tasks[tile]
            if not _can_do(inv, op):
                if PASS_ACC is not None:
                    PASS_ACC["blocked"] += 1
                # CHAINED ERRAND. Dropping the pair here is what makes
                # multi-step play impossible: a unit that is not carrying
                # wheat never sees the hungry animal, so "fetch it, then feed
                # it" is a decision nobody can take. Measured on a real
                # episode: we pick up as much wheat as v48 -232 against 238-
                # and convert it into 0.83 feedings against their 1.46, and we
                # pick an animal out of the shed 158 times to place it 12.
                #
                # If the shed holds what the operation consumes, the pair
                # stays, valued by the END of the chain and discounted by the
                # WHOLE path -- unit to shed, shed to tile. The route is
                # mechanics, exactly like the Manhattan distance already here;
                # what the task is worth is still the network's.
                # OVERRIDE PARA MEDIR. `CHAIN_VALUE` viene del macro, que nunca
                # se entreno con el, asi que en inferencia esta en su defecto
                # 0,0004 -- apagado. KAG_CADENA lo fuerza para poder medir el
                # mecanismo sin reentrenar. Lo que arregla: sin el, la casilla
                # que pide FEED desaparece cuando nadie lleva trigo, y entonces
                # nadie va a buscar trigo; la accion previa que habilita la
                # siguiente se vuelve invisible.
                import os as _os_c
                _cv = _os_c.environ.get("KAG_CADENA", "")
                _CV = float(_cv) if _cv else CHAIN_VALUE
                if _CV <= 0.0:
                    continue              # stays 0: loses to the dummy
                _memo = chain_ctx.setdefault("_chain_memo", {}) if chain_ctx is not None else None
                _mk = (pos, tile, op[0], tuple(sorted((k, int(v)) for k, v in (inv or {}).items() if v))) if _memo is not None else None
                if _memo is not None and _mk in _memo:
                    _ch = _memo[_mk]
                else:
                    _ch = _chain_for(pos, tile, op, inv, chain_ctx)
                    if _memo is not None:
                        _memo[_mk] = _ch
                if _ch is None:
                    continue
                _acc, _item, _qty, _dtot = _ch
                row[j] = _CV * v * (STEP_DISCOUNT ** _dtot)
                _chain[(i, j)] = (_acc, _item, _qty)
                if PASS_ACC is not None:
                    PASS_ACC["offers"] += 1
                if _EZ is not None:
                    row[j] *= float(_EZ[i, j])
                if stickiness and previous is not None and previous.get(i) == tile:
                    row[j] *= (1.0 + stickiness)
                continue
            if PASS_ACC is not None:
                PASS_ACC["offers"] += 1
            if _zones and op[0] not in ("DROP", "PICKUP", "PLACE") and _zone_of(tile) != _home[i]:
                # ZONES (a plan field): each unit owns a quadrant, like a route
                # agent; work in another quadrant is discounted so the crew
                # stops criss-crossing (measured: 60 % of unit-turns were
                # moves in the 60-tile block, v48 makes 1.08 moves per work).
                v = v * (1.0 - _zones)
            if _share is not None and op[0] == "DROP" and i < len(_share):
                v = v * _share[i]
                if _load > 0 and _carried_n[i] < _load:
                    # The plan says how full a hand goes to the shed; the
                    # deadline overrides it (a trip that cannot wait).
                    _forced = _deadline is not None and _deadline <= _shed_dist(pos) + 2
                    if not _forced:
                        continue
            if _deadline is not None and op[0] == "WATER":
                # Watering on the last day pays only through a harvest that
                # still reaches the market.
                _need = dist(pos, tile) + 2 + _shed_dist(tile) + 1 + 1
                if _need > _deadline:
                    continue
            if _deadline is not None and op[0] == "HARVEST":
                # LAST DAY: the score is cash, and a harvest reaches it only
                # through walk + HARVEST + walk to the shed + DROP + a SELL
                # order the turn after. What cannot make that trip is worth 0.
                _need = dist(pos, tile) + 1 + _shed_dist(tile) + 1 + 1
                if _need > _deadline:
                    continue
            row[j] = v * (STEP_DISCOUNT ** dist(pos, tile))
            _base_j = row[j]
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
            # COORDINACION: MULTIPLICATIVA O ADITIVA.
            #
            # Medido el 2026-09-23: el valor de la tarea pasa por
            # copysign(expm1(min(20,|r|)), r), o sea que puede llegar a 4,8e8,
            # mientras los dos terminos de coordinacion son factores de orden
            # uno -- el emparejamiento aprendido da _EZ entre 0,94 y 1,15, y la
            # continuidad que la red emite es x1,02. No pueden competir: para
            # cambiar una decision tendrian que valer tanto como una diferencia
            # de dolares que ya paso por una exponencial.
            #
            # Con KAG_COORD=aditiva entran SUMANDO en la misma escala que el
            # valor, como una fraccion de la celda base, para que puedan pelear
            # de tu a tu. Sigue siendo la red quien decide cuanto: `_EZ` sale
            # de sus claves y consultas, y la continuidad de su dial.
            #
            # AVISO: "mas coordinacion" NO es obviamente mejor -- forzar la
            # continuidad a 0,25 perdio -1.828 $ (t -1,4, 60 semillas
            # pareadas). Esto no sube el dial, le da ESCALA; solo se mide
            # reentrenando.
            _st = globals().get("_STICKY_FORZADO", None)
            _st = stickiness if _st is None else _st
            _mismo = previous is not None and previous.get(i) == tile
            if _COORD == "aditiva":
                # ESCALA GLOBAL DEL TURNO, no de la propia celda.
                # Con `abs(_base_j)` esto era una IDENTIDAD ALGEBRAICA --
                # row*(1+(EZ-1)) == row + row*(EZ-1) -- y el interruptor salia
                # inerte: mismo dinero y mismas operaciones hasta el digito.
                # Verificado y corregido el 2026-09-23.
                #
                # Con una escala comun del turno el bono es ABSOLUTO, que es
                # lo que permite que una tarea CONTINUADA y barata gane a una
                # NUEVA y cara. Si es proporcional a la celda, nunca puede.
                _esc = _ESCALA_TURNO[0]
                if _EZ is not None:
                    row[j] += _esc * (float(_EZ[i, j]) - 1.0)
                if _st and _mismo:
                    row[j] += _esc * _st
            else:
                if _EZ is not None:
                    row[j] *= float(_EZ[i, j])
                if _st and _mismo:
                    row[j] *= (1.0 + _st)

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

    # RESERVED TILES. A unit already in flight keeps its destination while
    # that destination is still worth at least COMMIT_FRACTION of the best
    # tile open to it, and the tile is taken out of the pool so nobody can
    # outbid it. With the fraction at its default of 2.0 nothing is ever
    # reserved -the destination is one of the alternatives, so it cannot be
    # twice the best of them- and the matching is the one of always.
    # OVERRIDE FOR MEASUREMENT. `COMMIT_FRACTION` comes from the macro, which
    # has never been trained with it, so at inference it sits at its 2.0
    # default and commitment is off. KAG_COMMIT forces a value so the dial can
    # be A/B'd without retraining. Why it matters: 23.8% of destinations are
    # changed EN ROUTE, and against the top replays we spend 63% of our
    # operations walking where they spend 31-38%.
    import os as _os
    _cf = _os.environ.get("KAG_COMMIT", "")
    _COM = float(_cf) if _cf else COMMIT_FRACTION
    if _COM < 1.0 and previous is not None:
        for i, pos in enumerate(units):
            t = previous.get(i)
            if t is None or t not in tasks or t == pos:
                continue                      # no errand, or already arrived
            j = tiles.index(t)
            if value[i][j] <= 0.0:
                continue
            _best = max(value[i][:len(tiles)])
            if value[i][j] >= _COM * _best:
                for i2 in range(n):           # the tile is reserved
                    if i2 != i:
                        value[i2][j] = 0.0
                for j2 in range(len(tiles)):  # and the unit is committed
                    if j2 != j:
                        value[i][j2] = 0.0

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
            _cj = _chain.get((i, j))
            if _cj is not None:
                # First leg of the errand: reach the shed and pick the item
                # up. The DESTINATION recorded is still the tile, so the
                # stickiness bonus and the commitment reservation keep the
                # unit on the errand instead of re-deciding next turn.
                _acc, _item, _qty = _cj
                if units[i] == _acc:
                    actions.append(["PICKUP", _item, int(_qty)])
                else:
                    actions.append([step_toward(units[i], _acc)])
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
    # CONTEXT FOR THE CHAINS: what the shed holds and where it can be reached.
    # Without this a unit that is not carrying what an operation consumes
    # simply never sees that tile, and "fetch it, then use it" is not a
    # decision anybody can take.
    _remaining = spec.EPISODE_STEPS - 1 - int(obs["step"])     # turns after this one
    _chain_ctx = {"shed": obs["private"].get("shed", {}) or {},
                  "access": sorted(_shed_access_set())}
    if value_map is None:
        # THE TACTICAL MECHANICS BELOW ARE FOR AGENTS WITHOUT A VALUE MAP
        # (the plan executor, the heuristic agent). Under the deployed
        # network they change the semantics its map was trained on -one
        # DROP column, no deadline- and a frozen policy cannot be judged on
        # a widened space: measured paired v5 vs v48, 200 episodes, they
        # cost -5,356 $ (t -4.8, worse on 127). They reach the network only
        # by retraining with them on. See docs/DEBT.md.
        _chain_ctx.update({"carried": _carried_values(obs),
                           "deadline": _remaining if days_left(obs) <= 1 else None,
                           "load": _plan_load(obs),
                           "zones": _plan_zones(obs), "home": _home_zones(obs, units)})
    adh = 0.0
    if macro is not None:
        from ..macro import assignment_stickiness
        adh = assignment_stickiness(macro)
    return _assign_hungarian(units, tasks, invs, previous, adh,
                             key_map, query_map, _chain_ctx)
