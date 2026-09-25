"""Market timing: where the game is decided.

Measured margins between comparable agents are below 1.6%, and 96.5% of unit
actions are identical across opponents. The competition is not in farming: it
is in **selling before the other side crashes the price**.

The rules come from two engine facts, not from intuition:

  1. **Only the town drains the market.** The centre: 1 of each non-fertiliser
     product every 24 turns. Each shop instance: 1 of each product it demands
     every 4 turns (x2 if it is a single-product shop). Nothing else.
  2. **Nobody demands fertiliser** -neither the centre nor any shop- so its
     price only ever falls. And melon is taken only by the centre: 1 every 24
     turns.

Everything follows: sell wheat late (the town drains it faster than the players
produce it, so it appreciates), sell fertiliser early or not at all, and treat
melon as a one-shot market.
"""
from __future__ import annotations

import math

from .. import spec

# spec.TURNS_PER_DAY is read at call time (see spec.set_turns_per_day): as a
# module alias it froze at import and stopped following `turnsPerDay`, silently
# desynchronising the executor from the engine.
# Share of the cash that may go on labour in one day. Labour multiplies
# everything else, but the fibonacci cost explodes (15 hands = $1,596/day) and
# with no brake it bankrupts the farm in two days.
LABOUR_BUDGET_FRACTION = 0.15
# EVERY VALUE BELOW IS LEARNED. These are the module-level defaults; the live
# values are written by `macro.apply_params` once per turn from the policy's
# vector. They are kept here so an agent without a macro still runs.
SEED_BUDGET_FRACTION = 0.5    # seed spend cap as a share of the cash
HAND_MARGIN = 3.0             # how much a hand must return over its cost
FEED_STOCK_DAYS = 3.0         # feed reserve, in days
SAT_HIGH, SAT_LOW = 0.85, 0.60   # shed saturation thresholds when selling
LAND_RETURN, LAND_CASH = 2.0, 1.5  # quadrant purchase condition
# NOT an engine limit -that is maxMarketOrdersPerTurn = 10- but a pacing
# decision, and therefore learned.
MAX_ANIMALS_PER_TURN = 2
# THE ONES THAT WERE STILL INLINE, inside function bodies. Measured live in
# ops (4 seeds, 24h x 30d): zeroing the manure credit costs -22.5% and
# strangling the feed cash -21.6%. All three are now `f_*` of the vector.
MANURE_CREDIT = 0.5           # manure credit when valuing an animal
FEED_CASH_FRACTION = 0.25     # share of cash that may go on feed at once
LABOUR_FLOOR = 60.0           # dollar floor of the labour budget
LIQUIDATION_DAYS = 2.0        # liquidation days at the season close
# Third and fourth passes of the audit. See `Macro`.
SEED_STOCK_PER_UNIT = 2.0     # seeds per unit before buying stops
SEED_FLOOR = 2.0              # minimum seeds before the stock rule bites
ANIMAL_CASH_RESERVE = 300.0   # cash reserved before buying an animal
LAST_HIRE_HOUR = 3.0          # last hour of the day for hiring
MIN_HAND_DAYS = 2.0           # minimum days to pay a hand back
SHOP_INT = spec.DEFAULT_CONFIG["townShopSellInterval"]
CENTER_INT = spec.DEFAULT_CONFIG["townCenterSellInterval"]

# Expected demand for each product per shop instance, computed from the
# engine's SHOPS table: shops are drawn uniformly among the 8.
_PER_SHOP = {
    p: sum((2 if len(prods) == 1 else 1) / len(spec.SHOPS)
           for prods in spec.SHOPS.values() if p in prods)
    for p in spec.PRODUCTS
}


def drain_rate(obs, product: str) -> float:
    """Units the town absorbs per turn, with the shops OPEN right now."""
    n = len(obs["town"].get("unlocked_shops", []))
    r = _PER_SHOP[product] * n / SHOP_INT
    if product in spec.TOWN_CENTER_PRODUCTS:
        r += 1.0 / CENTER_INT
    return r


# PRECIOS MEMOIZADOS. `market_price(item, inventory, params)` es pura, y
# medido sobre un episodio completo `params` llega None en los 720 turnos, asi
# que cae al MARKET_PARAMS global del motor y el precio depende solo de
# (producto, inventario). Medido: 10.166 llamadas a `marginal_prices` sobre
# 1.957 argumentos distintos, 80,7% redundante, 13,2% del tiempo del agente.
#
# Esto es EXACTO, no una aproximacion: se cachea una funcion pura. Con params
# no nulo se salta la cache, por si alguna configuracion los trae.
#
# La cache se invalida si el motor cambia su tabla de parametros, que es lo
# unico de lo que depende el resultado y no esta en la clave.
_PRECIO: dict = {}
_PRECIO_TABLA = None


def _precio(product: str, inv: int, params) -> float:
    if params is not None:
        return spec.market_price(product, inv, params)
    global _PRECIO_TABLA
    _t = getattr(spec, "MARKET_PARAMS", None)
    if _t is not _PRECIO_TABLA:
        _PRECIO.clear()
        _PRECIO_TABLA = _t
    k = (product, inv)
    v = _PRECIO.get(k)
    if v is None:
        v = _PRECIO[k] = spec.market_price(product, inv, None)
    return v


def marginal_prices(obs, product: str, k: int) -> list[float]:
    """What the engine pays for each of k units, in order."""
    inv = obs["market"]["inventory"][product]
    params = obs["market"].get("params")
    return [_precio(product, inv + j, params) for j in range(k)]


def future_price(obs, product: str, horizon: int, opp_flow: float = 0.0) -> float:
    """Price forecast `horizon` turns from now.

    Inventory moves for two known reasons -what the town drains and what the
    opponent dumps- and the second is predicted by the world model. Without a
    model, zero flow is assumed, which is conservative: it underestimates the
    future drop and therefore pushes towards selling earlier.
    """
    inv = obs["market"]["inventory"][product]
    future = inv + (opp_flow - drain_rate(obs, product)) * horizon
    return _precio(product, int(round(future)), obs["market"].get("params"))


def turns_left(obs) -> int:
    return spec.EPISODE_STEPS - 1 - int(obs["step"])


def sell_orders(obs, opp_flow=None, horizon: int = 12, final_urgency: int = None,
                macro=None) -> list:
    """How much of each product to sell THIS turn.

    For each product the marginal price of the k-th unit is compared with the
    price forecast `horizon` turns ahead. It sells while taking the money now
    beats waiting. There is no hand-picked 'drip rate': the drip emerges on its
    own, because each unit sold lowers the price of the next until it stops
    paying.
    """
    if macro is not None:
        from ..macro import sell_horizon
        horizon = sell_horizon(obs, macro)
    if final_urgency is None:
        final_urgency = int(round(LIQUIDATION_DAYS * spec.TURNS_PER_DAY))
    shed = obs["private"].get("shed", {})
    remaining = turns_left(obs)
    cap = spec.DEFAULT_CONFIG["shedCapacity"]
    used = sum(shed.values())

    # FEED RESERVE. FEED consumes 1 wheat per animal per day, and after two
    # days without eating the animal escapes. Measured: without this reserve
    # 137 units of wheat were sold and 17 of the 25 animals bought escaped, at
    # ~$1,700 each. Selling ONE unit too many costs a whole animal.
    me = int(obs["player"])
    n_animals = sum(1 for row in obs["farms"][me]["tiles"] for t in row
                     if isinstance(t, dict) and t.get("animal"))
    reserve = {"WHEAT": n_animals * max(1, (remaining + 1) // spec.TURNS_PER_DAY)} if n_animals else {}

    # LEARNED MARKET FACTORS, one per product. It is the last decision the
    # network did not make: the board is already decided tile by tile by the
    # network, but what to sell and how long to hold came out of a fixed
    # formula.
    #
    # The gap it attacks is measured: we sell $59,872 against v48's $200,760,
    # and $71,170 of that difference is STRAWBERRIES we never touch and
    # $60,179 MILK where we do a fifth -84% in two products-.
    #
    # A factor > 1 lowers that product's target price (we trade more of it)
    # and raises its priority in the queue when there are more orders than
    # slots. Neutral = 1.0, so without a macro behaviour is unchanged.
    _levels = {}
    if macro is not None:
        try:
            from ..macro import market_factors
            _levels = market_factors(macro)
        except Exception:
            _levels = {}

    # PER-TURN SELLING RULE. `_levels` is the LEVEL of each product -how much
    # we want to trade it- and it is constant for all 24 hours, because the
    # macro is emitted once a day: one RL step is a DAY (see `environment.py`)
    # and this layer plays the whole day on its own.
    #
    # What was missing is not looking at the state -this function already
    # does- but being able to learn HOW MUCH to react to each signal. That is
    # what these five coefficients add:
    #
    #     factor_p(t) = level_p * exp( sum_j w_j * x_j(t) )
    #
    # The daily pass emits the RULE; the rule is evaluated against each turn's
    # state. That gives decisions per situation without multiplying network
    # passes by 24, which is what the competition's one second per turn
    # allows.
    #
    # The five signals are dimensionless and centred on 0, and the weights are
    # 0 by default (w = logit(0.5)), so at startup the factor equals the level
    # EXACTLY and behaviour is unchanged.
    from ..macro import turn_weights
    _w = turn_weights(macro)
    _live = any(abs(v) > 1e-9 for v in _w.values())
    _z0 = 0.0
    if _live:
        _pnow = obs["market"]["prices"]
        _shed_value = sum(float(_pnow.get(_q, 0)) * int(_c) for _q, _c in shed.items())
        _money = float(obs["farms"][me]["money"])
        _z0 = (_w["shed"] * (used / max(1, cap) - 0.5)
               + _w["season"] * ((1.0 - remaining / max(1, spec.EPISODE_STEPS)) - 0.5)
               + _w["cash"] * (_shed_value / max(1e-6, _shed_value + _money) - 0.5))

    def _factor(prod, z=0.0):
        """Product level times the turn reaction. No weights -> just the level."""
        f = _levels.get(prod, 1.0)
        if not _live:
            return f
        # The +-13.8 clip is the same numerical guard as `_logit`: it leaves a
        # reach of 1e6 times, i.e. no border for practical purposes.
        return f * math.exp(max(-13.8, min(13.8, _z0 + z)))

    orders = []
    for p in spec.PRODUCTS:
        n = int(shed.get(p, 0)) - int(reserve.get(p, 0))
        if n <= 0:
            continue

        # End of season: holding is worth nothing, the shed does not score,
        # and neither do the animals: the feed reserve is released too.
        if remaining <= final_urgency:
            orders.append((unit_value(obs, p) * _factor(p),
                            ["SELL", p, int(shed.get(p, 0))]))
            continue

        flow = float(opp_flow[spec.PRODUCT_IX[p]]) if opp_flow is not None else 0.0
        target = future_price(obs, p, min(horizon, remaining), flow)
        # The two product-dependent signals:
        #   price   how much it is expected to FALL -log(now/forecast)-.
        #           Positive = the forecast says it drops, i.e. a reason to
        #           sell now. The weight decides how much the policy trusts
        #           the forecast.
        #   rival   what share of the imminent flow comes from THEM, not the
        #           town. It already enters `future_price` with coefficient 1;
        #           the weight allows over-reacting or ignoring it.
        _z = 0.0
        if _live:
            _xp = max(-2.0, min(2.0, math.log(max(1e-6, unit_value(obs, p))
                                              / max(1e-6, target))))
            _xr = flow / max(1e-6, flow + drain_rate(obs, p)) - 0.5
            _z = _w["price"] * _xp + _w["rival"] * _xr
        _fp = _factor(p, _z)
        # Opportunity cost of capital: if the freed money is reinvested and
        # compounds, holding produce has to beat that growth too. Without it
        # the agent sits on 65 units with $122 in cash.
        target /= capital_discount(obs, min(horizon, remaining), macro)
        target /= max(1e-6, _fp)
        prices_ = marginal_prices(obs, p, n)
        k = sum(1 for pr in prices_ if pr >= target)

        # The shed overflows at 100 and the excess is DISCARDED at day end.
        if used > cap * SAT_HIGH:
            k = max(k, n - int(cap * SAT_LOW))
        if k > 0:
            orders.append((prices_[0] * _fp, ["SELL", p, k]))

    # If there are more orders than slots, the ones moving most money go first.
    orders.sort(key=lambda x: -x[0])
    return [o for _, o in orders]


def capital_discount(obs, horizon: int, macro=None) -> float:
    """By how much freed capital would multiply over `horizon` turns.

    It only counts if there is somewhere to reinvest it: with the farm full,
    freeing cash adds nothing and waiting for the better price is right.
    """
    from .tasks import best_crop, growth_factor
    me = int(obs["player"])
    farm_ = obs["farms"][me]
    empty = sum(1 for row in farm_["tiles"] for t in row if t is None)
    if empty <= 0:
        return 1.0
    # PORTFOLIO, not the "best" crop. This valuation estimates at what rate
    # reinvested capital compounds, and it used to hard-wire `best_crop`
    # -which no longer decides anything: since sowing is a learned portfolio,
    # assuming monoculture gives a number that does not match what the policy
    # does.
    from .tasks import plantable
    _f = _product_levels(macro)
    _viable = [k for k in spec.CROP_LIST if plantable(obs, k)]
    if not _viable:
        return 1.0
    _weights = {k: max(1e-6, _f.get(k, 1.0)) for k in _viable}
    _st = sum(_weights.values())
    c = max(_viable, key=lambda k: _weights[k])      # the portfolio leader
    if c is None:
        return 1.0
    return max(1.0, growth_factor(obs, c) ** (horizon / spec.TURNS_PER_DAY))


def unit_value(obs, product: str) -> float:
    return float(obs["market"]["prices"].get(product, 1))


def hire_orders(obs, n_max: int = 15, margin: float = None, macro=None,
                planned_seeds: int = 0) -> list:
    """Hire while a hand costs less than it returns in a day.

    The cost of the n-th hand of the day is `fib(n)` and resets daily. A hand
    contributes `turnsPerDay` actions, so the honest comparison is its cost
    against the value of those actions. `margin` demands that it return several
    times its cost before hiring, because the value per action is an optimistic
    estimate (units spend turns moving).
    """
    margin = HAND_MARGIN if margin is None else margin
    farm = obs["farms"][int(obs["player"])]
    if obs["hour"] > LAST_HIRE_HOUR:
        return []
    hired = int(farm["hires_today"])
    money = float(farm["money"])
    days = max(1, (turns_left(obs) + 1) // spec.TURNS_PER_DAY)
    from ..plan import get_plan as _get_plan
    _pl = _get_plan()
    if days < MIN_HAND_DAYS and _pl is None:
        return []                      # no time left to pay it back

    from kaggle_environments.envs.kaggriculture import kaggriculture as E
    action_value = _value_per_action(obs)

    # HIRE BY LABOUR DEMAND, not by what the wallet can take. Without this cap
    # the fibonacci cost makes 15 hands cost $1,596/day and the farm goes
    # bankrupt in two days paying idle labour: measured, with a policy that did
    # not plant, money went from $3,000 to $204 by turn 24. Demand is measured
    # over REAL work, not potential: an empty tile is work only if the policy
    # actually plants, and the rules alone turned $3,000 of wealth into $2,422.
    # Spending must follow demonstrated capacity, with a minimum start so a
    # competent policy can take off (you begin with 0 seeds).
    # Real work = what is EXECUTABLE now, which is not the same as potential.
    # An empty tile is work only if there is seed in hand to plant it;
    # otherwise hiring for "plantable" tiles pays for intentions. Counting zero
    # empty tiles created the opposite deadlock: no seed means no tasks, no
    # tasks means no hands, no hands means no planting, and it never starts.
    # Seed being bought THIS turn counts as work: hiring and buying seed are
    # decided together, and hiring first is right only if the seed follows.
    seeds_in_hand = sum(int(v) for v in obs["private"].get("seeds", {}).values()) + int(planned_seeds)
    tasks = 0
    for row in farm["tiles"]:
        for t in row:
            if t is None and seeds_in_hand > tasks:
                tasks += 1                      # plantable AND seed in hand
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT" and not t.get("watered_today"):
                    tasks += 1
                elif k == "WEED":
                    tasks += 1
                elif t.get("animal") and not t.get("fed_today"):
                    tasks += 1
    # Each unit tends several tiles a day; with less work than people, hiring
    # more is pure spending. The task cap turned out to be too coarse a proxy:
    # it gave 2 hands/day and with that a strong public agent -a ROUTE agent,
    # unit i runs route i- only got to execute 2 routes, planted 2 seeds in 84
    # turns and was left with $2,827 of its $26,203. It is replaced by a daily
    # labour
    # BUDGET cap: it self-limits, scales with wealth and presumes nothing about
    # how the policy organises the work.
    # The target is set by the policy; the engine sets the hard limit.
    # THE MACRO'S TARGET, NOT A DEMAND CAP. Measured 2026-09-24: capping
    # hires by the work executable now (tiles with seed in hand, unwatered
    # plants, unfed animals) made a 5-day solitaire consistent (0 idle hires)
    # and cost -39,212 $ (t -22, worse on every one of 60 boards) in the full
    # game. Hands create the work of a growing farm -building, land, moving-
    # and "executable now" does not see it. The macro's target stands.
    per_unit = max(1, spec.TURNS_PER_DAY // 3)
    if macro is not None:
        from ..macro import target_hands
        n_max = min(n_max, target_hands(obs, macro))
    else:
        STARTUP_HANDS = 2
        n_max = min(n_max, max(STARTUP_HANDS, -(-max(tasks, 1) // per_unit)))

    budget = max(LABOUR_FLOOR, money * LABOUR_BUDGET_FRACTION)

    orders = []
    for n in range(hired, n_max):
        cost = E._fib(n)
        if _pl is not None:                 # the plan says how many; only cash limits it
            if cost > money:
                break
            orders.append(["HIRE"])
            money -= cost
            continue
        # A hand contributes `turnsPerDay` actions. It is hired while its
        # cost is a fraction of what those actions yield. It is NOT capped by
        # a share of cash: labour is the multiplier of everything else, and
        # funding it comes before buying seed.
        if cost * margin > action_value * spec.TURNS_PER_DAY or cost > money:
            break
        if cost > budget:
            break
        orders.append(["HIRE"])
        money -= cost
        budget -= cost
    return orders


def _value_per_action(obs) -> float:
    """What one unit-turn is worth, measured by the best available crop."""
    from .tasks import cycle_days, cycle_profit, cycle_yield, plantable
    best = 0.0
    for c in spec.CROP_LIST:
        if not plantable(obs, c):
            continue
        turns = cycle_days(c) + 3          # water daily + plant + harvest
        best = max(best, cycle_profit(obs, c) / turns)
    # THE WORK THAT ALREADY EXISTS. A unit-turn is worth what the best crop
    # cycle yields per turn OR what realising the standing position yields
    # per unit-turn left: harvests, feeding, sales. The old floor of $1
    # stood in for the second term and hired hands with nothing to do in
    # short solitaires (3/9/22 hires in 2/3/5 days); removing it alone lost
    # $3,509 (t -11.5) in the full game because late hands harvest and sell.
    # `liquidation` counts exactly what has time to pay, so it is the right
    # measure of pending work; in a game where nothing pays it is the cash.
    from ..potential import liquidation
    farm = obs["farms"][int(obs["player"])]
    pending = liquidation(obs, int(obs["player"]), obs.get("private")) - float(farm["money"])
    units = 1 + len(farm["hands"]) + int(farm.get("hires_today", 0))
    per_unit_turn = max(0.0, pending) / max(1, units * max(1, turns_left(obs)))
    return max(best, per_unit_turn)


def quadrant_of_xy(x: int, y: int) -> str:
    h = spec.BOARD // 2
    return ("N" if y < h else "S") + ("W" if x < h else "E")


def land_orders(obs, macro=None) -> list:
    """Buy land while there is season left to pay it back.

    The ladder data is blunt: buying between days 4 and 7 is associated with a
    0.49 win rate, against 0.12-0.19 when buying later or never. The condition
    here is economic, not a date: the quadrant costs LAND_PRICES[n] and
    contributes 25 tiles for however many days remain.
    """
    farm = obs["farms"][int(obs["player"])]
    n = len(farm["unlocked_quadrants"]) - 1
    # THE HOUR TEST WAS OURS AND IT IS GONE. The engine handles BUY_LAND as an
    # atomic order in the market phase with no reference to the hour, so
    # `obs["hour"] != 0` was an invented rule: 24 chances a day cut to one,
    # competing for the turn's ten market orders. Measured on 200 paired seeds
    # against v48 it is worth nothing either way -- -$247, se 393, t -0.6 --
    # so it goes on principle rather than for money: a condition that does not
    # come from the engine and cannot be learned has no business deciding.
    if n >= len(spec.LAND_PRICES):
        return []
    cost = spec.LAND_PRICES[n]
    money = float(farm["money"])
    from ..plan import get_plan as _get_plan
    _pl = _get_plan()
    if _pl is not None:                 # the plan says how much land, literally
        return [["BUY_LAND"]] if n < _pl.land_on(int(obs["day"])) and money >= cost else []
    # Only expand if the land already owned is being used. An alternative
    # payback criterion was tried (buy if there is season left to pay for it)
    # and it is WORSE: -$1,000 on all three controls, because it buys 25 tiles
    # the policy never gets to use. Saturation is the right guard; land was not
    # the bottleneck.
    empty = sum(1 for row in farm["tiles"] for t in row if t is None)
    usable = sum(1 for y in range(spec.BOARD) for x in range(spec.BOARD)
                  if quadrant_of_xy(x, y) in farm["unlocked_quadrants"])
    threshold = 0.25 if macro is None else float(macro.expand)
    if usable and empty > usable * threshold:
        return []
    days = max(0, (turns_left(obs) + 1) // spec.TURNS_PER_DAY)
    from .tasks import best_crop, cycle_days, cycle_profit, plantable
    # Land is worth its BEST use, not only its crop use. Measured: with a pure
    # livestock strategy `best_crop` returns nothing, so the return was
    # compared against zero and the agent stayed on ONE quadrant, cramming 13.6
    # animals into 25 tiles with 44.3% of its actions idle. The expert plays
    # with 3 full quadrants.
    per_tile = 0.0
    # PORTFOLIO here too: this estimates what one more tile would yield, and
    # hard-wiring `best_crop` assumes monoculture, which is not what is played
    # any more.
    _fl = _product_levels(macro)
    _viable2 = [k for k in spec.CROP_LIST if plantable(obs, k)]
    c = (max(_viable2, key=lambda k: max(1e-6, _fl.get(k, 1.0))
             * (cycle_profit(obs, k) / max(1, cycle_days(k))))
         if _viable2 else None)
    if c is not None:
        per_tile = (cycle_profit(obs, c) / max(1, cycle_days(c))) * days
    best_animal = max((animal_net_value(obs, a) for a in spec.ANIMALS), default=0.0)
    if best_animal > 0:
        # An animal occupies one tile (its structure) and `animal_net_value`
        # is already net of the whole remaining season, feed included.
        per_tile = max(per_tile, best_animal)
    if per_tile <= 0:
        return []
    ret = 25 * per_tile
    if ret > cost * LAND_RETURN and money > cost * LAND_CASH:
        return [["BUY_LAND"]]
    return []


def _product_levels(macro):
    if macro is None:
        return {}
    try:
        from ..macro import market_factors
        return market_factors(macro)
    except Exception:
        return {}


def seed_orders(obs, tile_target: int, macro=None) -> list:
    """Buy seed to cover the plantable tiles, following the learned portfolio."""
    # REVEALED preference, not computed preference. `best_crop` chose MELON by
    # $/tile-day while the policy emitted PLANT WHEAT: the seed bought was
    # useless and nothing got planted. Measured, under those rules a strong
    # public agent fell from $26,203 to $2,012, exactly what not playing gives.
    # What the policy DEMONSTRATES it plants is what gets bought; only if it
    # has planted nothing yet is the calculation used, biased to the cheapest
    # to start.
    from .tasks import best_crop
    _levels = _product_levels(macro)
    farm = obs["farms"][int(obs["player"])]
    planted_by_crop: dict = {}
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                planted_by_crop[t["crop"]] = planted_by_crop.get(t["crop"], 0) + 1
    if macro is not None:
        from ..macro import target_tiles, target_crop
        tile_target = target_tiles(obs, macro)
        c = target_crop(obs, macro)
    elif planted_by_crop:
        c = max(planted_by_crop, key=lambda k: planted_by_crop[k])
    else:
        c = min(spec.CROP_LIST, key=lambda k: spec.CROPS[k]["seed"])
    if c is None:
        return []
    held = int(obs["private"].get("seeds", {}).get(c, 0))
    # Do not buy seed if what is already held is not being planted. Measured:
    # with a policy that does not plant, this spent $980 on seed still sitting
    # in hand 120 turns later. Spending must follow demonstrated capacity, not
    # intention.
    # Capped by SOWING CAPACITY, not by a fixed number. The cap of 2 starved a
    # competent agent: under these rules a strong public agent only reached 3
    # crops, against the 57 it manages under its own.
    unused = sum(int(v) for v in obs["private"].get("seeds", {}).values())
    n_units = 1 + len(farm["hands"])
    from ..plan import get_plan as _get_plan
    _pl = _get_plan()
    if _pl is not None:                 # the plan: its crops, the tiles it asks, all the cash
        from .tasks import plantable as _plantable
        cash = float(farm["money"])
        out = []
        for crop_, want in _pl.crop_targets(int(obs["day"])).items():
            if not _plantable(obs, crop_):
                continue
            missing = want - planted_by_crop.get(crop_, 0) - int(obs["private"].get("seeds", {}).get(crop_, 0))
            price = max(1, spec.CROPS[crop_]["seed"])
            n_take = min(missing, int(cash // price))
            if n_take > 0:
                out.append(["BUY_SEED", crop_, n_take])
                cash -= n_take * price
        return out
    # LEARNED. This `return []` ABORTS the whole seed purchase, and with 8.4
    # units the threshold came out at 16.8 while we carried 25.1 seeds on
    # average: for much of the episode nothing was bought.
    if unused >= max(SEED_FLOOR, SEED_STOCK_PER_UNIT * n_units):
        return []
    # PORTFOLIO, not monoculture. Seed used to be bought for ONE crop -whatever
    # `target_crop` said- so giving the network freedom to plant what it wanted
    # was useless: there was no seed for anything else. The bottleneck sat
    # three layers above where planting happens.
    #
    # The split uses the LEARNED market factors, which include the five crops.
    # With all neutral the budget is split among the viable ones; when the
    # network learns a crop is worth more, it buys more of that one.
    #
    # What was measured is kept: only VIABLE crops, the ones with time to be
    # harvested, because buying seed that never gets planted wasted $980.
    from .tasks import plantable
    money = float(farm["money"])
    budget = money * SEED_BUDGET_FRACTION
    viable = [k for k in spec.CROP_LIST if plantable(obs, k)]
    if not viable:
        return []
    weights = {k: max(1e-6, _levels.get(k, 1.0)) for k in viable}
    _total = sum(weights.values())
    missing_total = max(0, tile_target - sum(
        int(obs["private"].get("seeds", {}).get(k, 0)) for k in viable))
    if missing_total <= 0:
        return []
    # TWO PASSES, and the second is the one that was missing. Each crop used to
    # get a FIXED slice of the budget and whatever was left was DISCARDED. With
    # the learned portfolio in its normal regime -measured: strawberry x87, up
    # to x511, wheat x0.33- nearly all the money goes to $100 seed, and when
    # it runs out the remaining tiles stay EMPTY even though there is cash left
    # to fill them with $10 wheat.
    #
    # The measured symptom: we met 47% of our own tile target (14.6 planted
    # against 31.2 asked for) and kept 15.4 live plants against v48's 44
    # -2.9x fewer, which is exactly the money ratio-. We
    # planted the SAME amount as it (190 against 206) but bought half the seed
    # (112 against 212), because we bought it expensive.
    #
    # This does NOT decide the portfolio: the preference order is still the one
    # the network emits, and whoever weighs most picks first. What changes is
    # that leaving the farm empty stops being unavoidable when the favourite
    # cannot be afforded.
    seed_ords = []
    _order = sorted(viable, key=lambda x: -weights[x])
    _cash_left, _tiles_left, _n = budget, missing_total, {}
    for _pass_ in (1, 2):
        for k in _order:
            if _tiles_left <= 0 or _cash_left <= 0:
                break
            _price = max(1, spec.CROPS[k]["seed"])
            # 1st pass: its share of tiles by weight. 2nd: whatever is left.
            _cap = (int(missing_total * weights[k] / _total) if _pass_ == 1
                     else _tiles_left)
            n_take = min(_cap, _tiles_left, int(_cash_left // _price))
            if n_take > 0:
                _n[k] = _n.get(k, 0) + n_take
                _tiles_left -= n_take
                _cash_left -= n_take * _price
    for k in _order:
        if _n.get(k, 0) > 0:
            seed_ords.append(["BUY_SEED", k, _n[k]])
    return seed_ords


def animal_net_value(obs, animal: str) -> float:
    """Net value of buying this animal NOW, with the engine's exact economics.

    An animal produces `1/interval` units per day from `first_yield_day`, and
    eats 1 wheat per day -if it goes two days without eating, it escapes-. It
    also leaves manure, which is worth something because it doubles the
    watering bonus. None of this is estimated: it comes from the engine tables.
    """
    d = spec.ANIMALS[animal]
    days = (turns_left(obs) + 1) // spec.TURNS_PER_DAY
    prod_days = max(0, days - d["first_yield_day"])
    if prod_days <= 0:
        return -1.0
    prices_ = obs["market"]["prices"]
    units = int(prod_days / d["interval"])
    if units <= 0:
        return -1.0
    # MARGINAL PRICE, not nominal. Each unit sold lowers the price of the next:
    # valuing 26 eggs at $50 each overestimates the income and makes an animal
    # look profitable when it is not. Measured: at nominal value we bought
    # animals with 14 days left and the result fell from $4,310 to $345.
    income = sum(marginal_prices(obs, d["product"], units))
    feed = days * float(prices_.get("WHEAT", 0))
    fert = prod_days * float(prices_.get("FERTILIZER", 0)) * MANURE_CREDIT
    return income + fert - d["cost"] - feed


def animal_orders(obs, max_per_turn: int = None, macro=None) -> list:
    """Buy animals while they pay off and there is room to place them.

    Measured: the 2945 expert reaches 17 animals and we reach 0. Each animal
    also generates 4 daily tasks (eat, care, collect fertiliser, harvest), so
    it is both income and work with which to justify more hands -our units
    spend 22.3% of turns on PASS for lack of tasks, against the expert's
    7.1%-.
    """
    max_per_turn = (MAX_ANIMALS_PER_TURN if max_per_turn is None
                     else max_per_turn)
    farm = obs["farms"][int(obs["player"])]
    priv = obs["private"]
    money = float(farm["money"])
    cap = spec.DEFAULT_CONFIG["shedCapacity"]
    if sum(priv["shed"].values()) >= cap:
        return []

    # Room: empty structures plus free tiles to build them on.
    free_slots = {"COOP": 0, "PASTURE": 0}
    empty = 0
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                empty += 1
            elif isinstance(t, dict) and t.get("kind") in free_slots and not t.get("animal"):
                free_slots[t["kind"]] += 1
    # animals already bought, waiting to be placed
    pending = sum(int(priv["shed"].get(a, 0)) for a in spec.ANIMALS)
    pending += sum(int(inv.get(a, 0)) for inv in priv.get("inventories", []) for a in spec.ANIMALS)

    # CAPPED BY ATTENTION CAPACITY. Each animal costs ~3 actions a day (eat,
    # care/collect, harvest) and each crop ~1 (water). Measured: without this
    # cap, 25 animals were bought with 4 units -75 daily tasks for 96 actions
    # counting movement- and 17 escaped.
    n_units = 1 + len(farm["hands"])
    crops = sum(1 for row in farm["tiles"] for t in row
                   if isinstance(t, dict) and t.get("kind") == "PLANT")
    alive = sum(1 for row in farm["tiles"] for t in row
                if isinstance(t, dict) and t.get("animal"))
    # ~50% of the budget goes on moving: that is what the expert measures
    # (42.3%) and what we measure (56.5%), so half is optimistic and therefore
    # prudent.
    capacity = n_units * spec.TURNS_PER_DAY * 0.5
    _fallback_per_animal = 3.0      # only for the no-macro path below
    if macro is not None:
        from ..macro import target_animals
        room = target_animals(obs, macro) - alive - pending
    else:
        room = int((capacity - crops) // _fallback_per_animal) - alive - pending
    if room <= 0:
        return []

    # WHICH animal to buy was a hand-written formula -`animal_net_value`-, the
    # same case as `best_crop` with crops. And it is where the other measured
    # hole lives: v48 makes $60,179 from MILK and we make $13,536.
    #
    # It is weighted by the LEARNED factor of the product each animal gives,
    # which already lives in the macro vector (EGG, MILK, WOOL). Neutral = 1.0,
    # so without a macro the ordering is the usual one.
    _lvlA = _product_levels(macro)
    candidates = sorted(
        spec.ANIMALS,
        key=lambda a: -(animal_net_value(obs, a)
                        * _lvlA.get(spec.ANIMALS[a]["product"], 1.0)))
    # UNDER A PLAN the plan says which kinds and how many; the executor's
    # valuation (manure as a watering credit, not a product) refuses every
    # animal below 14 days, while a goose pays back in 2-3 days through
    # eggs and fertiliser (measured: 4 geese, 14 days, 3,000 -> 8,658).
    from ..plan import get_plan as _get_plan
    _pl = _get_plan()
    _want = _pl.animal_targets(int(obs["day"])) if _pl is not None else None
    if _want is not None:
        alive_by = {}
        for row in farm["tiles"]:
            for t in row:
                if isinstance(t, dict) and t.get("animal"):
                    alive_by[t["animal"]] = alive_by.get(t["animal"], 0) + 1
        for a in spec.ANIMALS:
            alive_by[a] = alive_by.get(a, 0) + int(priv["shed"].get(a, 0)) \
                + sum(int(inv.get(a, 0)) for inv in priv.get("inventories", []))
        candidates = [a for a in spec.ANIMALS if _want.get(a, 0) > alive_by.get(a, 0)]
    orders = []
    for a in candidates:
        if len(orders) >= min(max_per_turn, room):
            break
        d = spec.ANIMALS[a]
        if _pl is None and animal_net_value(obs, a) <= 0:
            continue
        room = free_slots[d["structure"]] + empty
        if room - pending <= 0:
            continue
        # Reserve cash: running out of money for seed and feed ruins the
        # investment, because an animal that does not eat for two days escapes.
        if money - d["cost"] < ANIMAL_CASH_RESERVE:
            continue
        orders.append(["BUY_ANIMAL", a, 1])
        money -= d["cost"]
        pending += 1
    return orders


def feed_orders(obs, stock_days: int = None) -> list:
    """Buy wheat to feed. Without this the animals starve.

    `FEED` consumes 1 wheat per animal per day, and after two days without
    eating the animal escapes. But the wheat has to EXIST in the shed, and if
    the farm grows something else it never appears.

    Measured before this function existed: forcing `macro.animals` up,
    13.4 animals per episode were bought and 0.8 were left alive, with
    `shed_wheat = 0` at every checkpoint and ZERO FEED executions. Money fell
    from $41,598 to $18,714: we bought livestock to watch it escape.

    This also explains the `BUY_PRODUCT WHEAT 20` a strong public agent emits
    on turn 0, which had been dismissed here as "an expensive way to get 5
    wheat": it is feed.

    The stock is exact arithmetic, not a parameter: animals x days. It is
    bought with `stock_days` of cushion because the wheat has to be fetched
    from the shed and carried on foot.
    """
    # int(): the parameter is a learnable real but downstream it feeds
    # quantities that reach `range()`. Without this, a TypeError only when
    # there is livestock -at 5 days there is none, so it never fired in the
    # small cell-.
    stock_days = int(round(FEED_STOCK_DAYS if stock_days is None
                           else stock_days))
    me = int(obs["player"])
    farm = obs["farms"][me]
    priv = obs["private"]
    n_animals = sum(1 for row in farm["tiles"] for t in row
                     if isinstance(t, dict) and t.get("animal"))
    # including the ones bought and waiting to be placed
    n_animals += sum(int(priv["shed"].get(a, 0)) for a in spec.ANIMALS)
    if n_animals <= 0:
        return []

    days = max(1, (turns_left(obs) + 1) // spec.TURNS_PER_DAY)
    target = n_animals * min(stock_days, days)
    held = int(priv["shed"].get("WHEAT", 0))
    held += sum(int(inv.get("WHEAT", 0)) for inv in priv.get("inventories", []))
    missing = target - held
    if missing <= 0:
        return []

    cap = spec.DEFAULT_CONFIG["shedCapacity"]
    room = max(0, cap - sum(priv["shed"].values()))
    n = min(missing, room)
    if n <= 0:
        return []
    # Do not go broke buying feed: an unfed animal is worth 0, but a farm with
    # no cash does not produce either.
    cost = sum(marginal_prices(obs, "WHEAT", n))
    money = float(farm["money"])
    while n > 1 and cost > money * FEED_CASH_FRACTION:
        n -= 1
        cost = sum(marginal_prices(obs, "WHEAT", n))
    return [["BUY_PRODUCT", "WHEAT", n]] if n > 0 and cost <= money else []
