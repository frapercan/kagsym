"""The macro decision vector: everything the policy decides once per day.

Why this file exists. The scripted layer accumulated eleven hand-set constants
-how many animals per unit of attention, what fraction of the cash goes to
labour, what margin to demand from a hand, at what saturation to expand...-.
That violates the project's principle ("nothing guessed") and, worse, it
occupies exactly the place where learning should decide: those are STRATEGY
decisions, not mechanics.

Mechanics -routes, optimal assignment, legality, watering the day you plant,
the feed reserve, final liquidation- is derivable from the engine and stays in
the executor. What is not derivable is HOW MUCH of each thing, and above all
WHEN to dump into the shared market, which depends on the opponent. That is
this vector.

Every component lives in [0,1] so that a clipped Gaussian policy can emit them
without invented scales. The translation into exact quantities happens in the
resolver functions below, which is where the engine's real limits live.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

from . import spec

N_LEVELS = 8         # how much of each thing
CATEGORIES = ["land", "feed", "animal", "sell", "seed", "hand"]
N_PRIORITIES = len(CATEGORIES)
N_EXPOSED = 36       # the hand-set ones; see the block in `Macro`
N_MARKET = 9         # one learned value multiplier per product
N_TURN = 5           # coefficients of the PER-TURN selling rule
N_MACRO = N_LEVELS + N_PRIORITIES + N_EXPOSED + N_MARKET + N_TURN
# The 16th hand of the day costs $987 on its own -measured- but that is an
# ECONOMIC reason the search can find by itself, not an engine limit: the
# engine does not cap hiring. There is no constant any more; see `target_hands`.
# The resale horizon has no ceiling either; its default is one day
# (`spec.TURNS_PER_DAY`, del motor). Ver `horizonte_venta`.


@dataclass
class Macro:
    """Targets, not orders. The executor decides how to reach them."""

    tiles: float = 0.3333     # -> scale over the watering ceiling, BORDERLESS
                               #    (exp(logit(1/3)) = 0.5, the previous value)
    animals: float = 0.5      # -> share of attention capacity given to livestock
    hands: float = 0.35       # -> target hands per day
    selling: float = 0.25        # -> agresividad: 0 vender ya, 1 acumular al maximo
    crop: float = 0.0       # -> sesgo hacia cultivo caro (1) o barato y rapido (0)
    expand: float = 0.25    # -> saturation required before buying land
    stickiness: float = 0.3333 # -> stickiness scale, borderless
                               #    (exp(logit(1/3)) = 0.5, the previous value)
    fertilize: float = 0.0    # -> cuanto vale fertilizar frente a las demas tareas
    # PRIORITIES. The seven above say HOW MUCH of each thing; these say WHAT
    # GETS SACRIFICED when there is not enough for everything, which is 31% of
    # turns: measured, lack of cash blocks buying land on 31%, feed on 19% and
    # animals on 15%. That used to be decided by four hand-set constants (a
    # $300 reserve, 25% of the cash, cost x1.5, 15% on labour) and the order of
    # a list written by hand.
    #
    # They go through a softmax: what matters is the relative split, not the
    # scale.
    p_land: float = 0.5
    p_feed: float = 0.5
    p_animal: float = 0.5
    p_sell: float = 0.5
    p_seed: float = 0.5
    p_hand: float = 0.5
    # ------------------------------------------------------------------
    # WHAT USED TO BE HAND-SET. First audit: fourteen constants spread across
    # market_ops.py, tasks.py, executor.py and this file had never entered any
    # search. Exposing them raised the ceiling of the 12h x 5d cell from $939
    # to $1,258 (+34%), more than everything tried that day on the learning
    # side put together.
    #
    # They live here and not as globals because the correct value DEPENDS ON
    # THE STATE -how much cash is left, how many days, which opponent- and that
    # is the definition of what the module docstring says must be learned.
    #
    # All in [0,1]; `params()` maps them to their real range.
    f_labour: float = 0.137         # reproduces the old 0.15
    f_seed: float = 0.479        # 0.50
    f_hand_margin: float = 0.359    # 3.0
    f_feed_days: float = 0.385    # 3.0
    f_sat_high: float = 0.797       # 0.85
    f_sat_low: float = 0.588       # 0.60
    f_land_return: float = 0.273   # 2.0
    f_land_cash: float = 0.286      # 1.5
    f_step_discount: float = 0.653 # 0.82
    f_watering: float = 0.318   # 0.50
    f_turns_init: float = 0.375     # 3.0
    f_turns_min: float = 0.333     # 2.0
    f_dig_value: float = 0.421      # 0.90
    # INTERFACE between the learned and the symbolic: `expm1` is exponential,
    # so the gain decides whether the network's map SUGGESTS or IMPOSES, and
    # the cap decides where it is clipped. Nobody ever searched them.
    f_map_gain: float = 0.231  # 1.0
    f_map_cap: float = 0.655      # 20.0
    f_max_animal: float = 0.111     # 2  (the engine limit is 10, not 2)
    # THRESHOLD for tiles the heuristic declares empty and the engine
    # considers legal -5.81 per turn against 8.94 offered-. It is what such a
    # tile has to be worth, in dollars emitted by the network, to deserve
    # occupying a unit. Learned like the rest: there is no way to derive it
    # from the engine, so it cannot be set by hand.
    #
    # The default 0.5 -> $2,500, far above any real task, so at startup NO
    # extra tile enters and behaviour is identical to before. Training lowers
    # it if it pays.
    f_extra_threshold: float = 0.5
    # MARKET HEAD: one multiplier per product over its sale value.
    #
    # The board is already decided tile by tile by the network, but WHAT to
    # sell and how long to hold was decided by `market_ops.py` from about ten
    # scalars. That is where a measured gap lives: we sell $59,872 against
    # v48's $200,760, with $71,170 in STRAWBERRIES we never touch and $60,179
    # in MILK where we do a fifth -84% of the difference in two products-.
    #
    # Same pattern that worked for the micro head: a MULTIPLICATIVE residual
    # over the exact valuation, not a replacement. 0.5 -> factor 1.0, i.e.
    # neutral, so the default reproduces the previous behaviour.
    m_wheat: float = 0.5
    m_carrot: float = 0.5
    m_tomato: float = 0.5
    m_strawberry: float = 0.5
    m_melon: float = 0.5
    m_egg: float = 0.5
    m_milk: float = 0.5
    m_wool: float = 0.5
    m_fertilizer: float = 0.5
    # ------------------------------------------------------------------
    # THE ONES THAT WERE STILL INLINE. Second audit: four numbers written
    # INSIDE function bodies rather than as module constants, so the previous
    # audit -the one that exposed fourteen and gave +34%- never saw them.
    #
    # MEASURED LIVE in ops, 4 seeds, 24h x 30d, 720 turns:
    #     manure credit      0.5 -> 0.0    -22.5%
    #     cash for feed     0.25 -> 0.02   -21.6%
    # i.e. each one decides more than almost anything else in the vector.
    # `labour_floor` measured 0.0% at this starting cash, but it is a FLOOR IN
    # DOLLARS: with another starting cash it bites, and the model has to work
    # in another league.
    f_manure_credit: float = 0.5   # 0.50  credito de estiercol al valorar un animal
    f_feed_cash: float = 0.5   # 0.25  share of cash that may go on feed
    f_labour_floor: float = 0.5    # 60.0  floor of the labour budget
    f_liquidation: float = 0.5     # 2.0   liquidation days at the season close
    # ------------------------------------------------------------------
    # PER-TURN SELLING RULE. Until this, everything learned about the market
    # was a LEVEL per product, constant across all 24 hours: the network is
    # called once a day -one RL step is a DAY, see `environment.py`- and the
    # symbolic layer plays the whole day on its own.
    #
    # The selling rule DOES look at each turn's state, but through a formula I
    # wrote: sell while the current price beats the forecast. What the policy
    # could not move is HOW MUCH to react to each signal.
    #
    # These five are those coefficients. The daily pass emits the RULE and the
    # rule is evaluated against EACH TURN's state. That gives decisions per
    # situation without multiplying the network passes by 24, which is what a
    # competition with one second per turn demands.
    #
    # w = logit(f), so 0.5 -> w = 0 -> exp(0) = 1: at startup this is EXACTLY
    # the previous behaviour. No scale constant, for the same reason as in
    # `params()`: sigmoid and logit cancel.
    w_price: float = 0.5       # reaction to the forecast price drop
    w_rival: float = 0.5       # reaction to what the opponent is about to dump
    w_shed: float = 0.5        # reaction to storage pressure
    w_season: float = 0.5      # reaction to how far the season has advanced
    w_cash: float = 0.5        # reaction to how much wealth is tied up
    # ------------------------------------------------------------------
    # FOURTH PASS, from the AST audit. Two scales that were still fixed:
    #
    #   `f_priority_temp`  the softmax temperature over the six priorities.
    #                      A temperature decides how MUCH the ordering is
    #                      honoured -flat means the order barely matters, sharp
    #                      means the first category takes everything- and that
    #                      is policy, not mechanics. Measured dead at the
    #                      current operating point (the emitted priorities are
    #                      nearly uniform) but it gates how much the priority
    #                      vector can ever do.
    #   `f_seed_floor`     the hard floor of 2 seeds in the "stop buying seed"
    #                      rule. Measured LIVE: raising it to 40 costs -7%.
    f_priority_temp: float = 0.5   # 3.0  softmax temperature over priorities
    f_seed_floor: float = 0.5      # 2.0  minimum seeds before the stock rule bites
    # ------------------------------------------------------------------
    # TERCERA HORNADA, 2026-09-22. Auditoria ESTRUCTURAL: las dos anteriores
    # buscaban constantes de modulo y literales; esta recorre cada punto de
    # decision site in the tree (`herramientas/auditoria_cascada.py`) and
    # classifies it. Of 156 sites, these eight were policy decisions nobody
    # could move.
    #
    # The worst is `seed_stock`: `seed_orders` ABORTS entirely if you
    # already hold 2 seeds per unit. Measured, we carry 25.1 seeds on average
    # with 8.4 units -threshold 16.8- meaning that for much of the episode NO
    # SEED IS BOUGHT. It matches the symptom being chased: 15.4 live plants
    # against v48's 44 while planting the same amount.
    f_seed_stock: float = 0.5      # 2.0   seeds per unit before buying stops
    f_animal_reserve: float = 0.5  # 300.0 caja a reservar antes de comprar un animal
    f_actions_per_animal: float = 0.5 # 3.0   acciones/dia que cuesta sostener un animal
    f_last_hire_hour: float = 0.5  # 3.0   last hour of the day for hiring
    f_min_hand_days: float = 0.5   # 2.0   minimum days to pay a hand back
    f_cost_rise: float = 0.5       # 0.5   how much cost/tile rises when plants dry
    f_cost_decay: float = 0.5      # 0.93  how much it falls when nothing dries
    f_fert_per_trip: float = 0.5   # 4.0   fertiliser picked up in one trip
    # ------------------------------------------------------------------
    # FIFTH PASS. Five multipliers over the item price written INSIDE
    # `tile_task`, which is why neither the module-constant audits nor the
    # structural one reported them: they are not globals and they are not
    # thresholds, they are what an operation is WORTH. How much a feeding is
    # worth against a harvest is preference, so it is learned.
    f_fert_trip: float = 0.5       # 2.0   value of a fertiliser pickup
    f_wheat_trip: float = 0.5      # 2.0   value of a wheat pickup
    f_water_idle: float = 0.5      # 0.4   watering with no yield in sight
    f_feed_value: float = 0.5      # 2.0   feeding an animal
    f_care_value: float = 0.5      # 0.5   caring for an animal

    @staticmethod
    def default() -> "Macro":
        """The values that reproduce the scripted layer exactly as measured."""
        return Macro()

    @staticmethod
    def from_vector(v) -> "Macro":
        vals = [min(1.0, max(0.0, float(x))) for x in v]
        return Macro(*vals[:N_MACRO])

    def to_vector(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


# TOPE DE CURRICULO. None = sin tope. Limita cuantos peones puede planificar
# OUR agent, exactly as `public_with_cap` limits the opponent. It serves to
# entrenar en un juego mas pequeno y coherente: menos unidades = ejecutor mas
# cheaper (the executor is 94% of the cost) and a smaller action space, without
# truncating the episode -which flips the sign of the advantage, measured-.
HAND_CAP = None
# WATERING FACTOR. "half the budget goes on moving: measured 42.3% in the
# expert" -but 0.5 was a hand-set constant that had never entered any search,
# like the three market ones that gave +12.8% when exposed. Learned now.
WATERING_FACTOR = 0.5
# Actions per day it costs to sustain one animal. Learned (`f_actions_per_animal`).
ACTIONS_PER_ANIMAL = 3.0
# Softmax temperature over the six priorities. Learned (`f_priority_temp`).
PRIORITY_TEMP = 3.0


# (name, default, kind).
# PARAMETRIZACION SIN BORDES.
#
# It used to be `value = lo + range * f`: two hand-set constants per parameter
# -36 numbers- and, worse, a HARD floor and ceiling. If another league or
# another opponent needed a value outside, the model could not even express it.
# Measured: `dig_value` pinned itself to the floor in 5 of 8 learned vectors.
#
# There are no borders now. Since the network emits `macro_mu` and everywhere
# f = sigmoid(macro_mu), it follows that logit(f) == macro_mu EXACTLY, so:
#
#     positive   value = default * exp(logit(f))           -> (0, +inf)
#     fraction   value = sigmoid(logit(default) + logit(f)) -> (0, 1)
#
# f = 0.5 returns the exact default and the extremes reach the whole domain.
# No scale constant remains: the factor that would multiply logit(f) is 1
# because sigmoid and logit cancel, not because someone chose it.
#
# The DOMAIN is not taste: a budget fraction lives in [0,1] because of what it
# means. And where the ENGINE sets the limit, the engine's limit is used -see
# `max_animal` en `aplica_parametros`-.
#
# QUE NO ESTA AQUI: `horiz_fert` se fue. Era un HECHO DEL MOTOR disfrazado de
# parametro -el fertilizante dura 3 dias exactos- y dejarlo aprender permitia
# letting the policy believe it lasts seven and mis-value every fertilisation.
# It lives in `spec.FERTILIZER_DAYS`, with the other engine facts.
#
# `extra_threshold` has no original because it is new: its default is the p95
# of what the tasks the heuristic proposes are worth ($424 of 6,431).
PARAM_TABLE = [
    ("labour",      0.15,  "fraction"),
    ("seed",        0.50,  "fraction"),
    ("hand_margin",    3.00,  "positive"),
    ("feed_days",    3.00,  "positive"),
    ("sat_high",       0.85,  "fraction"),
    ("sat_low",       0.60,  "fraction"),
    ("land_return",   2.00,  "positive"),
    ("land_cash",      1.50,  "positive"),
    ("step_discount", 0.82,  "fraction"),
    ("watering",   0.50,  "positive"),
    ("extra_threshold", 424.0,   "positive"),
    ("turns_init",     3.00,  "positive"),
    ("turns_min",     2.00,  "positive"),
    ("dig_value",      0.90,  "positive"),
    ("map_gain",  1.00,  "positive"),
    ("map_cap",     20.00,  "positive"),
    ("max_animal",     2.00,  "positive"),
    ("manure_credit",    0.50,  "fraction"),
    ("feed_cash",    0.25,  "fraction"),
    ("labour_floor",    60.00,  "positive"),
    ("liquidation",       2.00,  "positive"),
    ("seed_stock",  2.00,  "positive"),
    ("animal_reserve", 300.0, "positive"),
    ("actions_per_animal", 3.00, "positive"),
    ("last_hire_hour",  3.00,  "positive"),
    ("min_hand_days",      2.00,  "positive"),
    ("cost_rise",     0.50,  "fraction"),
    ("cost_decay",     0.93,  "fraction"),
    ("fert_per_trip",     4.00,  "positive"),
    ("priority_temp",     3.00,  "positive"),
    ("seed_floor",        2.00,  "positive"),
    ("fert_trip",         2.00,  "positive"),
    ("wheat_trip",        2.00,  "positive"),
    ("water_idle",        0.40,  "positive"),
    ("feed_value",        2.00,  "positive"),
    ("care_value",        0.50,  "positive"),
]

# Coefficients of the per-turn selling rule. They are NOT in PARAM_TABLE
# because their form is different: they are ADDITIVE in the exponent, not
# multipliers of a default. The default of an additive coefficient is 0, and
# `default * exp(z)` degenerates to 0 forever. The natural form is w = logit(f)
# directly, which satisfies the same property: f = 0.5 reproduces the previous
# behaviour and there are no borders.
TURN_WEIGHTS = ("price", "rival", "shed", "season", "cash")


def turn_weights(macro) -> dict:
    """How much selling reacts to each turn signal. Default 0 = today's rule."""
    if macro is None:
        return {k: 0.0 for k in TURN_WEIGHTS}
    # NO default on getattr. It used to be `getattr(macro, "w_"+k, 0.5)`, and
    # when a field was renamed and this tuple was not, the lookup silently fell
    # back to the neutral value and the whole per-turn rule switched itself off
    # without a single error. An AttributeError here is the correct outcome.
    return {k: _logit(getattr(macro, "w_" + k)) for k in TURN_WEIGHTS}


def market_factors(macro: Macro) -> dict:
    """Learned multiplier per product. 0.5 -> 1.0 (neutral), borderless."""
    from . import spec as _sp
    out = {}
    for pr in _sp.PRODUCTS:
        # no default: a renamed field must raise, not silently
        # fall back to neutral (see `turn_weights`)
        f = getattr(macro, "m_" + pr.lower())
        out[pr] = math.exp(_logit(f))
    return out


def _logit(f: float) -> float:
    """logit with a numerical guard. The 1e-6 is not modelling: it is the
    epsilon that avoids infinity. It leaves a reach of exp(+-13.8) = 1e6 times
    the default, which for practical purposes is no border at all."""
    f = min(1.0 - 1e-6, max(1e-6, float(f)))
    return math.log(f / (1.0 - f))


def params(macro: Macro) -> dict:
    """The hand-set ones, in their real units and WITHOUT borders."""
    out = {}
    for n, default, kind in PARAM_TABLE:
        z = _logit(getattr(macro, "f_" + n))
        if kind == "fraction":
            out[n] = 1.0 / (1.0 + math.exp(-(_logit(default) + z)))
        else:
            out[n] = default * math.exp(z)
    return out


def apply_params(macro: Macro) -> None:
    """Write the parameters where the layers read them. One single place.

    Done this way -rather than threading the macro through six signatures-
    because `tasks.py` and `executor.py` read them from functions that do not
    receive the macro, and changing those signatures would touch far more code
    than this change justifies.
    """
    if macro is None:
        return
    from .symbolic import market_ops as _M, tasks as _T, executor as _E
    p = params(macro)
    _M.LABOUR_BUDGET_FRACTION = p["labour"]
    _M.SEED_BUDGET_FRACTION      = p["seed"]
    _M.HAND_MARGIN           = p["hand_margin"]
    _M.FEED_STOCK_DAYS     = p["feed_days"]
    _M.SAT_HIGH              = p["sat_high"]
    _M.SAT_LOW              = p["sat_low"]
    _M.LAND_RETURN          = p["land_return"]
    _M.LAND_CASH             = p["land_cash"]
    _T.STEP_DISCOUNT    = p["step_discount"]
    _T.DIG_VALUE             = p["dig_value"]
    _E.TURNS_PER_TILE_INIT    = p["turns_init"]
    _E.TURNS_PER_TILE_MIN    = p["turns_min"]
    _T.MAP_GAIN         = p["map_gain"]
    _T.MAP_CAP             = p["map_cap"]
    # The CEILING comes from the engine, not from us: `maxMarketOrdersPerTurn`.
    # How many animals to buy per turn is policy; how many FIT is mechanics.
    _M.MAX_ANIMALS_PER_TURN      = max(1, min(int(spec.DEFAULT_CONFIG["maxMarketOrdersPerTurn"]),
                                          int(round(p["max_animal"]))))
    # Engine fact, not a parameter: fertiliser lasts exactly 3 days.
    _T.FERTILIZER_HORIZON  = int(spec.FERTILIZER_DAYS)
    _M.MANURE_CREDIT           = p["manure_credit"]
    _M.FEED_CASH_FRACTION           = p["feed_cash"]
    _M.LABOUR_FLOOR            = p["labour_floor"]
    _M.LIQUIDATION_DAYS         = p["liquidation"]
    _M.SEED_STOCK_PER_UNIT         = p["seed_stock"]
    _M.ANIMAL_CASH_RESERVE        = p["animal_reserve"]
    _M.LAST_HIRE_HOUR         = p["last_hire_hour"]
    _M.MIN_HAND_DAYS             = p["min_hand_days"]
    _E.COST_RISE            = p["cost_rise"]
    _E.COST_DECAY            = p["cost_decay"]
    _T.FERT_PER_TRIP            = p["fert_per_trip"]
    _M.SEED_FLOOR              = p["seed_floor"]
    _T.FERT_TRIP_VALUE         = p["fert_trip"]
    _T.WHEAT_TRIP_VALUE        = p["wheat_trip"]
    _T.WATER_IDLE_VALUE        = p["water_idle"]
    _T.FEED_VALUE              = p["feed_value"]
    _T.CARE_VALUE              = p["care_value"]
    global PRIORITY_TEMP
    PRIORITY_TEMP = p["priority_temp"]
    global ACTIONS_PER_ANIMAL
    ACTIONS_PER_ANIMAL = p["actions_per_animal"]
    _T.EXTRA_TILE_THRESHOLD          = p["extra_threshold"]
    global WATERING_FACTOR
    WATERING_FACTOR = p["watering"]


def target_hands(obs, macro: Macro) -> int:
    # NO CEILING, and the engine backs that up: it does not limit hiring
    # -`_hire_cost(hires_today)` only makes it MORE EXPENSIVE- so the 15 that
    # used to be here was a wall I put over a learnable level. Same mechanism
    # as the parameters: 0.5 gives the usual value (7.5 -> 8) and the extremes
    # reach any workforce, which is what is needed for the model to work in
    # leagues with another starting cash and another horizon.
    n = max(0, int(round(7.5 * math.exp(_logit(macro.hands)))))
    return n if HAND_CAP is None else min(n, HAND_CAP)


def target_tiles(obs, macro: Macro) -> int:
    """Cuantas casillas plantadas mantener. Tope exacto: las que se pueden regar.

    A plant left unwatered for two days turns into a weed, so the real ceiling
    is not land but actions: each tile costs one watering per day.
    """
    me = int(obs["player"])
    farm = obs["farms"][me]
    # PLANNED, not the ones present right now. The macro is decided at hour 0,
    # when yesterday's hands have been cleared (`farm["hands"] = []` nightly)
    # y los de hoy aun no se han contratado: `len(farm["hands"])` vale SIEMPRE 0
    # there. With that the ceiling was pinned at 12 tiles however many hands
    # were hired afterwards, and the farm could not grow. CEM did not choose a
    # granja de 7 casillas y 1.9 unidades: era la unica alcanzable.
    n_units = 1 + target_hands(obs, macro)
    plantable = sum(1 for y in range(spec.BOARD) for x in range(spec.BOARD)
                      if farm["tiles"][y][x] != "LOCKED")
    # half the budget goes on moving: measured 42.3% in the expert
    # SIN BORDE, como en `peones_objetivo`. Antes era
    #     macro.casillas * min(cultivables, techo_riego)
    # y ahi hay DOS techos de naturaleza distinta metidos en el mismo `min`:
    #
    #   * `cultivables` es un limite DURO DEL MOTOR: en una casilla LOCKED no
    #     se puede plantar, y punto. Se queda como limite.
    #   * `techo_riego` NO lo es: es una ESTIMACION mia de cuantas casillas se
    #     pueden atender, y entraba como tope infranqueable. La politica no
    #     could not ask for more even when it paid -for instance with crops
    #     that do not need daily watering, or with more units than planned-.
    #     Same flaw as the 15-hand cap removed right next to this.
    #
    # Now the watering ceiling is the SCALE and the policy multiplies it
    # without a border: f = 1/3 reproduces the previous value
    # (exp(logit(1/3)) = 0.5), f = 0.5 asks for the whole watering ceiling, and
    # the extremes reach any farm the engine allows. The only clipping left is
    # the engine's.
    watering_cap = n_units * spec.TURNS_PER_DAY * WATERING_FACTOR
    wanted = watering_cap * math.exp(_logit(macro.tiles))
    return max(0, int(min(plantable, wanted)))


def target_animals(obs, macro: Macro) -> int:
    """Cuantos animales sostener. Cada uno cuesta ~3 acciones al dia."""
    me = int(obs["player"])
    farm = obs["farms"][me]
    n_units = 1 + target_hands(obs, macro)   # planificadas, ver arriba
    # DOUBLE COUNTING FIXED. It had `* 0.5` ("half goes on moving") AND
    # `/ ACTIONS_PER_ANIMAL = 3`, which already includes the travel. The result
    # was 4 tasks per unit per day, half the real figure.
    #
    # A strong public agent sustains 58 crops + 17 animals = 75 tasks with 9.4
    # units: 8 tasks per unit per day, exactly 24/3. Measured: with
    # el `0.5`, los animales saturaban en ~19 pasara lo que pasara -incluso con
    # 4 cuadrantes y 100 casillas libres-.
    capacity = n_units * spec.TURNS_PER_DAY
    planted = sum(1 for row in farm["tiles"] for t in row
                    if isinstance(t, dict) and t.get("kind") == "PLANT")
    room = max(0.0, capacity - planted) / max(1e-6, ACTIONS_PER_ANIMAL)
    return max(0, int(macro.animals * room))


def sell_horizon(obs, macro: Macro) -> int:
    """How many turns to look ahead before deciding to sell.

    Es la decision acoplada al rival: aguantar producto solo compensa si el
    the opponent does not crash the price first. It is also where the world
    model has measured signal (opponent money +19%, their spending +35%).
    """
    # The default is ONE DAY of horizon -`spec.TURNS_PER_DAY`, from the engine-
    # no ceiling: the old `2 *` limited it to two days by my decision, and
    # holding produce longer is exactly the play that can pay against an
    # opponent who is not crashing the price.
    return max(1, int(round(spec.TURNS_PER_DAY
                            * math.exp(_logit(macro.selling)))))


def target_crop(obs, macro: Macro):
    """Interpolate between the fastest crop and the most profitable per tile-day.

    Not a free choice: picking a crop that has no time to mature is a certain
    loss, so only the viable ones enter, which is exact.
    """
    from .symbolic.tasks import cycle_days, cycle_profit
    days = spec.EPISODE_STEPS // spec.TURNS_PER_DAY - int(obs["day"])
    viable = [c for c in spec.CROP_LIST if cycle_days(c) <= days]
    if not viable:
        return None
    if macro.crop <= 0.0:
        return min(viable, key=lambda c: cycle_days(c))
    by_value = sorted(viable, key=lambda c: cycle_profit(obs, c) / max(1, cycle_days(c)))
    i = min(len(by_value) - 1, int(macro.crop * len(by_value)))
    return by_value[i]


def assignment_stickiness(macro: Macro) -> float:
    """Multiplicative bonus for keeping the previous turn's destination.

    El humgaro reasigna desde cero cada turno y eso es miope: medido, el 23.8%
    de las decisiones de destino son un cambio estando YA EN RUTA, y los pasos
    dice are rolled. The remedy was measured too, with the CEM vector:

        0.00 -> 23 793 $     0.25 -> 27 177 $  (+14%)
        0.10 -> 26 887 $     0.50 -> 24 869 $
                             1.00 -> 22 755 $  (demasiado pegado: ignora urgencias)

    Tiene optimo interior, asi que no es un "cuanto mas mejor" y no se puede
    fixed by reasoning. The policy decides it.

    Design note: this is a term per (UNIT, tile), not per tile. A 10x10 value
    map cannot express it -units differ not only in position and inventory but
    also in their prior commitment-.
    """
    # BORDERLESS. It used to be `2.0 * macro.adherencia`, linear in [0,1]
    # and therefore capped at 2.0 -a ceiling nobody searched-. Now it is
    # the same borderless form as every other parameter: f = 0.5 gives
    # 1.0, the old midpoint, and the extremes reach any value.
    return math.exp(_logit(macro.stickiness))


def priorities(macro: Macro) -> dict:
    """Split of the cash across categories. Softmax over the 6 components.

    Returns fractions summing to 1. The learned temperature lets the policy
    concentrate almost everything on one category (with one component at 1 and
    the rest at 0 it takes 73% at the default temperature) without the uniform
    split being a strange point: with all equal, each gets 1/6.
    """
    import math
    vals = [getattr(macro, "p_" + c) for c in CATEGORIES]
    e = [math.exp(PRIORITY_TEMP * v) for v in vals]
    total = sum(e) or 1.0
    return {c: x / total for c, x in zip(CATEGORIES, e)}


def category_order(macro: Macro) -> list:
    """Categorias ordenadas de mas a menos prioritaria.

    It decides both the cash split and the POSITION in the order list, which
    matters because the engine only accepts `maxMarketOrdersPerTurn` and the
    se cae en silencio.
    """
    p = priorities(macro)
    return sorted(CATEGORIES, key=lambda c: -p[c])


def fertilize_weight(macro: Macro) -> float:
    """What fertilising is worth, as a multiplier of its computed value.

    It is a DIAL, not a switch. Tried as a global switch with the frozen vector
    it gave -$21,998 +- 4,563, but that vector was optimised for a world
    WITHOUT fertiliser: it had 7 hands because it needed no more. Judging a
    capability without re-optimising shows it in its worst light.

    Con un mando, la busqueda decide: si no paga lo deja en 0 y no cuesta nada
    have it; if it pays with more labour, it will find it together with the
    hands it needs. That is the only way to capture the interaction.
    """
    # Sin techo: el 3.0 de antes era un maximo elegido a ojo. 0.5 da 1.5, que
    # which is what the old midpoint gave.
    return 1.5 * math.exp(_logit(macro.fertilize))
