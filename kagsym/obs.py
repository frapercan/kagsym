"""Observation encoding for the networks.

Everything here is derived from the engine, never guessed: the per-tile
channels mirror the engine's tile fields, the legality mask is derived one by
one from `_apply_unit_action`, and valuation uses the engine's marginal price.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import spec

BOARD = spec.BOARD

# ---------------------------------------------------------------------------
# Grid channels, per player. The order IS the tensor layout.
# ---------------------------------------------------------------------------
TILE_CHANNELS = [
    "locked", "empty", "weed", "plant", "coop", "pasture",
    *[f"crop_{c}" for c in spec.CROP_LIST],
    *[f"animal_{a}" for a in spec.ANIMAL_LIST],
    "watered_or_fed", "consec_missed", "yield_units", "age_days",
    "fertilized", "cared_today", "fertilizer_avail", "pending_care",
    "has_animal", "farmer_here", "hands_here",
]
TILE_CH = {name: i for i, name in enumerate(TILE_CHANNELS)}
N_TILE_CH = len(TILE_CHANNELS)
N_GRID_CH = 2 * N_TILE_CH          # [me, opponent]

# ---------------------------------------------------------------------------
# Global vector. Each block documents its range so the layout can be read.
# ---------------------------------------------------------------------------
def _global_layout() -> list[tuple[str, int]]:
    return [
        # +2: ABSOLUTE HORIZON AND CAP. Everything else in the "time" block is
        # NORMALISED by spec.N_DAYS / spec.EPISODE_STEPS, which are per-process
        # globals holding whatever the current episode lasts: a 5-day and a
        # 30-day game produce exactly the same signal (0 -> 1). The network
        # could not tell them apart, so training on mixed horizons averaged two
        # incompatible strategies. Measured: climbing 5->8->10 days with
        # win=1.00 in self-play, the real margin against the ladder fell from
        # -82.7% to -98.3%. And the optima genuinely differ: hands 0.057 at 5
        # days, 0.610 at 8, 0.378 at 30 -not monotonic-.
        # +3 (was +2): absolute HOURS PER DAY added. Measured: `hour /
        # spec.TURNS_PER_DAY` normalises the working day and erases it, just as
        # `day / N_DAYS` erased the horizon. Consequence: 8h x 13d (104 turns)
        # and 5h x 21d (105) differ by only 0.0014 in the horizon feature -0.26
        # in total L2- while their optima differ by 2.8x (5.113x against
        # 1.815x). To the network they were the same cell. And hours per day is
        # the axis that matters MOST, because it decides what fraction of the
        # day the commute eats: at 5 h/day a freshly hired hand spends the
        # whole day walking to the open quadrant and never arrives -measured:
        # 83 of 104 actions are NORTH/WEST- whereas at 24 h/day that same
        # commute is 8 of 24 actions.
        ("time", 5 + 1 + len(spec.CROP_LIST) + len(spec.ANIMALS) + 1 + 3 + 3),
        # day, hour, sin h, cos h, step  +  EPISODE PHASE:
        #   normalised days remaining
        #   exact per-crop viability     (does it mature before the close?)
        #   exact per-animal viability   (is there time to pay it back?)
        #   liquidation urgency          (the shed scores 0 at the close)
        ("money", 4),                          # yo/rival x {lineal, log}
        ("mkt_inv", len(spec.PRODUCTS)),
        ("mkt_price", len(spec.PRODUCTS)),
        ("town", len(spec.SHOP_LIST)),
        ("shed", len(spec.SHED_ITEMS)),
        ("seeds", len(spec.CROP_LIST)),
        ("carried", len(spec.SHED_ITEMS)),
        ("labour", 4),                         # hires/hands x jugador
        ("land", 6),                           # 3 cuadrantes x jugador
        ("shed_fill", 1),
    ]


GLOBAL_LAYOUT = _global_layout()
GLOBAL_SLICES: dict[str, slice] = {}
_off = 0
for _name, _n in GLOBAL_LAYOUT:
    GLOBAL_SLICES[_name] = slice(_off, _off + _n)
    _off += _n
N_GLOBAL = _off

# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
ACTION_TILE_CHANNELS = [*spec.UNIT_OPS, *[f"plant_{c}" for c in spec.CROP_LIST]]
ACTION_TILE_CH = {n: i for i, n in enumerate(ACTION_TILE_CHANNELS)}
N_ACTION_GRID_CH = len(ACTION_TILE_CHANNELS)
N_ACTION_GLOBAL = spec.N_MARKET_SLOTS

# Normalisation scales (chosen from the typical range, not from dataset
# statistics: that way the encoder does not depend on the data it was trained
# on).
INV_SCALE = 500.0        # inventory moves by ~hundreds around I0
MONEY_SCALE = 3000.0
YIELD_SCALE = 6.0
# The targets (flow, spending) are emitted in PHYSICAL UNITS: product units
# and dollars. There is no invented scale because none is needed: the loss
# learns one sigma per head, so the scale is estimated from the data instead of
# being fixed by hand.


def flow_revenue(product: str, inv0: int, n: float, params=None) -> float:
    """Money moved by a flow of `n` units of `product` starting at `inv0`.

    Valued unit by unit with the engine's EXACT price function, because that
    is how the engine quotes: each unit sold lowers the price of the next. With
    wheat, selling 10 units moves it from 25 to 23 (-8%), so valuing at the
    initial price falls short on large flows.

    n > 0 = the opponent sells (supply enters); n < 0 = they buy back.
    """
    k = int(round(n))
    if k == 0:
        return 0.0
    total = 0.0
    if k > 0:
        for j in range(k):
            total += spec.market_price(product, inv0 + j, params)
    else:
        # The engine quotes BUY_PRODUCT at the already-decremented inventory.
        for j in range(-k):
            total -= spec.market_price(product, inv0 - 1 - j, params)
    return total


def flow_value(flow: np.ndarray, inv0: np.ndarray, params=None) -> float:
    """Total value in dollars of a 9-product flow vector."""
    return sum(flow_revenue(p, int(inv0[i]), float(flow[i]), params)
               for i, p in enumerate(spec.PRODUCTS) if abs(flow[i]) >= 0.5)


def market_inventory(obs) -> np.ndarray:
    return np.array([obs["market"]["inventory"][p] for p in spec.PRODUCTS],
                    dtype=np.float64)


def _quadrant_flags(farm) -> list[float]:
    q = set(farm["unlocked_quadrants"])
    return [1.0 if k in q else 0.0 for k in spec.LAND_ORDER]


def encode_farm_grid(farm, day: int, out: np.ndarray) -> None:
    """Write a farm's N_TILE_CH channels into `out` (N_TILE_CH,B,B)."""
    tiles = farm["tiles"]
    for y in range(BOARD):
        row = tiles[y]
        for x in range(BOARD):
            t = row[x]
            if t is None:
                out[TILE_CH["empty"], y, x] = 1.0
                continue
            if t == "LOCKED":
                out[TILE_CH["locked"], y, x] = 1.0
                continue
            kind = t.get("kind")
            if kind == "WEED":
                out[TILE_CH["weed"], y, x] = 1.0
            elif kind == "PLANT":
                out[TILE_CH["plant"], y, x] = 1.0
                out[TILE_CH[f"crop_{t['crop']}"], y, x] = 1.0
                out[TILE_CH["watered_or_fed"], y, x] = float(t["watered_today"])
                out[TILE_CH["consec_missed"], y, x] = t["consecutive_unwatered"] / 2.0
                out[TILE_CH["yield_units"], y, x] = t["yield_units"] / YIELD_SCALE
                out[TILE_CH["age_days"], y, x] = (day - t["planted_day"]) / spec.N_DAYS
                out[TILE_CH["fertilized"], y, x] = float(t["fertilized_until_day"] >= day)
            elif kind in ("COOP", "PASTURE"):
                out[TILE_CH["coop" if kind == "COOP" else "pasture"], y, x] = 1.0
                animal = t.get("animal")
                if animal:
                    out[TILE_CH["has_animal"], y, x] = 1.0
                    out[TILE_CH[f"animal_{animal}"], y, x] = 1.0
                    out[TILE_CH["watered_or_fed"], y, x] = float(t.get("fed_today", False))
                    out[TILE_CH["consec_missed"], y, x] = t.get("consecutive_unfed", 0) / 2.0
                    out[TILE_CH["yield_units"], y, x] = t.get("yield_units", 0) / YIELD_SCALE
                    out[TILE_CH["age_days"], y, x] = (day - t.get("placed_day", day)) / spec.N_DAYS
                    out[TILE_CH["cared_today"], y, x] = float(t.get("cared_today", False))
                    out[TILE_CH["fertilizer_avail"], y, x] = float(t.get("fertilizer_available", False))
                    out[TILE_CH["pending_care"], y, x] = t.get("pending_care_bonus", 0) / YIELD_SCALE

    fx, fy = farm["farmer"]
    out[TILE_CH["farmer_here"], fy, fx] = 1.0
    for hx, hy in farm["hands"]:
        out[TILE_CH["hands_here"], hy, hx] += 0.25


def _tope_peones():
    """Curriculum cap on OUR hands, or None. Imported late to avoid an
    import cycle between obs and macro."""
    try:
        from . import macro as _M
        return _M.HAND_CAP
    except Exception:
        return None


def encode_obs(obs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Observation -> (grid (N_GRID_CH,B,B), global (N_GLOBAL,)), float32.

    Our own farm always occupies the first block of channels, so the network
    does not have to learn to tell who is who.
    """
    me = int(obs["player"])
    opp = 1 - me
    farms = obs["farms"]
    day, hour, step = obs["day"], obs["hour"], obs["step"]

    grid = np.zeros((N_GRID_CH, BOARD, BOARD), dtype=np.float32)
    encode_farm_grid(farms[me], day, grid[:N_TILE_CH])
    encode_farm_grid(farms[opp], day, grid[N_TILE_CH:])

    g = np.zeros(N_GLOBAL, dtype=np.float32)
    # TIME REMAINING, not elapsed. What decides in this game is how much is
    # left: planting only pays if the crop matures before turn 720, an animal
    # only pays back if days remain, and liquidation must finish before the
    # close because the shed scores ZERO. Given only increasing counters, the
    # network would have to learn that 720 is the end and subtract, when
    # viability is exactly computable from the engine tables.
    remaining = max(0, spec.EPISODE_STEPS - 1 - step)
    dias_q = remaining / spec.TURNS_PER_DAY
    viables_cultivo = []
    for c in spec.CROP_LIST:
        cd = spec.CROPS[c]
        ciclo = cd["first_yield_day"] if cd["ongoing"] else cd["max_yield_day"]
        viables_cultivo.append(1.0 if dias_q >= ciclo else 0.0)
    viables_animal = []
    for a in spec.ANIMALS:
        d = spec.ANIMALS[a]
        viables_animal.append(1.0 if dias_q >= d["first_yield_day"] + d["interval"] else 0.0)
    # urgency: 1 on the last day, 0 while season remains
    urgencia = max(0.0, 1.0 - dias_q / 2.0)
    # PHASE OF THE DAY, also in decision terms. The hour has hard structure in
    # the engine: at hour 0 the hands are cleared and the fibonacci hiring cost
    # resets, and at the CLOSE of the day
    # inventories are flushed (whatever does not fit is discarded) and every
    # unwatered plant dies.
    #
    # The third is the one that really decides: watering slack. If fewer
    # unit-actions remain today than there are unwatered plants, some will die
    # tonight and there is a choice to make. It is exactly computable.
    horas_q = (spec.TURNS_PER_DAY - hour) / spec.TURNS_PER_DAY
    # NOT an engine rule. `_do_hire` has no hour restriction: verified in the
    # installed engine. This is just an "early in the day" indicator, and the
    # actual hiring window is the learned `LAST_HIRE_HOUR`. It is kept because
    # it is a cheap, monotone feature and removing it would change the
    # observation layout and invalidate every checkpoint for no gain -the same
    # information is already carried by `hour`, `sin h` and `cos h`-.
    temprano = 1.0 if hour <= 3 else 0.0
    n_unid = 1 + len(farms[me]["hands"])
    action_q = n_unid * (spec.TURNS_PER_DAY - hour)
    sin_regar = sum(1 for row in farms[me]["tiles"] for t in row
                    if isinstance(t, dict) and t.get("kind") == "PLANT"
                    and not t.get("watered_today"))
    holgura = 0.0 if sin_regar == 0 else max(-1.0, min(1.0,
              (action_q / 3.0 - sin_regar) / max(1.0, sin_regar)))
    g[GLOBAL_SLICES["time"]] = ([
        day / spec.N_DAYS,
        hour / spec.TURNS_PER_DAY,
        math.sin(2 * math.pi * hour / spec.TURNS_PER_DAY),
        math.cos(2 * math.pi * hour / spec.TURNS_PER_DAY),
        step / spec.EPISODE_STEPS,
        dias_q / spec.N_DAYS,
    ] + viables_cultivo + viables_animal
        + [urgencia, horas_q, temprano, holgura,
           # ABSOLUTE, with 30 days and 15 hands as the fixed reference.
           # EPISODE_STEPS, not N_DAYS: `set_episode_steps` only updates the
           # former and N_DAYS stays pinned at 30 forever.
           spec.EPISODE_STEPS / 720.0,
           (_tope_peones() or spec.HANDS_REF) / float(spec.HANDS_REF),
           # AT THE END OF THE BLOCK on purpose: `migrate_ckpt` inserts the
           # new columns at `GLOBAL_SLICES["time"].stop - n_new`. Putting it
           # anywhere else would shift the wrong weights, silently.
           spec.TURNS_PER_DAY / 24.0])
    m_me, m_opp = float(farms[me]["money"]), float(farms[opp]["money"])
    g[GLOBAL_SLICES["money"]] = [
        m_me / MONEY_SCALE, math.log1p(max(0.0, m_me)) / 10.0,
        m_opp / MONEY_SCALE, math.log1p(max(0.0, m_opp)) / 10.0,
    ]
    inv, prices = obs["market"]["inventory"], obs["market"]["prices"]
    g[GLOBAL_SLICES["mkt_inv"]] = [
        (inv[p] - spec.MARKET_I0) / INV_SCALE for p in spec.PRODUCTS]
    g[GLOBAL_SLICES["mkt_price"]] = [
        prices[p] / spec.MARKET_PARAMS[p]["base"] for p in spec.PRODUCTS]

    shops = obs["town"].get("unlocked_shops", [])
    town = np.zeros(len(spec.SHOP_LIST), dtype=np.float32)
    for s in shops:
        town[spec.SHOP_IX[s]] += 1.0 / spec.MAX_SHOP_INSTANCES
    g[GLOBAL_SLICES["town"]] = town

    private = obs["private"]
    shed = private.get("shed", {})
    g[GLOBAL_SLICES["shed"]] = [shed.get(i, 0) / 20.0 for i in spec.SHED_ITEMS]
    g[GLOBAL_SLICES["seeds"]] = [
        private.get("seeds", {}).get(c, 0) / 10.0 for c in spec.CROP_LIST]
    carried: dict = {}
    for invd in private.get("inventories", []):
        for k, v in invd.items():
            carried[k] = carried.get(k, 0) + v
    g[GLOBAL_SLICES["carried"]] = [carried.get(i, 0) / 10.0 for i in spec.SHED_ITEMS]
    g[GLOBAL_SLICES["labour"]] = [
        farms[me]["hires_today"] / 12.0, len(farms[me]["hands"]) / 12.0,
        farms[opp]["hires_today"] / 12.0, len(farms[opp]["hands"]) / 12.0,
    ]
    g[GLOBAL_SLICES["land"]] = _quadrant_flags(farms[me]) + _quadrant_flags(farms[opp])
    g[GLOBAL_SLICES["shed_fill"]] = sum(shed.values()) / spec.DEFAULT_CONFIG["shedCapacity"]
    return grid, g


HIST_TURNS = 4
HORIZONS = (4, 12, 24, 48)
N_OPP_HIST = (spec.TURNS_PER_DAY
              + len(HORIZONS) * len(spec.PRODUCTS)
              + HIST_TURNS * len(spec.PRODUCTS))


SHAPES = {
    "grid": (N_GRID_CH, BOARD, BOARD),
    "global": (N_GLOBAL,),
    "action_grid": (N_ACTION_GRID_CH, BOARD, BOARD),
    "action_global": (N_ACTION_GLOBAL,),
    "flow": (len(spec.PRODUCTS),),
    "spend": (1,),
}


# --- unit action legality ---------------------------------------------------
# Derived ONE BY ONE from the engine's `_apply_unit_action`. Not a heuristic:
# wherever the engine returns a silent no-op, the action is forbidden here.
#
# Why it was needed: measured over 336 turns, the policy emitted 573 PLANT and
# achieved 0.7 live crops on average (the expert: 108 PLANT -> 29.5 crops).
# More than 99% of its actions were no-ops. A policy gradient cannot correct
# that, because the engine returns the same state whatever it does: an illegal
# action produces no learning signal, it only consumes the exploration budget.
def legal_ops(obs) -> "np.ndarray":
    """(MAX_UNITS, N_UNIT_OPS) boolean. Rows of non-existent units: all
    False except PASS."""
    import numpy as np

    from . import bc
    me = int(obs["player"])
    farm = obs["farms"][me]
    priv = obs["private"]
    B = spec.BOARD
    day = int(obs["day"])
    tiles = farm["tiles"]
    ix = spec.UNIT_OP_IX

    acceso = _shed_access(B)
    pos = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
    n = min(len(pos), bc.MAX_UNITS)
    m = np.zeros((bc.MAX_UNITS, spec.N_UNIT_OPS), dtype=bool)
    m[:, ix["PASS"]] = True

    seeds = priv.get("seeds", {}) or {}
    has_seed = any(int(v) > 0 for v in seeds.values())
    cobertizo = priv.get("shed", {}) or {}
    hay_en_cobertizo = any(int(v) > 0 for v in cobertizo.values())
    invs = priv.get("inventories", priv.get("inventory", [])) or []

    for i in range(n):
        x, y = pos[i]
        inv = invs[i] if i < len(invs) and isinstance(invs[i], dict) else {}
        # movement: legal if it stays on the board (LOCKED is allowed)
        for op, (dx, dy) in (("NORTH", (0, -1)), ("SOUTH", (0, 1)),
                             ("EAST", (1, 0)), ("WEST", (-1, 0))):
            if 0 <= x + dx < B and 0 <= y + dy < B:
                m[i, ix[op]] = True

        junto_cobertizo = (x, y) in acceso
        if junto_cobertizo:
            if hay_en_cobertizo:
                m[i, ix["PICKUP"]] = True
            if inv:
                m[i, ix["PLACE"]] = True

        t = tiles[y][x]
        if t == "LOCKED":
            continue

        if t is None:
            if has_seed:
                m[i, ix["PLANT"]] = True
            m[i, ix["BUILD_COOP"]] = True
            m[i, ix["BUILD_PASTURE"]] = True
            continue

        if isinstance(t, dict):
            animal = t.get("animal")
            if not animal:
                m[i, ix["DIG"]] = True
            if t.get("kind") == "PLANT":
                if not t.get("watered_today"):
                    m[i, ix["WATER"]] = True
                if int(inv.get("FERTILIZER", 0)) > 0:
                    m[i, ix["FERTILIZE"]] = True
                if int(t.get("yield_units", 0)) > 0:
                    cd = spec.CROPS[t["crop"]]
                    if day - int(t["planted_day"]) >= cd["first_yield_day"]:
                        m[i, ix["HARVEST"]] = True
            if animal:
                if not t.get("fed_today") and int(inv.get("WHEAT", 0)) > 0:
                    m[i, ix["FEED"]] = True
                if not t.get("cared_today"):
                    m[i, ix["CARE"]] = True
                if t.get("fertilizer_available"):
                    m[i, ix["COLLECT_FERTILIZER"]] = True
                if int(t.get("yield_units", 0)) > 0:
                    m[i, ix["HARVEST"]] = True
            # PLACE an animal onto a compatible empty structure
            if t.get("kind") in ("COOP", "PASTURE") and not animal:
                if any(int(inv.get(a, 0)) > 0 for a in spec.ANIMALS):
                    m[i, ix["PLACE"]] = True
    return m


def _shed_access(board: int):
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    return {tuple(p) for p in K._shed_access_tiles(board)}


# --- THE OPPONENT'S IMMINENT SUPPLY -----------------------------------------
# Why it exists. The market is shared: when the opponent dumps produce, the
# marginal price falls and our income drops. Measured: our behaviour moves
# their score by 57%, and their liquidation accounts for 99.1% of the variance
# of the daily reward -which is why putting it in the shaping sank the critic
# to R2 -2.535-. The lesson was "shaping can only carry what the state
# predicts"; this attacks the other half of that sentence: making the state
# predict it.
#
# The input already existed -N_HIST = 4 x N_PRODUCTS, "opponent flow over 4
# backward windows"- and the trainer filled it with ZEROS. It is reinterpreted
# as FUTURE supply over 4 horizons, which is what causally precedes our income,
# rather than as past history.
#
# And it is predictable from what we see: their board is fully observable, so
# we are not asking the network to guess their strategy, only to read their
# harvest.
RIVAL_WINDOWS = (1, 2, 4, 8)          # dias vista
N_HIST_RIVAL = len(RIVAL_WINDOWS) * len(spec.PRODUCTS)

# AUXILIARY TASK HORIZONS, in Fibonacci. Predicting only tomorrow is nearly
# trivial -one day of growth is deterministic- and forces the encoder to do
# nothing. What has to be modelled are the CYCLES: wheat 2-4 days, tomato 8,
# strawberry and melon 10, and strawberry repeats every 2. A geometric scale
# covers them with few targets and spreads the difficulty the way real
# information degrades: predictable near, coarse far.
AUX_HORIZONS = (1, 2, 3, 5, 8, 13)
N_AUX_RIVAL = len(AUX_HORIZONS) * len(spec.PRODUCTS)


def rival_ready(obs, opp: int = None) -> np.ndarray:
    """Units per product the opponent has READY today. This is the auxiliary
    target: the network predicts this vector at 1, 2, 3, 5, 8 and 13 days."""
    f = rival_flow(obs, opp)
    return f.reshape(len(RIVAL_WINDOWS), -1)[0].copy()


def rival_flow(obs, opp: int = None) -> np.ndarray:
    """Units per product the opponent will have ready in each window."""
    me = int(obs.get("player", 0))
    opp = (1 - me) if opp is None else opp
    dia = int(obs.get("day", 0))
    out = np.zeros((len(RIVAL_WINDOWS), len(spec.PRODUCTS)), dtype=np.float32)
    try:
        tiles = obs["farms"][opp]["tiles"]
    except Exception:
        return out.reshape(-1)
    for row in tiles:
        for t in row:
            if not isinstance(t, dict):
                continue
            listo = float(t.get("yield_units", 0) or 0)
            if t.get("kind") == "PLANT":
                cd = spec.CROPS.get(t.get("crop"))
                if cd is None:
                    continue
                prod, age = t["crop"], dia - int(t.get("planted_day", dia))
                # how many days until it bears fruit
                missing = max(0, int(cd["first_yield_day"]) - age)
            elif "animal" in t:
                a = spec.ANIMALS.get(t.get("animal"))
                if a is None:
                    continue
                prod = a["product"]
                missing = max(0, int(a["first_yield_day"])
                            - (dia - int(t.get("placed_day", dia))))
            else:
                continue
            j = spec.PRODUCTS.index(prod) if prod in spec.PRODUCTS else None
            if j is None:
                continue
            for k, v in enumerate(RIVAL_WINDOWS):
                if missing <= v:
                    out[k, j] += max(listo, 1.0)
    return out.reshape(-1)
