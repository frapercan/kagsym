"""Game constants, imported live from the installed engine.

Nothing is copied by hand. If Kaggle ships a version of kaggle-environments
with different tables (base prices, growth times, shops), this module follows
it automatically and `verify()` reports any encoder assumption that breaks.
"""
from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as _eng

# --- engine tables ----------------------------------------------------------
CROPS: dict = _eng.CROPS
ANIMALS: dict = _eng.ANIMALS
PRODUCTS: list = _eng.PRODUCTS
SHOPS: dict = _eng.SHOPS
MARKET_PARAMS: dict = _eng.MARKET_PARAMS
LAND_ORDER: list = _eng.LAND_ORDER
LAND_PRICES: list = _eng.LAND_PRICES
MARKET_I0: int = _eng.MARKET_I0
PRICE_FLOOR: int = _eng.PRICE_FLOOR
MAX_SHOP_INSTANCES: int = _eng.MAX_SHOP_INSTANCES
TOWN_CENTER_PRODUCTS: list = _eng.TOWN_CENTER_PRODUCTS

market_price = _eng.market_price

# Fertiliser duration, in DAYS. Not a parameter and not learned: the engine
# fixes it in the FERTILIZE handler -"Active for `day`, `day+1`, `day+2`
# (3 days inclusive)"; `fertilized_until_day = day + 2`-. It is written here,
# among the other engine facts, because the engine does not expose it under a
# name; not because it is a choice of ours.
FERTILIZER_DAYS: int = 3

# --- ordered vocabularies (the order IS the tensor layout) ------------------
CROP_LIST = sorted(CROPS)                      # CARROT, MELON, STRAWBERRY, TOMATO, WHEAT
ANIMAL_LIST = sorted(ANIMALS)                  # COW, GOOSE, SHEEP
SHOP_LIST = sorted(SHOPS)
SHED_ITEMS = PRODUCTS + sorted(ANIMALS)        # what fits in the shed
STRUCTURES = ["COOP", "PASTURE"]

CROP_IX = {c: i for i, c in enumerate(CROP_LIST)}
ANIMAL_IX = {a: i for i, a in enumerate(ANIMAL_LIST)}
SHOP_IX = {s: i for i, s in enumerate(SHOP_LIST)}
PRODUCT_IX = {p: i for i, p in enumerate(PRODUCTS)}
SHED_IX = {p: i for i, p in enumerate(SHED_ITEMS)}

# --- action space -----------------------------------------------------------
MOVE_OPS = ["NORTH", "SOUTH", "EAST", "WEST"]
UNIT_OPS = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST",
    "PLANT", "WATER", "HARVEST", "FERTILIZE",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
    "DIG", "PICKUP", "PLACE",
]
UNIT_OP_IX = {op: i for i, op in enumerate(UNIT_OPS)}
N_UNIT_OPS = len(UNIT_OPS)

MARKET_OPS = ["BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"]
BUYABLE_PRODUCTS = ["WHEAT", "FERTILIZER"]   # the only ones with BUY_PRODUCT

# Flat layout of the market action vector.
MARKET_ACTION_SLOTS: list[tuple[str, str | None]] = (
    [("BUY_SEED", c) for c in CROP_LIST]
    + [("BUY_PRODUCT", p) for p in BUYABLE_PRODUCTS]
    + [("BUY_ANIMAL", a) for a in ANIMAL_LIST]
    + [("SELL", p) for p in SHED_ITEMS]
    + [("HIRE", None), ("BUY_LAND", None)]
)
MARKET_SLOT_IX = {k: i for i, k in enumerate(MARKET_ACTION_SLOTS)}
N_MARKET_SLOTS = len(MARKET_ACTION_SLOTS)

# --- configuration defaults (those of kaggriculture.json) -------------------
DEFAULT_CONFIG = {
    "episodeSteps": 720,
    "actTimeout": 1,
    "boardSize": 10,
    # Read from the environment: the parallel env spawns child processes and
    # each child builds FastEnv with episodeSteps only, inheriting the rest
    # from here. An outer loop sets the starting cash via os.environ.
    "startingMoney": int(__import__("os").environ.get("KAG_CAJA", "3000")),
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
}

BOARD = DEFAULT_CONFIG["boardSize"]
TURNS_PER_DAY = DEFAULT_CONFIG["turnsPerDay"]
N_DAYS = DEFAULT_CONFIG["episodeSteps"] // TURNS_PER_DAY


def verify() -> None:
    """Fail loudly if the installed engine no longer matches the encoder."""
    problems = []
    if len(CROP_LIST) != 5:
        problems.append(f"expected 5 crops, found {len(CROP_LIST)}: {CROP_LIST}")
    if len(ANIMAL_LIST) != 3:
        problems.append(f"expected 3 animals, found {len(ANIMAL_LIST)}")
    if len(PRODUCTS) != 9:
        problems.append(f"expected 9 products, found {len(PRODUCTS)}")
    if len(SHOP_LIST) != 8:
        problems.append(f"expected 8 shops, found {len(SHOP_LIST)}")
    missing = [a["structure"] for a in ANIMALS.values() if a["structure"] not in STRUCTURES]
    if missing:
        problems.append(f"unknown animal structures: {set(missing)}")
    for c, d in CROPS.items():
        if not {"seed", "first_yield_day", "max_yield_day", "ongoing"} <= set(d):
            problems.append(f"crop {c} is missing expected fields")
    if problems:
        raise RuntimeError(
            "The installed kaggriculture engine does not match kagsym/spec.py:\n  - "
            + "\n  - ".join(problems)
        )


# REAL episode length. `DEFAULT_CONFIG` says 720, but evaluating at another
# length with this value fixed makes the agent plan against a horizon that does
# not exist: it buys animals that never pay back and holds product past the
# close. `set_episode_steps` sets it from the configuration.
EPISODE_STEPS = DEFAULT_CONFIG["episodeSteps"]


def set_turns_per_day(n) -> None:
    """Change the hours per day.

    No module may cache an alias of `TURNS_PER_DAY`: it is read at call time
    precisely so that this function takes effect everywhere.
    """
    global TURNS_PER_DAY
    if n:
        TURNS_PER_DAY = int(n)
        DEFAULT_CONFIG["turnsPerDay"] = int(n)


def set_episode_steps(n) -> None:
    global EPISODE_STEPS
    if n:
        EPISODE_STEPS = int(n)

# FIXED reference for normalising the hand cap in the observation. It cannot
# depend on the curriculum: if it moved, the signal would stop being absolute.
HANDS_REF = 15
