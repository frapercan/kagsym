"""Constantes del juego, importadas en vivo del motor instalado.

Nada se copia a mano. Si Kaggle publica una version de kaggle-environments con
tablas distintas (precios base, tiempos de cultivo, tiendas), este modulo la
sigue sola y `verify()` avisa si alguna suposicion del encoder se rompe.
"""
from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as _eng

# --- tablas del motor -------------------------------------------------------
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

# Duracion del fertilizante, en DIAS. No es un parametro y no se aprende: lo
# fija el motor en el manejador de FERTILIZE -"Active for `day`, `day+1`,
# `day+2` (3 days inclusive)"; `fertilized_until_day = day + 2`-. Se escribe
# aqui, junto al resto de hechos del motor, porque el motor no lo expone con
# nombre; no porque sea una eleccion nuestra.
FERTILIZER_DAYS: int = 3

# --- vocabularios ordenados (el orden ES el layout del tensor) --------------
CROP_LIST = sorted(CROPS)                      # CARROT, MELON, STRAWBERRY, TOMATO, WHEAT
ANIMAL_LIST = sorted(ANIMALS)                  # COW, GOOSE, SHEEP
SHOP_LIST = sorted(SHOPS)
SHED_ITEMS = PRODUCTS + sorted(ANIMALS)        # lo que cabe en el cobertizo
STRUCTURES = ["COOP", "PASTURE"]

CROP_IX = {c: i for i, c in enumerate(CROP_LIST)}
ANIMAL_IX = {a: i for i, a in enumerate(ANIMAL_LIST)}
SHOP_IX = {s: i for i, s in enumerate(SHOP_LIST)}
PRODUCT_IX = {p: i for i, p in enumerate(PRODUCTS)}
SHED_IX = {p: i for i, p in enumerate(SHED_ITEMS)}

# --- espacio de acciones ----------------------------------------------------
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
BUYABLE_PRODUCTS = ["WHEAT", "FERTILIZER"]   # unicos con BUY_PRODUCT

# Layout plano del vector de accion de mercado.
MARKET_ACTION_SLOTS: list[tuple[str, str | None]] = (
    [("BUY_SEED", c) for c in CROP_LIST]
    + [("BUY_PRODUCT", p) for p in BUYABLE_PRODUCTS]
    + [("BUY_ANIMAL", a) for a in ANIMAL_LIST]
    + [("SELL", p) for p in SHED_ITEMS]
    + [("HIRE", None), ("BUY_LAND", None)]
)
MARKET_SLOT_IX = {k: i for i, k in enumerate(MARKET_ACTION_SLOTS)}
N_MARKET_SLOTS = len(MARKET_ACTION_SLOTS)

# --- defaults de configuracion (los del kaggriculture.json) -----------------
DEFAULT_CONFIG = {
    "episodeSteps": 720,
    "actTimeout": 1,
    "boardSize": 10,
    # Leible del entorno: `EntornoParalelo` lanza procesos hijos y el
    # entorno crea FastEnv solo con episodeSteps, heredando lo demas de
    # aqui. Un bucle exterior fija la caja con os.environ antes de lanzar.
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
    """Falla ruidosamente si el motor instalado ya no encaja con el encoder."""
    problems = []
    if len(CROP_LIST) != 5:
        problems.append(f"se esperaban 5 cultivos, hay {len(CROP_LIST)}: {CROP_LIST}")
    if len(ANIMAL_LIST) != 3:
        problems.append(f"se esperaban 3 animales, hay {len(ANIMAL_LIST)}")
    if len(PRODUCTS) != 9:
        problems.append(f"se esperaban 9 productos, hay {len(PRODUCTS)}")
    if len(SHOP_LIST) != 8:
        problems.append(f"se esperaban 8 tiendas, hay {len(SHOP_LIST)}")
    missing = [a["structure"] for a in ANIMALS.values() if a["structure"] not in STRUCTURES]
    if missing:
        problems.append(f"estructuras de animal desconocidas: {set(missing)}")
    for c, d in CROPS.items():
        if not {"seed", "first_yield_day", "max_yield_day", "ongoing"} <= set(d):
            problems.append(f"al cultivo {c} le faltan campos esperados")
    if problems:
        raise RuntimeError(
            "El motor de kaggriculture instalado no encaja con kagworld/spec.py:\n  - "
            + "\n  - ".join(problems)
        )


# Duracion REAL del episodio. `DEFAULT_CONFIG` son 720, pero evaluar a otra
# duracion con este valor fijo hace que el agente planifique contra un horizonte
# que no existe: compra animales que no llegan a amortizar y retiene producto
# mas alla del cierre. Lo fija `set_episode_steps` desde la configuracion.
EPISODE_STEPS = DEFAULT_CONFIG["episodeSteps"]


def set_turns_per_day(n) -> None:
    """Cambia las horas por dia. Ningun modulo puede guardar un alias de
    `TURNS_PER_DAY`: se leen en tiempo de llamada precisamente para esto."""
    global TURNS_PER_DAY
    if n:
        TURNS_PER_DAY = int(n)
        DEFAULT_CONFIG["turnsPerDay"] = int(n)


def set_episode_steps(n) -> None:
    global EPISODE_STEPS
    if n:
        EPISODE_STEPS = int(n)

# Referencia FIJA para normalizar el tope de peones en la observacion.
# No puede depender del curriculo: si cambiara, la señal dejaria de ser absoluta.
PEONES_REF = 15
