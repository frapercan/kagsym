"""Codificacion de la observacion para las redes.

Extraido de `kagworld/encode.py` dejando fuera la maquinaria de objetivos
residuales del world model (`residual_targets`, `opp_history`, `encode_action`):
esa rama esta medida como casi sin valor y vive en `kagworld/` por
reproducibilidad. Aqui queda lo que el camino vivo usa: `encode_obs`,
`legal_ops`, y la valoracion exacta a precio marginal.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import spec

BOARD = spec.BOARD

# ---------------------------------------------------------------------------
# Canales de la rejilla, por jugador. El orden ES el layout del tensor.
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
N_GRID_CH = 2 * N_TILE_CH          # [yo, rival]

# ---------------------------------------------------------------------------
# Vector global. Cada bloque documenta su rango para poder leer el residual.
# ---------------------------------------------------------------------------
def _global_layout() -> list[tuple[str, int]]:
    return [
        # +2: HORIZONTE Y TOPE ABSOLUTOS. Todo lo demas del bloque "time" esta
        # NORMALIZADO por spec.N_DAYS / spec.EPISODE_STEPS, que son globales por
        # proceso y valen lo que dure la partida en curso: una de 5 dias y una
        # de 30 producen exactamente la misma señal (0 -> 1). La red no podia
        # distinguirlas, asi que al entrenar a horizontes mezclados promediaba
        # dos estrategias incompatibles. Medido: ascendiendo por 5->8->10 dias
        # con win=1,00 en autojuego, el margen real contra la escalera cayo de
        # -82,7 % a -98,3 %. Y los optimos son distintos de verdad: peones
        # 0,057 a 5 dias, 0,610 a 8, 0,378 a 30 -no monotono-.
        # +3 (era +2): se anade HORAS POR DIA absolutas. Medido el 2026-09-21:
        # `hour / spec.TURNS_PER_DAY` normaliza la jornada y la borra, igual que
        # `day / N_DAYS` borraba el horizonte. Consecuencia: 8h x 13d (104
        # turnos) y 5h x 21d (105) se separan solo por 0,0014 en el rasgo de
        # horizonte -L2 total 0,26- y sus optimos difieren 2,8x (5,113x contra
        # 1,815x). Para la red eran la misma celda. Y las horas por dia son el
        # eje que MAS manda, porque deciden que fraccion de la jornada se come
        # el trayecto: a 5 h/dia el peon recien contratado gasta el dia entero
        # andando hasta el cuadrante abierto y no llega nunca -medido: 83 de
        # 104 acciones son NORTH/WEST-, mientras que a 24 h/dia ese mismo
        # trayecto son 8 de 24 acciones.
        ("time", 5 + 1 + len(spec.CROP_LIST) + len(spec.ANIMALS) + 1 + 3 + 3),
        # day, hour, sin h, cos h, step  +  FASE DEL EPISODIO:
        #   dias restantes normalizados
        #   viabilidad exacta por cultivo   (¿madura antes del cierre?)
        #   viabilidad exacta por animal    (¿da tiempo a amortizarlo?)
        #   urgencia de liquidacion         (el cobertizo vale 0 al cerrar)
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
# Accion
# ---------------------------------------------------------------------------
ACTION_TILE_CHANNELS = [*spec.UNIT_OPS, *[f"plant_{c}" for c in spec.CROP_LIST]]
ACTION_TILE_CH = {n: i for i, n in enumerate(ACTION_TILE_CHANNELS)}
N_ACTION_GRID_CH = len(ACTION_TILE_CHANNELS)
N_ACTION_GLOBAL = spec.N_MARKET_SLOTS

# Escalas de normalizacion (elegidas por rango tipico, no por estadistica del
# dataset: asi el encoder no depende de los datos con los que se entreno).
INV_SCALE = 500.0        # el inventario se mueve ~cientos alrededor de I0
MONEY_SCALE = 3000.0
YIELD_SCALE = 6.0
# Los objetivos (flujo, gasto) se emiten en UNIDADES FISICAS: unidades de
# producto y dolares. No hay escala inventada porque no hace falta: la perdida
# aprende una sigma por cabeza, asi que la escala se estima de los datos en vez
# de fijarla yo. Las entradas se normalizan con estadisticas medidas del buffer
# (ver model.Normalizer), no con divisores a ojo.


def flow_revenue(product: str, inv0: int, n: float, params=None) -> float:
    """Dinero que mueve un flujo de `n` unidades de `product` desde `inv0`.

    Se valora unidad a unidad con la funcion de precio EXACTA del motor, porque
    el motor cotiza asi: cada unidad vendida baja el precio de la siguiente.
    Con trigo, vender 10 unidades lo mueve de 25 a 23 (-8%), asi que valorar al
    precio inicial se queda corto en flujos grandes.

    n > 0 = el rival vende (entra oferta); n < 0 = recompra.
    """
    k = int(round(n))
    if k == 0:
        return 0.0
    total = 0.0
    if k > 0:
        for j in range(k):
            total += spec.market_price(product, inv0 + j, params)
    else:
        # El motor cotiza BUY_PRODUCT al inventario ya descontado.
        for j in range(-k):
            total -= spec.market_price(product, inv0 - 1 - j, params)
    return total


def flow_value(flow: np.ndarray, inv0: np.ndarray, params=None) -> float:
    """Valor total en $ de un vector de flujo de 9 productos."""
    return sum(flow_revenue(p, int(inv0[i]), float(flow[i]), params)
               for i, p in enumerate(spec.PRODUCTS) if abs(flow[i]) >= 0.5)


def market_inventory(obs) -> np.ndarray:
    return np.array([obs["market"]["inventory"][p] for p in spec.PRODUCTS],
                    dtype=np.float64)


def _quadrant_flags(farm) -> list[float]:
    q = set(farm["unlocked_quadrants"])
    return [1.0 if k in q else 0.0 for k in spec.LAND_ORDER]


def encode_farm_grid(farm, day: int, out: np.ndarray) -> None:
    """Escribe los N_TILE_CH canales de una granja en `out` (N_TILE_CH,B,B)."""
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
    """Tope de curriculo sobre NUESTROS peones, o None. Se lee tarde para no
    crear un ciclo de importacion entre obs y macro."""
    try:
        from . import macro as _M
        return _M.TOPE_PEONES
    except Exception:
        return None


def encode_obs(obs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Observacion -> (grid (N_GRID_CH,B,B), global (N_GLOBAL,)), float32.

    La granja propia va siempre en el primer bloque de canales, asi que la red
    no tiene que aprender a desambiguar quien es quien.
    """
    me = int(obs["player"])
    opp = 1 - me
    farms = obs["farms"]
    day, hour, step = obs["day"], obs["hour"], obs["step"]

    grid = np.zeros((N_GRID_CH, BOARD, BOARD), dtype=np.float32)
    encode_farm_grid(farms[me], day, grid[:N_TILE_CH])
    encode_farm_grid(farms[opp], day, grid[N_TILE_CH:])

    g = np.zeros(N_GLOBAL, dtype=np.float32)
    # TIEMPO RESTANTE, no transcurrido. Lo que decide en este juego es cuanto
    # queda: plantar solo compensa si el cultivo madura antes del turno 720, un
    # animal solo amortiza si quedan dias, y la liquidacion tiene que terminar
    # antes del cierre porque el cobertizo puntua CERO. Dando solo contadores
    # crecientes, la red tendria que aprender que 720 es el final y restar,
    # cuando la viabilidad es exactamente calculable de las tablas del motor.
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
    # urgencia: 1 en el ultimo dia, 0 mientras sobre temporada
    urgencia = max(0.0, 1.0 - dias_q / 2.0)
    # FASE DEL DIA, tambien en terminos de decision. La hora tiene estructura
    # dura en el motor: a la hora 0 se limpian los peones y se reinicia el coste
    # fibonacci de contratar, la ventana de contratacion se cierra a la hora 3,
    # y al CIERRE del dia se vuelcan los inventarios (lo que no cabe se tira) y
    # muere toda planta sin regar.
    #
    # La tercera es la que de verdad decide: holgura de riego. Si quedan menos
    # acciones-unidad hoy que plantas sin regar, esta noche mueren algunas y hay
    # que priorizar. Es exactamente calculable.
    horas_q = (spec.TURNS_PER_DAY - hour) / spec.TURNS_PER_DAY
    puede_contratar = 1.0 if hour <= 3 else 0.0
    n_unid = 1 + len(farms[me]["hands"])
    acciones_q = n_unid * (spec.TURNS_PER_DAY - hour)
    sin_regar = sum(1 for row in farms[me]["tiles"] for t in row
                    if isinstance(t, dict) and t.get("kind") == "PLANT"
                    and not t.get("watered_today"))
    holgura = 0.0 if sin_regar == 0 else max(-1.0, min(1.0,
              (acciones_q / 3.0 - sin_regar) / max(1.0, sin_regar)))
    g[GLOBAL_SLICES["time"]] = ([
        day / spec.N_DAYS,
        hour / spec.TURNS_PER_DAY,
        math.sin(2 * math.pi * hour / spec.TURNS_PER_DAY),
        math.cos(2 * math.pi * hour / spec.TURNS_PER_DAY),
        step / spec.EPISODE_STEPS,
        dias_q / spec.N_DAYS,
    ] + viables_cultivo + viables_animal
        + [urgencia, horas_q, puede_contratar, holgura,
           # ABSOLUTOS, con 30 dias y 15 peones como referencia fija.
           # EPISODE_STEPS, no N_DAYS: `set_episode_steps` solo actualiza el
           # primero y N_DAYS se queda clavado en 30 para siempre.
           spec.EPISODE_STEPS / 720.0,
           (_tope_peones() or spec.HANDS_REF) / float(spec.HANDS_REF),
           # AL FINAL DEL BLOQUE a proposito: `migrar_ckpt` inserta las columnas
           # nuevas en `GLOBAL_SLICES["time"].stop - N_NUEVAS`. Ponerlo en otro
           # sitio desplazaria los pesos equivocados, en silencio.
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
HORIZONTES = (4, 12, 24, 48)
N_OPP_HIST = (spec.TURNS_PER_DAY
              + len(HORIZONTES) * len(spec.PRODUCTS)
              + HIST_TURNS * len(spec.PRODUCTS))


SHAPES = {
    "grid": (N_GRID_CH, BOARD, BOARD),
    "global": (N_GLOBAL,),
    "action_grid": (N_ACTION_GRID_CH, BOARD, BOARD),
    "action_global": (N_ACTION_GLOBAL,),
    "flow": (len(spec.PRODUCTS),),
    "spend": (1,),
}


# --- legalidad de acciones de unidad ----------------------------------------
# Derivada UNA A UNA de `_apply_unit_action` del motor. No es heuristica: si el
# motor devuelve no-op silencioso, aqui la accion esta prohibida.
#
# Por que hacia falta: medido sobre 336 turnos, la politica emitia 573 PLANT y
# conseguia 0.7 cultivos vivos de media (el experto: 108 PLANT -> 29.5 cultivos).
# Mas del 99% de sus acciones eran no-ops. Un gradiente de politica no puede
# corregir eso, porque el motor devuelve el mismo estado haga lo que haga: la
# accion ilegal no produce senal de aprendizaje, solo consume el presupuesto de
# exploracion.
def legal_ops(obs) -> "np.ndarray":
    """(MAX_UNITS, N_UNIT_OPS) booleana. Filas de unidades inexistentes: todo False salvo PASS."""
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
    hay_semilla = any(int(v) > 0 for v in seeds.values())
    cobertizo = priv.get("shed", {}) or {}
    hay_en_cobertizo = any(int(v) > 0 for v in cobertizo.values())
    invs = priv.get("inventories", priv.get("inventory", [])) or []

    for i in range(n):
        x, y = pos[i]
        inv = invs[i] if i < len(invs) and isinstance(invs[i], dict) else {}
        # movimiento: legal si no sale del tablero (LOCKED si se permite)
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
            if hay_semilla:
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
            # PLACE de animal sobre estructura vacia compatible
            if t.get("kind") in ("COOP", "PASTURE") and not animal:
                if any(int(inv.get(a, 0)) > 0 for a in spec.ANIMALS):
                    m[i, ix["PLACE"]] = True
    return m


def _shed_access(board: int):
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    return {tuple(p) for p in K._shed_access_tiles(board)}


# --- OFERTA INMINENTE DEL RIVAL ------------------------------------------
# Por que existe. El mercado es compartido: cuando el rival vuelca genero, el
# precio marginal cae y nuestro ingreso baja. Medido: nuestra conducta mueve su
# puntuacion un 57 %, y su liquidacion aporta el 99,1 % de la varianza de la
# recompensa diaria -por eso meterla en el shaping hundia el critico a R2
# -2,535-. La leccion fue "el shaping solo puede llevar lo que el estado
# predice"; esto ataca la otra mitad de la frase: hacer que el estado lo prediga.
#
# La entrada existia -N_HIST = 4 x N_PRODUCTOS, "flujo del rival en 4 ventanas
# hacia atras"- y el entrenador la rellenaba con CEROS. Se reinterpreta como
# oferta FUTURA en 4 horizontes, que es lo causalmente anterior a nuestro
# ingreso, y no como historia pasada.
#
# Y es predecible desde lo que vemos: su tablero se observa entero, asi que no
# le pedimos adivinar su estrategia, solo leer su cosecha.
VENTANAS_RIVAL = (1, 2, 4, 8)          # dias vista
N_HIST_RIVAL = len(VENTANAS_RIVAL) * len(spec.PRODUCTS)

# HORIZONTES DE LA TAREA AUXILIAR, en Fibonacci. Predecir solo manana es casi
# trivial -el crecimiento de un dia es determinista- y no obliga al codificador
# a nada. Lo que hay que modelar son los CICLOS: trigo 2-4 dias, tomate 8,
# fresa y melon 10, y la fresa repite cada 2. Una escala geometrica los cubre
# con pocos objetivos y reparte la dificultad como se degrada la informacion
# real: cerca predecible, lejos grueso.
HORIZONTES_AUX = (1, 2, 3, 5, 8, 13)
N_AUX_RIVAL = len(HORIZONTES_AUX) * len(spec.PRODUCTS)


def rival_ready(obs, opp: int = None) -> np.ndarray:
    """Unidades por producto que el rival tiene LISTAS hoy. Es el objetivo
    auxiliar: se le pide predecir este vector a 1, 2, 3, 5, 8 y 13 dias."""
    f = rival_flow(obs, opp)
    return f.reshape(len(VENTANAS_RIVAL), -1)[0].copy()


def rival_flow(obs, opp: int = None) -> np.ndarray:
    """Unidades por producto que el rival tendra listas en cada ventana."""
    me = int(obs.get("player", 0))
    opp = (1 - me) if opp is None else opp
    dia = int(obs.get("day", 0))
    out = np.zeros((len(VENTANAS_RIVAL), len(spec.PRODUCTS)), dtype=np.float32)
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
                # cuantos dias faltan para que de fruto
                falta = max(0, int(cd["first_yield_day"]) - age)
            elif "animal" in t:
                a = spec.ANIMALS.get(t.get("animal"))
                if a is None:
                    continue
                prod = a["product"]
                falta = max(0, int(a["first_yield_day"])
                            - (dia - int(t.get("placed_day", dia))))
            else:
                continue
            j = spec.PRODUCTS.index(prod) if prod in spec.PRODUCTS else None
            if j is None:
                continue
            for k, v in enumerate(VENTANAS_RIVAL):
                if falta <= v:
                    out[k, j] += max(listo, 1.0)
    return out.reshape(-1)
