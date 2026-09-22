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
    """Un movimiento greedy. No hay obstaculos: las casillas son todas pasables."""
    x, y = pos
    tx, ty = target
    if x != tx:
        return "EAST" if x < tx else "WEST"
    if y != ty:
        return "SOUTH" if y < ty else "NORTH"
    return "PASS"


def dist(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- valor de un cultivo ----------------------------------------------------
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
    """Beneficio neto de un ciclo al precio de mercado ACTUAL."""
    return cycle_yield(crop) * unit_price(obs, crop) - spec.CROPS[crop]["seed"]


def days_left(obs) -> int:
    total = spec.EPISODE_STEPS // spec.TURNS_PER_DAY
    return total - obs["day"]


def plantable(obs, crop: str) -> bool:
    """No tiene sentido plantar lo que no dara tiempo a cosechar."""
    return cycle_days(crop) < days_left(obs)


def growth_factor(obs, crop: str) -> float:
    """Factor by which capital multiplies per day when reinvested.

    Un ciclo convierte `seed` dolares en `unidades x precio` dolares en
    `cycle_days` dias, asi que el factor diario es (retorno/coste)^(1/dias).
    Con trigo: 10 -> 100 en 4 dias = x1.78 al dia. Con melon: 80 -> 1500 en 12
    dias = x1.28. El melon gana por casilla-dia y pierde por capital-dia, y
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
    barato = min(spec.CROPS[c]["seed"] for c in spec.CROP_LIST if plantable(obs, c)) \
        if any(plantable(obs, c) for c in spec.CROP_LIST) else 1
    return money < free_tiles * barato


def best_crop(obs, seeds_only: bool = False, free_tiles: int = 0) -> str | None:
    """El mejor cultivo SEGUN CUAL SEA LA RESTRICCION.

    With tiles to spare and little money the constraint is capital and the
    que mas rapido lo multiplica. Con dinero de sobra y pocas casillas, la
    constraint is land and the winner is whatever yields most per tile-day.

    Confusing the two costs the game: optimising tile-day from turn 0
    lleva a plantar melon, que no paga nada durante 12 dias, quedarse sin caja y
    no poder ni contratar peones.
    """
    seeds = obs["private"].get("seeds", {})
    por_capital = capital_is_binding(obs, free_tiles)
    best, value = None, 0.0
    for c in spec.CROP_LIST:
        if not plantable(obs, c):
            continue
        if seeds_only and seeds.get(c, 0) <= 0:
            continue
        v = growth_factor(obs, c) if por_capital else cycle_profit(obs, c) / max(1, cycle_days(c))
        if v > value:
            best, value = c, v
    return best


# --- tareas -----------------------------------------------------------------
def _acceso_cobertizo():
    """Casillas de acceso al cobertizo, calculadas UNA vez.

    The set was rebuilt on every call, and it is called 44,275 times per
    episode (once per tile per turn). It is a board constant.
    """
    global _ACCESO
    if _ACCESO is None:
        import kaggle_environments.envs.kaggriculture.kaggriculture as K
        _ACCESO = frozenset(tuple(p) for p in K._shed_access_tiles(BOARD))
    return _ACCESO


_ACCESO = None


def _is_shed_access(x: int, y: int) -> bool:
    return (x, y) in _acceso_cobertizo()


def _shed_task(obs, farm, ctx=None, macro=None):
    """What to take out of the shed. Without this the animal chain never closes:
    se compra, cae en el cobertizo y se queda ahi para siempre.

    Y el trigo importa igual: FEED consume 1 trigo DEL INVENTARIO DE LA UNIDAD,
    so an animal nobody brings wheat to escapes after two days.
    """
    priv = obs["private"]
    shed = priv["shed"]
    # 1) an animal to place, if there is or could be room
    if ctx is None:
        ctx = contexto_turno(obs, farm)
    # Only the animal actually in the shed is valued.
    hay = [a for a in spec.ANIMALS if int(shed.get(a, 0)) > 0]
    for a in sorted(hay, key=lambda a: -animal_value(ctx, a, macro)):
        if ctx.free_slots.get(spec.ANIMALS[a]["structure"], 0) > 0:
            return (max(1.0, animal_value(ctx, a, macro) / ctx.days),
                    ["PICKUP", a, 1])
    # 1b) DROP. The harvest stays in the unit's inventory until the close of
    # the day; with DROP it reaches the shed NOW and can be sold the same day,
    # besides avoiding the nightly flush overflowing the 100-unit shed and
    # discarding the excess. The opponent uses it 417 times per episode and we
    # estaba en el repertorio.
    #
    # VALOR MARGINAL. NO vale lo que se lleva encima: el motor vuelca los
    # inventories to the shed only at the close of the day, so dropping early
    # does not change that the goods end up there. All it adds is being able to
    # SELL IT TODAY, before the price falls. Valuing it gross made all the
    # unidades corrieran al cobertizo: medido, 39 558 -> 13 250 $.
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
                ahora = float(sum(marginal_prices(obs, item, int(n_))))
                luego = float(future_price(obs, item, spec.TURNS_PER_DAY)) * int(n_)
                v_ += max(0.0, ahora - luego)      # only the drop avoided
        best = max(best, v_)
    if best > 0:
        return (best, ["DROP"])

    # 2) fertiliser, if there are plants to fertilise and it is in the shed
    fert = int(shed.get("FERTILIZER", 0))
    if fert > 0:
        fertilizables = sum(1 for row in farm["tiles"] for t in row
                            if isinstance(t, dict) and t.get("kind") == "PLANT"
                            and t.get("fertilized_until_day", -1) < obs["day"])
        if fertilizables > 0:
            n = min(fert, fertilizables, int(FERT_PER_TRIP))
            return (2.0 * unit_price(obs, "FERTILIZER"), ["PICKUP", "FERTILIZER", n])

    # 3) wheat to feed the animals that have not eaten yet
    hambrientos = sum(1 for row in farm["tiles"] for t in row
                      if isinstance(t, dict) and t.get("animal") and not t.get("fed_today"))
    if hambrientos > 0 and int(shed.get("WHEAT", 0)) > 0:
        n = min(hambrientos, int(shed["WHEAT"]))
        return (2.0 * unit_price(obs, "WHEAT"), ["PICKUP", "WHEAT", n])
    return None


# What a unit must be CARRYING for the operation not to be a no-op.
REQUIERE = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}


def _can_drop(inv) -> bool:
    """DROP only makes sense while carrying something."""
    return any(int(v) > 0 for v in (inv or {}).values())


def _animales_esperando(obs) -> list:
    """Animals bought but not yet placed (in the shed or in hand)."""
    priv = obs["private"]
    fuera = []
    for a in spec.ANIMALS:
        n = int(priv["shed"].get(a, 0))
        n += sum(int(inv.get(a, 0)) for inv in priv.get("inventories", []))
        fuera += [a] * n
    return fuera


def _estructuras_libres(farm) -> dict:
    free_slots = {"COOP": 0, "PASTURE": 0}
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") in free_slots and not t.get("animal"):
                free_slots[t["kind"]] += 1
    return free_slots


def _tasa_riego(farm) -> float:
    """Fraction of plants ALREADY watered today.

    Es la probabilidad empirica de que el riego futuro ocurra, y por tanto el
    the factor by which the fertiliser bonus must be discounted: if only 39%
    of plants get watered, fertilising returns 39% of what it promises.
    Sale del propio estado, no de una constante.
    """
    total = regadas = 0
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                total += 1
                regadas += 1 if t.get("watered_today") else 0
    return (regadas / total) if total else 0.0


class contexto_turno:
    """Facts shared by all 100 tiles of the turn, computed LAZILY.

    Dos medidas, las dos con cProfile:
      - calcularlos dentro de `tile_task`: 10547 escaneos de tablero y 15215
        animal revaluations per episode -> 2,172 steps/s.
      - calcularlos todos al entrar al turno: PEOR, 1853 pasos/s, porque la
        most turns have no pending animal and the cost was paid
        escaneo igualmente.
    Lazy captures the best of both: zero cost when not needed, and one
    sola vez cuando hace falta. (El motor solo hace 38567 pasos/s: el cuello
    siempre estuvo aqui, no en la simulacion.)
    """

    __slots__ = ("obs", "farm", "_pend", "_libres", "_val", "_dias")

    def __init__(self, obs, farm):
        self.obs, self.farm = obs, farm
        self._pend = self._libres = self._val = self._dias = None

    @property
    def pending_animals(self):
        if self._pend is None:
            self._pend = _animales_esperando(self.obs)
        return self._pend

    @property
    def free_slots(self):
        if self._libres is None:
            self._libres = _estructuras_libres(self.farm)
        return self._libres

    @property
    def days(self):
        if self._dias is None:
            self._dias = max(1, days_left(self.obs))
        return self._dias

    def value(self, animal):
        if self._val is None:
            self._val = {}
        if animal not in self._val:
            from .market_ops import animal_net_value
            self._val[animal] = animal_net_value(self.obs, animal)
        return self._val[animal]


def animal_value(ctx, a, macro):
    """Animal value WEIGHTED by the learned factor of its product.

    `ctx.valor` es `animal_net_value`, una formula escrita a mano. Decidir con
    it WHICH animal is picked up, which is placed and which shed is built is
    the same family of hard-wired decisions as `best_crop` and `seed_orders`,
    and it is where the other measured hole lives: v48 makes $60,179 from MILK
    and we make
    13.536. El factor por producto (EGG, MILK, WOOL) ya vive en el vector, asi
    que la preferencia pasa a la red. Neutro = 1.0: sin macro, el orden de
    siempre.

    It exists as a function rather than repeated at each site because it
    already happened once:
    se pondero UN sitio -PICKUP en `tile_options`- y los otros cuatro se
    stayed unweighted for a whole session without anyone noticing.
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
    """(valor en $, operacion) de la mejor accion posible en esa casilla."""
    tile = farm["tiles"][y][x]
    day = obs["day"]
    seeds = obs["private"].get("seeds", {})
    if ctx is None:
        ctx = contexto_turno(obs, farm)

    if _is_shed_access(x, y):
        t = _shed_task(obs, farm, ctx, macro)
        if t is not None:
            return t

    if tile is None:
        # An animal waiting in the shed is worth nothing until there is
        # ponerlo. Construir desbloquea un ingreso de ~1700 $ ya pagado.
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
        return (v * DIG_VALUE, ["DIG"])    # fraccion expuesta, no fija

    if kind == "PLANT":
        cd = spec.CROPS[tile["crop"]]
        age = day - tile["planted_day"]
        price = unit_price(obs, tile["crop"])
        maduro = age >= cd["first_yield_day"] and tile["yield_units"] > 0

        if not tile["watered_today"]:
            # `consecutive_unwatered` NACE EN 1 al plantar: una planta sin regar
            # el mismo dia se convierte en hierba esa noche. Por eso >= 1 ya es
            # an emergency, not a warning. Verified against the engine.
            if tile["consecutive_unwatered"] >= 1:
                restante = max(1, cycle_yield(tile["crop"]) - tile["yield_units"])
                return (restante * price, ["WATER"])
            if not cd["ongoing"]:
                w0 = (cd["max_yield_day"] + 1) // 2
                if w0 <= age <= cd["max_yield_day"] and tile["yield_units"] < cd["max_yield"]:
                    bonus = 2 if tile["fertilized_until_day"] >= day else 1
                    return (bonus * price, ["WATER"])
            else:
                if age >= cd["first_yield_day"] - 1:
                    return (price, ["WATER"])
            return (0.4 * price, ["WATER"])

        # FERTILISE: only where the NETWORK provides the value.
        #
        # La capacidad hacia falta -recogiamos fertilizante 550 veces por
        # partida sin usarlo nunca, y el motor da `bonus = 2 if fertilized`-
        # pero NO se como valorarla. Tres intentos a ojo, los tres peores:
        #   valor bruto (dias x precio)  -> gana siempre a regar: 22 768 $
        #   valor marginal x tasa_riego  -> sigue robando riegos: 25 770 $
        #   movido despues de regar      -> 11 086 $
        # Comparacion PAREADA sobre 8 semillas: -21 998 $ +- 4 563 (4.8 sigma),
        # con dos partidas hundidas a 568 $ y 33 $.
        #
        # So the operation is left AVAILABLE for the network to put a
        # precio en modo directo, y fuera de la heuristica escrita a mano. Es
        # exactly the case that motivates the learned micro head: an action
        # whose value depends on the state in a way I cannot write down.
        # Fertilise dial: 0 turns it off, and the search decides.
        from ..macro import peso_fertilizar
        _pf = peso_fertilizar(macro) if macro is not None else 0.0
        if _pf > 0.0:
            # FERTILIZAR, DESPUES de regar. Estaba ANTES y hacia `return`, asi que
            # a plant needing both got fertilised and the watering
            # nunca llegaba a evaluarse: moria esa noche. El sintoma era plantar
            # 563 and watered 386 -more plantings than waterings is a death
            # sentence-.
            if (not cd["ongoing"] and tile.get("fertilized_until_day", -1) < day):
                w0 = (cd["max_yield_day"] + 1) // 2
                if w0 <= age + 1 <= cd["max_yield_day"] and tile["yield_units"] < cd["max_yield"]:
                    dias_utiles = min(FERTILIZER_HORIZON,
                                      cd["max_yield_day"] - age, days_left(obs))
                    if dias_utiles > 0:
                        # MARGINAL VALUE, not gross. The bonus adds +1 unit per day
                        # REGADO, no por dia. Valorarlo a `dias * precio` lo hacia
                        # ganar siempre a regar (1-2 x precio): medido, el riego
                        # caia de 1 791 a 904, las cosechas de 768 a 360 y el dinero
                        # de 36 566 a 22 768 $.
                        return (_pf * dias_utiles * price * _tasa_riego(farm), ["FERTILIZE"])
            elif cd["ongoing"] and tile.get("fertilized_until_day", -1) < day:
                if age >= cd["first_yield_day"] - 1 and days_left(obs) > 1:
                    return (_pf * min(FERTILIZER_HORIZON, days_left(obs))
                            * price * _tasa_riego(farm), ["FERTILIZE"])

        if maduro and (cd["ongoing"] or age >= cd["max_yield_day"]):
            return (tile["yield_units"] * price, ["HARVEST"])
        if maduro and tile["yield_units"] >= cd["max_yield"]:
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
            return (2 * price, ["FEED"])          # sin comer dos dias, escapa
        if tile.get("yield_units", 0) > 0:
            return (tile["yield_units"] * price, ["HARVEST"])
        if tile.get("fertilizer_available"):
            return (unit_price(obs, "FERTILIZER"), ["COLLECT_FERTILIZER"])
        if not tile.get("cared_today"):
            return (0.5 * price, ["CARE"])
    return None


STEP_DISCOUNT = 0.82   # un paso cuesta un turno: se descuenta el valor
# Two more that were hand-set. Exposing 12 similar constants raised the cell
# ceiling from $939 to $1,201 (+27.9%), so none is taken on trust without
# having entered a search.
DIG_VALUE = 0.9             # desbrozar, como fraccion del valor de plantar
# FORMA CON QUE EL MAPA DE LA RED ENTRA EN EL VALOR. `expm1` es exponencial, asi
# so GAIN decides whether the map SUGGESTS or IMPOSES, and CAP decides where
# it is clipped. Nobody ever searched them. This may explain why the map
# saturated at 4 parameters: perhaps 4 were not enough, the transform was
# limiting their effect.
MAP_GAIN = 1.0   # cuanto pesa lo que emite la red (los tres modos)
MAP_CAP = 20.0      # corte en `ops`/`directo`, dentro de expm1
RESIDUAL_CAP = 3.0    # corte en `residuo`: exp(3) = 20x de amplificacion o
                      # of damping over the heuristic. It decides whether the
                      # network SUGGESTS or IMPOSES.
FERTILIZER_HORIZON = 3    # dias que se le cuentan al bono del fertilizante
FERT_PER_TRIP = 4.0            # fertilizante que se recoge de una vez. Aprendido.


# ---------------------------------------------------------------------------
# ENUMERACION LEGAL. Lo contrario de `tile_task`.
#
# `tile_task` es una cascada de `return`: mezcla LEGALIDAD (que permite el
# motor) con PREFERENCIA (que prefiero yo). El orden de los `return` y las
# value formulas were eleven hand-written constants, and measured at the time
# their optimum was NOT TO PLANT: sweeping the tile target paired over 8 seeds,
# planting cost $41-50k (8-11 sigma) because crops displaced
# la economia animal, que rinde 831 uds de producto frente a 316.
#
# Only legality lives here, which is derivable from the engine. The verb is
# red; el sustantivo (que cultivo, que animal) sale de aritmetica exacta
# -`best_crop`, `ctx.valor`-, que es mecanica, no preferencia.
# UN VERBO POR CULTIVO. Antes habia un solo "PLANT" y QUE se plantaba lo
# decidia `best_crop`, una funcion escrita a mano; el comentario de arriba lo
# llamaba "mecanica, no preferencia". Pero es preferencia, y es la que decide
# 84% of the gap: v48 makes $71,170 from STRAWBERRIES we never touch and
# $60,179 from MILK where we do a fifth. With a single verb the network could
# elegir SI plantar, nunca QUE, y ninguna cabeza aguas abajo podia arreglarlo:
# the market head cannot sell what is not produced -measured, moving its
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

# MASCARA DE DIMENSIONES QUE DE VERDAD DECIDEN.
#
# The micro log-prob is a SUM over 1+N_OPS channels x 100 tiles = 1,614
# Gaussian dimensions, but only those on tiles with
# operaciones legales, y el logit de un verbo solo importa si HABIA con quien
# compararlo. Medido: sin enmascarar, el 99.4 % de los cocientes de importancia
# de PPO se saturan en el tope de +-10 tras UN paso de gradiente -el cociente es
# un producto sobre ~1 570 dimensiones de puro ruido-. No es una eleccion de
# diseno incluirlas: es un fallo.
MASK_ACC = None
FILTER_BY_INVENTORY = False

# MUESTREO DEL VERBO. Por defecto apagado: el ejecutor UMBRALIZA con argmax,
# which is correct for deterministic evaluation. Turning it on samples the
# verb from a softmax over the logits of the LEGAL options, which is what gives
# gradient to a categorical head. `_VERB_RNG` is seeded per episode from
# outside so comparisons stay paired.
import numpy as np
# The Hungarian is the only assignment method (see `assign_units`).
METODO_ASIGNACION = "humgaro"

# UMBRAL de las casillas EXTRA. Lo fija `macro.aplica_parametros` una vez por
# turn from the learned `f_extra_threshold` parameter; this value is only the
# de arranque y equivale a "ninguna casilla extra entra".
#
# What the extra tiles are. Measured at championship scale: the heuristic
# offers 8.94 tasks per turn for 10.88 units and leaves out another 5.81 the
# engine DOES consider legal. Turns are passed 52.8% of the time -v48 passes
# 5.3%- and 44.5 of those points are from having no task to assign, not from
# choosing badly. On 22.7% of turns the heuristic offers nothing and yet
# hay jugadas legales. Verbos desaprovechados: FERTILIZE 3010, HARVEST 837,
# DROP 327, PICKUP 164 -and we harvest 111 times per episode against 420
# de v48-.
#
# Por que la red no podia arreglarlo: en `residuo` el mapa MULTIPLICA el valor
# of existing tasks, but the tile with
# `tile_task is None` no entra en el diccionario y es invisible al aprendizaje.
#
# What decides here: LEGALITY comes from the engine, the VERB is chosen by the
# network among the legal options and the VALUE is emitted by the network.
# There is no hand-written preference order or value: that would put a
# heuristic back exactly where one is being removed.
EXTRA_TILE_THRESHOLD = 2500.0
SAMPLE_VERB = False
VERB_TEMP = 1.0
_VERB_RNG = np.random.default_rng(0)


GRAD_ACC = None


def siembra_verbo(seed: int) -> None:
    global _VERB_RNG
    _VERB_RNG = np.random.default_rng(seed)


def activa_grad(n_ops: int) -> None:
    global GRAD_ACC
    GRAD_ACC = np.zeros(n_ops, dtype=np.float64)


def recoge_grad():
    global GRAD_ACC
    g, GRAD_ACC = GRAD_ACC, None
    return g


def enable_mask():
    import numpy as np
    global MASK_ACC
    MASK_ACC = np.zeros((1 + N_OPS, BOARD, BOARD), dtype=np.float32)


def collect_mask():
    global MASK_ACC
    m = MASK_ACC
    MASK_ACC = None
    return m


def tile_options(obs, farm, x: int, y: int, free_capacity: int, ctx=None,
                 macro=None) -> list:
    """All LEGAL operations on that tile, unvalued and unordered.

    Devuelve [(indice_en_OPS_VOCAB, operacion_completa), ...].
    """
    tile = farm["tiles"][y][x]
    day = obs["day"]
    if ctx is None:
        ctx = contexto_turno(obs, farm)
    out = []

    if _is_shed_access(x, y):
        shed = obs["private"]["shed"]
        hay = [a for a in spec.ANIMALS if int(shed.get(a, 0)) > 0]
        if hay:
            # WHICH animal to take from the shed was another hand-written
            # cuarta de la misma familia: `best_crop`, `seed_orders`,
            # formula. It is weighted by the LEARNED factor of the product
            # the animal gives, which already lives in the macro vector.
            _fa = {}
            if macro is not None:
                try:
                    from ..macro import market_factors
                    _fa = market_factors(macro)
                except Exception:
                    _fa = {}
            best = max(hay, key=lambda a: animal_value(ctx, a, macro))
            out.append((OPS_IX["PICKUP_ANIMAL"], ["PICKUP", best, 1]))
        # CANTIDAD, no solo verbo. `OPS_VOCAB` tiene `PICKUP_WHEAT` como un
        # verbo sin argumento, asi que esto emitia SIEMPRE 1 mientras la
        # heuristica coge `min(hambrientos, trigo_en_cobertizo)`. Medido el
        # measured by comparing the two task tables over THE SAME state:
        # 96,8 % identicas, 0 % de casillas perdidas, 0 % de valores distintos,
        # y el 3,2 % restante eran exactamente esto -`PICKUP WHEAT 2` contra
        # `PICKUP WHEAT 1`-. With one unit bringing a single wheat per trip,
        # the livestock feeding chain runs at half capacity, and livestock is
        # the entire economy at this scale.
        #
        # La cantidad NO es una decision estrategica: la fijan cuantos animales
        # tienen hambre y cuanto hay en el cobertizo. Es del lado simbolico,
        # like legality and routing. The network still chooses the VERB.
        trigo = int(shed.get("WHEAT", 0))
        if trigo > 0:
            hambrientos = sum(1 for row in farm["tiles"] for t in row
                              if isinstance(t, dict) and t.get("animal")
                              and not t.get("fed_today"))
            out.append((OPS_IX["PICKUP_WHEAT"],
                        ["PICKUP", "WHEAT", max(1, min(hambrientos, trigo))]))
        fert = int(shed.get("FERTILIZER", 0))
        if fert > 0:
            fertilizables = sum(1 for row in farm["tiles"] for t in row
                                if isinstance(t, dict) and t.get("kind") == "PLANT"
                                and t.get("fertilized_until_day", -1) < obs["day"])
            out.append((OPS_IX["PICKUP_FERTILIZER"],
                        ["PICKUP", "FERTILIZER", max(1, min(fert, fertilizables, int(FERT_PER_TRIP)))]))
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
        if True:
            # UNA OPCION POR CULTIVO con semilla disponible y que dé tiempo a
            # mature. Legality and viability still come from the engine;
            # la PREFERENCIA pasa a la red.
            _sem = obs["private"].get("seeds", {})
            for _c in spec.CROP_LIST:
                if int(_sem.get(_c, 0)) <= 0:
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
    _pendientes = []          # casillas invisibles, se resuelven al final
    free = free_capacity
    ctx = contexto_turno(obs, farm)
    _invs_turno = (obs["private"].get("inventories") or []
                   )
    _invs_turno = [iv for iv in _invs_turno if isinstance(iv, dict)] or [{}]
    for y in range(BOARD):
        for x in range(BOARD):
            if not unlocked(farm, x, y):
                continue
            if verb_map is not None:
                # END TO END: LEGALITY from the engine, the VERB from the
                # network. NOTE: this is only entered WITH a verb map.
                # al proceso, asi que un agente SIN red -el rival de una liga,
                # un publico- caia aqui y elegia `opciones[0]`, la primera
                # legal, o sea al azar. Medido: el experto de 8 dias hacia
                # 2.340 $ de rival cuando vale 4.861, con 1 unidad en vez de 9,
                # and we won 1.000 against an opponent lobotomised by OUR own
                # training setup. With no map the heuristic is used.
                # Nada de `tile_task` entra aqui -ni su orden de preferencia ni
                # sus formulas de valor-.
                import math
                options = tile_options(obs, farm, x, y, free, ctx, macro)
                # `_puede` ES LEGALIDAD, no una preocupacion de asignacion: el
                # the engine IGNORES the action if the unit is not carrying
                # what it consumes.
                # Con la heuristica daba igual -ofrecia otra cosa-, pero al
                # committing to ONE verb per tile decides. Measured without
                # filtro: la red elegia PLACE en el 95 % de las casillas, el
                # 66.9% of the tasks could be executed by nobody, 86% of
                # PASS y 3 036 $.
                # REVERTIDO tras medirlo: filtrar aqui cambia un bloqueo por
                # otro. Sin filtro, 10.7 tareas/turno y 6 727 $; con filtro,
                # 0 % de tareas inejecutables pero 3.8 tareas/turno y 4 484 $,
                # because a tile asking for FEED DISAPPEARS when nobody
                # lleva trigo y entonces nadie va a buscarlo. El `carried`
                # agregado esta en la observacion, asi que evitarlo es
                # learnable; teaching it by CLONING is not, because the expert
                # nunca esta en esa situacion.
                if FILTER_BY_INVENTORY and _invs_turno:
                    options = [o_ for o_ in options
                                if any(_can_do(iv, o_[1]) for iv in _invs_turno)]
                if not options:
                    continue
                if verb_map is not None:
                    if SAMPLE_VERB:
                        # MUESTREAR en vez de UMBRALIZAR. El argmax hace que
                        # moving a logit changes NOTHING until it crosses another
                        # verbo: derivada cero en casi todo punto, por
                        # construccion. Medido en 12h x 5d: la subida local en
                        # el espacio de verbos acepta 0,1 mejoras en 250
                        # evaluaciones -es plano- y acaba POR DEBAJO de la
                        # inaccion. Con softmax, subir un logit baja a los demas
                        # SIEMPRE, asi que el retorno esperado si depende de
                        # all of them and there is gradient to follow.
                        _lg = np.array([float(verb_map[o[0]][y][x])
                                        for o in options], dtype=np.float64)
                        _lg -= _lg.max()
                        _pr = np.exp(_lg / max(1e-6, VERB_TEMP))
                        _pr /= _pr.sum()
                        _sel = int(_VERB_RNG.choice(len(options), p=_pr))
                        k, op = options[_sel]
                        if GRAD_ACC is not None:
                            # d/dlogit_j de log p(elegido) = [j==elegido] - p_j,
                            # only over this tile's LEGAL options. Summed over
                            # every decision of the episode it gives the
                            # score-function gradient REINFORCE uses.
                            for _i, (_k, _) in enumerate(options):
                                GRAD_ACC[_k] += (1.0 if _i == _sel else 0.0) - _pr[_i]
                    else:
                        k, op = max(options,
                                    key=lambda o: float(verb_map[o[0]][y][x]))
                    # Solo el APRENDIZ trae mapa_ops; el rival no, asi que esto
                    # tells who accumulates without passing flags down the pipe.
                    if MASK_ACC is not None:
                        MASK_ACC[0, y, x] = 1.0        # el valor decidio aqui
                        if len(options) > 1:
                            for _k, _ in options:
                                MASK_ACC[1 + _k, y, x] = 1.0   # hubo comparacion
                else:
                    k, op = options[0]
                r = float(value_map[y][x]) if value_map is not None else 0.0
                v = math.copysign(
                    math.expm1(abs(min(MAP_CAP, MAP_GAIN * r))), r)
                tasks[(x, y)] = (v, op)
                if op[0] == "PLANT":
                    free -= 1
                continue
            t = tile_task(obs, farm, x, y, free, ctx, macro)
            if t is None and verb_map is not None and value_map is not None:
                # SEGUNDA PASADA, y el orden importa. Resolviendolo aqui, una
                # extra tile with PLANT consumed the planting budget and
                # strangled the plantings the heuristic would have proposed:
                # medido, 8,62 tareas/turno caian a 5,78 -el mecanismo QUITABA
                # en vez de anadir-. Apuntandolas y resolviendolas al final,
                # the heuristic decides first at full capacity and the extra
                # tiles fill whatever is left. That makes it purely additive
                # and the learned threshold the only dial.
                _pendientes.append((x, y))
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
    # `f_extra_threshold`. No hay orden de preferencia ni valor escritos a mano.
    #
    # It comes after the main sweep on purpose: resolving them inline, an
    # extra tile with PLANT consumed the budget and strangled the
    # plantar de la heuristica -medido, 8,62 tareas/turno caian a 5,78-. Aqui
    # the heuristic has already decided at full capacity and this only fills.
    for _x, _y in _pendientes:
        _ex = tile_options(obs, farm, _x, _y, free, ctx, macro)
        if not _ex:
            continue
        import math
        _lg = np.array([float(verb_map[o[0]][_y][_x]) for o in _ex],
                       dtype=np.float64)
        _lg -= _lg.max()
        _pr = np.exp(_lg / max(1e-6, VERB_TEMP))
        _pr /= _pr.sum()
        if SAMPLE_VERB:
            _sel = int(_VERB_RNG.choice(len(_ex), p=_pr))
            if GRAD_ACC is not None:
                # score-function, as in the main sweep: without sampling, argmax
                # deja derivada cero en casi todo punto.
                for _i, (_k2, _) in enumerate(_ex):
                    GRAD_ACC[_k2] += (1.0 if _i == _sel else 0.0) - _pr[_i]
        else:
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
    req = REQUIERE.get(op[0])
    if req is not None:
        return int(inv.get(req, 0)) > 0
    if op[0] == "PLACE" and len(op) > 1:
        return int(inv.get(op[1], 0)) > 0
    return True


# NOT IMPLEMENTED, written down so it is not rediscovered. There used to be a
# constant STICKINESS = 0.0 here -a bonus for keeping the previous turn's
# destination- declared and never read: tuning it did absolutely nothing. The
# measurement that motivated it still stands and is still interesting: 23.8%
# de las decisiones de destino son un cambio estando YA EN RUTA, porque el
# the Hungarian reassigns from scratch every turn. A change is not
# desperdicio -puede aparecer algo urgente-, asi que si algun dia se implementa,
# the weight must be LEARNED like the others, not set by hand.

# KEPT FOR COMPATIBILITY; it no longer branches anything. Three modes existed
# -"residuo", "directo", "ops"- so they could be measured against each other.
# They are measured: `ops` -the network emits value AND verb- is the only one
# with full learning freedom and the only one used. Historical scripts still
# assign this, so the name survives without changing behaviour.
MICRO_MODE = "ops"


def _assign_hungarian(units, tasks, invs=None, previous=None, adherencia=0.0) -> list:
    """Minimum-cost assignment: maximises the TOTAL discounted value.

    El greedy falla de una forma concreta y frecuente: la primera unidad se
    takes a task that was equally close to another unit, and leaves that other
    one with nothing nearby. Since the order of units means nothing, that loss
    is pure arbitrariness.

    Las `n` columnas ficticias de valor 0 hacen dos cosas a la vez: garantizan
    that there are at least as many columns as rows, and they ensure no unit
    accepts a negative-value task -a free dummy always remains, which
    es mejor-. `best_crop` puede devolver un cultivo de beneficio negativo
    when capital dominates, so the case does happen.
    """
    tiles = list(tasks)
    n = len(units)
    m = len(tiles) + n
    value = [[0.0] * m for _ in range(n)]
    invs = invs or [{}] * len(units)
    for i, pos in enumerate(units):
        row = value[i]
        inv = invs[i] if i < len(invs) and isinstance(invs[i], dict) else {}
        for j, tile in enumerate(tiles):
            v, op = tasks[tile]
            if not _can_do(inv, op):
                continue                  # stays 0: loses to the dummy
            row[j] = v * (STEP_DISCOUNT ** dist(pos, tile))
            if adherencia and previous is not None and previous.get(i) == tile:
                row[j] *= (1.0 + adherencia)

    actions = []
    for i, j in enumerate(max_assignment(value)):
        if j >= len(tiles) or value[i][j] <= 0.0:
            actions.append(["PASS"])
            if previous is not None:
                previous[i] = None
        else:
            actions.append(_action_for(units[i], tiles[j], tasks))
            if previous is not None:
                previous[i] = tiles[j]
    return actions


def assign_units(obs, free_capacity: int, method=None,
                 value_map=None, previous=None, macro=None, verb_map=None) -> list:
    """One action per unit, maximising the discounted value of the set.

    The budget is not the problem it was feared to be: 16 units x 116 columns
    solve exactly in 0.23 ms, against the ~83 ms average allowed by the
    bolsa de 60 s para 720 turnos (ver `tests/test_assign.py`).
    """
    # ONE METHOD ONLY. The greedy and the route assigner were kept to be
    # measured against; they are measured and the Hungarian wins. Three
    # unused paths are just surface for a bug to hide in.
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
    return _assign_hungarian(units, tasks, invs, previous, adh)
