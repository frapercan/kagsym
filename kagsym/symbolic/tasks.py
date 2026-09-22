"""Gestion de la granja: valorar tareas en dolares y asignar unidades.

El principio es el mismo que rige el resto del proyecto: **nada adivinado**.
Cada prioridad sale de lo que esa accion vale de verdad segun el motor, no de
un numero elegido a ojo.

  - Regar dentro de la ventana de bonus vale **+1 unidad** de ese cultivo, asi
    que su valor es el precio de mercado de una unidad.
  - Regar una planta con `consecutive_unwatered >= 1` salva la planta entera:
    vale todo su rendimiento futuro, porque manana se convierte en hierba.
  - Cosechar realiza el rendimiento acumulado y libera la casilla.
  - Plantar inicia un ciclo cuyo valor es el beneficio neto del cultivo.

Como las unidades se mueven una casilla por turno, el valor se descuenta por
distancia: una tarea de 40 $ a cuatro pasos rinde menos que una de 20 $ al lado.
"""
from __future__ import annotations

from .. import spec

# spec.TURNS_PER_DAY se lee en tiempo de llamada (ver spec.set_turns_per_day):
# como alias de modulo se congelaba al importar y no seguia a
# `turnsPerDay`, desincronizando el ejecutor del motor sin avisar.
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
    """Unidades que da un ciclo completo, segun las mecanicas exactas del motor."""
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
    """Factor por el que multiplica el capital cada dia, reinvirtiendo.

    Un ciclo convierte `seed` dolares en `unidades x precio` dolares en
    `cycle_days` dias, asi que el factor diario es (retorno/coste)^(1/dias).
    Con trigo: 10 -> 100 en 4 dias = x1.78 al dia. Con melon: 80 -> 1500 en 12
    dias = x1.28. El melon gana por casilla-dia y pierde por capital-dia, y
    cuando el capital es la restriccion la que manda es la segunda.
    """
    cost = max(1, spec.CROPS[crop]["seed"])
    ret = cycle_yield(crop) * unit_price(obs, crop)
    d = max(1, cycle_days(crop))
    return (ret / cost) ** (1.0 / d)


def capital_is_binding(obs, free_tiles: int) -> bool:
    """Falta dinero para llenar las casillas que se podrian atender?"""
    if free_tiles <= 0:
        return False
    money = float(obs["farms"][int(obs["player"])]["money"])
    barato = min(spec.CROPS[c]["seed"] for c in spec.CROP_LIST if plantable(obs, c)) \
        if any(plantable(obs, c) for c in spec.CROP_LIST) else 1
    return money < free_tiles * barato


def best_crop(obs, seeds_only: bool = False, free_tiles: int = 0) -> str | None:
    """El mejor cultivo SEGUN CUAL SEA LA RESTRICCION.

    Con casillas de sobra y poco dinero, la restriccion es el capital y gana el
    que mas rapido lo multiplica. Con dinero de sobra y pocas casillas, la
    restriccion es la tierra y gana el que mas rinde por casilla-dia.

    Confundir las dos cuesta la partida: optimizar casilla-dia desde el turno 0
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

    Se reconstruia el conjunto en cada llamada, y se llama 44 275 veces por
    partida (una por casilla y turno). Es una constante del tablero.
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
    """Que sacar del cobertizo. Sin esto la cadena del animal no se cierra:
    se compra, cae en el cobertizo y se queda ahi para siempre.

    Y el trigo importa igual: FEED consume 1 trigo DEL INVENTARIO DE LA UNIDAD,
    asi que un animal sin que nadie le lleve trigo se escapa a los dos dias.
    """
    priv = obs["private"]
    shed = priv["shed"]
    # 1) animal por colocar, si hay o puede haber sitio
    if ctx is None:
        ctx = contexto_turno(obs, farm)
    # Solo se valora el animal que de verdad esta en el cobertizo.
    hay = [a for a in spec.ANIMALS if int(shed.get(a, 0)) > 0]
    for a in sorted(hay, key=lambda a: -animal_value(ctx, a, macro)):
        if ctx.free_slots.get(spec.ANIMALS[a]["structure"], 0) > 0:
            return (max(1.0, animal_value(ctx, a, macro) / ctx.days),
                    ["PICKUP", a, 1])
    # 1b) DESCARGAR. La cosecha se queda en el inventario de la unidad hasta el
    # cierre del dia; con DROP llega al cobertizo YA y puede venderse el mismo
    # dia, ademas de evitar que el volcado nocturno desborde las 100 unidades y
    # tire el exceso. El rival la usa 417 veces por partida y nosotros 0: no
    # estaba en el repertorio.
    #
    # VALOR MARGINAL. NO vale lo que se lleva encima: el motor vuelca los
    # inventarios al cobertizo solo al cierre del dia, asi que descargar antes
    # no cambia que el genero acabe alli. Lo unico que anade es poder VENDERLO
    # HOY, antes de que el precio baje. Valorarlo bruto hacia que todas las
    # unidades corrieran al cobertizo: medido, 39 558 -> 13 250 $.
    #
    # Es el tercer sitio hoy donde confundo valor bruto con marginal (antes: el
    # precio nominal del animal y el bonus del fertilizante).
    from .market_ops import future_price, marginal_prices
    priv_inv = priv.get("inventories") or []
    best = 0.0
    for inv in priv_inv:
        v_ = 0.0
        for item, n_ in (inv or {}).items():
            if item in spec.PRODUCTS and n_:
                ahora = float(sum(marginal_prices(obs, item, int(n_))))
                luego = float(future_price(obs, item, spec.TURNS_PER_DAY)) * int(n_)
                v_ += max(0.0, ahora - luego)      # solo la caida que se evita
        best = max(best, v_)
    if best > 0:
        return (best, ["DROP"])

    # 2) fertilizante, si hay plantas que fertilizar y esta en el cobertizo
    fert = int(shed.get("FERTILIZER", 0))
    if fert > 0:
        fertilizables = sum(1 for row in farm["tiles"] for t in row
                            if isinstance(t, dict) and t.get("kind") == "PLANT"
                            and t.get("fertilized_until_day", -1) < obs["day"])
        if fertilizables > 0:
            n = min(fert, fertilizables, int(FERT_PER_TRIP))
            return (2.0 * unit_price(obs, "FERTILIZER"), ["PICKUP", "FERTILIZER", n])

    # 3) trigo para alimentar a los animales que aun no han comido
    hambrientos = sum(1 for row in farm["tiles"] for t in row
                      if isinstance(t, dict) and t.get("animal") and not t.get("fed_today"))
    if hambrientos > 0 and int(shed.get("WHEAT", 0)) > 0:
        n = min(hambrientos, int(shed["WHEAT"]))
        return (2.0 * unit_price(obs, "WHEAT"), ["PICKUP", "WHEAT", n])
    return None


# Lo que una unidad debe LLEVAR ENCIMA para que la operacion no sea un no-op.
REQUIERE = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}


def _can_drop(inv) -> bool:
    """DROP solo tiene sentido con algo encima."""
    return any(int(v) > 0 for v in (inv or {}).values())


def _animales_esperando(obs) -> list:
    """Animales comprados que aun no estan colocados (cobertizo o en mano)."""
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
    """Fraccion de plantas que YA se ha conseguido regar hoy.

    Es la probabilidad empirica de que el riego futuro ocurra, y por tanto el
    factor por el que hay que descontar el bonus del fertilizante: si solo se
    riega el 39 % de las plantas, fertilizar rinde el 39 % de lo que promete.
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
    """Hechos iguales para las 100 casillas del turno, calculados PEREZOSAMENTE.

    Dos medidas, las dos con cProfile:
      - calcularlos dentro de `tile_task`: 10547 escaneos de tablero y 15215
        revaloraciones de animales por partida -> 2172 pasos/s.
      - calcularlos todos al entrar al turno: PEOR, 1853 pasos/s, porque la
        mayoria de turnos no tiene ningun animal pendiente y se pagaba el
        escaneo igualmente.
    Perezoso captura lo mejor de ambos: cero coste cuando no hace falta, y una
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
    """Valor del animal PONDERADO por el factor aprendido de su producto.

    `ctx.valor` es `animal_net_value`, una formula escrita a mano. Decidir con
    ella QUE animal se recoge, cual se coloca y que establo se construye es la
    misma familia de decisiones cableadas que `best_crop` y `seed_orders`, y es
    donde esta el otro agujero medido: v48 saca 60.179 $ de LECHE y nosotros
    13.536. El factor por producto (EGG, MILK, WOOL) ya vive en el vector, asi
    que la preferencia pasa a la red. Neutro = 1.0: sin macro, el orden de
    siempre.

    Existe como funcion y no repetida en cada sitio porque ya paso una vez:
    se pondero UN sitio -PICKUP en `tile_options`- y los otros cuatro se
    quedaron sin ponderar toda una sesion sin que nadie lo viera.
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
        # Un animal esperando en el cobertizo no vale nada hasta que hay donde
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
        # Se prorratea: plantar hoy captura el ciclo entero, pero el valor se
        # compara con acciones que rinden ya, asi que se mide por dia.
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
            # una emergencia, no un aviso. Verificado contra el motor.
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

        # FERTILIZAR: SOLO en modo directo, donde la RED valora.
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
        # Asi que la operacion se deja DISPONIBLE para que la red le ponga
        # precio en modo directo, y fuera de la heuristica escrita a mano. Es
        # justo el caso que motiva el micro aprendido: una accion cuyo valor
        # depende del estado de forma que yo no se escribir.
        # Mando de fertilizar: 0 lo apaga, y lo decide la busqueda.
        from ..macro import peso_fertilizar
        _pf = peso_fertilizar(macro) if macro is not None else 0.0
        if _pf > 0.0:
            # FERTILIZAR, DESPUES de regar. Estaba ANTES y hacia `return`, asi que
            # una planta que necesitaba las dos cosas se fertilizaba y el riego
            # nunca llegaba a evaluarse: moria esa noche. El sintoma era plantar
            # 563 y regar 386 -mas siembras que riegos es una sentencia-.
            if (not cd["ongoing"] and tile.get("fertilized_until_day", -1) < day):
                w0 = (cd["max_yield_day"] + 1) // 2
                if w0 <= age + 1 <= cd["max_yield_day"] and tile["yield_units"] < cd["max_yield"]:
                    dias_utiles = min(FERTILIZER_HORIZON,
                                      cd["max_yield_day"] - age, days_left(obs))
                    if dias_utiles > 0:
                        # VALOR MARGINAL, no bruto. El bonus anade +1 unidad por dia
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
# Dos mas que estaban a ojo. Auditoria del 2026-09-21: exponer 12 constantes
# similares subio el techo de la celda de 939 a 1201 $ (+27,9 %), asi que
# ninguna se da por buena sin haber entrado en una busqueda.
DIG_VALUE = 0.9             # desbrozar, como fraccion del valor de plantar
# FORMA CON QUE EL MAPA DE LA RED ENTRA EN EL VALOR. `expm1` es exponencial, asi
# que GANANCIA decide si el mapa SUGIERE o IMPONE, y TOPE donde se corta. Nadie
# las busco nunca. Puede explicar por que el mapa saturaba con 4 parametros:
# quiza no era que 4 bastaran, sino que la transformacion limita su efecto.
MAP_GAIN = 1.0   # cuanto pesa lo que emite la red (los tres modos)
MAP_CAP = 20.0      # corte en `ops`/`directo`, dentro de expm1
RESIDUAL_CAP = 3.0    # corte en `residuo`: exp(3) = 20x de amplificacion o
                      # de hundimiento sobre la heuristica. Decide si la red
                      # SUGIERE o IMPONE, y es el unico camino que ejecutamos.
FERTILIZER_HORIZON = 3    # dias que se le cuentan al bono del fertilizante
FERT_PER_TRIP = 4.0            # fertilizante que se recoge de una vez. Aprendido.


# ---------------------------------------------------------------------------
# ENUMERACION LEGAL. Lo contrario de `tile_task`.
#
# `tile_task` es una cascada de `return`: mezcla LEGALIDAD (que permite el
# motor) con PREFERENCIA (que prefiero yo). El orden de los `return` y las
# formulas de valor son once constantes escritas a mano, y medido el 2026-09-20
# su optimo es NO PLANTAR: barriendo `macro.casillas` en pareado sobre 8
# semillas, plantar cuesta 41-50 k$ (8-11 sigma) porque el cultivo desplaza a
# la economia animal, que rinde 831 uds de producto frente a 316.
#
# Aqui solo va la legalidad, que es derivable del motor. El verbo lo elige la
# red; el sustantivo (que cultivo, que animal) sale de aritmetica exacta
# -`best_crop`, `ctx.valor`-, que es mecanica, no preferencia.
# UN VERBO POR CULTIVO. Antes habia un solo "PLANT" y QUE se plantaba lo
# decidia `best_crop`, una funcion escrita a mano; el comentario de arriba lo
# llamaba "mecanica, no preferencia". Pero es preferencia, y es la que decide
# el 84 % del hueco: v48 saca 71.170 $ de FRESAS que nosotros no tocamos y
# 60.179 de LECHE donde hacemos la quinta parte. Con un solo verbo la red podia
# elegir SI plantar, nunca QUE, y ninguna cabeza aguas abajo podia arreglarlo:
# la de mercado no puede vender lo que no se produce -medido, mover su factor
# de fresas no cambia un solo dolar-.
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
# La log-prob del micro es una SUMA sobre 1+N_OPS canales x 100 casillas = 1 614
# dimensiones gaussianas, pero solo influyen en la accion las de casillas con
# operaciones legales, y el logit de un verbo solo importa si HABIA con quien
# compararlo. Medido: sin enmascarar, el 99.4 % de los cocientes de importancia
# de PPO se saturan en el tope de +-10 tras UN paso de gradiente -el cociente es
# un producto sobre ~1 570 dimensiones de puro ruido-. No es una eleccion de
# diseno incluirlas: es un fallo.
MASK_ACC = None
FILTER_BY_INVENTORY = False

# MUESTREO DEL VERBO. Por defecto apagado: el ejecutor UMBRALIZA con argmax,
# que es lo correcto en evaluacion determinista. Encenderlo hace que el verbo
# se sortee de una softmax sobre los logits de las opciones LEGALES, que es lo
# que da gradiente a una cabeza categorica. `_RNG_VERBO` se siembra por episodio
# desde fuera para que las comparaciones sean pareadas.
import numpy as np
# The Hungarian is the only assignment method (see `assign_units`).
METODO_ASIGNACION = "humgaro"

# UMBRAL de las casillas EXTRA. Lo fija `macro.aplica_parametros` una vez por
# turno desde el parametro aprendido `f_extra_threshold`; este valor solo es el
# de arranque y equivale a "ninguna casilla extra entra".
#
# Que son las casillas extra. Medido el 2026-09-21 en campeonato: la heuristica
# ofrece 8,94 tareas por turno para 10,88 unidades y deja fuera otras 5,81 que
# el motor SI considera legales. Se pasa turno el 52,8 % de las veces -v48 pasa
# el 5,3 %- y 44,5 de esos puntos son por no tener tarea que asignar, no por
# elegir mal. En el 22,7 % de los turnos la heuristica no ofrece nada y aun asi
# hay jugadas legales. Verbos desaprovechados: FERTILIZE 3010, HARVEST 837,
# DROP 327, PICKUP 164 -y cosechamos 111 veces por partida frente a las 420
# de v48-.
#
# Por que la red no podia arreglarlo: en `residuo` el mapa MULTIPLICA el valor
# de las tareas existentes y en `directo` lo SUSTITUYE, pero la casilla con
# `tile_task is None` no entra en el diccionario y es invisible al aprendizaje.
#
# Lo que decide aqui: la LEGALIDAD la da el motor, el VERBO lo elige la red
# entre las opciones legales y el VALOR lo emite la red. No hay orden de
# preferencia ni valor escritos a mano: seria volver a meter una heuristica
# donde justamente se esta quitando.
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
    """Todas las operaciones LEGALES en esa casilla, sin valorarlas ni ordenarlas.

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
            # QUE animal sacar del cobertizo era otra formula a mano -la
            # cuarta de la misma familia: `best_crop`, `seed_orders`,
            # `animal_net_value` y esta-. Se pondera por el factor APRENDIDO
            # del producto que da el animal, que ya vive en el vector macro.
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
        # 2026-09-21 comparando las dos tablas de tareas sobre EL MISMO estado:
        # 96,8 % identicas, 0 % de casillas perdidas, 0 % de valores distintos,
        # y el 3,2 % restante eran exactamente esto -`PICKUP WHEAT 2` contra
        # `PICKUP WHEAT 1`-. Con una unidad trayendo un trigo por viaje, la
        # cadena de alimentacion del ganado va a la mitad de capacidad, y el
        # ganado es la economia entera a esta escala.
        #
        # La cantidad NO es una decision estrategica: la fijan cuantos animales
        # tienen hambre y cuanto hay en el cobertizo. Es del lado simbolico,
        # como la legalidad y el enrutado. La red sigue eligiendo el VERBO.
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
            # madurar. La legalidad y la viabilidad las sigue poniendo el motor;
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
    """(x,y) -> (valor en $, operacion) para cada casilla que ofrece algo."""
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
                # E2E: la LEGALIDAD la da el motor, el VERBO lo elige la red.
                # OJO: solo se entra aqui CON mapa_ops. `MODO_MICRO` es global
                # al proceso, asi que un agente SIN red -el rival de una liga,
                # un publico- caia aqui y elegia `opciones[0]`, la primera
                # legal, o sea al azar. Medido: el experto de 8 dias hacia
                # 2.340 $ de rival cuando vale 4.861, con 1 unidad en vez de 9,
                # y ganabamos 1.000 contra un rival lobotomizado por NUESTRO
                # modo de entrenamiento. Sin mapa se usa la heuristica.
                # Nada de `tile_task` entra aqui -ni su orden de preferencia ni
                # sus formulas de valor-.
                import math
                options = tile_options(obs, farm, x, y, free, ctx, macro)
                # `_puede` ES LEGALIDAD, no una preocupacion de asignacion: el
                # motor IGNORA la accion si la unidad no lleva lo que consume.
                # Con la heuristica daba igual -ofrecia otra cosa-, pero al
                # comprometer UN verbo por casilla decide. Medido sin este
                # filtro: la red elegia PLACE en el 95 % de las casillas, el
                # 66.9 % de las tareas no las podia ejecutar nadie, 86 % de
                # PASS y 3 036 $.
                # REVERTIDO tras medirlo: filtrar aqui cambia un bloqueo por
                # otro. Sin filtro, 10.7 tareas/turno y 6 727 $; con filtro,
                # 0 % de tareas inejecutables pero 3.8 tareas/turno y 4 484 $,
                # porque una casilla que pide FEED DESAPARECE cuando nadie
                # lleva trigo y entonces nadie va a buscarlo. El `carried`
                # agregado esta en la observacion, asi que evitarlo es
                # aprendible; ensenarlo por CLONACION no, porque el experto
                # nunca esta en esa situacion.
                if FILTER_BY_INVENTORY and _invs_turno:
                    options = [o_ for o_ in options
                                if any(_can_do(iv, o_[1]) for iv in _invs_turno)]
                if not options:
                    continue
                if verb_map is not None:
                    if SAMPLE_VERB:
                        # MUESTREAR en vez de UMBRALIZAR. El argmax hace que
                        # mover un logit no cambie NADA hasta que cruza a otro
                        # verbo: derivada cero en casi todo punto, por
                        # construccion. Medido en 12h x 5d: la subida local en
                        # el espacio de verbos acepta 0,1 mejoras en 250
                        # evaluaciones -es plano- y acaba POR DEBAJO de la
                        # inaccion. Con softmax, subir un logit baja a los demas
                        # SIEMPRE, asi que el retorno esperado si depende de
                        # todos ellos y hay gradiente que seguir.
                        _lg = np.array([float(verb_map[o[0]][y][x])
                                        for o in options], dtype=np.float64)
                        _lg -= _lg.max()
                        _pr = np.exp(_lg / max(1e-6, VERB_TEMP))
                        _pr /= _pr.sum()
                        _sel = int(_VERB_RNG.choice(len(options), p=_pr))
                        k, op = options[_sel]
                        if GRAD_ACC is not None:
                            # d/dlogit_j de log p(elegido) = [j==elegido] - p_j,
                            # solo sobre las opciones LEGALES de esta casilla.
                            # Sumado sobre todas las decisiones del episodio da
                            # el gradiente de score-function que usa REINFORCE.
                            for _i, (_k, _) in enumerate(options):
                                GRAD_ACC[_k] += (1.0 if _i == _sel else 0.0) - _pr[_i]
                    else:
                        k, op = max(options,
                                    key=lambda o: float(verb_map[o[0]][y][x]))
                    # Solo el APRENDIZ trae mapa_ops; el rival no, asi que esto
                    # distingue quien acumula sin pasar banderas por la tuberia.
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
                # casilla extra con PLANT consumia `libre` y estrangulaba las
                # tareas de plantar que la heuristica habria propuesto despues:
                # medido, 8,62 tareas/turno caian a 5,78 -el mecanismo QUITABA
                # en vez de anadir-. Apuntandolas y resolviendolas al final,
                # la heuristica decide primero con toda su capacidad y lo extra
                # rellena con lo que sobre. Asi es puramente aditivo y el
                # umbral aprendido es el unico mando.
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
    # SEGUNDA PASADA: las casillas que la heuristica declaro vacias y el motor
    # considera legales. La LEGALIDAD la da el motor, el VERBO lo elige la red
    # entre las opciones legales y el VALOR lo emite la red; el UMBRAL que
    # decide si merece ocupar una unidad es el parametro aprendido
    # `f_extra_threshold`. No hay orden de preferencia ni valor escritos a mano.
    #
    # Va despues del barrido principal a proposito: resolviendolas en linea,
    # una casilla extra con PLANT consumia `libre` y estrangulaba las tareas de
    # plantar de la heuristica -medido, 8,62 tareas/turno caian a 5,78-. Aqui
    # la heuristica ya decidio con toda su capacidad y esto solo rellena.
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
                # score-function, igual que en `ops`: sin muestrear, el argmax
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
    """Ejecutar la tarea si ya se esta encima; si no, dar un paso hacia ella."""
    if tile == pos:
        return tasks[tile][1]
    return [step_toward(pos, tile)]


def _can_do(inv, op) -> bool:
    """Si la unidad NO lleva lo que la operacion consume, el motor la ignora.

    FEED gasta 1 trigo del inventario DE LA UNIDAD, FERTILIZE 1 fertilizante, y
    PLACE necesita el animal encima. Asignar esas tareas a quien no lleva nada
    es regalar el turno: no-op silencioso. Se anula el valor del par para que la
    asignacion prefiera cualquier otra cosa, incluida la columna ficticia.
    """
    if op[0] == "DROP":
        return _can_drop(inv)
    req = REQUIERE.get(op[0])
    if req is not None:
        return int(inv.get(req, 0)) > 0
    if op[0] == "PLACE" and len(op) > 1:
        return int(inv.get(op[1], 0)) > 0
    return True


# SIN IMPLEMENTAR, y se deja escrito para que no se vuelva a descubrir. Habia
# aqui una constante ADHERENCIA = 0.0 -bonus por conservar el destino del turno
# anterior- declarada y jamas leida: ajustarla no hacia absolutamente nada.
# La medida que la motivaba sigue en pie y sigue siendo interesante: el 23,8 %
# de las decisiones de destino son un cambio estando YA EN RUTA, porque el
# hungaro reasigna desde cero cada turno. Un cambio no es automaticamente
# desperdicio -puede aparecer algo urgente-, asi que si algun dia se implementa,
# el peso tiene que APRENDERSE como los otros 18, no fijarse a ojo.

# KEPT FOR COMPATIBILITY; it no longer branches anything. Three modes existed
# -"residuo", "directo", "ops"- so they could be measured against each other.
# They are measured: `ops` -the network emits value AND verb- is the only one
# with full learning freedom and the only one used. Historical scripts still
# assign this, so the name survives without changing behaviour.
MICRO_MODE = "ops"


def _assign_hungarian(units, tasks, invs=None, previous=None, adherencia=0.0) -> list:
    """Asignacion de coste minimo: maximiza el valor descontado TOTAL.

    El greedy falla de una forma concreta y frecuente: la primera unidad se
    lleva una tarea que estaba igual de cerca de otra, y deja a esa otra sin
    nada que hacer cerca. Como el orden de las unidades no significa nada, esa
    perdida es pura arbitrariedad.

    Las `n` columnas ficticias de valor 0 hacen dos cosas a la vez: garantizan
    que haya al menos tantas columnas como filas, y aseguran que ninguna unidad
    acepte una tarea de valor negativo -siempre le queda una ficticia libre, que
    es mejor-. `best_crop` puede devolver un cultivo de beneficio negativo
    cuando manda el capital, asi que el caso ocurre de verdad.
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
                continue                  # queda en 0: pierde contra la ficticia
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
    """Una accion por unidad, maximizando el valor descontado del conjunto.

    El presupuesto no es el problema que se temia: 16 unidades x 116 columnas
    se resuelven exacto en 0.23 ms, frente a los ~83 ms de media que da la
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
