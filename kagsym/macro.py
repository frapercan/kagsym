"""El vector de decision macro: lo unico que la politica decide.

Por que existe este fichero. La capa guionizada acumulo once constantes puestas
a ojo -cuantos animales por unidad de capacidad, que fraccion de la caja en mano
de obra, cuanto margen exigir a un peon, a partir de que saturacion expandir...-.
Eso viola el principio del proyecto ("nada adivinado") y, peor, ocupa justo el
sitio donde deberia decidir el aprendizaje: son decisiones de ESTRATEGIA, no de
mecanica.

La mecanica -rutas, asignacion optima, legalidad, regar el dia que se planta,
reserva de comida, liquidacion final- es derivable del motor y se queda en el
ejecutor. Lo que no es derivable es CUANTO de cada cosa, y sobre todo cuando
volcar al mercado compartido, que depende del rival. Eso es este vector.

Todas las componentes viven en [0,1] para que una politica gaussiana recortada
pueda emitirlas sin escalas inventadas. La traduccion a cantidades exactas la
hace `resolver`, que es donde estan los limites reales del motor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

from . import spec

N_LEVELS = 8        # cuanto de cada cosa
CATEGORIES = ["land", "feed", "animal", "sell", "seed", "hand"]
N_PRIORITIES = len(CATEGORIES)
N_EXPOSED = 30       # las que estaban a ojo; ver el bloque en `Macro`
N_MARKET = 9        # un multiplicador de valor por producto, aprendido
N_TURN = 5          # coeficientes de la regla de venta POR TURNO
N_MACRO = N_LEVELS + N_PRIORITIES + N_EXPOSED + N_MARKET + N_TURN
# El coste del peon 16 del dia es 987 $ el solo -medido-, pero eso es una razon
# ECONOMICA que la busqueda puede descubrir sola, no un limite del motor: el
# motor no topa las contrataciones. Ya no hay constante; ver `peones_objetivo`.
# El horizonte de reventa ya no tiene techo; su defecto es un dia
# (`spec.TURNS_PER_DAY`, del motor). Ver `horizonte_venta`.


@dataclass
class Macro:
    """Objetivos, no ordenes. El ejecutor decide como alcanzarlos."""

    tiles: float = 0.3333   # -> escala sobre el techo de riego, SIN borde
                               #    (exp(logit(1/3)) = 0.5, el valor de antes)
    animals: float = 0.5      # -> fraccion de la capacidad de atencion dedicada a ganado
    hands: float = 0.35       # -> peones objetivo por dia
    venta: float = 0.25        # -> agresividad: 0 vender ya, 1 acumular al maximo
    crop: float = 0.0       # -> sesgo hacia cultivo caro (1) o barato y rapido (0)
    expandir: float = 0.25     # -> saturacion exigida antes de comprar tierra
    adherencia: float = 0.25   # -> cuanto se premia conservar el destino de ayer
    fertilizar: float = 0.0    # -> cuanto vale fertilizar frente a las demas tareas
    # PRIORIDADES. Los siete de arriba dicen CUANTO de cada cosa; estos dicen
    # QUE SE SACRIFICA cuando no llega para todo, que es el 31 % de los turnos:
    # medido, la falta de caja impide comprar tierra en el 31 %, pienso en el
    # 19 % y animales en el 15 %. Hoy eso lo deciden cuatro constantes mias
    # (reserva de 300, 25 % de la caja, coste x1.5, 15 % en mano de obra) y el
    # orden de una lista que escribi a mano.
    #
    # Pasan por softmax: lo que importa es el reparto relativo, no la escala.
    p_land: float = 0.5
    p_feed: float = 0.5
    p_animal: float = 0.5
    p_sell: float = 0.5
    p_seed: float = 0.5
    p_hand: float = 0.5
    # ------------------------------------------------------------------
    # LO QUE ESTABA A OJO. Auditoria del 2026-09-21: catorce constantes
    # repartidas por mercado.py, tareas.py, ejecutor.py y este fichero
    # nunca habian entrado en ninguna busqueda. Exponerlas subio el techo
    # de la celda 12h x 5d de 939 a 1.258 $ (+34 %), mas que todo lo que
    # se probo ese dia del lado del aprendizaje junto.
    #
    # Viven aqui y no como globales porque el valor correcto DEPENDE DEL
    # ESTADO -cuanta caja queda, cuantos dias, que rival- y eso es la
    # definicion de lo que el docstring de arriba manda aprender.
    #
    # Todas en [0,1]; `parametros()` las lleva a su rango real.
    f_labour: float = 0.137      # 0.15 / 1.09   -> reproduce el valor viejo
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
    # INTERFAZ entre lo aprendido y lo simbolico: `expm1` es exponencial, asi
    # que la ganancia decide si el mapa de la red SUGIERE o IMPONE, y el tope
    # donde se corta. Nadie las busco nunca.
    f_map_gain: float = 0.231  # 1.0
    f_map_cap: float = 0.655      # 20.0
    f_max_animal: float = 0.111     # 2  (el limite del motor es 10, no 2)
    f_residual_cap: float = 0.2784   # 3.0 -> exp(3) = 20x sobre la heuristica
    # UMBRAL de las casillas que la heuristica declara vacias y el motor
    # considera legales -5,81 por turno frente a 8,94 ofrecidas-. Es lo que
    # esa casilla tiene que valer, en dolares emitidos por la red, para
    # merecer ocupar una unidad. Aprendido como los otros 18: no hay forma de
    # derivarlo del motor, asi que no puede ir a ojo.
    # Por defecto 0.5 -> 2.500 $, muy por encima de cualquier tarea real, o
    # sea que al arrancar NINGUNA casilla extra entra y la conducta es
    # identica a la anterior. El entrenamiento lo baja si compensa.
    f_extra_threshold: float = 0.5
    # CABEZA DE MERCADO: un multiplicador por producto sobre su valor de venta.
    #
    # Es la ULTIMA capa de decision que no se aprendia. El tablero ya lo decide
    # la red casilla a casilla, pero que vender y cuanto aguantar lo decidia
    # `mercado.py` a partir de unos diez escalares. Y ahi esta el hueco medido:
    # vendemos 59.872 $ contra los 200.760 de v48, con 71.170 en FRESAS que no
    # tocamos y 60.179 en LECHE donde hacemos la quinta parte -el 84 % de la
    # diferencia en dos productos-.
    #
    # Mismo patron que funciono en la cabeza micro: residuo MULTIPLICATIVO
    # sobre la valoracion exacta, no sustituirla. 0.5 -> factor 1.0, o sea
    # neutro, asi que el defecto reproduce la conducta anterior.
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
    # LAS QUE SEGUIAN INLINE. Auditoria del 2026-09-22: cuatro numeros
    # escritos DENTRO del cuerpo de las funciones, no como constantes de
    # modulo, asi que la auditoria anterior -la que expuso catorce y dio
    # +34 %- no los vio.
    #
    # MEDIDOS VIVOS en ops, 4 semillas, 24h x 30d, 720 turnos:
    #     credito de estiercol  0.5 -> 0.0    -22.5 %
    #     caja para pienso     0.25 -> 0.02   -21.6 %
    # o sea que cada uno decide mas que casi cualquier otra cosa del vector.
    # `suelo_obra` midio 0.0 % a esta caja, pero es un SUELO EN DOLARES: con
    # otra caja inicial muerde, y el modelo tiene que servir en otra liga.
    f_manure_credit: float = 0.5   # 0.50  credito de estiercol al valorar un animal
    f_feed_cash: float = 0.5   # 0.25  fraccion de caja que puede irse en pienso
    f_labour_floor: float = 0.5    # 60.0  suelo del presupuesto de mano de obra
    f_liquidation: float = 0.5      # 2.0   dias de liquidacion al cerrar la temporada
    # ------------------------------------------------------------------
    # REGLA DE VENTA POR TURNO. Hasta aqui, todo lo aprendido del mercado era
    # un NIVEL por producto, constante durante las 24 horas: la red se llama
    # una vez al dia -un paso de RL es un DIA, ver el docstring de
    # `entorno.py`- y la capa simbolica juega el dia entero sola.
    #
    # La regla de venta SI mira el estado de cada turno, pero por una formula
    # que escribi yo: vender mientras el precio de ahora bata al previsto. Lo
    # que la politica no podia mover es CUANTO reaccionar a cada senal.
    #
    # Estos cinco son esos coeficientes. La pasada diaria emite la REGLA y la
    # regla se evalua con el estado de CADA TURNO. Da decision por situacion
    # sin multiplicar por 24 las pasadas de red, que es lo que pide una
    # competicion con un segundo por turno.
    #
    # w = logit(f), asi que 0.5 -> w = 0 -> exp(0) = 1: al arrancar esto es
    # EXACTAMENTE la conducta anterior. Sin constante de escala, por el mismo
    # motivo que en `parametros()`: sigmoide y logit se cancelan.
    w_price: float = 0.5      # reaccion a la caida de precio prevista
    w_rival: float = 0.5       # reaccion a lo que el rival esta a punto de verter
    w_shed: float = 0.5   # reaccion a la presion de almacen
    w_season: float = 0.5    # reaccion al avance de la temporada
    w_cash: float = 0.5        # reaccion a cuanta riqueza esta inmovilizada
    # ------------------------------------------------------------------
    # TERCERA HORNADA, 2026-09-22. Auditoria ESTRUCTURAL: las dos anteriores
    # buscaban constantes de modulo y literales; esta recorre cada punto de
    # decision del arbol (`runs/ligas/auditoria_cascada.py`) y clasifica. De
    # 156 sitios, estos ocho eran decisiones de politica que nadie podia mover.
    #
    # El que mas duele es `stock_semilla`: `seed_orders` ABORTA entera si ya
    # tienes 2 semillas por unidad. Medido, llevamos 25,1 semillas de media con
    # 8,4 unidades -umbral 16,8-, o sea que buena parte de la partida NO SE
    # COMPRA SEMILLA. Encaja con el sintoma que perseguiamos: 15,4 plantas
    # vivas contra las 44 de v48 con la misma siembra.
    f_seed_stock: float = 0.5   # 2.0   semillas por unidad antes de dejar de comprar
    f_animal_reserve: float = 0.5  # 300.0 caja a reservar antes de comprar un animal
    f_actions_per_animal: float = 0.5 # 3.0   acciones/dia que cuesta sostener un animal
    f_last_hire_hour: float = 0.5   # 3.0   ultima hora del dia en que se contrata
    f_min_hand_days: float = 0.5       # 2.0   dias minimos para amortizar un peon
    f_cost_rise: float = 0.5      # 0.5   cuanto sube el coste/casilla al secarse
    f_cost_decay: float = 0.5      # 0.93  cuanto baja cuando no se seca nada
    f_fert_per_trip: float = 0.5      # 4.0   fertilizante que se coge de una vez

    @staticmethod
    def default() -> "Macro":
        """Los valores que reproducen la capa guionizada tal y como esta medida."""
        return Macro()

    @staticmethod
    def from_vector(v) -> "Macro":
        vals = [min(1.0, max(0.0, float(x))) for x in v]
        return Macro(*vals[:N_MACRO])

    def to_vector(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


# TOPE DE CURRICULO. None = sin tope. Limita cuantos peones puede planificar
# NUESTRO agente, igual que `publico_con_tope` limita al rival. Sirve para
# entrenar en un juego mas pequeno y coherente: menos unidades = ejecutor mas
# barato (el ejecutor es el 94 % del coste) y espacio de accion menor, sin
# truncar la partida -que invierte el signo de la ventaja, medido-.
HAND_CAP = None
# FACTOR DE RIEGO. "la mitad del presupuesto se va en moverse: medido 42.3% en
# el experto" -pero 0.5 es una constante a ojo y nunca ha entrado en ninguna
# busqueda, igual que las tres del mercado que al exponerlas dieron +12,8 %.
WATERING_FACTOR = 0.5
# Acciones al dia que cuesta sostener un animal. Aprendido (`f_actions_per_animal`).
ACTIONS_PER_ANIMAL = 3.0


# (nombre, minimo, rango). Los rangos salen de la busqueda que encontro 1.258 $.
# PARAMETRIZACION SIN BORDES.
#
# Antes era `valor = lo + rango * f`: dos constantes a ojo por parametro -36
# numeros- y, peor, un suelo y un techo DUROS. Si otra liga u otro rival pedian
# un valor fuera, el modelo no podia ni expresarlo. Medido: `valor_dig` se
# pegaba al suelo en 5 de 8 vectores aprendidos.
#
# Ahora no hay bordes. Como la red emite `macro_mu` y en todas partes se hace
# f = sigmoid(macro_mu), resulta que logit(f) == macro_mu EXACTAMENTE, asi que:
#
#     positivo   valor = defecto * exp(logit(f))           -> (0, +inf)
#     fraccion   valor = sigmoid(logit(defecto) + logit(f)) -> (0, 1)
#
# f = 0.5 devuelve el defecto exacto y los extremos alcanzan todo el dominio.
# No queda ninguna constante de escala: el factor que multiplicaria a logit(f)
# es 1 porque sigmoide y logit se cancelan, no porque nadie lo elija.
#
# El DOMINIO no es gusto: una fraccion de presupuesto vive en [0,1] por lo que
# significa. Y donde el limite lo pone el MOTOR se usa el del motor -ver
# `max_animal` en `aplica_parametros`-.
#
# QUE NO ESTA AQUI: `horiz_fert` se fue. Era un HECHO DEL MOTOR disfrazado de
# parametro -el fertilizante dura 3 dias exactos- y dejarlo aprender permitia
# que la politica creyera que dura siete y valorase mal cada fertilizacion.
# Vive en `spec.FERTILIZER_DAYS`, con los demas hechos del motor.
#
# `umbral_extra` no tiene original porque es nuevo: su defecto es el p95 MEDIDO
# de lo que valen las tareas que propone la heuristica (424 $ sobre 6.431).
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
    ("residual_cap",   3.00,  "positive"),
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
]

# Coeficientes de la regla de venta por turno. NO estan en RANGOS_F porque su
# forma es otra: son ADITIVOS en el exponente, no multiplicadores de un
# defecto. El defecto de un coeficiente aditivo es 0, y `defecto * exp(z)` se
# degenera en 0 para siempre. La forma natural es w = logit(f) directamente,
# que cumple lo mismo: f = 0.5 reproduce la conducta anterior y no hay bordes.
TURN_WEIGHTS = ("price", "rival", "shed", "season", "cash")


def turn_weights(macro) -> dict:
    """Cuanto reacciona la venta a cada senal del turno. Defecto 0 = regla de hoy."""
    if macro is None:
        return {k: 0.0 for k in TURN_WEIGHTS}
    # NO default on getattr. It used to be `getattr(macro, "w_"+k, 0.5)`, and
    # when a field was renamed and this tuple was not, the lookup silently fell
    # back to the neutral value and the whole per-turn rule switched itself off
    # without a single error. An AttributeError here is the correct outcome.
    return {k: _logit(getattr(macro, "w_" + k)) for k in TURN_WEIGHTS}


def market_factors(macro: Macro) -> dict:
    """Multiplicador aprendido por producto. 0.5 -> 1.0 (neutro), sin bordes."""
    from . import spec as _sp
    out = {}
    for pr in _sp.PRODUCTS:
        # no default: a renamed field must raise, not silently
        # fall back to neutral (see `turn_weights`)
        f = getattr(macro, "m_" + pr.lower())
        out[pr] = math.exp(_logit(f))
    return out


def _logit(f: float) -> float:
    """logit con guarda numerica. El 1e-6 no es modelado: es el epsilon que
    evita el infinito. Deja un alcance de exp(+-13.8) = 1e6 veces el defecto,
    que a efectos practicos es no tener borde."""
    f = min(1.0 - 1e-6, max(1e-6, float(f)))
    return math.log(f / (1.0 - f))


def params(macro: Macro) -> dict:
    """Los que estaban a ojo, ya en sus unidades reales y SIN borde."""
    out = {}
    for n, default, kind in PARAM_TABLE:
        z = _logit(getattr(macro, "f_" + n))
        if kind == "fraction":
            out[n] = 1.0 / (1.0 + math.exp(-(_logit(default) + z)))
        else:
            out[n] = default * math.exp(z)
    return out


def apply_params(macro: Macro) -> None:
    """Escribe los parametros donde los leen las capas. Un solo sitio.

    Se hace asi -y no pasando el macro por seis firmas- porque `tareas.py` y
    `ejecutor.py` los consultan desde funciones que no reciben el macro, y
    cambiar sus firmas tocaria mucho mas codigo del que este cambio justifica.
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
    # El TECHO lo pone el motor, no nosotros: `maxMarketOrdersPerTurn`. Cuantos
    # animales comprar por turno es politica; cuantos CABEN es mecanica.
    _M.MAX_ANIMALS_PER_TURN      = max(1, min(int(spec.DEFAULT_CONFIG["maxMarketOrdersPerTurn"]),
                                          int(round(p["max_animal"]))))
    # Hecho del motor, no parametro: el fertilizante dura 3 dias exactos.
    _T.FERTILIZER_HORIZON  = int(spec.FERTILIZER_DAYS)
    _T.RESIDUAL_CAP          = p["residual_cap"]
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
    global ACTIONS_PER_ANIMAL
    ACTIONS_PER_ANIMAL = p["actions_per_animal"]
    _T.EXTRA_TILE_THRESHOLD          = p["extra_threshold"]
    global WATERING_FACTOR
    WATERING_FACTOR = p["watering"]


def target_hands(obs, macro: Macro) -> int:
    # SIN TECHO, y el motor respalda que no lo haya: no limita las
    # contrataciones -`_hire_cost(hires_today)` solo las ENCARECE-, asi que el
    # 15 de antes era un muro puesto por mi sobre un nivel aprendible. Mismo
    # mecanismo que los parametros: 0.5 da el valor de siempre (7,5 -> 8) y los
    # extremos alcanzan cualquier plantilla, que es lo que hace falta para que
    # el modelo sirva en ligas con otra caja y otro horizonte.
    n = max(0, int(round(7.5 * math.exp(_logit(macro.hands)))))
    return n if HAND_CAP is None else min(n, HAND_CAP)


def target_tiles(obs, macro: Macro) -> int:
    """Cuantas casillas plantadas mantener. Tope exacto: las que se pueden regar.

    Una planta sin regar dos dias se convierte en hierba, asi que el techo real
    no es la tierra sino las acciones: cada casilla cuesta un riego al dia.
    """
    me = int(obs["player"])
    farm = obs["farms"][me]
    # PLANIFICADAS, no las que hay ahora mismo. El macro se decide en la hora 0,
    # cuando los peones de ayer ya se limpiaron (`farm["hands"] = []` cada noche)
    # y los de hoy aun no se han contratado: `len(farm["hands"])` vale SIEMPRE 0
    # ahi. Con eso el techo quedaba clavado en 12 casillas por muchos peones que
    # se contrataran despues, y la granja no podia crecer. El CEM no eligio una
    # granja de 7 casillas y 1.9 unidades: era la unica alcanzable.
    n_units = 1 + target_hands(obs, macro)
    cultivables = sum(1 for y in range(spec.BOARD) for x in range(spec.BOARD)
                      if farm["tiles"][y][x] != "LOCKED")
    # la mitad del presupuesto se va en moverse: medido 42.3% en el experto
    # SIN BORDE, como en `peones_objetivo`. Antes era
    #     macro.casillas * min(cultivables, techo_riego)
    # y ahi hay DOS techos de naturaleza distinta metidos en el mismo `min`:
    #
    #   * `cultivables` es un limite DURO DEL MOTOR: en una casilla LOCKED no
    #     se puede plantar, y punto. Se queda como limite.
    #   * `techo_riego` NO lo es: es una ESTIMACION mia de cuantas casillas se
    #     pueden atender, y entraba como tope infranqueable. La politica no
    #     podia pedir mas aunque le conviniera -por ejemplo con cultivos que no
    #     piden riego todos los dias, o con mas unidades de las planificadas-.
    #     Mismo defecto que el tope de 15 peones que se quito de aqui al lado.
    #
    # Ahora `techo_riego` es la ESCALA y la politica la multiplica sin borde:
    # f = 1/3 reproduce el valor de antes (exp(logit(1/3)) = 0.5), f = 0.5 pide
    # el techo de riego entero, y los extremos alcanzan cualquier granja que el
    # motor permita. El unico recorte que queda es el del motor.
    techo_riego = n_units * spec.TURNS_PER_DAY * WATERING_FACTOR
    deseadas = techo_riego * math.exp(_logit(macro.tiles))
    return max(0, int(min(cultivables, deseadas)))


def target_animals(obs, macro: Macro) -> int:
    """Cuantos animales sostener. Cada uno cuesta ~3 acciones al dia."""
    me = int(obs["player"])
    farm = obs["farms"][me]
    n_units = 1 + target_hands(obs, macro)   # planificadas, ver arriba
    # DOBLE CONTEO CORREGIDO. Estaba `* 0.5` ("la mitad se va en moverse") y
    # ademas `/ ACCIONES_POR_ANIMAL = 3`, que ya incluye el desplazamiento. El
    # resultado era 4 tareas por unidad y dia, la mitad de lo real.
    #
    # El experto 2945 sostiene 58 cultivos + 17 animales = 75 tareas con 9.4
    # unidades: 8 tareas por unidad y dia, que es exactamente 24/3. Medido: con
    # el `0.5`, los animales saturaban en ~19 pasara lo que pasara -incluso con
    # 4 cuadrantes y 100 casillas libres-.
    capacity = n_units * spec.TURNS_PER_DAY
    planted = sum(1 for row in farm["tiles"] for t in row
                    if isinstance(t, dict) and t.get("kind") == "PLANT")
    room = max(0.0, capacity - planted) / max(1e-6, ACTIONS_PER_ANIMAL)
    return max(0, int(macro.animals * room))


def sell_horizon(obs, macro: Macro) -> int:
    """Turnos que se mira hacia delante antes de decidir vender.

    Es la decision acoplada al rival: aguantar producto solo compensa si el
    rival no hunde el precio antes. Es tambien donde el world model tiene senal
    medida (dinero del rival +19%, su gasto +35%).
    """
    # El defecto es UN DIA de horizonte -`spec.TURNS_PER_DAY`, del motor-, y
    # sin techo: antes el `2 *` limitaba a dos dias por decision mia, y aguantar
    # producto mas tiempo es justo la jugada que puede pagar contra un rival
    # que no hunde el precio.
    return max(1, int(round(spec.TURNS_PER_DAY
                            * math.exp(_logit(macro.venta)))))


def target_crop(obs, macro: Macro):
    """Interpola entre el cultivo mas rapido y el mas rentable por casilla-dia.

    No es una categoria libre: elegir un cultivo que no da tiempo a madurar es
    perdida segura, asi que solo entran los viables, que es exacto.
    """
    from .symbolic.tasks import cycle_days, cycle_profit
    days = spec.EPISODE_STEPS // spec.TURNS_PER_DAY - int(obs["day"])
    viable = [c for c in spec.CROP_LIST if cycle_days(c) <= days]
    if not viable:
        return None
    if macro.crop <= 0.0:
        return min(viable, key=lambda c: cycle_days(c))
    por_valor = sorted(viable, key=lambda c: cycle_profit(obs, c) / max(1, cycle_days(c)))
    i = min(len(por_valor) - 1, int(macro.crop * len(por_valor)))
    return por_valor[i]


def assignment_stickiness(macro: Macro) -> float:
    """Bonus multiplicativo por conservar el destino del turno anterior.

    El humgaro reasigna desde cero cada turno y eso es miope: medido, el 23.8%
    de las decisiones de destino son un cambio estando YA EN RUTA, y los pasos
    dados se tiran. Medido tambien el remedio, con el vector del CEM:

        0.00 -> 23 793 $     0.25 -> 27 177 $  (+14%)
        0.10 -> 26 887 $     0.50 -> 24 869 $
                             1.00 -> 22 755 $  (demasiado pegado: ignora urgencias)

    Tiene optimo interior, asi que no es un "cuanto mas mejor" y no se puede
    fijar por razonamiento. Lo decide la politica.

    Nota de diseno: esto es un termino por (UNIDAD, casilla), no por casilla.
    Un mapa de valor 10x10 no puede expresarlo -las unidades no solo difieren en
    posicion e inventario, tambien en su compromiso previo-.
    """
    return 2.0 * float(macro.adherencia)


def priorities(macro: Macro) -> dict:
    """Reparto de la caja entre categorias. Softmax sobre las 6 componentes.

    Devuelve fracciones que suman 1. La temperatura 3.0 hace que la politica
    pueda llegar a concentrar casi todo en una categoria (con una componente a 1
    y el resto a 0, esa se lleva el 73 %) sin que el reparto uniforme sea un
    punto raro: con todas iguales, cada una recibe 1/6.
    """
    import math
    vals = [getattr(macro, "p_" + c) for c in CATEGORIES]
    e = [math.exp(3.0 * v) for v in vals]
    total = sum(e) or 1.0
    return {c: x / total for c, x in zip(CATEGORIES, e)}


def category_order(macro: Macro) -> list:
    """Categorias ordenadas de mas a menos prioritaria.

    Decide tanto el reparto de caja como la POSICION en la lista de ordenes,
    que importa porque el motor solo acepta `maxMarketOrdersPerTurn` y el resto
    se cae en silencio.
    """
    p = priorities(macro)
    return sorted(CATEGORIES, key=lambda c: -p[c])


def peso_fertilizar(macro: Macro) -> float:
    """Cuanto vale fertilizar, como multiplicador de su valor calculado.

    Es un MANDO, no un interruptor. Probado como interruptor global con el
    vector congelado dio -21 998 $ +- 4 563, pero ese vector estaba optimizado
    para un mundo SIN fertilizante: tenia 7 peones porque no le hacian falta
    mas. Juzgar una capacidad sin reoptimizar la pone en su peor luz.

    Con un mando, la busqueda decide: si no paga lo deja en 0 y no cuesta nada
    tenerlo; si paga con mas mano de obra, lo encontrara junto con los peones
    que necesita. Es la unica forma de capturar la interaccion.
    """
    # Sin techo: el 3.0 de antes era un maximo elegido a ojo. 0.5 da 1.5, que
    # es lo que salia antes en el centro del rango.
    return 1.5 * math.exp(_logit(macro.fertilizar))
