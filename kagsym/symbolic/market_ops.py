"""Temporizado de mercado: donde se decide la partida.

Los margenes medidos entre agentes comparables estan por debajo del 1.6%, y el
96.5% de las acciones de unidad son identicas entre rivales. No se compite en
agricultura: se compite en **vender antes de que el otro hunda el precio**.

Las reglas salen de dos hechos del motor, no de intuicion:

  1. **Solo el pueblo drena el mercado.** Centro: 1 de cada producto no
     fertilizante cada 24 turnos. Cada instancia de tienda: 1 de cada producto
     que demanda cada 4 turnos (x2 si es de un solo producto). Nada mas.
  2. **El fertilizante no lo demanda nadie** -ni el centro ni ninguna tienda-
     asi que su precio solo baja, para siempre. Y el melon solo lo toma el
     centro: 1 cada 24 turnos.

De ahi sale todo: vender trigo tarde (el pueblo lo drena mas rapido de lo que
los jugadores lo producen, asi que se aprecia), vender fertilizante pronto o no
venderlo, y tratar el melon como un mercado de un solo tiro.
"""
from __future__ import annotations

import math

from .. import spec

# spec.TURNS_PER_DAY se lee en tiempo de llamada (ver spec.set_turns_per_day):
# como alias de modulo se congelaba al importar y no seguia a
# `turnsPerDay`, desincronizando el ejecutor del motor sin avisar.
# Fraccion de la caja que puede irse en mano de obra en un dia. Es el unico
# tope: la mano de obra multiplica todo lo demas, pero el coste fibonacci se
# dispara (15 peones = 1596 $/dia) y sin freno quiebra la granja en dos dias.
LABOUR_BUDGET_FRACTION = 0.15
# CONSTANTES A OJO QUE LA POLITICA NO PODIA TOCAR. El docstring de `macro.py`
# dice que existe porque "la capa guionizada acumulo once constantes puestas a
# ojo... eso viola el principio del proyecto (nada adivinado)". Movio catorce y
# dejo estas. Nunca han entrado en ninguna busqueda.
SEED_BUDGET_FRACTION = 0.5      # tope del gasto en semilla como fraccion de la caja
HAND_MARGIN = 3.0           # cuanto debe rendir un peon sobre su coste
FEED_STOCK_DAYS = 3.0     # reserva de comida, en dias
SAT_HIGH, SAT_LOW = 0.85, 0.60   # umbrales de saturacion del mercado al vender
LAND_RETURN, LAND_CASH = 2.0, 1.5  # condicion de compra de cuadrante
# NO es un limite del motor -ese es maxMarketOrdersPerTurn = 10-, es una
# decision de ritmo escrita a mano. Por tanto, aprendible.
MAX_ANIMALS_PER_TURN = 2
# LAS QUE SEGUIAN INLINE, dentro del cuerpo de las funciones. Medidas vivas en
# ops (4 semillas, 24h x 30d): anular el credito de estiercol cuesta -22.5 % y
# estrangular la caja de pienso -21.6 %. Ahora las tres son `f_*` del vector.
MANURE_CREDIT = 0.5     # credito de estiercol al valorar un animal
FEED_CASH_FRACTION = 0.25    # fraccion de caja que puede irse en pienso de golpe
LABOUR_FLOOR = 60.0     # suelo en dolares del presupuesto de mano de obra
LIQUIDATION_DAYS = 2.0   # dias de liquidacion al cerrar la temporada
# TERCERA HORNADA (auditoria estructural del 2026-09-22). Ver `Macro`.
SEED_STOCK_PER_UNIT = 2.0
SEED_FLOOR = 2.0      # minimo de semillas antes de que la regla de stock actue   # semillas por unidad antes de dejar de comprar
ANIMAL_CASH_RESERVE = 300.0  # caja reservada antes de comprar un animal
LAST_HIRE_HOUR = 3.0   # ultima hora del dia en que se contrata
MIN_HAND_DAYS = 2.0       # dias minimos para amortizar un peon
SHOP_INT = spec.DEFAULT_CONFIG["townShopSellInterval"]
CENTER_INT = spec.DEFAULT_CONFIG["townCenterSellInterval"]

# Demanda esperada de cada producto por instancia de tienda, calculada de la
# tabla SHOPS del motor: las tiendas se sortean uniformemente entre las 8.
_POR_TIENDA = {
    p: sum((2 if len(prods) == 1 else 1) / len(spec.SHOPS)
           for prods in spec.SHOPS.values() if p in prods)
    for p in spec.PRODUCTS
}


def drain_rate(obs, product: str) -> float:
    """Unidades que el pueblo absorbe por turno, con las tiendas ABIERTAS ahora."""
    n = len(obs["town"].get("unlocked_shops", []))
    r = _POR_TIENDA[product] * n / SHOP_INT
    if product in spec.TOWN_CENTER_PRODUCTS:
        r += 1.0 / CENTER_INT
    return r


def marginal_prices(obs, product: str, k: int) -> list[float]:
    """Lo que paga el motor por cada una de k unidades, en orden."""
    inv = obs["market"]["inventory"][product]
    params = obs["market"].get("params")
    return [spec.market_price(product, inv + j, params) for j in range(k)]


def future_price(obs, product: str, horizon: int, opp_flow: float = 0.0) -> float:
    """Precio previsto dentro de `horizon` turnos.

    El inventario se mueve por dos cosas conocidas -lo que drena el pueblo y lo
    que vierte el rival- y la segunda la predice el world model. Sin modelo se
    supone flujo cero, que es conservador: infravalora la caida futura y por
    tanto empuja a vender antes.
    """
    inv = obs["market"]["inventory"][product]
    futuro = inv + (opp_flow - drain_rate(obs, product)) * horizon
    return spec.market_price(product, int(round(futuro)), obs["market"].get("params"))


def turns_left(obs) -> int:
    return spec.EPISODE_STEPS - 1 - int(obs["step"])


def sell_orders(obs, opp_flow=None, horizon: int = 12, urgencia_final: int = None,
                macro=None) -> list:
    """Cuanto vender de cada producto ESTE turno.

    Para cada producto se compara el precio marginal de la unidad k-esima con el
    precio previsto dentro de `horizon` turnos. Se vende mientras cobrar ahora
    sea mejor que esperar. No hay 'tasa de goteo' elegida a ojo: el goteo emerge
    solo, porque cada unidad vendida baja el precio de la siguiente hasta que
    deja de compensar.
    """
    if macro is not None:
        from ..macro import sell_horizon
        horizon = sell_horizon(obs, macro)
    if urgencia_final is None:
        urgencia_final = int(round(LIQUIDATION_DAYS * spec.TURNS_PER_DAY))
    shed = obs["private"].get("shed", {})
    remaining = turns_left(obs)
    cap = spec.DEFAULT_CONFIG["shedCapacity"]
    used = sum(shed.values())

    # RESERVA DE COMIDA. FEED consume 1 trigo por animal y dia, y a los dos dias
    # sin comer el animal se escapa. Medido: sin esta reserva se vendian 137
    # unidades de trigo y se escapaban 17 de los 25 animales comprados, a ~1700 $
    # cada uno. Vender UNA unidad de mas cuesta un animal entero.
    me = int(obs["player"])
    n_animals = sum(1 for row in obs["farms"][me]["tiles"] for t in row
                     if isinstance(t, dict) and t.get("animal"))
    reserve = {"WHEAT": n_animals * max(1, (remaining + 1) // spec.TURNS_PER_DAY)} if n_animals else {}

    # FACTORES DE MERCADO APRENDIDOS, uno por producto. Es la ultima decision
    # que no se aprendia: el tablero ya lo decide la red casilla a casilla,
    # pero que vender y cuanto aguantar salia de una formula fija.
    #
    # El hueco que ataca esta medido: vendemos 59.872 $ contra los 200.760 de
    # v48, y 71.170 de esa diferencia son FRESAS que no tocamos y 60.179 LECHE
    # donde hacemos la quinta parte -el 84 % en dos productos-.
    #
    # Factor > 1 baja el precio objetivo de ese producto (comerciamos mas de
    # el) y sube su prioridad en la cola cuando hay mas ordenes que cupo.
    # Neutro = 1.0, asi que sin macro la conducta es la de siempre.
    _fac = {}
    if macro is not None:
        try:
            from ..macro import market_factors
            _fac = market_factors(macro)
        except Exception:
            _fac = {}

    # REGLA DE VENTA POR TURNO. `_fac` es el NIVEL de cada producto -cuanto nos
    # interesa comerciarlo- y es constante las 24 horas, porque el macro se
    # emite una vez al dia: un paso de RL es un DIA (ver `entorno.py`) y esta
    # capa juega el dia entero sola.
    #
    # Lo que faltaba no es mirar el estado -esta funcion ya lo mira- sino poder
    # aprender CUANTO reaccionar a cada senal. Eso es lo que anaden estos cinco
    # coeficientes:
    #
    #     factor_p(t) = nivel_p * exp( sum_j w_j * x_j(t) )
    #
    # La pasada diaria emite la REGLA; la regla se evalua con el estado de cada
    # turno. Asi hay decision por situacion sin multiplicar por 24 las pasadas
    # de red, que es lo que permite el segundo por turno de la competicion.
    #
    # Las cinco senales son adimensionales y estan centradas en 0, y los pesos
    # valen 0 por defecto (w = logit(0.5)), asi que al arrancar `fp` == `_fac`
    # EXACTAMENTE y la conducta es la de antes.
    from ..macro import turn_weights
    _w = turn_weights(macro)
    _vivo = any(abs(v) > 1e-9 for v in _w.values())
    _z0 = 0.0
    if _vivo:
        _pnow = obs["market"]["prices"]
        _val_shed = sum(float(_pnow.get(_q, 0)) * int(_c) for _q, _c in shed.items())
        _dinero = float(obs["farms"][me]["money"])
        _z0 = (_w["shed"] * (used / max(1, cap) - 0.5)
               + _w["season"] * ((1.0 - remaining / max(1, spec.EPISODE_STEPS)) - 0.5)
               + _w["cash"] * (_val_shed / max(1e-6, _val_shed + _dinero) - 0.5))

    def _factor(prod, z=0.0):
        """Nivel del producto por la reaccion del turno. Sin pesos -> el nivel."""
        f = _fac.get(prod, 1.0)
        if not _vivo:
            return f
        # El corte a +-13.8 es la misma guarda numerica que `_logit`: deja un
        # alcance de 1e6 veces, o sea no tener borde a efectos practicos.
        return f * math.exp(max(-13.8, min(13.8, _z0 + z)))

    orders = []
    for p in spec.PRODUCTS:
        n = int(shed.get(p, 0)) - int(reserve.get(p, 0))
        if n <= 0:
            continue

        # Fin de temporada: guardar no vale nada, el cobertizo no puntua, y los
        # animales tampoco: se libera tambien la reserva de comida.
        if remaining <= urgencia_final:
            orders.append((unit_value(obs, p) * _factor(p),
                            ["SELL", p, int(shed.get(p, 0))]))
            continue

        flow = float(opp_flow[spec.PRODUCT_IX[p]]) if opp_flow is not None else 0.0
        target = future_price(obs, p, min(horizon, remaining), flow)
        # Las dos senales que dependen del producto:
        #   precio  cuanto se espera que CAIGA -log(ahora/previsto)-. Positivo
        #           = el pronostico dice que baja, o sea razon para vender ya.
        #           El peso decide cuanto se fia la politica del pronostico.
        #   rival   que parte del flujo inminente lo pone EL, no el pueblo.
        #           Ya entra en `future_price` con coeficiente 1; el peso deja
        #           sobre-reaccionar o ignorarlo.
        _z = 0.0
        if _vivo:
            _xp = max(-2.0, min(2.0, math.log(max(1e-6, unit_value(obs, p))
                                              / max(1e-6, target))))
            _xr = flow / max(1e-6, flow + drain_rate(obs, p)) - 0.5
            _z = _w["price"] * _xp + _w["rival"] * _xr
        _fp = _factor(p, _z)
        # Coste de oportunidad del capital: si el dinero liberado se reinvierte
        # y compone, retener producto tiene que batir tambien a ese crecimiento.
        # Sin esto el agente se sienta sobre 65 unidades con 122 $ en caja.
        target /= capital_discount(obs, min(horizon, remaining), macro)
        target /= max(1e-6, _fp)
        prices_ = marginal_prices(obs, p, n)
        k = sum(1 for pr in prices_ if pr >= target)

        # El cobertizo desborda a 100 y lo que sobra se TIRA al final del dia.
        if used > cap * SAT_HIGH:
            k = max(k, n - int(cap * SAT_LOW))
        if k > 0:
            orders.append((prices_[0] * _fp, ["SELL", p, k]))

    # Si hay mas ordenes que cupo, primero las que mas dinero mueven.
    orders.sort(key=lambda x: -x[0])
    return [o for _, o in orders]


def capital_discount(obs, horizon: int, macro=None) -> float:
    """Cuanto multiplicaria el capital liberado en `horizon` turnos.

    Solo cuenta si hay donde reinvertirlo: con la granja llena, liberar caja no
    aporta nada y conviene esperar al mejor precio.
    """
    from .tasks import best_crop, growth_factor
    me = int(obs["player"])
    farm_ = obs["farms"][me]
    empty = sum(1 for row in farm_["tiles"] for t in row if t is None)
    if empty <= 0:
        return 1.0
    # CARTERA, no el "mejor" cultivo. Esta valoracion estima a cuanto compone
    # el capital reinvertido, y cableaba `best_crop` -que ya no decide nada:
    # desde que la siembra es una cartera aprendida, suponer monocultivo da un
    # numero que no corresponde a lo que la politica hace.
    from .tasks import plantable
    _f = _factores(macro)
    _vi = [k for k in spec.CROP_LIST if plantable(obs, k)]
    if not _vi:
        return 1.0
    _pes = {k: max(1e-6, _f.get(k, 1.0)) for k in _vi}
    _st = sum(_pes.values())
    c = max(_vi, key=lambda k: _pes[k])      # el dominante de la cartera
    if c is None:
        return 1.0
    return max(1.0, growth_factor(obs, c) ** (horizon / spec.TURNS_PER_DAY))


def unit_value(obs, product: str) -> float:
    return float(obs["market"]["prices"].get(product, 1))


def hire_orders(obs, n_max: int = 15, margin: float = None, macro=None) -> list:
    """Contratar mientras el peon cueste menos de lo que rinde en un dia.

    El coste del n-esimo peon del dia es `fib(n)` y se reinicia cada dia. Un
    peon aporta 24 acciones, asi que la comparacion honesta es su coste contra
    el valor de esas acciones. `margen` exige que rinda varias veces su coste
    antes de contratarlo, porque el valor por accion es una estimacion optimista
    (las unidades gastan turnos moviendose).
    """
    margin = HAND_MARGIN if margin is None else margin
    farm = obs["farms"][int(obs["player"])]
    if obs["hour"] > LAST_HIRE_HOUR:
        return []
    ya = int(farm["hires_today"])
    money = float(farm["money"])
    days = max(1, (turns_left(obs) + 1) // spec.TURNS_PER_DAY)
    if days < MIN_HAND_DAYS:
        return []                      # no da tiempo a amortizarlo

    from kaggle_environments.envs.kaggriculture import kaggriculture as E
    valor_accion = _value_per_action(obs)

    # Contratar SEGUN LA DEMANDA DE TRABAJO, no segun lo que el bolsillo aguante.
    # Sin este tope, el coste fibonacci hace que 15 peones cuesten 1596 $/dia y
    # la granja quiebra en dos dias pagando mano de obra ociosa. Medido: con una
    # politica que no plantaba, el dinero pasaba de 3000 a 204 en el turno 24.
    # Y la demanda se mide sobre trabajo REAL, no potencial. Una casilla vacia
    # es trabajo solo si la politica de verdad planta; si no, contratar para
    # "plantables" es pagar por intenciones. Medido: con la politica quieta, las
    # reglas solas convertian 3000 $ de patrimonio en 2422 $. El gasto debe
    # seguir a la capacidad demostrada, con un arranque minimo para que una
    # politica competente pueda despegar (se empieza con 0 semillas).
    # Trabajo real = lo EJECUTABLE ahora, que no es lo mismo que lo potencial.
    # Una casilla vacia es trabajo solo si hay semilla en mano para plantarla;
    # si no, contratar para "plantables" es pagar por intenciones. Contar cero
    # casillas vacias creaba el bloqueo inverso: sin semilla no hay tareas, sin
    # tareas no hay peones, sin peones no se planta, y nunca se arranca.
    semillas_mano = sum(int(v) for v in obs["private"].get("seeds", {}).values())
    tasks = 0
    for row in farm["tiles"]:
        for t in row:
            if t is None and semillas_mano > tasks:
                tasks += 1                      # plantable Y con semilla
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT" and not t.get("watered_today"):
                    tasks += 1
                elif k == "WEED":
                    tasks += 1
                elif t.get("animal") and not t.get("fed_today"):
                    tasks += 1
    # cada unidad atiende varias casillas al dia; con menos trabajo que gente,
    # contratar mas es puro gasto
    # El tope por tareas resulto un proxy demasiado burdo: daba 2 peones/dia y
    # con eso el experto 2945 -agente de RUTAS, la unidad i ejecuta la ruta i-
    # solo llegaba a ejecutar 2 rutas, plantaba 2 semillas en 84 turnos y se
    # quedaba en 2827 $ de sus 26203 $. Se sustituye por un tope de PRESUPUESTO
    # diario de mano de obra: se autolimita, escala con la riqueza y no presupone
    # nada sobre como la politica organiza el trabajo.
    # El objetivo lo fija la politica; el motor pone el limite duro.
    if macro is not None:
        from ..macro import target_hands
        n_max = min(n_max, target_hands(obs, macro))
    else:
        ARRANQUE = 2
        por_unidad = max(1, spec.TURNS_PER_DAY // 3)
        n_max = min(n_max, max(ARRANQUE, -(-max(tasks, 1) // por_unidad)))
    budget = max(LABOUR_FLOOR, money * LABOUR_BUDGET_FRACTION)

    orders = []
    for n in range(ya, n_max):
        cost = E._fib(n)
        # Un peon aporta `turnsPerDay` acciones. Se contrata mientras su coste
        # sea una fraccion de lo que esas acciones rinden. NO se limita por
        # porcentaje de caja: la mano de obra es el multiplicador de todo lo
        # demas, y financiarla va antes que comprar semilla.
        if cost * margin > valor_accion * spec.TURNS_PER_DAY or cost > money:
            break
        if cost > budget:
            break
        orders.append(["HIRE"])
        money -= cost
        budget -= cost
    return orders


def _value_per_action(obs) -> float:
    """Cuanto vale un turno de unidad, medido por el mejor cultivo disponible."""
    from .tasks import cycle_days, cycle_profit, cycle_yield, plantable
    best = 0.0
    for c in spec.CROP_LIST:
        if not plantable(obs, c):
            continue
        turns = cycle_days(c) + 3          # regar cada dia + plantar + cosechar
        best = max(best, cycle_profit(obs, c) / turns)
    return max(1.0, best)


def quadrant_of_xy(x: int, y: int) -> str:
    h = spec.BOARD // 2
    return ("N" if y < h else "S") + ("W" if x < h else "E")


def land_orders(obs, macro=None) -> list:
    """Comprar terreno mientras quede temporada para amortizarlo.

    Los datos de la ladder son tajantes: comprar entre los dias 4 y 7 se asocia
    a 0.49 de tasa de victoria, frente a 0.12-0.19 comprando mas tarde o nunca.
    Aqui la condicion es economica, no una fecha: el cuadrante cuesta
    LAND_PRICES[n] y aporta 25 casillas durante los dias que queden.
    """
    farm = obs["farms"][int(obs["player"])]
    n = len(farm["unlocked_quadrants"]) - 1
    if n >= len(spec.LAND_PRICES) or obs["hour"] != 0:
        return []
    cost = spec.LAND_PRICES[n]
    money = float(farm["money"])
    # Solo expandir si la tierra que ya se tiene esta aprovechada. Probado el
    # criterio alternativo de amortizacion (comprar si queda temporada para
    # pagarlo) y es PEOR: -1000 $ en los tres controles, porque compra 25
    # casillas que la politica no llega a usar. La saturacion es la guarda
    # correcta; la tierra no era el cuello de botella.
    empty = sum(1 for row in farm["tiles"] for t in row if t is None)
    usables = sum(1 for y in range(spec.BOARD) for x in range(spec.BOARD)
                  if quadrant_of_xy(x, y) in farm["unlocked_quadrants"])
    umbral = 0.25 if macro is None else float(macro.expandir)
    if usables and empty > usables * umbral:
        return []
    days = max(0, (turns_left(obs) + 1) // spec.TURNS_PER_DAY)
    from .tasks import best_crop, cycle_days, cycle_profit, plantable
    # La tierra vale por el MEJOR uso que se le pueda dar, no solo por cultivo.
    # Medido: con estrategia ganadera pura `best_crop` no devuelve nada, asi que
    # el retorno se comparaba contra cero y el agente se quedaba en UN cuadrante
    # metiendo 13.6 animales en 25 casillas, con el 44.3 % de sus acciones
    # ociosas. El experto juega con 3 cuadrantes llenos.
    per_tile = 0.0
    # CARTERA tambien aqui: esto estima que renta daria una casilla mas, y
    # cablear `best_crop` supone monocultivo, que ya no es lo que jugamos.
    _fl = _factores(macro)
    _vl = [k for k in spec.CROP_LIST if plantable(obs, k)]
    c = (max(_vl, key=lambda k: max(1e-6, _fl.get(k, 1.0))
             * (cycle_profit(obs, k) / max(1, cycle_days(k))))
         if _vl else None)
    if c is not None:
        per_tile = (cycle_profit(obs, c) / max(1, cycle_days(c))) * days
    mejor_animal = max((animal_net_value(obs, a) for a in spec.ANIMALS), default=0.0)
    if mejor_animal > 0:
        # Un animal ocupa una casilla (su estructura) y `animal_net_value` ya es
        # el neto de toda la temporada restante, comida incluida.
        per_tile = max(per_tile, mejor_animal)
    if per_tile <= 0:
        return []
    ret = 25 * per_tile
    if ret > cost * LAND_RETURN and money > cost * LAND_CASH:
        return [["BUY_LAND"]]
    return []


def _factores(macro):
    if macro is None:
        return {}
    try:
        from ..macro import market_factors
        return market_factors(macro)
    except Exception:
        return {}


def seed_orders(obs, tile_target: int, macro=None) -> list:
    """Comprar semilla del mejor cultivo para cubrir las casillas plantables."""
    # PREFERENCIA REVELADA, no preferencia calculada. `best_crop` elegia MELON
    # por $/casilla-dia mientras la politica emitia PLANT WHEAT: la semilla
    # comprada era inservible y no se plantaba nada. Medido: con mis reglas, el
    # experto 2945 caia de 26203 $ a 2012 $, exactamente lo mismo que no jugar.
    # Se compra lo que la politica DEMUESTRA plantar; solo si no ha plantado
    # nada aun se recurre al calculo, sesgado al mas barato para arrancar.
    from .tasks import best_crop
    _fac = _factores(macro)
    farm = obs["farms"][int(obs["player"])]
    plantados: dict = {}
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                plantados[t["crop"]] = plantados.get(t["crop"], 0) + 1
    if macro is not None:
        from ..macro import target_tiles, target_crop
        tile_target = target_tiles(obs, macro)
        c = target_crop(obs, macro)
    elif plantados:
        c = max(plantados, key=lambda k: plantados[k])
    else:
        c = min(spec.CROP_LIST, key=lambda k: spec.CROPS[k]["seed"])
    if c is None:
        return []
    tengo = int(obs["private"].get("seeds", {}).get(c, 0))
    # No comprar semilla si la que hay no se esta plantando. Medido: con una
    # politica que no planta, esto gastaba 980 $ en semillas que seguian en la
    # mano 120 turnos despues. El gasto debe seguir a la capacidad demostrada,
    # no a la intencion.
    # Tope por CAPACIDAD DE SIEMBRA, no por un numero fijo. El tope de 2 mataba
    # de hambre a un agente competente: con mis reglas el experto 2945 solo
    # llegaba a 3 cultivos, frente a los 57 que hace con las suyas.
    sin_usar = sum(int(v) for v in obs["private"].get("seeds", {}).values())
    n_units = 1 + len(farm["hands"])
    # APRENDIDO. Este `return []` ABORTA la compra de semilla entera, y con
    # 8,4 unidades el umbral salia 16,8 mientras llevabamos 25,1 semillas de
    # media: buena parte de la partida no se compraba nada.
    if sin_usar >= max(SEED_FLOOR, SEED_STOCK_PER_UNIT * n_units):
        return []
    # CARTERA, no monocultivo. Antes se compraba semilla de UN cultivo -el que
    # dijera `cultivo_objetivo`-, asi que dar libertad a la red para plantar lo
    # que quisiera no servia de nada: no habia semilla de lo demas. El cuello
    # estaba tres capas por encima de donde se planta.
    #
    # El reparto usa los factores de mercado APRENDIDOS, que incluyen los cinco
    # cultivos. Con todos neutros el presupuesto se reparte entre los viables;
    # cuando la red aprende que un cultivo vale mas, compra mas de ese.
    #
    # Se conserva lo medido: solo cultivos VIABLES -que dé tiempo a cosechar- y
    # el tope por capacidad de siembra de arriba, que existe porque comprar
    # semilla que no se planta gastaba 980 $ en vano.
    from .tasks import plantable
    money = float(farm["money"])
    budget = money * SEED_BUDGET_FRACTION
    viable = [k for k in spec.CROP_LIST if plantable(obs, k)]
    if not viable:
        return []
    weights = {k: max(1e-6, _fac.get(k, 1.0)) for k in viable}
    _tot = sum(weights.values())
    faltan_tot = max(0, tile_target - sum(
        int(obs["private"].get("seeds", {}).get(k, 0)) for k in viable))
    if faltan_tot <= 0:
        return []
    # DOS PASADAS, y la segunda es la que faltaba. Antes cada cultivo recibia
    # una rebanada FIJA del presupuesto y lo que sobraba se TIRABA. Con la
    # cartera aprendida en su regimen normal -medido: fresa x87, hasta x511,
    # trigo x0,33- eso significa que casi todo el dinero va a semilla de 100 $,
    # y cuando se agota las casillas restantes se quedan VACIAS aunque quede
    # caja para llenarlas de trigo a 10 $.
    #
    # El sintoma medido: cumpliamos el 47 % de nuestro propio objetivo de
    # casillas (14,6 plantadas contra 31,2 pedidas) y manteniamos 15,4 plantas
    # vivas contra las 44 de v48 -2,9x menos, que es justo la razon de dinero-.
    # Plantabamos lo MISMO que el (190 contra 206) pero comprabamos la mitad de
    # semilla (112 contra 212), porque la comprabamos cara.
    #
    # Esto NO decide la cartera: el orden de preferencia sigue siendo el que la
    # red emite, y quien mas pesa elige primero. Lo que cambia es que dejar la
    # granja vacia deja de ser inevitable cuando el favorito no se puede pagar.
    ordenes_s = []
    _orden = sorted(viable, key=lambda x: -weights[x])
    _queda_caja, _queda_cas, _n = budget, faltan_tot, {}
    for _pasada in (1, 2):
        for k in _orden:
            if _queda_cas <= 0 or _queda_caja <= 0:
                break
            _precio = max(1, spec.CROPS[k]["seed"])
            # 1a pasada: su cuota de casillas segun el peso. 2a: lo que quede.
            _tope = (int(faltan_tot * weights[k] / _tot) if _pasada == 1
                     else _queda_cas)
            n_k = min(_tope, _queda_cas, int(_queda_caja // _precio))
            if n_k > 0:
                _n[k] = _n.get(k, 0) + n_k
                _queda_cas -= n_k
                _queda_caja -= n_k * _precio
    for k in _orden:
        if _n.get(k, 0) > 0:
            ordenes_s.append(["BUY_SEED", k, _n[k]])
    return ordenes_s


def animal_net_value(obs, animal: str) -> float:
    """Valor neto de comprar este animal AHORA, con la economia exacta del motor.

    Un animal produce `1/interval` unidades por dia desde `first_yield_day`, y
    come 1 trigo al dia -si pasa dos dias sin comer, escapa-. Ademas deja
    fertilizante, que vale porque duplica el bonus de riego. Nada de esto se
    estima: sale de las tablas del motor.
    """
    d = spec.ANIMALS[animal]
    days = (turns_left(obs) + 1) // spec.TURNS_PER_DAY
    dias_prod = max(0, days - d["first_yield_day"])
    if dias_prod <= 0:
        return -1.0
    prices_ = obs["market"]["prices"]
    uds = int(dias_prod / d["interval"])
    if uds <= 0:
        return -1.0
    # PRECIO MARGINAL, no nominal. Cada unidad vendida baja el precio de la
    # siguiente: valorar 26 huevos a 50 $ cada uno sobrestima el ingreso y hace
    # que un animal parezca rentable cuando no lo es. Medido: con valor nominal
    # comprabamos animales a 14 dias y el resultado caia de 4310 $ a 345 $.
    income = sum(marginal_prices(obs, d["product"], uds))
    comida = days * float(prices_.get("WHEAT", 0))
    fert = dias_prod * float(prices_.get("FERTILIZER", 0)) * MANURE_CREDIT
    return income + fert - d["cost"] - comida


def animal_orders(obs, max_per_turn: int = None, macro=None) -> list:
    """Comprar animales mientras salgan a cuenta y haya sitio donde ponerlos.

    Medido: el experto 2945 llega a 17 animales y nosotros a 0. Cada animal
    genera ademas 4 tareas diarias (comer, cuidar, recoger fertilizante,
    cosechar), asi que es a la vez ingreso y trabajo con el que justificar mas
    peones -nuestras unidades pasan el 22.3% de los turnos en PASS por falta de
    tareas, frente al 7.1% del experto-.
    """
    max_per_turn = (MAX_ANIMALS_PER_TURN if max_per_turn is None
                     else max_per_turn)
    farm = obs["farms"][int(obs["player"])]
    priv = obs["private"]
    money = float(farm["money"])
    cap = spec.DEFAULT_CONFIG["shedCapacity"]
    if sum(priv["shed"].values()) >= cap:
        return []

    # Sitio: estructuras vacias mas casillas libres donde construirlas.
    free_slots = {"COOP": 0, "PASTURE": 0}
    empty = 0
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                empty += 1
            elif isinstance(t, dict) and t.get("kind") in free_slots and not t.get("animal"):
                free_slots[t["kind"]] += 1
    # animales ya comprados esperando colocacion
    pending = sum(int(priv["shed"].get(a, 0)) for a in spec.ANIMALS)
    pending += sum(int(inv.get(a, 0)) for inv in priv.get("inventories", []) for a in spec.ANIMALS)

    # TOPE POR CAPACIDAD DE ATENCION. Cada animal cuesta ~3 acciones al dia
    # (comer, cuidar/recoger, cosechar) y cada cultivo ~1 (regar). Medido: sin
    # este tope se compraban 25 animales con 4 unidades -75 tareas diarias para
    # 96 acciones contando el movimiento- y se escapaban 17.
    n_units = 1 + len(farm["hands"])
    crops = sum(1 for row in farm["tiles"] for t in row
                   if isinstance(t, dict) and t.get("kind") == "PLANT")
    alive = sum(1 for row in farm["tiles"] for t in row
                if isinstance(t, dict) and t.get("animal"))
    # ~50% del presupuesto se va en moverse: es lo que mide el experto (42.3%)
    # y nosotros (56.5%), asi que la mitad es optimista y por tanto prudente.
    capacity = n_units * spec.TURNS_PER_DAY * 0.5
    ACCIONES_POR_ANIMAL = 3.0
    if macro is not None:
        from ..macro import target_animals
        room = target_animals(obs, macro) - alive - pending
    else:
        room = int((capacity - crops) // ACCIONES_POR_ANIMAL) - alive - pending
    if room <= 0:
        return []

    # QUE animal comprar era una formula escrita a mano -`animal_net_value`-,
    # el mismo caso que `best_crop` con los cultivos. Y es donde esta el otro
    # agujero medido: v48 saca 60.179 $ de LECHE y nosotros 13.536.
    #
    # Se pondera por el factor APRENDIDO del producto que da cada animal, que
    # ya existe en el vector macro (EGG, MILK, WOOL). Neutro = 1.0, asi que sin
    # macro el orden es el de siempre.
    _facA = _factores(macro)
    candidates = sorted(
        spec.ANIMALS,
        key=lambda a: -(animal_net_value(obs, a)
                        * _facA.get(spec.ANIMALS[a]["product"], 1.0)))
    orders = []
    for a in candidates:
        if len(orders) >= min(max_per_turn, room):
            break
        d = spec.ANIMALS[a]
        if animal_net_value(obs, a) <= 0:
            continue
        sitio = free_slots[d["structure"]] + empty
        if sitio - pending <= 0:
            continue
        # Reservar caja: quedarse sin dinero para semilla y comida arruina la
        # inversion, porque un animal sin comer dos dias se escapa.
        if money - d["cost"] < ANIMAL_CASH_RESERVE:
            continue
        orders.append(["BUY_ANIMAL", a, 1])
        money -= d["cost"]
        pending += 1
    return orders


def feed_orders(obs, stock_days: int = None) -> list:
    """Comprar trigo para alimentar. Sin esto los animales se mueren de hambre.

    `FEED` consume 1 trigo por animal y dia, y a los dos dias sin comer el
    animal se escapa. Pero el trigo tiene que EXISTIR en el cobertizo, y si la
    granja cultiva otra cosa nunca aparece.

    Medido antes de existir esta funcion: forzando `macro.animales` se compraban
    13.4 animales por partida y quedaban 0.8 vivos, con `trigo_cobertizo = 0`
    en todos los hitos y CERO ejecuciones de FEED. El dinero caia de 41 598 $ a
    18 714 $: comprabamos ganado para verlo escapar.

    Esto explica tambien el `BUY_PRODUCT WHEAT 20` que el experto 2945 emite en
    el turno 0 y que aqui se habia descartado como "una forma cara de conseguir
    5 trigos": es pienso.

    El stock es aritmetica exacta, no un parametro: animales x dias. Se compra
    con `dias_stock` de colchon porque el trigo hay que ir a recogerlo al
    cobertizo y llevarlo andando.
    """
    # int(): el parametro es un real aprendible pero aguas abajo alimenta
    # cantidades que llegan a `range()`. Sin esto, TypeError solo cuando hay
    # ganaderia -a 5 dias no la hay, asi que no saltaba en la celda pequena-.
    stock_days = int(round(FEED_STOCK_DAYS if stock_days is None
                           else stock_days))
    me = int(obs["player"])
    farm = obs["farms"][me]
    priv = obs["private"]
    n_animals = sum(1 for row in farm["tiles"] for t in row
                     if isinstance(t, dict) and t.get("animal"))
    # tambien los que estan comprados esperando colocacion
    n_animals += sum(int(priv["shed"].get(a, 0)) for a in spec.ANIMALS)
    if n_animals <= 0:
        return []

    days = max(1, (turns_left(obs) + 1) // spec.TURNS_PER_DAY)
    target = n_animals * min(stock_days, days)
    tengo = int(priv["shed"].get("WHEAT", 0))
    tengo += sum(int(inv.get("WHEAT", 0)) for inv in priv.get("inventories", []))
    faltan = target - tengo
    if faltan <= 0:
        return []

    cap = spec.DEFAULT_CONFIG["shedCapacity"]
    room = max(0, cap - sum(priv["shed"].values()))
    n = min(faltan, room)
    if n <= 0:
        return []
    # No arruinarse comprando pienso: un animal sin comer vale 0, pero una
    # granja sin caja tampoco produce.
    cost = sum(marginal_prices(obs, "WHEAT", n))
    money = float(farm["money"])
    while n > 1 and cost > money * FEED_CASH_FRACTION:
        n -= 1
        cost = sum(marginal_prices(obs, "WHEAT", n))
    return [["BUY_PRODUCT", "WHEAT", n]] if n > 0 and cost <= money else []
