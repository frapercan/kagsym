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
PRESUPUESTO_MANO_OBRA = 0.15
# CONSTANTES A OJO QUE LA POLITICA NO PODIA TOCAR. El docstring de `macro.py`
# dice que existe porque "la capa guionizada acumulo once constantes puestas a
# ojo... eso viola el principio del proyecto (nada adivinado)". Movio catorce y
# dejo estas. Nunca han entrado en ninguna busqueda.
FRACCION_SEMILLA = 0.5      # tope del gasto en semilla como fraccion de la caja
MARGEN_PEON = 3.0           # cuanto debe rendir un peon sobre su coste
DIAS_STOCK_PIENSO = 3.0     # reserva de comida, en dias
SAT_ALTA, SAT_BAJA = 0.85, 0.60   # umbrales de saturacion del mercado al vender
LAND_RETORNO, LAND_CAJA = 2.0, 1.5  # condicion de compra de cuadrante
# NO es un limite del motor -ese es maxMarketOrdersPerTurn = 10-, es una
# decision de ritmo escrita a mano. Por tanto, aprendible.
MAX_ANIMAL_TURNO = 2
# LAS QUE SEGUIAN INLINE, dentro del cuerpo de las funciones. Medidas vivas en
# ops (4 semillas, 24h x 30d): anular el credito de estiercol cuesta -22.5 % y
# estrangular la caja de pienso -21.6 %. Ahora las tres son `f_*` del vector.
FERT_ANIMAL = 0.5     # credito de estiercol al valorar un animal
CAJA_PIENSO = 0.25    # fraccion de caja que puede irse en pienso de golpe
SUELO_OBRA = 60.0     # suelo en dolares del presupuesto de mano de obra
URGENCIA_DIAS = 2.0   # dias de liquidacion al cerrar la temporada
# TERCERA HORNADA (auditoria estructural del 2026-09-22). Ver `Macro`.
STOCK_SEMILLA = 2.0   # semillas por unidad antes de dejar de comprar
RESERVA_ANIMAL = 300.0  # caja reservada antes de comprar un animal
HORA_CONTRATA = 3.0   # ultima hora del dia en que se contrata
DIAS_PEON = 2.0       # dias minimos para amortizar un peon
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
        from ..macro import horizonte_venta
        horizon = horizonte_venta(obs, macro)
    if urgencia_final is None:
        urgencia_final = int(round(URGENCIA_DIAS * spec.TURNS_PER_DAY))
    shed = obs["private"].get("shed", {})
    restantes = turns_left(obs)
    cap = spec.DEFAULT_CONFIG["shedCapacity"]
    ocupado = sum(shed.values())

    # RESERVA DE COMIDA. FEED consume 1 trigo por animal y dia, y a los dos dias
    # sin comer el animal se escapa. Medido: sin esta reserva se vendian 137
    # unidades de trigo y se escapaban 17 de los 25 animales comprados, a ~1700 $
    # cada uno. Vender UNA unidad de mas cuesta un animal entero.
    me = int(obs["player"])
    n_animales = sum(1 for fila in obs["farms"][me]["tiles"] for t in fila
                     if isinstance(t, dict) and t.get("animal"))
    reserva = {"WHEAT": n_animales * max(1, (restantes + 1) // spec.TURNS_PER_DAY)} if n_animales else {}

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
            from ..macro import mercado_factores
            _fac = mercado_factores(macro)
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
    from ..macro import pesos_turno
    _w = pesos_turno(macro)
    _vivo = any(abs(v) > 1e-9 for v in _w.values())
    _z0 = 0.0
    if _vivo:
        _pnow = obs["market"]["prices"]
        _val_shed = sum(float(_pnow.get(_q, 0)) * int(_c) for _q, _c in shed.items())
        _dinero = float(obs["farms"][me]["money"])
        _z0 = (_w["cobertizo"] * (ocupado / max(1, cap) - 0.5)
               + _w["estacion"] * ((1.0 - restantes / max(1, spec.EPISODE_STEPS)) - 0.5)
               + _w["caja"] * (_val_shed / max(1e-6, _val_shed + _dinero) - 0.5))

    def _factor(prod, z=0.0):
        """Nivel del producto por la reaccion del turno. Sin pesos -> el nivel."""
        f = _fac.get(prod, 1.0)
        if not _vivo:
            return f
        # El corte a +-13.8 es la misma guarda numerica que `_logit`: deja un
        # alcance de 1e6 veces, o sea no tener borde a efectos practicos.
        return f * math.exp(max(-13.8, min(13.8, _z0 + z)))

    ordenes = []
    for p in spec.PRODUCTS:
        n = int(shed.get(p, 0)) - int(reserva.get(p, 0))
        if n <= 0:
            continue

        # Fin de temporada: guardar no vale nada, el cobertizo no puntua, y los
        # animales tampoco: se libera tambien la reserva de comida.
        if restantes <= urgencia_final:
            ordenes.append((unit_value(obs, p) * _factor(p),
                            ["SELL", p, int(shed.get(p, 0))]))
            continue

        flujo = float(opp_flow[spec.PRODUCT_IX[p]]) if opp_flow is not None else 0.0
        objetivo = future_price(obs, p, min(horizon, restantes), flujo)
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
                                              / max(1e-6, objetivo))))
            _xr = flujo / max(1e-6, flujo + drain_rate(obs, p)) - 0.5
            _z = _w["precio"] * _xp + _w["rival"] * _xr
        _fp = _factor(p, _z)
        # Coste de oportunidad del capital: si el dinero liberado se reinvierte
        # y compone, retener producto tiene que batir tambien a ese crecimiento.
        # Sin esto el agente se sienta sobre 65 unidades con 122 $ en caja.
        objetivo /= capital_discount(obs, min(horizon, restantes), macro)
        objetivo /= max(1e-6, _fp)
        precios = marginal_prices(obs, p, n)
        k = sum(1 for pr in precios if pr >= objetivo)

        # El cobertizo desborda a 100 y lo que sobra se TIRA al final del dia.
        if ocupado > cap * SAT_ALTA:
            k = max(k, n - int(cap * SAT_BAJA))
        if k > 0:
            ordenes.append((precios[0] * _fp, ["SELL", p, k]))

    # Si hay mas ordenes que cupo, primero las que mas dinero mueven.
    ordenes.sort(key=lambda x: -x[0])
    return [o for _, o in ordenes]


def capital_discount(obs, horizon: int, macro=None) -> float:
    """Cuanto multiplicaria el capital liberado en `horizon` turnos.

    Solo cuenta si hay donde reinvertirlo: con la granja llena, liberar caja no
    aporta nada y conviene esperar al mejor precio.
    """
    from .tareas import best_crop, growth_factor
    me = int(obs["player"])
    farm_ = obs["farms"][me]
    vacias = sum(1 for fila in farm_["tiles"] for t in fila if t is None)
    if vacias <= 0:
        return 1.0
    # CARTERA, no el "mejor" cultivo. Esta valoracion estima a cuanto compone
    # el capital reinvertido, y cableaba `best_crop` -que ya no decide nada:
    # desde que la siembra es una cartera aprendida, suponer monocultivo da un
    # numero que no corresponde a lo que la politica hace.
    from .tareas import plantable
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


def hire_orders(obs, n_max: int = 15, margen: float = None, macro=None) -> list:
    """Contratar mientras el peon cueste menos de lo que rinde en un dia.

    El coste del n-esimo peon del dia es `fib(n)` y se reinicia cada dia. Un
    peon aporta 24 acciones, asi que la comparacion honesta es su coste contra
    el valor de esas acciones. `margen` exige que rinda varias veces su coste
    antes de contratarlo, porque el valor por accion es una estimacion optimista
    (las unidades gastan turnos moviendose).
    """
    margen = MARGEN_PEON if margen is None else margen
    farm = obs["farms"][int(obs["player"])]
    if obs["hour"] > HORA_CONTRATA:
        return []
    ya = int(farm["hires_today"])
    dinero = float(farm["money"])
    dias = max(1, (turns_left(obs) + 1) // spec.TURNS_PER_DAY)
    if dias < DIAS_PEON:
        return []                      # no da tiempo a amortizarlo

    from kaggle_environments.envs.kaggriculture import kaggriculture as E
    valor_accion = _valor_por_accion(obs)

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
    tareas = 0
    for fila in farm["tiles"]:
        for t in fila:
            if t is None and semillas_mano > tareas:
                tareas += 1                      # plantable Y con semilla
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT" and not t.get("watered_today"):
                    tareas += 1
                elif k == "WEED":
                    tareas += 1
                elif t.get("animal") and not t.get("fed_today"):
                    tareas += 1
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
        from ..macro import peones_objetivo
        n_max = min(n_max, peones_objetivo(obs, macro))
    else:
        ARRANQUE = 2
        por_unidad = max(1, spec.TURNS_PER_DAY // 3)
        n_max = min(n_max, max(ARRANQUE, -(-max(tareas, 1) // por_unidad)))
    presupuesto = max(SUELO_OBRA, dinero * PRESUPUESTO_MANO_OBRA)

    ordenes = []
    for n in range(ya, n_max):
        coste = E._fib(n)
        # Un peon aporta `turnsPerDay` acciones. Se contrata mientras su coste
        # sea una fraccion de lo que esas acciones rinden. NO se limita por
        # porcentaje de caja: la mano de obra es el multiplicador de todo lo
        # demas, y financiarla va antes que comprar semilla.
        if coste * margen > valor_accion * spec.TURNS_PER_DAY or coste > dinero:
            break
        if coste > presupuesto:
            break
        ordenes.append(["HIRE"])
        dinero -= coste
        presupuesto -= coste
    return ordenes


def _valor_por_accion(obs) -> float:
    """Cuanto vale un turno de unidad, medido por el mejor cultivo disponible."""
    from .tareas import cycle_days, cycle_profit, cycle_yield, plantable
    mejor = 0.0
    for c in spec.CROP_LIST:
        if not plantable(obs, c):
            continue
        turnos = cycle_days(c) + 3          # regar cada dia + plantar + cosechar
        mejor = max(mejor, cycle_profit(obs, c) / turnos)
    return max(1.0, mejor)


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
    coste = spec.LAND_PRICES[n]
    dinero = float(farm["money"])
    # Solo expandir si la tierra que ya se tiene esta aprovechada. Probado el
    # criterio alternativo de amortizacion (comprar si queda temporada para
    # pagarlo) y es PEOR: -1000 $ en los tres controles, porque compra 25
    # casillas que la politica no llega a usar. La saturacion es la guarda
    # correcta; la tierra no era el cuello de botella.
    vacias = sum(1 for fila in farm["tiles"] for t in fila if t is None)
    usables = sum(1 for y in range(spec.BOARD) for x in range(spec.BOARD)
                  if quadrant_of_xy(x, y) in farm["unlocked_quadrants"])
    umbral = 0.25 if macro is None else float(macro.expandir)
    if usables and vacias > usables * umbral:
        return []
    dias = max(0, (turns_left(obs) + 1) // spec.TURNS_PER_DAY)
    from .tareas import best_crop, cycle_days, cycle_profit, plantable
    # La tierra vale por el MEJOR uso que se le pueda dar, no solo por cultivo.
    # Medido: con estrategia ganadera pura `best_crop` no devuelve nada, asi que
    # el retorno se comparaba contra cero y el agente se quedaba en UN cuadrante
    # metiendo 13.6 animales en 25 casillas, con el 44.3 % de sus acciones
    # ociosas. El experto juega con 3 cuadrantes llenos.
    por_casilla = 0.0
    # CARTERA tambien aqui: esto estima que renta daria una casilla mas, y
    # cablear `best_crop` supone monocultivo, que ya no es lo que jugamos.
    _fl = _factores(macro)
    _vl = [k for k in spec.CROP_LIST if plantable(obs, k)]
    c = (max(_vl, key=lambda k: max(1e-6, _fl.get(k, 1.0))
             * (cycle_profit(obs, k) / max(1, cycle_days(k))))
         if _vl else None)
    if c is not None:
        por_casilla = (cycle_profit(obs, c) / max(1, cycle_days(c))) * dias
    mejor_animal = max((animal_net_value(obs, a) for a in spec.ANIMALS), default=0.0)
    if mejor_animal > 0:
        # Un animal ocupa una casilla (su estructura) y `animal_net_value` ya es
        # el neto de toda la temporada restante, comida incluida.
        por_casilla = max(por_casilla, mejor_animal)
    if por_casilla <= 0:
        return []
    retorno = 25 * por_casilla
    if retorno > coste * LAND_RETORNO and dinero > coste * LAND_CAJA:
        return [["BUY_LAND"]]
    return []


def _factores(macro):
    if macro is None:
        return {}
    try:
        from ..macro import mercado_factores
        return mercado_factores(macro)
    except Exception:
        return {}


def seed_orders(obs, objetivo_casillas: int, macro=None) -> list:
    """Comprar semilla del mejor cultivo para cubrir las casillas plantables."""
    # PREFERENCIA REVELADA, no preferencia calculada. `best_crop` elegia MELON
    # por $/casilla-dia mientras la politica emitia PLANT WHEAT: la semilla
    # comprada era inservible y no se plantaba nada. Medido: con mis reglas, el
    # experto 2945 caia de 26203 $ a 2012 $, exactamente lo mismo que no jugar.
    # Se compra lo que la politica DEMUESTRA plantar; solo si no ha plantado
    # nada aun se recurre al calculo, sesgado al mas barato para arrancar.
    from .tareas import best_crop
    _fac = _factores(macro)
    farm = obs["farms"][int(obs["player"])]
    plantados: dict = {}
    for fila in farm["tiles"]:
        for t in fila:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                plantados[t["crop"]] = plantados.get(t["crop"], 0) + 1
    if macro is not None:
        from ..macro import casillas_objetivo, cultivo_objetivo
        objetivo_casillas = casillas_objetivo(obs, macro)
        c = cultivo_objetivo(obs, macro)
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
    n_unidades = 1 + len(farm["hands"])
    # APRENDIDO. Este `return []` ABORTA la compra de semilla entera, y con
    # 8,4 unidades el umbral salia 16,8 mientras llevabamos 25,1 semillas de
    # media: buena parte de la partida no se compraba nada.
    if sin_usar >= max(2.0, STOCK_SEMILLA * n_unidades):
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
    from .tareas import plantable
    dinero = float(farm["money"])
    presupuesto = dinero * FRACCION_SEMILLA
    viables = [k for k in spec.CROP_LIST if plantable(obs, k)]
    if not viables:
        return []
    pesos = {k: max(1e-6, _fac.get(k, 1.0)) for k in viables}
    _tot = sum(pesos.values())
    faltan_tot = max(0, objetivo_casillas - sum(
        int(obs["private"].get("seeds", {}).get(k, 0)) for k in viables))
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
    _orden = sorted(viables, key=lambda x: -pesos[x])
    _queda_caja, _queda_cas, _n = presupuesto, faltan_tot, {}
    for _pasada in (1, 2):
        for k in _orden:
            if _queda_cas <= 0 or _queda_caja <= 0:
                break
            _precio = max(1, spec.CROPS[k]["seed"])
            # 1a pasada: su cuota de casillas segun el peso. 2a: lo que quede.
            _tope = (int(faltan_tot * pesos[k] / _tot) if _pasada == 1
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
    dias = (turns_left(obs) + 1) // spec.TURNS_PER_DAY
    dias_prod = max(0, dias - d["first_yield_day"])
    if dias_prod <= 0:
        return -1.0
    precios = obs["market"]["prices"]
    uds = int(dias_prod / d["interval"])
    if uds <= 0:
        return -1.0
    # PRECIO MARGINAL, no nominal. Cada unidad vendida baja el precio de la
    # siguiente: valorar 26 huevos a 50 $ cada uno sobrestima el ingreso y hace
    # que un animal parezca rentable cuando no lo es. Medido: con valor nominal
    # comprabamos animales a 14 dias y el resultado caia de 4310 $ a 345 $.
    ingreso = sum(marginal_prices(obs, d["product"], uds))
    comida = dias * float(precios.get("WHEAT", 0))
    fert = dias_prod * float(precios.get("FERTILIZER", 0)) * FERT_ANIMAL
    return ingreso + fert - d["cost"] - comida


def animal_orders(obs, max_por_turno: int = None, macro=None) -> list:
    """Comprar animales mientras salgan a cuenta y haya sitio donde ponerlos.

    Medido: el experto 2945 llega a 17 animales y nosotros a 0. Cada animal
    genera ademas 4 tareas diarias (comer, cuidar, recoger fertilizante,
    cosechar), asi que es a la vez ingreso y trabajo con el que justificar mas
    peones -nuestras unidades pasan el 22.3% de los turnos en PASS por falta de
    tareas, frente al 7.1% del experto-.
    """
    max_por_turno = (MAX_ANIMAL_TURNO if max_por_turno is None
                     else max_por_turno)
    farm = obs["farms"][int(obs["player"])]
    priv = obs["private"]
    dinero = float(farm["money"])
    cap = spec.DEFAULT_CONFIG["shedCapacity"]
    if sum(priv["shed"].values()) >= cap:
        return []

    # Sitio: estructuras vacias mas casillas libres donde construirlas.
    libres = {"COOP": 0, "PASTURE": 0}
    vacias = 0
    for fila in farm["tiles"]:
        for t in fila:
            if t is None:
                vacias += 1
            elif isinstance(t, dict) and t.get("kind") in libres and not t.get("animal"):
                libres[t["kind"]] += 1
    # animales ya comprados esperando colocacion
    esperando = sum(int(priv["shed"].get(a, 0)) for a in spec.ANIMALS)
    esperando += sum(int(inv.get(a, 0)) for inv in priv.get("inventories", []) for a in spec.ANIMALS)

    # TOPE POR CAPACIDAD DE ATENCION. Cada animal cuesta ~3 acciones al dia
    # (comer, cuidar/recoger, cosechar) y cada cultivo ~1 (regar). Medido: sin
    # este tope se compraban 25 animales con 4 unidades -75 tareas diarias para
    # 96 acciones contando el movimiento- y se escapaban 17.
    n_unidades = 1 + len(farm["hands"])
    cultivos = sum(1 for fila in farm["tiles"] for t in fila
                   if isinstance(t, dict) and t.get("kind") == "PLANT")
    vivos = sum(1 for fila in farm["tiles"] for t in fila
                if isinstance(t, dict) and t.get("animal"))
    # ~50% del presupuesto se va en moverse: es lo que mide el experto (42.3%)
    # y nosotros (56.5%), asi que la mitad es optimista y por tanto prudente.
    capacidad = n_unidades * spec.TURNS_PER_DAY * 0.5
    ACCIONES_POR_ANIMAL = 3.0
    if macro is not None:
        from ..macro import animales_objetivo
        hueco = animales_objetivo(obs, macro) - vivos - esperando
    else:
        hueco = int((capacidad - cultivos) // ACCIONES_POR_ANIMAL) - vivos - esperando
    if hueco <= 0:
        return []

    # QUE animal comprar era una formula escrita a mano -`animal_net_value`-,
    # el mismo caso que `best_crop` con los cultivos. Y es donde esta el otro
    # agujero medido: v48 saca 60.179 $ de LECHE y nosotros 13.536.
    #
    # Se pondera por el factor APRENDIDO del producto que da cada animal, que
    # ya existe en el vector macro (EGG, MILK, WOOL). Neutro = 1.0, asi que sin
    # macro el orden es el de siempre.
    _facA = _factores(macro)
    candidatos = sorted(
        spec.ANIMALS,
        key=lambda a: -(animal_net_value(obs, a)
                        * _facA.get(spec.ANIMALS[a]["product"], 1.0)))
    ordenes = []
    for a in candidatos:
        if len(ordenes) >= min(max_por_turno, hueco):
            break
        d = spec.ANIMALS[a]
        if animal_net_value(obs, a) <= 0:
            continue
        sitio = libres[d["structure"]] + vacias
        if sitio - esperando <= 0:
            continue
        # Reservar caja: quedarse sin dinero para semilla y comida arruina la
        # inversion, porque un animal sin comer dos dias se escapa.
        if dinero - d["cost"] < RESERVA_ANIMAL:
            continue
        ordenes.append(["BUY_ANIMAL", a, 1])
        dinero -= d["cost"]
        esperando += 1
    return ordenes


def feed_orders(obs, dias_stock: int = None) -> list:
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
    dias_stock = int(round(DIAS_STOCK_PIENSO if dias_stock is None
                           else dias_stock))
    me = int(obs["player"])
    farm = obs["farms"][me]
    priv = obs["private"]
    n_animales = sum(1 for fila in farm["tiles"] for t in fila
                     if isinstance(t, dict) and t.get("animal"))
    # tambien los que estan comprados esperando colocacion
    n_animales += sum(int(priv["shed"].get(a, 0)) for a in spec.ANIMALS)
    if n_animales <= 0:
        return []

    dias = max(1, (turns_left(obs) + 1) // spec.TURNS_PER_DAY)
    objetivo = n_animales * min(dias_stock, dias)
    tengo = int(priv["shed"].get("WHEAT", 0))
    tengo += sum(int(inv.get("WHEAT", 0)) for inv in priv.get("inventories", []))
    faltan = objetivo - tengo
    if faltan <= 0:
        return []

    cap = spec.DEFAULT_CONFIG["shedCapacity"]
    hueco = max(0, cap - sum(priv["shed"].values()))
    n = min(faltan, hueco)
    if n <= 0:
        return []
    # No arruinarse comprando pienso: un animal sin comer vale 0, pero una
    # granja sin caja tampoco produce.
    coste = sum(marginal_prices(obs, "WHEAT", n))
    dinero = float(farm["money"])
    while n > 1 and coste > dinero * CAJA_PIENSO:
        n -= 1
        coste = sum(marginal_prices(obs, "WHEAT", n))
    return [["BUY_PRODUCT", "WHEAT", n]] if n > 0 and coste <= dinero else []
