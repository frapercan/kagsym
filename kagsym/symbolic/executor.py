"""El agente: une la granja y el mercado dentro del presupuesto de 1 s/turno.

El world model es OPCIONAL a proposito. En el entorno de submission puede no
haber torch, y un agente que falle por eso vale cero. Sin modelo, la capa de
mercado supone flujo del rival nulo -conservador: infravalora la caida futura
del precio y por tanto empuja a vender antes- y el resto funciona igual.
"""
from __future__ import annotations

import os
import time

from .. import spec
from . import market_ops, tasks

# spec.TURNS_PER_DAY se lee en tiempo de llamada (ver spec.set_turns_per_day):
# como alias de modulo se congelaba al importar y no seguia a
# `turnsPerDay`, desincronizando el ejecutor del motor sin avisar.


# Usar el flujo del rival estimado de su tablero en la valoracion de venta.
# La entrada existia (`sell_orders(opp_flow=...)`) y por defecto iba a CERO,
# que el propio codigo describe como "conservador: infravalora la caida futura
# y por tanto empuja a vender antes".
USA_FLUJO_RIVAL = bool(int(__import__("os").environ.get("KAG_FLUJO_MERCADO", "0")))

TURNOS_CASILLA_INI = 3.0    # a ojo, nunca buscado
TURNOS_CASILLA_MIN = 2.0    # idem
# Como se ADAPTA el coste por casilla. Aprendidos (`f_sube_coste`,
# `f_baja_coste`): el 0.5 topaba la subida y el 0.93 la bajada, y ninguno de
# los dos habia entrado nunca en una busqueda.
SUBE_COSTE = 0.5
BAJA_COSTE = 0.93


def sustainable_tiles(obs, n_units: int, turnos_por_casilla: float) -> int:
    """Cuantas casillas puede mantener vivas esta plantilla.

    `turnos_por_casilla` NO es una constante: el agente la mide durante la
    partida (ver `Agent._recalibrar`). Una formula a priori no sabe cuanto
    tiempo pierden las unidades moviendose, y pasarse de casillas es fatal:
    dos dias sin regar y la planta se convierte en hierba, perdiendo semilla,
    casilla y el trabajo invertido.
    """
    return max(1, int(n_units * spec.TURNS_PER_DAY / max(1.0, turnos_por_casilla)))


class Agent:
    """Mantiene estado entre turnos: el world model y su puerta de confianza."""

    def __init__(self, model_path: str | None = None, horizon: int = 12,
                 use_model: bool = False, episode_steps: int | None = None,
                 macro=None, rival=None, micro=None):
        spec.set_episode_steps(episode_steps)
        # None = capa guionizada tal cual. Un `Macro` = la politica decide.
        self.macro = macro
        self.horizon = horizon
        # Punto de inyeccion del modelo de rival (fase 4). None = capa exacta pura.
        self.rival = rival
        # Micro: funcion obs -> mapa 10x10 de residuo de valor (fase 1).
        # None = valoracion heuristica tal cual, que es el liston de 22 298 $.
        self.micro = micro
        # Destino asignado a cada unidad el turno anterior, para la adherencia.
        self._destinos = {}
        self.prev = None          # (obs, accion) del turno anterior
        self.t_total = 0.0
        self.n_turnos = 0
        # Turnos de unidad que cuesta mantener una casilla. Se recalibra sola:
        # sube si hay plantas sin regar (nos hemos pasado), baja si no las hay y
        # la granja esta llena (podemos abarcar mas).
        self.turnos_por_casilla = TURNOS_CASILLA_INI
        self._ultimo_dia = -1

    # -- prediccion del rival (punto de inyeccion, fase 4) ------------------
    def _opponent_flow(self, obs, accion_provisional):
        """Vertido esperado del rival, o None si no hay modelo enchufado.

        `rival` es un objeto opcional con `flujo(obs, accion) -> np.ndarray` y
        `actualizar(obs_prev, accion_prev, obs)`. Se inyecta, no se carga de
        disco: asi el ejecutor no depende de que exista ningun checkpoint y la
        capa exacta sigue siendo exacta sin el.

        Medido antes de separarlo: el modelo acierta el NIVEL del vertido del
        rival (+19% en su dinero, +35% en su gasto) pero es PEOR que no
        corregir turno a turno (-26%). Quien se enchufe aqui debe predecir
        nivel acumulado, nunca el turno concreto.
        """
        if self.rival is None:
            if not USA_FLUJO_RIVAL:
                return None
            # Del TABLERO del rival, que se observa entero. Se convierte el
            # NIVEL acumulado -unidades que tendra listas dentro del horizonte-
            # en la TASA POR TURNO que espera `future_price`, que hace
            # `opp_flow * horizon`.
            #
            # La conversion no es un detalle: el docstring de arriba avisa de
            # que predecir el turno concreto salio PEOR que no corregir (-26 %)
            # y que aqui solo vale el nivel acumulado. Enchufar el nivel como
            # si fuera tasa lo multiplicaria por el horizonte.
            try:
                from .. import obs as _O
                import numpy as _np
                days = max(1.0, self.horizon / float(spec.TURNS_PER_DAY))
                f = _O.rival_flow(obs).reshape(len(_O.VENTANAS_RIVAL), -1)
                # la ventana mas cercana que cubre el horizonte
                k = min(range(len(_O.VENTANAS_RIVAL)),
                        key=lambda i: abs(_O.VENTANAS_RIVAL[i] - days))
                return _np.asarray(f[k], dtype=_np.float64) / max(1.0, self.horizon)
            except Exception:
                return None
        try:
            return self.rival.flow(obs, accion_provisional)
        except Exception:
            return None

    def _update_gate(self, obs) -> None:
        if self.rival is None or self.prev is None:
            return
        try:
            self.rival.actualizar(self.prev[0], self.prev[1], obs)
        except Exception:
            pass

    def _recalibrar(self, obs) -> None:
        """Ajusta la capacidad de riego con lo que de verdad ha pasado hoy."""
        if obs["day"] == self._ultimo_dia or obs["hour"] != spec.TURNS_PER_DAY - 1:
            return
        self._ultimo_dia = obs["day"]
        mi = obs["farms"][int(obs["player"])]
        dry = planted = 0
        for row in mi["tiles"]:
            for t in row:
                if isinstance(t, dict) and t.get("kind") == "PLANT":
                    planted += 1
                    if not t["watered_today"]:
                        dry += 1
        if planted == 0:
            return
        if dry > 0:
            # Nos hemos pasado: cada casilla cuesta mas de lo que creiamos.
            self.turnos_por_casilla *= 1.0 + min(SUBE_COSTE, dry / planted)
        else:
            self.turnos_por_casilla = max(TURNOS_CASILLA_MIN,
                                          self.turnos_por_casilla * BAJA_COSTE)

    # -- turno ---------------------------------------------------------------
    def __call__(self, obs) -> dict:
        t0 = time.perf_counter()
        self._ultima_accion = None
        me = int(obs["player"])
        mi = obs["farms"][me]

        self._update_gate(obs)
        self._recalibrar(obs)
        # Los 14 parametros que estaban a ojo viven ahora en el macro y los
        # leen tres modulos distintos desde funciones que no lo reciben. Se
        # escriben aqui, una vez por turno, en lugar de pasar el macro por
        # seis firmas mas.
        if self.macro is not None:
            from ..macro import apply_params
            apply_params(self.macro)

        # Unidades PLANIFICADAS, no las que hay en este instante. Tercera vez
        # que muerde la misma trampa: `len(mi["hands"])` vale SIEMPRE 0 en la
        # hora 0, que es justo cuando se decide. Con eso `sostenibles` salia 9-12
        # mientras habia 17-18 plantadas, o sea `libre = 0` y el ejecutor NO
        # dejaba plantar nunca mas. Por eso subir `casillas` solo compraba
        # semilla que no se plantaba: 178 semillas para 3.5 cultivos vivos,
        # ~4 000 $ tirados por partida y la ganaderia sin caja.
        n_units = 1 + len(mi["hands"])
        if self.macro is not None:
            from ..macro import target_hands
            n_units = max(n_units, 1 + target_hands(obs, self.macro))
        sustainable = sustainable_tiles(obs, n_units, self.turnos_por_casilla)
        planted = sum(1 for row in mi["tiles"] for t in row
                        if isinstance(t, dict) and t.get("kind") == "PLANT")
        free = max(0, sustainable - planted)

        # En modo "ops" el micro devuelve DOS mapas: valor 10x10 y logits
        # N_OPS x 10 x 10. En los demas modos devuelve solo el valor.
        _mv, _mo = None, None
        if self.micro:
            _salida = self.micro(obs)
            if isinstance(_salida, tuple):
                _mv, _mo = _salida
            else:
                _mv = _salida
        units = tasks.assign_units(
            obs, free,
            value_map=_mv, verb_map=_mo,
            previous=self._destinos,
            macro=self.macro)
        provisional = {"farmer": units[0] if units else ["PASS"],
                       "hands": units[1:], "market": []}

        flow = self._opponent_flow(obs, provisional)
        # El ORDEN importa: el motor procesa las ordenes en secuencia, asi que
        # vender primero financia contratar, y contratar va antes que comprar
        # semilla porque sin manos que rieguen la semilla se pierde.
        orders = []
        mac = self.macro
        # ORDEN POR VALOR EN JUEGO, no por el orden en que se escribieron las
        # llamadas. El motor solo acepta `maxMarketOrdersPerTurn` (10) por turno
        # y el resto se cae en silencio. Medido: `land_orders` iba la ultima y
        # BUY_LAND solo se puede emitir en la hora 0, que es justo cuando
        # `hire_orders` emite 8 contrataciones. El agente emitia la orden de
        # comprar cuadrante desde el dia 15 y NUNCA se ejecutaba: se quedaba en
        # 1 cuadrante metiendo 17 animales en 25 casillas.
        #
        # El criterio es que no puede esperar y cuanto vale:
        #   tierra   1000 $ por 25 casillas (~14 000 $ de retorno), solo hora 0
        #   pienso   sin el, el animal se escapa en dos dias (~1700 $ cada uno)
        #   animales la inversion de mayor retorno por casilla
        #   ventas   sensibles al precio, pero repetibles el turno siguiente
        #   semillas repetibles
        #   peones   el n-esimo del dia cuesta fib(n); se pueden emitir en las
        #            horas 0-3, asi que son las que mejor toleran esperar
        # El ORDEN lo decide la politica, no una lista que escribi yo. Medido:
        # la caja impide comprar tierra en el 31 % de los turnos, pienso en el
        # 19 % y animales en el 15 %, asi que quien va primero decide quien se
        # queda sin. Con prioridades uniformes se recupera el orden anterior.
        from ..macro import orden_categorias
        cajas = {
            "tierra": lambda: market_ops.land_orders(obs, macro=mac),
            "pienso": lambda: market_ops.feed_orders(obs),
            "animal": lambda: market_ops.animal_orders(obs, macro=mac),
            "venta": lambda: market_ops.sell_orders(obs, opp_flow=flow,
                                                 horizon=self.horizon, macro=mac),
            "semilla": lambda: market_ops.seed_orders(obs, tile_target=min(free, 12),
                                                   macro=mac),
            "peon": lambda: market_ops.hire_orders(obs, macro=mac),
        }
        cats = orden_categorias(mac) if mac is not None else list(cajas)
        for c in cats:
            orders += cajas[c]()
        orders = orders[: spec.DEFAULT_CONFIG["maxMarketOrdersPerTurn"]]

        action = dict(provisional, market=orders)
        self._ultima_accion = action
        # La copia solo la consume `_update_gate`, que se rinde de inmediato si
        # no hay modelo de rival. Copiar la observacion ENTERA -dos granjas de
        # 100 casillas, mercado, pueblo, inventarios- cada turno cuando nadie
        # la va a leer era el 14,4 % del tiempo de partida perfilado el
        # 2026-09-21, mas su parte de `isinstance` y `dict.get`. Ningun camino
        # de entrenamiento ni de evaluacion pasa `rival=`.
        if self.rival is not None:
            from ..fastenv import _fast_copy
            self.prev = (_fast_copy(obs), _fast_copy(action))
        self.t_total += time.perf_counter() - t0
        self.n_turnos += 1
        return action

    @property
    def ms_por_turno(self) -> float:
        return 1000 * self.t_total / max(1, self.n_turnos)


_AGENTE = None


def agent(obs, config=None) -> dict:
    """Punto de entrada que espera kaggle-environments."""
    global _AGENTE
    if _AGENTE is None:
        _AGENTE = Agent()
    if config is not None:
        spec.set_episode_steps(
            config.get("episodeSteps") if isinstance(config, dict)
            else getattr(config, "episodeSteps", None))
    return _AGENTE(obs)
