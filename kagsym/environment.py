"""N partidas en paralelo, avanzadas de DIA en dia.

Un paso = un dia. Es la decision de diseno que hace tratable el credit
assignment: el episodio son 30 pasos en vez de 720, y la accion de hoy cae en
la misma ventana que su consecuencia. Medido: con decision por turno y
recompensa de patrimonio, PPO convergia a 2 794 $, por debajo de los 3 000 $
que da no jugar.

Instrumenta ademas la FRACCION UTIL, que es la puerta de la fase 1: el termino
dominante de la brecha con el experto (23.3 % nuestro contra 50.6 % suyo).
"""
from __future__ import annotations

import numpy as np

from . import obs as O
from . import spec
from .symbolic.executor import Agent
from .fastenv import FastEnv
from .macro import Macro
from .reward import ProductionLedger
from .reward import RIVAL_WEIGHT as _RIVAL_W
from .reward import ILLEGAL_WEIGHT as _ILLEGAL_W

# spec.TURNS_PER_DAY se lee en tiempo de llamada (ver spec.set_turns_per_day):
# como alias de modulo se congelaba al importar y no seguia a
# `turnsPerDay`, desincronizando el ejecutor del motor sin avisar.
MOV = {"NORTH", "SOUTH", "EAST", "WEST"}


# Escalera de rivales, ORDENADA POR FUERZA MEDIDA en la matriz de la liga:
#   the-2945 gana 100% a todos
#   v16-rc5 y v48 empatan entre si (50%) y pierden con 2945
#   shop-router pierde con todos
# El criterio real de la competicion es victoria/derrota contra rivales de tu
# nivel (Elo + torneo Bradley-Terry final), no dinero medio contra un pasivo.
# ESCALERA REAL, por tope de peones. La anterior no era una escalera: medido
# el 2026-09-20 contra un rival pasivo, v48 hace 179.514 $, v16-rc5 171.878 y
# el 2945 171.392 -los tres valen lo mismo- y "shop-router-0909" esta ROTO (le
# falta agents_pub/actions.json). Con eso se perdia el 100 % de las partidas en
# cualquier nivel usable y `win_rate` valia 0,000 siempre.
#
# Atenuar (pasar turno al azar) NO sirve: al 80 % de sus acciones v48 cae de
# 179.514 $ a 312 $. Son ejecutores de PLANES acoplados, no politicas
# reactivas. Limitar los peones si conserva la coherencia, porque su propia
# logica se dimensiona al numero de unidades. Medido contra pasivo:
#   tope  3 ->  16.746 $      tope  8 ->  79.908 $
#   tope  5 ->  42.867 $      tope 15 -> 179.514 $
# Por debajo de 3 se derrumba (tope 2 -> 1 $): no puede ni arrancar la granja.
LADDER_CAPS = [None, 3, 5, 8, None]

LADDER = [
    None,                                              # pasivo
    "v48-fast-routes",
    "v48-fast-routes",
    "v48-fast-routes",
    "the-2945-farm-96-vs-the-top-10-public-bots",
    # v16-rc5 entra el 2026-09-22. Medido en la escalera de dificultad: a plena
    # potencia saca MENOS dinero que v48 (133.912 contra 153.720) y sin embargo
    # es el que MENOS nos deja a nosotros (36.139 contra 40.416). O sea que el
    # mas duro para nuestra politica no es el que mas puntua, y entrenar solo
    # contra v48 dejaba fuera precisamente al que peor se nos da.
    "v16-rc5-high-score-8c-4s-premium-market-lead",
    # ---- 2026-09-22: diez agentes publicos mas, de `runs/ligas/baja_escalera.py`.
    #
    # POR QUE HACIAN FALTA. Con seis peldanos ganabamos ocho de once al 88-100 %
    # y perdiamos el ultimo al 100 %: el salto era demasiado grande y no habia
    # relleno. Se bajaron 598 cuadernos publicos y se valido cada uno jugando
    # una partida entera; 63 juegan, y de esos solo 40 son distintos -el resto
    # son republicaciones exactas del mismo codigo, que dan el mismo dolar-.
    #
    # ORDENADOS POR FUERZA MEDIDA contra rival pasivo, no por puntuacion: la
    # correlacion entre puntuacion de leaderboard y dinero absoluto es r=+0,15.
    # Todos cultivan casi igual; los 1.000 puntos de diferencia salen del cara
    # a cara. Por eso la graduacion REAL la da el tope de peones -medida:
    # 3->16.825, 4->24.003, 5->42.099, 7->61.361, 9->92.281, sin tope->177.315-
    # y estos diez aportan DIVERSIDAD DE ESTILO, que es lo que el tope no da.
    "testkaggriculture-hamburger",                      #   80613 $, score 789
    "kaggriculture-adaptive-land-allocator",            #   90046 $, score 404
    "kaggriculture-weedproof-clone-market",             #  163933 $, score 1823
    "best-market-agent-high-strategy",                  #  166815 $, score 2498
    "kaggle-frontier-lab-strategy-improvement",         #  169238 $, score 2127
    "v29-r1-adaptive-market-hysteresis",                #  175069 $, score 1842
    "kaggriculture-conservative-market-router-v5",      #  180783 $, score 2042
    "kaggriculture-v6",                                 #  185827 $, score 2269
    "kaggriculture-v45-first-turn-wheat-round-trip",    #  186625 $, score 2498
    "your-market-list-is-an-order-book",                #  188274 $, score 2671
]

def public_with_cap(name_, max_hands: int):
    """Un agente publico al que se le limita cuantos peones puede contratar.

    ATENUAR NO SIRVE: medido, dejar pasar turno al azar al 20 % de las acciones
    hunde a v48 de 179.514 $ a 312 $. No es un continuo, es un precipicio -estos
    agentes son ejecutores de PLANES acoplados, no politicas reactivas: mueven
    una unidad varios turnos hacia una casilla y luego actuan, asi que perder un
    turno rompe la cadena entera-.

    Limitar los peones si conserva la coherencia del plan: su propia logica se
    dimensiona sola al numero de unidades que tiene.
    """
    base = load_public(name_)

    def jugar(obs):
        a = base(obs)
        ya = len(obs["farms"][int(obs["player"])]["hands"])
        room = max(0, max_hands - ya)
        orders = []
        for o in (a.get("market") or []):
            if o[0] == "HIRE":
                n_ = int(o[2]) if len(o) > 2 else 1
                n_ = min(n_, room)
                room -= max(0, n_)
                if n_ > 0:
                    orders.append([o[0], o[1], n_] if len(o) > 2 else o)
            else:
                orders.append(o)
        return dict(a, market=orders)
    return jugar


def publico_atenuado(name_, p: float, seed: int = 0):
    """Un agente publico que solo ACTUA con probabilidad `p`; si no, pasa turno.

    Por que existe. La ESCALERA no era una escalera: medido el 2026-09-20 contra
    un rival pasivo, v48 hace 179.514 $, v16-rc5 171.878 y el 2945 171.392 -los
    tres valen lo mismo- y el peldano 1 (shop-router-0909) esta ROTO, le falta
    agents_pub/actions.json. Con eso perdemos el 100 % de las partidas en
    cualquier nivel usable, `win_rate` vale 0,000 y el termino de victoria de la
    recompensa es una constante que no aporta gradiente.

    Atenuar es preferible a inventarse un bot intermedio: el comportamiento
    sigue siendo el de un agente real y bueno, solo que actua menos a menudo.
    Las ordenes de mercado se atenuan igual que las unidades, porque si no el
    agente sigue comprando tierra y peones que luego no usa.
    """
    import random as _rnd
    base = load_public(name_)
    rng = _rnd.Random(seed)

    def jugar(obs):
        a = base(obs)
        if rng.random() < p:
            return a
        return {"farmer": ["PASS"],
                "hands": [["PASS"] for _ in (a.get("hands") or [])],
                "market": []}
    return jugar


_PUBLICOS = {}


def load_public(name_):
    """Carga un agente publico de `agents_pub/` como funcion obs -> accion.

    CON CACHE. Sin ella, `spec_from_file_location` + `exec_module` recompilaban
    el modulo en CADA llamada: medido, 553 ms, el 64 % del coste de una partida
    de 36 turnos (858 ms en total, de los que el motor y el ejecutor son 250 y
    la red solo 57). Como se llama una vez por partida, una iteracion del CEM
    con 40 candidatos x 3 semillas hacia 120 recompilaciones.
    """
    ag = _PUBLICOS.get(name_)
    if ag is None:
        import importlib.util
        import os
        path_ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents_pub", name_ + ".py")
        spec_ = importlib.util.spec_from_file_location(
            "pub_" + name_.replace("-", "_"), path_)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        ag = mod.agent
        _PUBLICOS[name_] = ag
    return ag


def _parte_micro(m):
    """En modo "ops" el mapa trae 1+N_OPS canales: canal 0 valor, resto verbos.

    En los demas modos es (10,10) y se devuelve tal cual, asi que la misma
    tuberia sirve para los tres sin ramificar en el resto del codigo.
    """
    if m is None:
        return None
    m = np.asarray(m, dtype=np.float32)
    return (m[0], m[1:]) if m.ndim == 3 else m


class DayEnv:
    def __init__(self, n, steps=720, seed0=1, scale=None, macro=None,
                 idx0=0, n_total=None,
                 rival_fn=None, level=0, potential=True, gamma=0.995):
        from .reward import SCALE as _ESC
        self.n, self.steps, self.seed0 = n, steps, seed0
        self.scale = float(_ESC if scale is None else scale)
        self.idx0 = int(idx0)
        self.n_total = int(n_total) if n_total else int(n)
        self.macro_fijo = macro
        self.rival_fn = rival_fn
        # Peldano de la escalera. Los agentes publicos guardan estado entre
        # turnos, asi que hay que construir uno NUEVO por partida: reutilizarlo
        # arrastra la granja de la partida anterior.
        self.level = level
        self.resultados = []
        # Shaping por potencial (Ng, Harada & Russell 1999):
        #     r' = r + gamma*Phi(s') - Phi(s)
        # con Phi = liquidacion exacta MIA menos la del RIVAL. Validado: el sesgo
        # terminal |Phi(s_T) - (caja_yo - caja_rival)| es del 0.7 %, frente a los
        # 611-917 $ que tenia el patrimonio neto. Por eso no cambia cual es la
        # politica optima: solo adelanta el credito.
        #
        # El termino negativo del rival hace que DESTRUIR su valor puntue igual
        # que crear el mio, que es lo coherente con un marcador de signo.
        #
        # Y quedarse con genero sin vender al cierre ya se penaliza solo: Phi
        # solo cuenta lo que DA TIEMPO a convertirse en caja, asi que el ultimo
        # salto cae a la caja. Un termino aparte lo contaria dos veces.
        # Estadisticos del RIVAL. Sin ellos "19 000 $" no significa nada:
        # puede ser empate o paliza. Medido contra v48: nosotros 36 566 $ con 26
        # casillas productivas de 60, el 89 037 $ con 55 de las mismas 60.
        self.riv_fin = []
        self.riv_cult = []
        self.riv_anim = []
        self.riv_uds = []
        self._riv_uds_max = [1] * n
        # Mascara de dimensiones que de verdad decidieron, un dia por env.
        # La usa PPO para que el cociente de importancia ignore las ~1 570
        # dimensiones gaussianas que no pueden cambiar ninguna accion.
        self._masks = [None] * n
        self.potential = potential
        self.gamma = gamma
        self._phi = [0.0] * n
        self.sin_liquidar = []
        self.envs, self.agents, self.counter, self.obs = [None] * n, [None] * n, [None] * n, [None] * n
        self._rival = [None] * n
        self.ep = 0
        self.finales, self.utiles = [], []
        self._micro = [None] * n
        for i in range(n):
            self._reset(i)

    def _reset(self, i):
        # SEMILLAS DISJUNTAS POR CONSTRUCCION. Antes cada trabajador arrancaba
        # en `seed0 + off*1000` y dentro avanzaba `ep*n`, asi que el trabajador
        # 0 alcanzaba el rango del 1 en el episodio 1.000 y varios procesos
        # jugaban LA MISMA partida: el lote dejaba de ser independiente sin
        # avisar. Particionando por residuo modulo el total de entornos, dos
        # entornos distintos no pueden coincidir jamas.
        s = self.seed0 + self.ep * self.n_total + self.idx0 + i
        self.envs[i] = FastEnv(configuration={"episodeSteps": self.steps}, seed=s)
        self.obs[i] = self.envs[i].reset()
        mac = self.macro_fijo if self.macro_fijo is not None else Macro.default()
        self.agents[i] = Agent(episode_steps=self.steps, macro=mac,
                                micro=lambda ob, k=i: _parte_micro(self._micro[k]))
        self.counter[i] = ProductionLedger()
        self._micro[i] = None
        self._rival[i] = self._nuevo_rival()
        self._phi[i] = self._compute_phi(i) if self.potential else 0.0

    def _compute_phi(self, i):
        from .potential import phi as _phi_fn
        try:
            ob = self.obs[i][0]
            return _phi_fn(ob, 0, ob["private"]) / self.scale
        except Exception:
            return 0.0

    def set_rival_policy(self, factory):
        """Rival = una POLITICA nuestra (auto-juego).

        Por que hace falta. Entrenando contra v48 perdemos el 100 % de las
        partidas, asi que el termino terminal de la recompensa es CONSTANTE
        en -1: no aporta un solo bit. Todo el gradiente venia del shaping, y el
        criterio que de verdad puntua -ganar- era invisible para el aprendizaje.

        Y no hay peldano intermedio entre los publicos: a shop-router le ganamos
        el 100 % y a los otros tres les perdemos el 100 %.

        Contra uno mismo la tasa de victorias es ~50 % por construccion, que es
        donde la senal victoria/derrota tiene maxima varianza y por tanto maxima
        informacion.
        """
        self._rival_factory = factory
        for i in range(self.n):
            self._rival[i] = factory()

    def set_rival_macro(self, vector):
        """Rival = NUESTRO ejecutor exacto con un macro dado.

        Los agentes publicos son especialistas de 30 dias: medido, en partidas
        de 5-10 dias hacen 42-2.629 $ mientras nuestra heuristica hace ~3.000 y
        gana 1,000 contra TODOS los niveles. Como rival de liga corta no valen
        -win=1,000 no tiene mas gradiente que win=0,000-. Un vector macro
        optimizado por CEM PARA ESE HORIZONTE si sabe jugar esos dias.
        """
        self._rival_macro = None if vector is None else list(vector)
        self.resultados.clear()
        # REASIGNAR YA. `_reset` corrio en el constructor y los rivales de los
        # episodios en curso son los antiguos: cambiar solo el atributo no
        # cambia nada hasta el siguiente reset, y la medida sale identica a la
        # de antes sin avisar.
        for i in range(self.n):
            self._rival[i] = self._nuevo_rival()

    def _nuevo_rival(self):
        if getattr(self, "_rival_macro", None) is not None:
            from .symbolic.executor import Agent as _Ag
            from .macro import Macro as _Mac
            # Agente NUEVO por episodio: guarda estado entre turnos
            # (`_destinos`, `turnos_por_casilla`) y reutilizarlo contamina.
            # `episode_steps=None` a proposito: no tocar el global, que ya lo
            # fijo nuestro propio agente con los pasos de esta liga.
            return _Ag(macro=_Mac.from_vector(self._rival_macro))
        if getattr(self, "_rival_factory", None) is not None:
            return self._rival_factory()
        if self.rival_fn is not None:
            return self.rival_fn
        name_ = LADDER[min(self.level, len(LADDER) - 1)]
        if name_ is None:
            from kaggle_environments.envs.kaggriculture import kaggriculture as E
            return E.pass_agent
        try:
            cap = LADDER_CAPS[min(self.level, len(LADDER_CAPS) - 1)]
            if cap is not None:
                return public_with_cap(name_, cap)
            return load_public(name_)
        except Exception:
            from kaggle_environments.envs.kaggriculture import kaggriculture as E
            return E.pass_agent

    def raise_level(self):
        self.level = min(self.level + 1, len(LADDER) - 1)
        self.resultados.clear()
        return LADDER[self.level]

    def win_rate(self, ultimos=60):
        r = self.resultados[-ultimos:]
        return float(np.mean(r)) if r else float("nan")

    def encode(self):
        """Rejilla, vector global y OFERTA INMINENTE DEL RIVAL.

        La tercera salida alimenta la entrada `hist` de la red, que existia
        desde el principio -N_HIST = 4 x N_PRODUCTOS- y el entrenador rellenaba
        con CEROS. Es lo causalmente anterior a nuestro ingreso en un mercado
        compartido, y es predecible desde el tablero del rival, que se observa.
        """
        G = np.zeros((self.n, *O.SHAPES["grid"]), dtype=np.float32)
        B = np.zeros((self.n, O.N_GLOBAL), dtype=np.float32)
        Hf = np.zeros((self.n, O.N_HIST_RIVAL), dtype=np.float32)
        for i, o in enumerate(self.obs):
            G[i], B[i] = O.encode_obs(o[0])
            Hf[i] = O.rival_flow(o[0])
        return G, B, Hf

    def step_day(self, mapas=None, macros=None):
        """Juega 24 turnos. `mapas` (n,10,10) es el residuo de valor del dia."""
        from kaggle_environments.envs.kaggriculture import kaggriculture as E
        rec = np.zeros(self.n, dtype=np.float32)
        fin = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            if mapas is not None:
                self._micro[i] = np.asarray(mapas[i], dtype=np.float32)
            if macros is not None:
                self.agents[i].macro = Macro.from_vector(macros[i])
            env, counter = self.envs[i], self.counter[i]
            from .symbolic import tasks as _Tm
            if _Tm.MICRO_MODE == "ops":
                _Tm.enable_mask()
            util = total = 0
            for _ in range(spec.TURNS_PER_DAY):
                if env.done:
                    break
                ob = self.obs[i][0]
                acc = self.agents[i](ob)
                _invs = (ob.get("private", {}).get("inventories") or []
                         if _ILLEGAL_W else [])
                for _j, l in enumerate([acc.get("farmer")]
                                       + list(acc.get("hands") or [])):
                    if l:
                        total += 1
                        util += 1 if (l[0] not in MOV and l[0] != "PASS") else 0
                        if _ILLEGAL_W and l[0] not in MOV and l[0] != "PASS":
                            _iv = (_invs[_j] if _j < len(_invs)
                                   and isinstance(_invs[_j], dict) else {})
                            if not _Tm._can_do(_iv, l):
                                rec[i] -= _ILLEGAL_W / self.scale
                counter.harvested(ob, acc, 0)
                income, bonus = counter.sold(ob, acc)
                rec[i] += income / self.scale + bonus
                try:
                    rival = self._rival[i](self.obs[i][1])
                except Exception:
                    rival = {"farmer": ["PASS"], "hands": [], "market": []}
                self.obs[i], d = env.step([acc, rival])
                # DENTRO del bucle de turnos, a proposito. Antes se tomaba una
                # vez por dia, fuera, y ahi los peones ya estan limpiados: daba
                # 1.0 siempre y parecia que el rival jugaba sin mano de obra.
                # Medido el 2026-09-21 muestreando los 719 turnos: v48 sostiene
                # 8,94 de media -y nosotros 9,88-, no 1. El comentario de abajo
                # decia que estaba arreglado y no lo estaba.
                try:
                    self._riv_uds_max[i] = max(
                        self._riv_uds_max[i],
                        1 + len(self.obs[i][1]["farms"][1]["hands"]))
                except Exception:
                    pass
                if d or env.done:
                    break
            if _Tm.MICRO_MODE == "ops":
                self._masks[i] = _Tm.collect_mask()
            if total:
                self.utiles.append(util / total)
            if self.potential and not env.done:
                nuevo = self._compute_phi(i)
                rec[i] += self.gamma * nuevo - self._phi[i]
                self._phi[i] = nuevo
            if env.done:
                r = env.rewards()
                if self.potential:
                    # Phi(s_T) = MI caja por construccion (sin el rival)
                    fin_phi = float(r[0]) / self.scale
                    rec[i] += self.gamma * fin_phi - self._phi[i]
                self.sin_liquidar.append(
                    int(sum(self.obs[i][0]["private"]["shed"].values())))
                # TERMINO COMPETITIVO, en el OBJETIVO y no en el shaping.
                # El de abajo (+-1) ya mira al rival, pero es un SIGNO: ganar
                # por 1 $ puntua igual que ganar por 50.000, asi que no dice
                # en que direccion apretar. Este es continuo en el MARGEN.
                # No va denso por dia a proposito: eso es el termino que
                # potencial.phi documenta como medido y retirado -el rival
                # aportaba el 99,1 % de la varianza diaria y el critico caia a
                # R2 -2,535-. El shaping solo puede llevar lo que el estado
                # predice; los saltos de su caja no lo son. Al cierre es un
                # escalar por episodio, la misma forma que el +-1 de siempre.
                if _RIVAL_W:
                    rec[i] -= _RIVAL_W * float(r[1]) / self.scale
                res = 1.0 if r[0] > r[1] else (0.5 if r[0] == r[1] else 0.0)
                rec[i] += 2.0 * res - 1.0
                self.resultados.append(res)
                self.finales.append(float(r[0]))
                self.riv_fin.append(float(r[1]))
                fr = self.obs[i][1]["farms"][1]
                # OJO: al cerrar la partida los peones ya se han limpiado, asi
                # que contarlos aqui da 1 siempre y parece que el rival juega
                # mutilado. Verificado midiendo aparte: contrata 87 veces y
                # sostiene 3.78 unidades de media. Sexta vez hoy que la hora
                # muerde; se usa el maximo visto durante la partida.
                self.riv_uds.append(max(1, self._riv_uds_max[i]))
                self._riv_uds_max[i] = 1
                self.riv_cult.append(sum(1 for fl in fr["tiles"] for t in fl
                                         if isinstance(t, dict) and t.get("kind") == "PLANT"))
                self.riv_anim.append(sum(1 for fl in fr["tiles"] for t in fl
                                         if isinstance(t, dict) and t.get("animal")))
                fin[i] = 1.0
                self.ep += 1
                self._reset(i)
        return rec, fin

    def masks(self):
        """(n, 1+N_OPS, B, B): que dimensiones del micro decidieron hoy.

        Fuera del modo "ops" no hay nada que enmascarar y se devuelve None.
        """
        from .symbolic import tasks as _Tm
        if _Tm.MICRO_MODE != "ops":
            return None
        z = np.zeros((1 + _Tm.N_OPS, spec.BOARD, spec.BOARD), dtype=np.float32)
        return np.stack([m if m is not None else z for m in self._masks])

    def mean_money(self, ultimos=50):
        return float(np.mean(self.finales[-ultimos:])) if self.finales else float("nan")

    def rival_stats(self, ultimos=50):
        m = lambda v: float(np.mean(v[-ultimos:])) if v else float("nan")
        return {"dinero": m(self.riv_fin), "cultivos": m(self.riv_cult),
                "animales": m(self.riv_anim), "unidades": m(self.riv_uds)}

    def mean_unsold(self, ultimos=50):
        """Unidades que quedaron en el cobertizo al cerrar. Deben ser 0."""
        v = self.sin_liquidar[-ultimos:]
        return float(np.mean(v)) if v else float("nan")

    def mean_useful(self, ultimos=400):
        return float(np.mean(self.utiles[-ultimos:])) if self.utiles else float("nan")
