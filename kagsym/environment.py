"""N episodes in parallel, advanced DAY by day.

One step = one day. This is the design decision that makes credit assignment
tractable: an episode is 30 steps instead of 720, and today's action falls in
the same window as its consequence. Measured: with a per-turn decision and a
net-worth reward, PPO converged to $2,794, below the $3,000 you get by not
playing.

It also instruments the USEFUL FRACTION, the share of unit-turns spent on
something other than moving or passing: the dominant term of the gap against
the expert (23.3% ours against 50.6% theirs).
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
from .reward import DENSE_WEIGHT as _DENSE_W
from .reward import WIN_WEIGHT as _WIN_W

# spec.TURNS_PER_DAY is read at call time (see spec.set_turns_per_day): as a
# module alias it froze at import and stopped following `turnsPerDay`, silently
# desynchronising the executor from the engine.
MOV = {"NORTH", "SOUTH", "EAST", "WEST"}


# Opponent ladder. The real competition criterion is win/loss against
# opponents at your level (Elo plus a final Bradley-Terry tournament), not mean
# money against a passive agent.
#
# THE REAL GRADATION IS THE HAND CAP, not the choice of agent. Measured against
# a passive opponent, the strong public agents all land within 5% of each other
# -$179,514, $171,878, $171,392- so swapping one for another changes nothing.
#
# Attenuating them (passing at random) does NOT work: at 80% of its actions v48
# falls from $179,514 to $312. They are executors of COUPLED PLANS, not
# reactive policies. Capping the hands does preserve coherence, because their
# own logic sizes itself to the number of units. Measured against passive:
#   cap  3 ->  $16,746      cap  8 ->  $79,908
#   cap  5 ->  $42,867      cap 15 -> $179,514
# Below 3 it collapses (cap 2 -> $1): it cannot even start the farm.
# PELDAÑOS 6 Y 7, QUE FALTABAN. Medido el 2026-09-23 con `prod` contra v48,
# 32 semillas por punto:
#
#     manos   nosotros   el rival    margen    WIN
#         5     56.591     21.725   +160,5%   1,00   <- agotado
#         6     52.972     37.660    +40,7%   0,91
#         7     53.018     52.234     +1,5%   0,69   <- AQUI hay gradiente
#         8     49.629     64.073    -22,5%   0,00   <- muro
#        14+    43.462    126.380    -65,6%   0,00   (v48 satura en 14 manos)
#
# La escalera tenia 3, 5 y 8: tres peldaños por debajo de la zona util y un
# salto por encima de ella. Entrenando en el 5 el win saturaba en 0,98 y la
# mejora NO transferia (-444 $, t -0,58 contra el v48 entero); saltando a los
# sin capar el win era 0,00 y el dinero BAJABA (29.732 -> 25.739 en 45 updates).
LADDER_CAPS = [None, 3, 5, 6, 7, 8, None]

LADDER = [
    None,                                              # pasivo
    "v48-fast-routes",
    "v48-fast-routes",
    "v48-fast-routes",                                 # cap 6
    "v48-fast-routes",                                 # cap 7  <- la zona util
    "v48-fast-routes",                                 # cap 8
    "the-2945-farm-96-vs-the-top-10-public-bots",
    # Measured on the difficulty ladder: at full power v16-rc5 makes LESS money
    # than v48 ($133,912 against $153,720) and yet it is the one that leaves US
    # the least ($36,139 against $40,416). The hardest opponent for our policy
    # is not the highest-scoring one, and training only against v48 left out
    # precisely the one we do worst against.
    "v16-rc5-high-score-8c-4s-premium-market-lead",
    # ---- ten more public agents, downloaded with `tools/download_ladder.py`.
    #
    # WHY THEY WERE NEEDED. With six rungs we won eight of eleven at 88-100% and
    # lost the last one 100%: the jump was too large and there was no filler.
    # 598 public notebooks were downloaded and each one validated by playing a
    # full episode; 63 play, and of those only 40 are distinct -the rest are
    # exact republications of the same code, giving the same dollar-.
    #
    # ORDERED BY MEASURED STRENGTH against a passive opponent, not by score:
    # the correlation between leaderboard score and absolute money is r=+0.15.
    # They all farm almost identically; the 1,000 points of difference come
    # from head-to-head play. That is why the REAL gradation comes from the
    # hand cap -measured: 3->$16,825, 4->$24,003, 5->$42,099, 7->$61,361,
    # 9->$92,281, uncapped->$177,315- and these ten contribute STYLE DIVERSITY,
    # which the cap cannot give.
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
    """A public agent limited in how many hands it may hire.

    ATTENUATING DOES NOT WORK: measured, passing at random on 20% of actions
    sinks v48 from $179,514 to $312. It is not a continuum, it is a cliff
    -these agents are executors of COUPLED PLANS, not reactive policies: they
    move a unit towards a tile over several turns and then act, so losing one
    turn breaks the whole chain-.

    Capping the hands does preserve the plan's coherence: their own logic sizes
    itself to the number of units they have.
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


def public_attenuated(name_, p: float, seed: int = 0):
    """A public agent that only ACTS with probability `p`; otherwise passes.

    KEPT FOR REFERENCE, NOT IN USE. Attenuating looks preferable to inventing
    an intermediate bot -the behaviour is still that of a real, good agent,
    just acting less often- but it was measured and it does not work: at 80% of
    its actions v48 falls from $179,514 to $312. These agents execute coupled
    plans, so dropping one turn breaks the whole chain. The hand cap is what
    grades them without breaking them (see `public_with_cap`).
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
    """Load a public agent from `agents_pub/` as a function obs -> action.

    CACHED. Without the cache, `spec_from_file_location` + `exec_module`
    recompiled the module on EVERY call: measured, 553 ms, 64% of the cost of a
    36-turn episode (858 ms total, of which engine and executor are 250 and the
    network only 57). Since it is called once per episode, one CEM iteration
    with 40 candidates x 3 seeds did 120 recompilations.
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


_RIV_FALLOS = [0]   # turnos en que el rival lanzo y jugo PASS en silencio


def _split_micro(m):
    """Split the micro map into value, verbs and -if present- keys/queries.

    Layout: channel 0 is the value, the next N_OPS are the verb logits, and
    the remaining 2K are K assignment keys followed by K queries. An agent
    without a network returns (10,10) or nothing, so the same pipeline serves
    both without branching elsewhere.
    """
    if m is None:
        return None
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 3:
        return m
    from .symbolic.tasks import N_OPS
    _rest = m.shape[0] - 1 - N_OPS
    if _rest >= 2 and _rest % 2 == 0:
        k = _rest // 2
        return (m[0], m[1:1 + N_OPS], m[1 + N_OPS:1 + N_OPS + k],
                m[1 + N_OPS + k:])
    return (m[0], m[1:])


class DayEnv:
    def __init__(self, n, steps=720, seed0=1, scale=None, macro=None,
                 idx0=0, n_total=None,
                 rival_fn=None, level=0, potential=None, gamma=None,
                 dos_asientos=False):
        # THE SHAPING GAMMA FOLLOWS THE TRAINER'S. It was pinned at 0.995
        # while `--gamma` moved the one used by GAE, so the two disagreed the
        # moment anybody touched it -- and `--grid` and `--leagues` change the
        # horizon, which is exactly when one would. A potential shaped with a
        # different gamma than the return is no longer policy-invariant.
        if gamma is None:
            import os as _os
            gamma = float(_os.environ.get("KAG_GAMMA", "0.995"))
        from .reward import SCALE as _ESC
        self.n, self.steps, self.seed0 = n, steps, seed0
        self.scale = float(_ESC if scale is None else scale)
        self.idx0 = int(idx0)
        self.n_total = int(n_total) if n_total else int(n)
        self.macro_fijo = macro
        self.rival_fn = rival_fn
        # Ladder rung. Public agents keep state across turns, so a NEW one has
        # to be built per episode: reusing it drags in the previous episode's
        # farm.
        self.level = level
        self.results = []
        # Potential-based shaping (Ng, Harada & Russell 1999):
        #     r' = r + gamma*Phi(s') - Phi(s)
        # with Phi = my exact liquidation minus the OPPONENT's. Validated: the
        # terminal bias |Phi(s_T) - (my_cash - their_cash)| is 0.7%, against
        # the $611-917 net worth had. That is why it does not change which
        # policy is optimal: it only moves credit earlier.
        #
        # The opponent's negative term makes DESTROYING their value score the
        # same as creating mine, which is what a sign-based scoreboard implies.
        #
        # And being left with unsold produce at the close already penalises
        # itself: Phi counts only what HAS TIME to become cash, so the last
        # jump lands on the cash. A separate term would count it twice.
        # OPPONENT statistics. Without them "$19,000" means nothing: it could
        # be a tie or a thrashing. Measured against v48: us $36,566 with 26
        # productive tiles of 60, them $89,037 with 55 of the same 60.
        self.rival_finals = []
        self.riv_cult = []
        self.riv_anim = []
        self.rival_units = []
        self._riv_uds_max = [1] * n
        # Mask of the dimensions that actually decided, one day per env. PPO
        # uses it so the importance ratio ignores the ~1,570 Gaussian
        # dimensions that cannot change any action.
        self._masks = [None] * n
        # SHAPING SWITCH, from the environment for the same reason the reward
        # weights are: the parallel env spawns child processes and threading a
        # flag through three layers of signatures would be more invasive. With
        # KAG_POTENCIAL=0 the reward is money and nothing else, which is the
        # only way to measure what the shaping is actually buying.
        if potential is None:
            import os as _os
            potential = _os.environ.get("KAG_POTENCIAL", "1") != "0"
        self.potential = potential
        # PESO DEL SHAPING, y con recocido. Hasta hoy era un on/off sin dial,
        # asi que no se podia medir cuanto compra ni apagarlo gradualmente.
        #
        # OJO con lo que el recocido compra AQUI: nuestro shaping es POTENCIAL
        # (Ng, Harada & Russell 1999), o sea invariante a la politica por
        # construccion -- no cambia cual es el optimo. El recocido existe para
        # recompensas densas que son un SUCEDANEO hackeable; la nuestra no lo
        # es. Asi que esto sirve para MEDIR su aportacion y para quitar su
        # varianza al final, no para arreglar un sesgo que no tiene.
        import os as _os2
        self.phi_w = float(_os2.environ.get("KAG_PHI_W", "1.0"))
        self.phi_w0 = self.phi_w
        # updates tras los cuales el peso llega a `KAG_PHI_FIN` (0 = sin recocido)
        self.phi_recocido = int(_os2.environ.get("KAG_PHI_RECOCIDO", "0"))
        self.phi_fin = float(_os2.environ.get("KAG_PHI_FIN", "0.0"))
        self.gamma = gamma
        self._phi = [0.0] * n
        self.unsold = []
        # Unidades vendidas por PRODUCTO, un dict por episodio cerrado. Es
        # CONTAR, no estimar, asi que no puede dar falso positivo -- y es la
        # unica forma de ver si un cambio mueve fresa y leche, que son el 84%
        # del hueco, en vez de juzgarlo contra un agregado.
        self.por_producto = []
        # PEDIDO contra REALIZADO. El macro emite una PETICION y nadie le dice
        # si se ejecuto: medido, el dial pide 42 manos el dia 20 y el ejecutor
        # da 12, porque atan las puertas de coste y presupuesto. El gradiente
        # sigue premiando o culpando a ese dial por el resultado del episodio
        # aunque no tuviera efecto -- credito atribuido a una palanca
        # desconectada. Esto no lo arregla, pero lo hace VISIBLE, que es el
        # paso que faltaba.
        self.saturacion = []
        self.envs, self.agents, self.counter, self.obs = [None] * n, [None] * n, [None] * n, [None] * n
        self._rival = [None] * n
        self.ep = 0
        self.finals, self.useful_frac = [], []
        self._micro = [None] * n
        # DOS ASIENTOS. Los dos jugadores los decide la red y los dos aportan
        # gradiente. Hasta ahora el motor y los DOS ejecutores ya corrian en
        # cada episodio y la trayectoria del asiento 1 se tiraba: esto es el
        # doble de datos por segundo de simulacion, que es el cuello.
        #
        # Y el rival queda a nuestro nivel POR CONSTRUCCION, que es lo que el
        # pool con Elo intentaba aproximar dando un rodeo.
        #
        # `encode`, `step_day` y `masks` pasan a devolver/aceptar 2n filas:
        # primero las n del asiento 0, luego las n del asiento 1.
        self.dos = bool(dos_asientos)
        self.agents2 = [None] * n
        self.counter2 = [None] * n
        self._micro2 = [None] * n
        self._phi2 = [0.0] * n
        self._masks2 = [None] * n
        for i in range(n):
            self._reset(i)

    @property
    def n_filas(self):
        """Filas que produce un update: 2n con dos asientos, n si no."""
        return self.n * (2 if self.dos else 1)

    def fijar_progreso(self, upd: int):
        """Recocido del peso del shaping. Lo llama el entrenador cada update."""
        if self.phi_recocido > 0:
            t = min(1.0, max(0.0, upd / float(self.phi_recocido)))
            self.phi_w = self.phi_w0 + t * (self.phi_fin - self.phi_w0)
        return self.phi_w

    def _reset(self, i):
        # DISJOINT SEEDS BY CONSTRUCTION. Each worker used to start at
        # `seed0 + off*1000` and advance by `ep*n` inside, so worker 0 reached
        # worker 1's range at episode 1,000 and several processes played THE
        # SAME episode: the batch silently stopped being independent.
        # Partitioning by residue modulo the total number of envs, two
        # different envs can never coincide.
        s = self.seed0 + self.ep * self.n_total + self.idx0 + i
        self.envs[i] = FastEnv(configuration={"episodeSteps": self.steps}, seed=s)
        self.obs[i] = self.envs[i].reset()
        mac = self.macro_fijo if self.macro_fijo is not None else Macro.default()
        self.agents[i] = Agent(episode_steps=self.steps, macro=mac,
                                micro=lambda ob, k=i: _split_micro(self._micro[k]))
        self.counter[i] = ProductionLedger()
        self._micro[i] = None
        self._rival[i] = None if self.dos else self._new_rival()
        self._phi[i] = self._compute_phi(i) if self.potential else 0.0
        if self.dos:
            # El asiento 1 es un agente COMPLETO e independiente: su ejecutor
            # guarda estado entre turnos (`_destinations`, `turns_per_tile`),
            # y compartirlo con el asiento 0 seria la misma trampa que ya
            # costo enterrar la busqueda una vez.
            self.agents2[i] = Agent(
                episode_steps=self.steps, macro=mac,
                micro=lambda ob, k=i: _split_micro(self._micro2[k]))
            self.counter2[i] = ProductionLedger()
            self._micro2[i] = None
            self._phi2[i] = self._compute_phi(i, 1) if self.potential else 0.0

    def _compute_phi(self, i, asiento=0):
        from .potential import phi as _phi_fn
        try:
            ob = self.obs[i][asiento]
            return _phi_fn(ob, asiento, ob["private"]) / self.scale
        except Exception:
            return 0.0

    def set_rival_policy(self, factory):
        """Opponent = one of OUR policies (self-play).

        Why it is needed. Training against a full-power public agent we lose
        100% of episodes, so the terminal reward term is CONSTANT at -1: it
        carries not one bit. All the gradient came from the shaping, and the
        criterion that actually scores -winning- was invisible to learning.

        Against a copy of ourselves the win rate is ~50% by construction, which
        is where the win/loss signal has maximum variance and therefore maximum
        information. That is also why self-play needs a FIXED quota rather than
        competing for one: p(1-p) would hand it the whole budget.
        """
        self._rival_factory = factory
        for i in range(self.n):
            self._rival[i] = factory()

    def set_rival_macro(self, vector):
        """Opponent = OUR exact executor with a given macro vector.

        The public agents are 30-day specialists: measured, in 5-10 day games
        they make $42-2,629 while our heuristic makes ~$3,000 and wins 1.000
        against EVERY level. As short-league opponents they are useless
        -win=1.000 carries no more gradient than win=0.000-. A macro vector
        optimised by CEM FOR THAT HORIZON does know how to play those days.
        """
        self._rival_macro = None if vector is None else list(vector)
        self.results.clear()
        # REASSIGN NOW. `_reset` ran in the constructor and the opponents of
        # in-flight episodes are the old ones: changing only the attribute
        # changes nothing until the next reset, and the measurement comes out
        # identical to the previous one without warning.
        for i in range(self.n):
            self._rival[i] = self._new_rival()

    def _new_rival(self):
        if getattr(self, "_rival_macro", None) is not None:
            from .symbolic.executor import Agent as _Ag
            from .macro import Macro as _Mac
            # A NEW agent per episode: it keeps state across turns
            # (`_destinations`, `turns_per_tile`) and reusing it contaminates.
            # `episode_steps=None` on purpose: do not touch the global, which
            # our own agent already set to this league's step count.
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
        self.results.clear()
        return LADDER[self.level]

    def win_rate(self, last=60):
        r = self.results[-last:]
        return float(np.mean(r)) if r else float("nan")

    def encode(self):
        """Rejilla, vector global y OFERTA INMINENTE DEL RIVAL.

        The third output feeds the network's `hist` input, which existed from
        the start -N_HIST = 4 x N_PRODUCTS- and the trainer filled with ZEROS.
        It is what causally precedes our income in a shared market, and it is
        predictable from the opponent's board, which is observable.
        """
        nf = self.n_filas
        G = np.zeros((nf, *O.SHAPES["grid"]), dtype=np.float32)
        B = np.zeros((nf, O.N_GLOBAL), dtype=np.float32)
        Hf = np.zeros((nf, O.N_HIST_RIVAL), dtype=np.float32)
        for i, o in enumerate(self.obs):
            G[i], B[i] = O.encode_obs(
                o[0], getattr(self.agents[i], "_destinations", None))
            Hf[i] = O.rival_flow(o[0])
            if self.dos:
                j = self.n + i
                # `encode_obs` es RELATIVA AL ASIENTO -mira obs["player"] y
                # pone tu granja en el primer bloque de canales-, asi que la
                # misma red sirve para los dos lados sin tocar nada.
                G[j], B[j] = O.encode_obs(
                    o[1], getattr(self.agents2[i], "_destinations", None))
                Hf[j] = O.rival_flow(o[1])
        return G, B, Hf

    def step_day(self, maps=None, macros=None):
        """Juega 24 turnos.

        Con `dos_asientos` las entradas y salidas llevan 2n filas: primero las
        n del asiento 0, luego las n del asiento 1. Los dos jugadores los
        decide la red y los dos producen trayectoria.
        """
        from kaggle_environments.envs.kaggriculture import kaggriculture as E
        from .symbolic import tasks as _Tm
        nf = self.n_filas
        rec = np.zeros(nf, dtype=np.float32)
        fin = np.zeros(nf, dtype=np.float32)
        for i in range(self.n):
            j = self.n + i                      # fila del asiento 1
            _sat_ped = _sat_real = 0.0
            _sat_n = 0
            if maps is not None:
                self._micro[i] = np.asarray(maps[i], dtype=np.float32)
                if self.dos:
                    self._micro2[i] = np.asarray(maps[j], dtype=np.float32)
            if macros is not None:
                self.agents[i].macro = Macro.from_vector(macros[i])
                if self.dos:
                    self.agents2[i].macro = Macro.from_vector(macros[j])
            env = self.envs[i]
            _mk = self._micro[i]
            _nk = 0
            if _mk is not None and getattr(_mk, "ndim", 0) == 3:
                _nk = max(0, (_mk.shape[0] - 1 - _Tm.N_OPS) // 2)
            self._n_keys = _nk
            # Un acumulador de mascara POR ASIENTO: el global se intercambia
            # alrededor de cada decision.
            _Tm.enable_mask(_nk)
            _m0 = _Tm.swap_mask(None)
            _m1 = None
            if self.dos:
                _Tm.enable_mask(_nk)
                _m1 = _Tm.swap_mask(None)
            util = total = 0

            def _cuenta(ob_s, acc_s, fila):
                """Acciones utiles/totales del asiento y penalizacion ilegal."""
                _u = _t = 0
                _invs = (ob_s.get("private", {}).get("inventories") or []
                         if _ILLEGAL_W else [])
                for _q, l in enumerate([acc_s.get("farmer")]
                                       + list(acc_s.get("hands") or [])):
                    if l:
                        _t += 1
                        _u += 1 if (l[0] not in MOV and l[0] != "PASS") else 0
                        if _ILLEGAL_W and l[0] not in MOV and l[0] != "PASS":
                            _iv = (_invs[_q] if _q < len(_invs)
                                   and isinstance(_invs[_q], dict) else {})
                            if not _Tm._can_do(_iv, l):
                                rec[fila] -= _ILLEGAL_W / self.scale
                return _u, _t

            for _ in range(spec.TURNS_PER_DAY):
                if env.done:
                    break
                ob = self.obs[i][0]
                # a media mañana: las manos de hoy ya estan contratadas y aun
                # no se han limpiado (a la hora 0 el conteo da siempre 1)
                if int(ob.get("hour", 0)) == max(1, spec.TURNS_PER_DAY // 4):
                    try:
                        from .macro import target_hands as _th
                        _sat_ped += float(_th(ob, self.agents[i].macro))
                        _sat_real += float(1 + len(ob["farms"][0]["hands"]))
                        _sat_n += 1
                    except Exception:
                        pass
                _Tm.swap_mask(_m0)
                acc = self.agents[i](ob)
                _m0 = _Tm.swap_mask(None)
                _u, _t = _cuenta(ob, acc, i)
                util += _u
                total += _t
                self.counter[i].harvested(ob, acc, 0)
                income, bonus = self.counter[i].sold(ob, acc)
                rec[i] += _DENSE_W * (income / self.scale) + bonus
                if self.dos:
                    ob1 = self.obs[i][1]
                    _Tm.swap_mask(_m1)
                    rival = self.agents2[i](ob1)
                    _m1 = _Tm.swap_mask(None)
                    _cuenta(ob1, rival, j)
                    self.counter2[i].harvested(ob1, rival, 1)
                    inc1, bon1 = self.counter2[i].sold(ob1, rival)
                    rec[j] += _DENSE_W * (inc1 / self.scale) + bon1
                else:
                    try:
                        rival = self._rival[i](self.obs[i][1])
                    except Exception as _e_riv:
                        # NUNCA EN SILENCIO. Tragarse la excepcion y jugar
                        # PASS convierte un rival roto en un rival debil, que
                        # es indistinguible de un rival al que ganamos: el
                        # autojuego media un handicap y lo llamaba ventaja.
                        _RIV_FALLOS[0] += 1
                        if _RIV_FALLOS[0] <= 3:
                            import traceback as _tb
                            print(f"  [RIVAL ROTO] {type(_e_riv).__name__}: "
                                  f"{_e_riv}", flush=True)
                            _tb.print_exc()
                        rival = {"farmer": ["PASS"], "hands": [], "market": []}
                self.obs[i], d = env.step([acc, rival])
                # DENTRO del bucle de turnos, a proposito. Fuera, las manos ya
                # se han limpiado: daba siempre 1,0 y hacia parecer que el
                # rival jugaba sin plantilla. Medido muestreando los 719
                # turnos: v48 sostiene 8,94 de media -y nosotros 9,88-, no 1.
                try:
                    self._riv_uds_max[i] = max(
                        self._riv_uds_max[i],
                        1 + len(self.obs[i][1]["farms"][1]["hands"]))
                except Exception:
                    pass
                if d or env.done:
                    break
            if _sat_n:
                self.saturacion.append((_sat_ped / _sat_n, _sat_real / _sat_n))
                self.saturacion = self.saturacion[-200:]
            self._masks[i] = _m0
            if self.dos:
                self._masks2[i] = _m1
            if total:
                self.useful_frac.append(util / total)
            if self.potential and not env.done:
                nuevo_ = self._compute_phi(i, 0)
                rec[i] += self.phi_w * (self.gamma * nuevo_ - self._phi[i])
                self._phi[i] = nuevo_
                if self.dos:
                    n2_ = self._compute_phi(i, 1)
                    rec[j] += self.phi_w * (self.gamma * n2_ - self._phi2[i])
                    self._phi2[i] = n2_
            if env.done:
                r = env.rewards()
                # Los dos asientos cierran con la MISMA estructura de premio,
                # cada uno con su dinero y el del otro invertido. Es el unico
                # reparto que deja el juego simetrico, que es lo que hace que
                # una copia contra si misma tenga que dar 0,5.
                _lados = [(i, float(r[0]), float(r[1]), self._phi)]
                if self.dos:
                    _lados.append((j, float(r[1]), float(r[0]), self._phi2))
                for _fila, _mio, _suyo, _ph in _lados:
                    if self.potential:
                        # Phi(s_T) = MI caja por construccion (sin termino rival)
                        rec[_fila] += self.phi_w * (
                            self.gamma * (_mio / self.scale) - _ph[i])
                    # TERMINO COMPETITIVO, en el OBJETIVO y no en el shaping.
                    # El +-1 de abajo ya mira al rival pero es un SIGNO: ganar
                    # por 1 $ puntua igual que por 50.000, asi que no dice
                    # hacia donde empujar. Este es continuo en el MARGEN, y no
                    # es denso por dia a proposito: el termino diario del
                    # rival se midio y hundio el critico a R2 -2,535.
                    if _RIVAL_W:
                        rec[_fila] -= _RIVAL_W * _suyo / self.scale
                    _res = 1.0 if _mio > _suyo else (0.5 if _mio == _suyo else 0.0)
                    rec[_fila] += _WIN_W * (2.0 * _res - 1.0)
                    fin[_fila] = 1.0
                self.unsold.append(
                    int(sum(self.obs[i][0]["private"]["shed"].values())))
                # ANTES del _reset, que recrea el ledger
                self.por_producto.append(
                    {"uds": dict(self.counter[i].vendidas),
                     "ing": dict(self.counter[i].ingreso)})
                self.por_producto = self.por_producto[-40:]
                res = 1.0 if r[0] > r[1] else (0.5 if r[0] == r[1] else 0.0)
                self.results.append(res)
                self.finals.append(float(r[0]))
                self.rival_finals.append(float(r[1]))
                fr = self.obs[i][1]["farms"][1]
                # OJO: al cierre las manos ya estan limpias, asi que contarlas
                # aqui da siempre 1. Se usa el maximo visto durante el episodio.
                self.rival_units.append(max(1, self._riv_uds_max[i]))
                self._riv_uds_max[i] = 1
                self.riv_cult.append(sum(1 for fl in fr["tiles"] for t in fl
                                         if isinstance(t, dict) and t.get("kind") == "PLANT"))
                self.riv_anim.append(sum(1 for fl in fr["tiles"] for t in fl
                                         if isinstance(t, dict) and t.get("animal")))
                self.ep += 1
                self._reset(i)
        return rec, fin

    def masks(self):
        """(n, 1+N_OPS+2K, B, B): which micro dimensions decided today.

        None when there is nothing to mask.
        """
        from .symbolic import tasks as _Tm
        _nk = int(getattr(self, "_n_keys", 0))
        z = np.zeros((1 + _Tm.N_OPS + 2 * _nk, spec.BOARD, spec.BOARD),
                     dtype=np.float32)
        _ms = list(self._masks) + (list(self._masks2) if self.dos else [])
        return np.stack([m if m is not None else z for m in _ms])

    def mean_money(self, last=50):
        return float(np.mean(self.finals[-last:])) if self.finals else float("nan")

    def rival_stats(self, last=50):
        m = lambda v: float(np.mean(v[-last:])) if v else float("nan")
        return {"money": m(self.rival_finals), "crops": m(self.riv_cult),
                "animals": m(self.riv_anim), "units": m(self.rival_units)}

    def manos_pedidas_reales(self, last=60):
        """(pedidas, reales) de media. Su cociente dice cuanto del dial es inerte."""
        d = self.saturacion[-last:]
        if not d:
            return (float("nan"), float("nan"))
        return (sum(x[0] for x in d) / len(d), sum(x[1] for x in d) / len(d))

    def producto_medio(self, last=20):
        """Unidades vendidas por producto, media de los ultimos episodios."""
        ds = self.por_producto[-last:]
        if not ds:
            return {}
        out = {}
        for campo in ("uds", "ing"):
            ks = set()
            for d in ds:
                ks |= set(d.get(campo) or {})
            out[campo] = {k: sum((d.get(campo) or {}).get(k, 0) for d in ds)
                          / len(ds) for k in ks}
        return out

    def mean_unsold(self, last=50):
        """Units left in the shed at the close. They should be 0."""
        v = self.unsold[-last:]
        return float(np.mean(v)) if v else float("nan")

    def mean_useful(self, last=400):
        return float(np.mean(self.useful_frac[-last:])) if self.useful_frac else float("nan")
