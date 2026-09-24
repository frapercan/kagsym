"""Rollouts across several processes. The last order-of-magnitude change.

Measured before this:

    engine alone .................. 38,567 steps/s
    engine + executor .............  3,264 steps/s
    end-to-end training loop ......    768 steps/s   <- one process, 12 cores

The split works well because the decision is made ONCE PER DAY: there are 30
synchronisations per episode, not 720. The main process does a single batched
forward on GPU at the start of the day, distributes the vectors, and the
workers run their 24 turns in parallel without talking to anyone.

Workers are PERSISTENT and keep their own environments: sending the engine
state down the pipe on every step would cost more than simulating it.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _worker(conn, n_envs, steps, seed0, macro_vec, level,
                hand_cap=None, hours=None, idx0=0, n_total=None, dos=False):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # ONE THREAD PER WORKER. torch starts with as many threads as cores, so 10
    # workers asked for 120 threads on 12 cores and spent their time fighting
    # each other. Measured: self-play cost 230 s/update when the same number of
    # environments against a public agent -also a full agent- cost 12 s. The
    # difference was not simulating two agents, it was contention.
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from kagsym.symbolic import tasks as _T
    if hours:
        # Like HAND_CAP: a per-process global, it does not arrive from the parent.
        from . import spec as _S
        _S.set_turns_per_day(hours)
    # The cap lives in a module global and workers are separate PROCESSES:
    # setting it in the parent does not reach here. It must be passed
    # explicitly.
    if hand_cap is not None:
        from . import macro as _M
        _M.HAND_CAP = hand_cap
    from kagsym.environment import DayEnv
    from kagsym.macro import Macro

    env = DayEnv(n_envs, steps=steps, seed0=seed0,
                     macro=Macro.from_vector(macro_vec), level=level,
                     idx0=idx0, n_total=n_total, dos_asientos=dos)
    while True:
        cmd, data = conn.recv()
        if cmd == "codifica":
            conn.send(env.encode())
        elif cmd == "paso":
            maps, macros = data
            rec, fin = env.step_day(maps=maps, macros=macros)
            conn.send((rec, fin, env.win_rate(), env.mean_money(),
                       env.mean_useful(), env.rival_stats(),
                       env.mean_unsold(), env.masks(),
                       env.producto_medio(), env.manos_pedidas_reales()))
        elif cmd == "autojuego":
            # Opponent weights: a snapshot of our own policy. They are sent
            # down the pipe each time a new version is frozen, not every step:
            # the cost is negligible against simulating.
            import io
            import torch
            from kagsym.symbolic.executor import Agent as _Ag
            from kagsym.macro import Macro as _Mac, N_MACRO
            from kagsym.nets.world import E2EAgent, WorldConfig
            from kagsym import obs as _O
            sd, cfgd = data
            _net = E2EAgent(WorldConfig(**{k: v for k, v in cfgd.items()
                                            if k in WorldConfig.__dataclass_fields__}))
            _net.load_state_dict(sd)
            _net.eval()

            def factory():
                # Decides ONCE PER DAY, exactly as we do. Calling the network
                # every turn is 24x more expensive and also ASYMMETRIC: the
                # opponent would play at a different decision frequency and it
                # would not be self-play.
                ag = _Ag(episode_steps=steps, macro=_Mac.from_vector([0.5] * N_MACRO))
                # SAME exploration as ours. With a deterministic opponent
                # (eps=0) the matchup is "us with noise" against "us without
                # noise", and we lose to the handicap, not to being worse:
                # measured, the noise costs $40,972 -> $18,967 with the same
                # vector. That biased the win signal, which is precisely the
                # only thing self-play was there to provide.
                eps_r = torch.randn(1, N_MACRO)
                # LA SIGMA APRENDIDA, que ademas trae la FORMA correcta.
                #
                # Esto calculaba `_nc = 1 + n_ops` = 20 canales, pero la
                # cabeza micro emite 28: sobran los canales de CLAVE
                # (1 + N_OPS + 2*n_keys). La suma `micro[0] + _sg*eps_u`
                # lanzaba "size of tensor a (28) must match b (20)" EL PRIMER
                # DIA DE CADA EPISODIO, y `step_day` se tragaba la excepcion y
                # jugaba PASS.
                #
                # Es decir: el rival de autojuego no jugaba MAL, NO JUGABA.
                # MEDIDO el 2026-09-23 con el caso nulo: 53.417 $ contra
                # 16.302 $ y win 1,000 frente a una copia EXACTA de nosotros
                # mismos. Con la misma red y el mismo codigo en los dos lados
                # la diferencia es -4.780 +- 4.374 (2 de 6), o sea cero.
                #
                # `log_sigma_micro` es un parametro del modelo, asi que su
                # forma sigue a la cabeza sola y los valores son los que la
                # politica usa de verdad -no los iniciales de la config-.
                _sg = _net.log_sigma_micro.exp().detach()
                _nc = int(_sg.shape[0])
                eps_u = torch.randn(_nc, 10, 10) if _nc > 1 else torch.randn(10, 10)
                estado = {"dia": None, "mapa": None}

                def jugar(ob):
                    # The DAY comes from the engine in the observation. This
                    # used to be `int(ob["step"]) // 24` with 24 hardcoded: in
                    # any cell that was not 24 turns/day the frozen opponent
                    # changed its decision every TWO game days and played
                    # crippled. Measured at 12h x 10d: an exact copy of a
                    # policy scoring $2,089 made $1,348.
                    d_ = int(ob["day"])
                    if estado["dia"] != d_:
                        # LAS TRES ENTRADAS, como nosotros. Faltaban dos:
                        #
                        #  * `hist` (N_HIST = 4 x N_PRODUCTS, el flujo del
                        #    rival en cuatro ventanas). `forward` lo rellena
                        #    con CEROS si no llega, asi que la copia jugaba
                        #    ciega a lo que el otro va a sacar al mercado. Y no
                        #    es un canal lateral: entra en `glob_enc`, que
                        #    modula por FiLM las cien casillas.
                        #  * `_destinations` del ejecutor, que nuestro
                        #    `encode_obs` si recibe: a donde van las unidades
                        #    ya asignadas.
                        #
                        # MEDIDO el 2026-09-23 con el caso nulo -politica
                        # congelada contra copia EXACTA de si misma, que por
                        # simetria debe dar 0,5-: daba win 1,000 y 53.417 $
                        # contra 16.302 $, el 31%. La copia rendia menos que un
                        # v48 capado. Con eso, el Elo del torneo, el "peldaño
                        # ganado al 100%" y la promocion puntuaban un
                        # handicap, no una diferencia de juego.
                        g, b = _O.encode_obs(
                            ob, getattr(ag, "_destinations", None))
                        hf = np.asarray(_O.rival_flow(ob), dtype=np.float32)
                        with torch.no_grad():
                            s_ = _net(torch.from_numpy(g).unsqueeze(0),
                                      torch.from_numpy(b).unsqueeze(0),
                                      torch.from_numpy(hf).unsqueeze(0))
                            am, _ = _net.macro_from(s_, eps_r)
                            estado["mapa"] = (s_["micro"][0]
                                              + _sg * eps_u).numpy()
                        ag.macro = _Mac.from_vector(am[0].numpy())
                        from .environment import _split_micro
                        ag.micro = lambda o2: _split_micro(estado["mapa"])
                        estado["dia"] = d_
                    return ag(ob)
                return jugar

            env.set_rival_policy(factory)
            conn.send(True)
        elif cmd == "rival_cap":
            # THE CAP APPLIES TO THE RUNG'S OWN AGENT. It used to hard-code
            # `v48-fast-routes` here, and since a grid ALWAYS calls
            # `set_rival_cap`, every worker ended up playing v48 no matter what
            # `--levels` said: eleven rungs of opponent variety collapsed into
            # one agent, and the ladder built for exactly that purpose never
            # played a single episode of training.
            #
            # Measured with two workers on rungs 0 and 15, 30 days: before the
            # cap call the opponent makes $3,000 (rung 0 is the PASSIVE agent,
            # to the dollar) and $141,374 (rung 15). After it, $82,388 and
            # $127,095 -the passive rung is now playing v48-.
            from .environment import public_with_cap as _pct, load_public as _cp
            from .environment import LADDER as _L
            _nm = _L[min(env.level, len(_L) - 1)]
            if _nm is None:
                # Rung 0 is the passive agent: it has no cap to apply, and
                # forcing a public agent on it destroys the only rung whose
                # opponent is policy-independent.
                from kaggle_environments.envs.kaggriculture import (
                    kaggriculture as _E)
                env.set_rival_policy(lambda: _E.pass_agent)
            elif data is None:
                env.set_rival_policy(lambda n=_nm: _cp(n))
            else:
                env.set_rival_policy(lambda n=_nm, t=data: _pct(n, t))
            conn.send(True)
        elif cmd == "rival_macro":
            env.set_rival_macro(data)
            conn.send(True)
        elif cmd == "olvida_resultados":
            # Clears the win history. Needed when PROMOTING the opponent:
            # `win_rate` averages the last 60 episodes, which are all wins
            # against the OLD opponent, so without this the threshold is
            # crossed again on the next check and promotion cascades.
            env.results.clear()
            conn.send(True)
        elif cmd == "level":
            env.level = data
            conn.send(True)
        elif cmd == "cerrar":
            conn.close()
            return


class ParallelEnv:
    """The same interface as `DayEnv`, spread across processes."""

    def __init__(self, n_envs, n_procs=8, steps=720, seed0=1, macro=None, level=2,
                 hand_cap=None, hours=None, dos_asientos=0):
        # DOS ASIENTOS POR TRABAJADOR, no global. Autojuego puro desde cero
        # tiene un equilibrio degenerado -los dos quietos- y el retorno es
        # RELATIVO: deja de subir en cuanto el rival mejora aunque los dos
        # esteis mejorando. Unos cuantos peldaños contra la escalera publica
        # son el ancla ABSOLUTA que impide esa deriva.
        #
        # Entero K = los K primeros trabajadores con dos asientos; -1 = todos.
        self._dos_arg = dos_asientos
        self.n_procs = min(n_procs, n_envs)
        self.n = n_envs
        # HORIZON PER WORKER. `steps` may be a list: each process plays
        # episodes of a different length and one update's batch mixes them.
        # This is possible because `spec.EPISODE_STEPS` is a PER-PROCESS
        # global, and workers are separate processes.
        #
        # Why MIXED matters and sequential does not: in sequence the policy
        # trains only at 5 days, then only at 8... and forgets, because nothing
        # in the gradient asks it to remember. Mixed, a single gradient has to
        # serve all three horizons at once, and the horizon signal in the
        # observation (obs.py, "time" block) lets it condition instead of
        # averaging.
        # `hours` and `hand_cap` accept lists for the same reason: `spec` and
        # `_M.HAND_CAP` are PER-PROCESS globals. So a batch can mix an entire
        # grid -hours, days and cap at once- which is what the two absolute
        # dimensions of the observation (EPISODE_STEPS/720 and cap/HANDS_REF)
        # let it condition on instead of average over.
        def _split_envs(v):
            if isinstance(v, (list, tuple)):
                # LOUD, because silent was expensive. The modulo only wraps
                # when the list is SHORTER than n_procs; when it is longer the
                # tail is dropped without a word, and a run then trains on a
                # grid nobody asked for. Measured cost of that: an overnight
                # campaign with 11 rungs and 5 processes trained on the first
                # two horizons against a passive agent, and its logs looked
                # right because the printout echoed the grid REQUESTED.
                if len(v) > self.n_procs:
                    raise SystemExit(
                        f"{len(v)} rungs asked for and only {self.n_procs} "
                        f"processes: rungs {list(v)[self.n_procs:]} would be "
                        f"dropped in silence. Raise --procs or shorten --grid.")
                return [v[i % len(v)] for i in range(self.n_procs)]
            return [v] * self.n_procs

        self.steps_proc = [int(x) for x in _split_envs(steps)]
        self.hours_proc = _split_envs(hours)
        self.cap_per_proc = _split_envs(hand_cap)

        # SPLIT BY COST, not per head. With a heterogeneous grid a 720-turn
        # rung costs 30 times a 24-turn one, and splitting episodes evenly
        # leaves ten workers waiting on the slow one on EVERY update. Episodes
        # are handed out in inverse proportion to their length, so everyone
        # takes roughly the same time and the batch still sums to `n_envs`.
        # Minimum one per worker: a rung with no episodes contributes no
        # gradient and the network would stop seeing that scale.
        weight = [1.0 / p for p in self.steps_proc]
        total = sum(weight)
        self.envs_per_proc = [max(1, int(n_envs * w / total)) for w in weight]
        sobran = n_envs - sum(self.envs_per_proc)
        i = 0
        while sobran != 0:                     # hand out the remainder by weight
            k = max(range(self.n_procs), key=lambda j: weight[j] / self.envs_per_proc[j])
            if sobran > 0:
                self.envs_per_proc[k] += 1; sobran -= 1
            else:
                k = min((j for j in range(self.n_procs) if self.envs_per_proc[j] > 1),
                        key=lambda j: weight[j], default=None)
                if k is None:
                    self.n = sum(self.envs_per_proc)
                    break
                self.envs_per_proc[k] -= 1; sobran += 1
            i += 1
            if i > 10 * n_envs:
                break
        self.n = sum(self.envs_per_proc)
        # FILAS, no envs: con dos asientos cada trabajador devuelve 2m filas
        # -primero sus m del asiento 0, luego sus m del asiento 1- y el orden
        # global queda [w0s0, w0s1, w1s0, w1s1, ...]. `encode` concatena y
        # `step_day` corta con el mismo reparto, asi que el emparejamiento
        # fila<->asiento se mantiene sin mas contabilidad.
        _k = self._dos_arg
        if isinstance(_k, (list, tuple)):
            self.dos_proc = [bool(_k[i % len(_k)]) for i in range(self.n_procs)]
        else:
            _k = int(_k)
            _k = self.n_procs if _k < 0 else _k
            self.dos_proc = [i < _k for i in range(self.n_procs)]
        self.dos = any(self.dos_proc)
        self.filas_per_proc = [m * (2 if d else 1)
                               for m, d in zip(self.envs_per_proc, self.dos_proc)]
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        off = 0
        vec = list(macro.to_vector()) if hasattr(macro, "to_vector") else list(macro)
        for k, m in enumerate(self.envs_per_proc):
            padre, hijo = ctx.Pipe()
            # DISJOINT seeds per worker: if they overlap, several processes
            # play the same episode and the batch stops being independent.
            p = ctx.Process(target=_worker,
                            args=(hijo, m, self.steps_proc[k], seed0, vec,
                                  (level[k % len(level)]
                                   if isinstance(level, (list, tuple)) else level),
                                  self.cap_per_proc[k],
                                  self.hours_proc[k], off, n_envs,
                                  self.dos_proc[k]),
                            daemon=True)
            p.start()
            self.conns.append(padre)
            self.procs.append(p)
            off += m
        self._wr = [float("nan")] * self.n_procs
        self._din = [float("nan")] * self.n_procs
        self._util = [float("nan")] * self.n_procs
        self._riv = [None] * self.n_procs
        self._sinliq = [float("nan")] * self.n_procs
        self._prod = [None] * self.n_procs
        self._sat = [None] * self.n_procs

    @property
    def n_filas(self):
        """Filas por update: 2n con dos asientos, n si no."""
        return sum(self.filas_per_proc)

    def encode(self):
        for c in self.conns:
            c.send(("codifica", None))
        partes = [c.recv() for c in self.conns]
        return (np.concatenate([p[0] for p in partes]),
                np.concatenate([p[1] for p in partes]),
                np.concatenate([p[2] for p in partes]))

    def step_day(self, maps=None, macros=None):
        off = 0
        for c, m in zip(self.conns, self.filas_per_proc):
            c.send(("paso", (None if maps is None else maps[off:off + m],
                             None if macros is None else macros[off:off + m])))
            off += m
        rec, fin, masks_ = [], [], []
        for k, c in enumerate(self.conns):
            r, f, wr, din, util, riv, sinliq, ms, prod_, sat_ = c.recv()
            self._prod[k] = prod_
            self._sat[k] = sat_
            rec.append(r); fin.append(f)
            if ms is not None:
                masks_.append(ms)
            self._wr[k], self._din[k], self._util[k] = wr, din, util
            self._riv[k], self._sinliq[k] = riv, sinliq
        self._masks = np.concatenate(masks_) if masks_ else None
        return np.concatenate(rec), np.concatenate(fin)

    def _rung_label(self, k):
        """Name of worker k's rung: "8h x 13d".

        It used to be labelled with `steps // 24`, which assumes 24 hours per
        day. With a grid the hours vary per worker, so that label merged
        different rungs under the same name: `5h x 21d` and `8h x 13d` are 105
        and 104 steps, i.e. "4 days" for both.
        """
        h = self.hours_proc[k] or 24
        return f"{h}h x {self.steps_proc[k] // h}d"

    def money_by_horizon(self):
        """{rung: mean money} instead of a single average.

        Averaging money over different horizons means NOTHING: a 15-day game
        earns less than a 30-day one by construction, so the mean rises or
        falls with whichever mix of episodes happened to close, not with how
        well the policy plays.
        """
        out = {}
        for k, steps in enumerate(self.steps_proc):
            d = self._din[k]
            if d is None or (isinstance(d, float) and d != d):
                continue
            out.setdefault(self._rung_label(k), []).append(float(d))
        return {k: sum(v) / len(v) for k, v in out.items() if v}

    def rival_by_horizon(self):
        """{rung: mean OPPONENT money}, broken down, not averaged."""
        out = {}
        for k, steps in enumerate(self.steps_proc):
            r = self._riv[k]
            if not isinstance(r, dict):
                continue
            d = r.get("money")
            if d is None or d != d:
                continue
            out.setdefault(self._rung_label(k), []).append(float(d))
        return {k: sum(v) / len(v) for k, v in out.items() if v}

    def masks(self):
        return getattr(self, "_masks", None)

    def set_selfplay(self, net):
        """Freeze the current policy as the opponent in every worker."""
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        cfgd = dict(vars(net.cfg))
        for c in self.conns:
            c.send(("autojuego", (sd, cfgd)))
        for c in self.conns:
            c.recv()

    def set_selfplay_on(self, idxs, net):
        """Freeze the policy as the opponent ONLY in the given workers.

        `set_selfplay` is global: it replaces the opponent on all rungs at
        once, so it cannot serve the MERIT-BASED relief, which is per rung.
        And without a per-rung version the relief had to fall back on
        `set_rival_macro`, which sends a VECTOR: our executor with the
        hand-written board heuristic and no micro head.

        That is not self-play. Measured with the SAME macro vector on both
        sides, 5 seeds of 24h x 30d: the network makes $64,762 and the
        vector-only opponent $29,924, i.e. +116% and 5 of 5. The opponent was
        worth half of us by construction, which is why the relief replaced it
        again and again -105 times in one overnight run- without it ever
        ceasing to lose.
        """
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        cfgd = dict(vars(net.cfg))
        for k in idxs:
            if 0 <= k < len(self.conns):
                self.conns[k].send(("autojuego", (sd, cfgd)))
        for k in idxs:
            if 0 <= k < len(self.conns):
                self.conns[k].recv()

    def set_rival_cap(self, cap):
        """Opponent = a public agent with that hand cap (None = uncapped).

        Set explicitly rather than via `level`, because the league ladder
        samples caps (3/5/8/11) that do not line up with the LADDER rungs.

        Accepts a LIST, one cap per worker, like `hours` and `steps`: in a grid
        the opponent cap is the rung's third axis and has to vary with it. A
        single value applies to all, as before.
        """
        vals = (list(cap) if isinstance(cap, (list, tuple))
                else [cap] * self.n_procs)
        for k, c in enumerate(self.conns):
            c.send(("rival_cap", vals[k % len(vals)]))
        for c in self.conns:
            c.recv()

    def set_rival_macro(self, vector):
        """Propagate the expert opponent to the workers (separate processes).

        Accepts a LIST, one vector per worker; `None` in a position leaves that
        worker on the public agent.

        Why it has to be per worker. Measured: `v48` makes $60,426 at 24h x 30d
        and essentially ZERO at any other scale (13h x 21d: $0; 8h x 13d: $0;
        6h x 6d: $7). It is a tape player indexed by step, built for 719 steps
        at 24 hours per day; change the hours or the days and the tape
        desynchronises, so the agent buys when it should be harvesting. In
        other words, across the whole reduced ladder there IS NO opponent: the
        win term is free and nobody drains the shared market.

        With one vector per rung, the opponent on the reduced rungs becomes OUR
        executor with the macro CEM found for that scale -which does play
        there, and reaches the measured ceiling- while the competition rung
        keeps the public agent.
        """
        if isinstance(vector, (list, tuple)) and vector and (
                vector[0] is None or hasattr(vector[0], "__len__")):
            vals = list(vector)
        else:
            vals = [vector] * self.n_procs
        for k, c in enumerate(self.conns):
            v = vals[k % len(vals)]
            c.send(("rival_macro", None if v is None else list(v)))
        for c in self.conns:
            c.recv()

    def raise_level(self, level):
        """Accepts a LIST, one level per worker.

        Why it is needed: the hand cap grades the opponent's DIFFICULTY, but
        the rungs would still all be the same agent. And the public agents do
        not resemble each other -measured: the one that hurts us most is
        v16-rc5 (it leaves us $36,139) even though v48 scores higher
        ($153,720)-. Training against a single style risks learning to beat
        THAT one rather than learning to play.
        """
        vals = (list(level) if isinstance(level, (list, tuple))
                else [level] * self.n_procs)
        for k, c in enumerate(self.conns):
            c.send(("level", vals[k % len(vals)]))
        for c in self.conns:
            c.recv()

    def _media(self, v):
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    def forget_results(self, idxs=None):
        """Clear the win history, optionally only in SOME workers.

        Used by opponent promotion: against a new opponent the rate has to be
        measured from scratch, not carried over from the previous one.

        `idxs` limits it to the workers whose opponent actually changed. Sin
        eso, el relevo por meritos -que cambia UN peldaño- borraba el historial
        de los ONCE. Medido el 2026-09-23: la tasa agregada quedaba en nan 7 de
        16 lecturas, lo que a su vez dejaba muerta la promocion -exige no-nan-
        y la actualizacion del Elo. Y el propio relevo volvia a leer el mismo
        peldaño con dos episodios y lo veia al 100% otra vez: siete relevos
        seguidos del peldaño 0 y de ningun otro.
        """
        ks = (range(len(self.conns)) if idxs is None
              else [k for k in idxs if 0 <= k < len(self.conns)])
        ks = list(ks)
        for k in ks:
            self.conns[k].send(("olvida_resultados", None))
        for k in ks:
            self.conns[k].recv()
        for k in ks:
            self._wr[k] = float("nan")

    def rival_per_rung(self):
        """Opponent money on EACH rung.

        The data already arrived per worker (`_riv[k]`) and was not exposed:
        without it there is no way to check that the ladder is graded or that
        each rung contributes anything.
        """
        out = []
        for d in self._riv:
            try:
                out.append(float(d.get("money", float("nan"))))
            except Exception:
                out.append(float("nan"))
        return out

    def win_rate_per_rung(self):
        """Win rate of EACH worker, that is, of each rung.

        Each worker plays a different rung of the ladder, so its `_wr` is
        exactly how exhausted that rung is. The global mean hides this: one
        rung won 100% and another lost 100% average the same as two at 50%,
        and those are opposite situations.
        """
        return list(self._wr)

    def win_rate(self, last=60):
        return self._media(self._wr)

    def money_per_worker(self):
        """Nuestro dinero en CADA trabajador.

        `mean_money` promedia los once, y con ocho en autojuego esa media no
        distingue "extraemos mas valor" de "farmeamos en un mundo donde el
        rival es igual de flojo". El trabajador ancla -el que juega contra el
        agente publico sin capar- es la unica lectura absoluta.
        """
        return [float(d) if d is not None else float("nan") for d in self._din]

    def mean_money(self, last=50):
        return self._media(self._din)

    def mean_useful(self, last=400):
        return self._media(self._util)

    def rival_stats(self, last=50):
        vs = [r for r in self._riv if r]
        if not vs:
            return {"money": float("nan"), "crops": float("nan"),
                    "animals": float("nan"), "units": float("nan")}
        return {k: self._media([r[k] for r in vs]) for k in vs[0]}

    def manos_pedidas_reales(self, idxs=None):
        ks_ = range(self.n_procs) if idxs is None else idxs
        ds = [self._sat[k] for k in ks_
              if k < len(self._sat) and self._sat[k]
              and self._sat[k][0] == self._sat[k][0]]
        if not ds:
            return (float("nan"), float("nan"))
        return (sum(d[0] for d in ds) / len(ds), sum(d[1] for d in ds) / len(ds))

    def producto_medio(self, idxs=None):
        """Unidades vendidas por producto, media sobre los trabajadores dados.

        `idxs` permite pedir SOLO los del rival externo: mezclar autojuego con
        v48 promedia dos mercados distintos.
        """
        ks_ = range(self.n_procs) if idxs is None else idxs
        ds = [self._prod[k] for k in ks_
              if k < len(self._prod) and self._prod[k]]
        if not ds:
            return {}
        out = {}
        for campo in ("uds", "ing"):
            ks = set()
            for d in ds:
                ks |= set(d.get(campo) or {})
            out[campo] = {k: sum((d.get(campo) or {}).get(k, 0.0) for d in ds)
                          / len(ds) for k in ks}
        return out

    def mean_unsold(self, last=50):
        return self._media(self._sinliq)

    def cerrar(self):
        for c in self.conns:
            try:
                c.send(("cerrar", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)
