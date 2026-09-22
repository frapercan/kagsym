"""Rollouts en varios procesos. El unico cambio de orden de magnitud que queda.

Medido antes de esto:

    motor solo .................... 38 567 pasos/s
    motor + ejecutor ..............  3 264 pasos/s
    bucle de entrenamiento e2e ....    768 pasos/s   <- un solo proceso, 12 cores

El reparto encaja bien porque la decision se toma UNA VEZ AL DIA: hay 30
sincronizaciones por episodio, no 720. El proceso principal hace un unico
forward por lote en GPU al empezar el dia, reparte los vectores, y los
trabajadores ejecutan sus 24 turnos en paralelo sin hablar con nadie.

Los trabajadores son PERSISTENTES y mantienen sus propios entornos: mandar el
estado del motor por la tuberia en cada paso costaria mas que simularlo.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _trabajador(conn, n_envs, steps, seed0, macro_vec, level, mode="residuo",
                tope_peones=None, hours=None, idx0=0, n_total=None):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # UN HILO POR TRABAJADOR. torch arranca con tantos hilos como cores, asi que
    # 10 trabajadores pedian 120 hilos sobre 12 nucleos y se pasaban el tiempo
    # peleandose. Medido: el auto-juego costaba 230 s/update cuando el mismo
    # numero de entornos contra v48 -que tambien es un agente completo- costaba
    # 12 s. La diferencia no era simular dos agentes, era la contencion.
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from kagsym.symbolic import tasks as _T
    if hours:
        # Igual que TOPE_PEONES: global por proceso, no llega desde el padre.
        from . import spec as _S
        _S.set_turns_per_day(hours)
    _T.MODO_MICRO = mode
    # El tope vive en un global de modulo y los trabajadores son PROCESOS
    # aparte: ponerlo en el padre no llega aqui. Hay que pasarlo explicito.
    if tope_peones is not None:
        from . import macro as _M
        _M.TOPE_PEONES = tope_peones
    from kagsym.environment import EntornoDia
    from kagsym.macro import Macro

    env = EntornoDia(n_envs, steps=steps, seed0=seed0,
                     macro=Macro.from_vector(macro_vec), level=level,
                     idx0=idx0, n_total=n_total)
    while True:
        cmd, datos = conn.recv()
        if cmd == "codifica":
            conn.send(env.encode())
        elif cmd == "paso":
            mapas, macros = datos
            rec, fin = env.step_day(mapas=mapas, macros=macros)
            conn.send((rec, fin, env.win_rate(), env.mean_money(),
                       env.mean_useful(), env.rival_stats(),
                       env.mean_unsold(), env.masks()))
        elif cmd == "autojuego":
            # Pesos del rival: una instantanea de nuestra propia politica. Se
            # mandan por la tuberia cada vez que se congela una version nueva,
            # no cada paso: el coste es despreciable frente a simular.
            import io
            import torch
            from kagsym.symbolic.executor import Agent as _Ag
            from kagsym.macro import Macro as _Mac, N_MACRO
            from kagsym.nets.world import AgenteE2E, MundoConfig
            from kagsym import obs as _O
            sd, cfgd = datos
            _net = AgenteE2E(MundoConfig(**{k: v for k, v in cfgd.items()
                                            if k in MundoConfig.__dataclass_fields__}))
            _net.load_state_dict(sd)
            _net.eval()

            def fabrica():
                # Decide UNA VEZ AL DIA, igual que nosotros. Llamar a la red
                # cada turno es 24x mas caro y ademas ASIMETRICO: el rival
                # jugaria con otra frecuencia de decision y no seria auto-juego.
                ag = _Ag(episode_steps=steps, macro=_Mac.from_vector([0.5] * N_MACRO))
                # MISMA exploracion que nosotros. Con el rival determinista
                # (eps=0) el emparejamiento es "nosotros con ruido" contra
                # "nosotros sin ruido", y perdemos por el handicap, no por ser
                # peores: medido, el ruido cuesta 40 972 -> 18 967 $ con el
                # mismo vector. Eso sesgaba la senal de victoria, que es
                # justamente lo unico que el auto-juego venia a aportar.
                eps_r = torch.randn(1, N_MACRO)
                _nc = 1 + getattr(_net, "n_ops", 0)
                eps_u = torch.randn(10, 10) if _nc == 1 else torch.randn(_nc, 10, 10)
                # UN SIGMA POR CANAL, igual que el entrenador. Antes aqui se
                # usaba 0.15 plano para los 16 canales, y el 0.15 esta
                # calibrado para el canal de VALOR en dolares-symlog: aplicado
                # a los logits de verbo son CINCO veces el sigma que usamos
                # nosotros (0.03), y con ese ruido el verbo es 96 % ruido -lo
                # dice el propio docstring de `sigma_ops`-. O sea que el rival
                # congelado no era una copia nuestra: eramos nosotros con la
                # cabeza de verbos lobotomizada. Medido: marcaba 1.413 $ contra
                # los 2.048 de la politica de la que era copia.
                _sg = float(cfgd.get("sigma_micro", 0.15))
                if _nc > 1:
                    _sg = torch.full((_nc, 1, 1),
                                     float(cfgd.get("sigma_ops", 0.03)))
                    _sg[0] = float(cfgd.get("sigma_micro", 0.15))
                estado = {"dia": None, "mapa": None}

                def jugar(ob):
                    # El DIA lo da el motor en la observacion. Antes esto era
                    # `int(ob["step"]) // 24` con el 24 cableado: en cualquier
                    # celda que no fuese de 24 turnos/dia el rival congelado
                    # cambiaba de decision cada DOS dias de juego y jugaba
                    # mutilado. Medido en 12h x 10d: una copia exacta de una
                    # politica que marca 2.089 $ sacaba 1.348.
                    d_ = int(ob["day"])
                    if estado["dia"] != d_:
                        g, b = _O.encode_obs(ob)
                        with torch.no_grad():
                            s_ = _net(torch.from_numpy(g).unsqueeze(0),
                                      torch.from_numpy(b).unsqueeze(0))
                            am, _ = _net.macro_from(s_, eps_r)
                            estado["mapa"] = (s_["micro"][0]
                                              + _sg * eps_u).numpy()
                        ag.macro = _Mac.from_vector(am[0].numpy())
                        from .environment import _parte_micro
                        ag.micro = lambda o2: _parte_micro(estado["mapa"])
                        estado["dia"] = d_
                    return ag(ob)
                return jugar

            env.set_rival_policy(fabrica)
            conn.send(True)
        elif cmd == "rival_tope":
            from .environment import public_with_cap as _pct, load_public as _cp
            _n = "v48-fast-routes"
            env.set_rival_policy(
                (lambda: _cp(_n)) if datos is None else (lambda t=datos: _pct(_n, t)))
            conn.send(True)
        elif cmd == "rival_macro":
            env.set_rival_macro(datos)
            conn.send(True)
        elif cmd == "olvida_resultados":
            # Vacia el historial de victorias. Hace falta al PROMOCIONAR el
            # rival: `win_rate` promedia los ultimos 60 episodios, que son
            # todos victorias contra el rival VIEJO, asi que sin esto el
            # umbral se vuelve a cruzar en la comprobacion siguiente y
            # promociona en cascada.
            env.resultados.clear()
            conn.send(True)
        elif cmd == "nivel":
            env.level = datos
            conn.send(True)
        elif cmd == "cerrar":
            conn.close()
            return


class EntornoParalelo:
    """Misma interfaz que `EntornoDia`, repartida entre procesos."""

    def __init__(self, n_envs, n_procs=8, steps=720, seed0=1, macro=None, level=2,
                 mode="residuo", tope_peones=None, hours=None):
        self.n_procs = min(n_procs, n_envs)
        self.n = n_envs
        # HORIZONTE POR TRABAJADOR. `pasos` puede ser una lista: cada proceso
        # juega partidas de una longitud distinta y el lote de un update mezcla
        # los tres. Es posible porque `spec.EPISODE_STEPS` es global POR
        # PROCESO, y los trabajadores son procesos aparte.
        #
        # Por que importa que sea MEZCLADO y no secuencial: en secuencia la
        # politica entrena solo a 5 dias, luego solo a 8... y olvida, porque
        # nada en el gradiente le pide recordar. Mezclado, un unico gradiente
        # tiene que servir para los tres horizontes a la vez, y la señal de
        # horizonte en la observacion (obs.py, bloque "time") le permite
        # condicionar en vez de promediar.
        # `horas` y `tope_peones` admiten lista por el mismo motivo: `spec` y
        # `_M.TOPE_PEONES` son globales POR PROCESO. Asi un lote puede mezclar
        # rejilla entera -horas, dias y tope a la vez-, que es lo que las dos
        # dimensiones absolutas de la observacion (EPISODE_STEPS/720 y
        # tope/HANDS_REF) permiten condicionar en vez de promediar.
        def _reparte(v):
            if isinstance(v, (list, tuple)):
                return [v[i % len(v)] for i in range(self.n_procs)]
            return [v] * self.n_procs

        self.pasos_proc = [int(x) for x in _reparte(steps)]
        self.horas_proc = _reparte(hours)
        self.tope_proc = _reparte(tope_peones)

        # REPARTO POR COSTE, no por cabeza. Con una rejilla heterogenea un
        # peldano de 720 turnos cuesta 30 veces uno de 24, y repartir los
        # episodios a partes iguales deja a diez trabajadores esperando al
        # lento en CADA update. Se dan episodios en proporcion inversa a su
        # longitud, asi que todos tardan mas o menos lo mismo y el lote sigue
        # sumando `n_envs`. Minimo uno por trabajador: un peldano sin episodios
        # no aporta gradiente y la red dejaria de ver esa escala.
        peso = [1.0 / p for p in self.pasos_proc]
        total = sum(peso)
        self.por_proc = [max(1, int(n_envs * w / total)) for w in peso]
        sobran = n_envs - sum(self.por_proc)
        i = 0
        while sobran != 0:                     # reparte el resto por peso
            k = max(range(self.n_procs), key=lambda j: peso[j] / self.por_proc[j])
            if sobran > 0:
                self.por_proc[k] += 1; sobran -= 1
            else:
                k = min((j for j in range(self.n_procs) if self.por_proc[j] > 1),
                        key=lambda j: peso[j], default=None)
                if k is None:
                    self.n = sum(self.por_proc)
                    break
                self.por_proc[k] -= 1; sobran += 1
            i += 1
            if i > 10 * n_envs:
                break
        self.n = sum(self.por_proc)
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        off = 0
        vec = list(macro.to_vector()) if hasattr(macro, "a_vector") else list(macro)
        for k, m in enumerate(self.por_proc):
            padre, hijo = ctx.Pipe()
            # Semillas DISJUNTAS por trabajador: si se solapan, varios procesos
            # juegan la misma partida y el lote deja de ser independiente.
            p = ctx.Process(target=_trabajador,
                            args=(hijo, m, self.pasos_proc[k], seed0, vec,
                                  (level[k % len(level)]
                                   if isinstance(level, (list, tuple)) else level),
                                  mode, self.tope_proc[k],
                                  self.horas_proc[k], off, n_envs),
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

    def encode(self):
        for c in self.conns:
            c.send(("codifica", None))
        partes = [c.recv() for c in self.conns]
        return (np.concatenate([p[0] for p in partes]),
                np.concatenate([p[1] for p in partes]),
                np.concatenate([p[2] for p in partes]))

    def step_day(self, mapas=None, macros=None):
        off = 0
        for c, m in zip(self.conns, self.por_proc):
            c.send(("paso", (None if mapas is None else mapas[off:off + m],
                             None if macros is None else macros[off:off + m])))
            off += m
        rec, fin, masc = [], [], []
        for k, c in enumerate(self.conns):
            r, f, wr, din, util, riv, sinliq, ms = c.recv()
            rec.append(r); fin.append(f)
            if ms is not None:
                masc.append(ms)
            self._wr[k], self._din[k], self._util[k] = wr, din, util
            self._riv[k], self._sinliq[k] = riv, sinliq
        self._masc = np.concatenate(masc) if masc else None
        return np.concatenate(rec), np.concatenate(fin)

    def _etiqueta(self, k):
        """Nombre del peldano del trabajador k: "8h x 13d".

        Antes se etiquetaba con `pasos // 24`, que da por hecho 24 horas al
        dia. Con la rejilla las horas cambian por trabajador, asi que esa
        etiqueta juntaba peldanos distintos bajo el mismo nombre: `5h x 21d` y
        `8h x 13d` son 105 y 104 pasos, o sea "4 dias" los dos.
        """
        h = self.horas_proc[k] or 24
        return f"{h}h x {self.pasos_proc[k] // h}d"

    def dinero_por_horizonte(self):
        """{dias: dinero medio} en vez de un solo promedio.

        Promediar dinero sobre horizontes distintos NO significa nada: una
        partida de 15 dias gana menos que una de 30 por construccion, asi que
        la media sube o baja segun que mezcla de episodios haya cerrado, no
        segun lo bien que juegue la politica.
        """
        out = {}
        for k, steps in enumerate(self.pasos_proc):
            d = self._din[k]
            if d is None or (isinstance(d, float) and d != d):
                continue
            out.setdefault(self._etiqueta(k), []).append(float(d))
        return {k: sum(v) / len(v) for k, v in out.items() if v}

    def rival_por_horizonte(self):
        """{dias: dinero medio del RIVAL} desglosado, no promediado."""
        out = {}
        for k, steps in enumerate(self.pasos_proc):
            r = self._riv[k]
            if not isinstance(r, dict):
                continue
            d = r.get("dinero")
            if d is None or d != d:
                continue
            out.setdefault(self._etiqueta(k), []).append(float(d))
        return {k: sum(v) / len(v) for k, v in out.items() if v}

    def masks(self):
        return getattr(self, "_masc", None)

    def set_selfplay(self, net):
        """Congela la politica actual como rival en todos los trabajadores."""
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        cfgd = dict(vars(net.cfg))
        for c in self.conns:
            c.send(("autojuego", (sd, cfgd)))
        for c in self.conns:
            c.recv()

    def set_selfplay_on(self, idxs, net):
        """Congela la politica como rival SOLO en los trabajadores dados.

        `pon_autojuego` es global: sustituye el rival de los once peldanos a la
        vez, asi que no sirve para el relevo POR MERITO, que es por peldano.
        Y sin una version por peldano el relevo tenia que caer en
        `pon_rival_macro`, que manda un VECTOR: nuestro ejecutor con la
        heuristica de tablero y sin cabeza micro.

        Eso no es auto-juego. Medido con el MISMO vector macro en los dos
        lados, 5 semillas de 24h x 30d: la red hace 64.762 $ y el rival de
        vector 29.924, o sea +116 % y 5 de 5. El rival valia la mitad que
        nosotros por construccion, y por eso el relevo lo sustituia una y otra
        vez -105 veces en la run de la noche- sin que dejase de perder.
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
        """Rival = v48 con ese tope de peones (None = sin tope).

        Se fija explicito en vez de por `nivel`, porque la escalera de ligas
        muestrea topes (3/5/8/11) que no coinciden con los peldanos de ESCALERA.

        Admite LISTA, un tope por trabajador, igual que `horas` y `pasos`: en
        una rejilla el tope del rival es el tercer eje del peldano y tiene que
        variar con el. Con un solo valor se aplica a todos, como antes.
        """
        vals = (list(cap) if isinstance(cap, (list, tuple))
                else [cap] * self.n_procs)
        for k, c in enumerate(self.conns):
            c.send(("rival_tope", vals[k % len(vals)]))
        for c in self.conns:
            c.recv()

    def set_rival_macro(self, vector):
        """Propaga el rival-experto a los trabajadores (procesos aparte).

        Admite LISTA, un vector por trabajador, y `None` en una posicion deja
        ese trabajador con el agente publico.

        Por que hace falta por trabajador. Medido el 2026-09-21: `v48` hace
        60.426 $ a 24h x 30d y practicamente CERO a cualquier otra escala
        (13h x 21d: 0 $; 8h x 13d: 0 $; 6h x 6d: 7 $). Es un reproductor de
        cinta indexado por paso, construida para 719 pasos a 24 horas al dia;
        al cambiar horas o dias la cinta se desincroniza y el agente compra
        cuando tocaba cosechar. O sea que en toda la escalera reducida NO HAY
        adversario: el termino de victoria es gratis y el mercado compartido no
        lo vacia nadie.

        Con un vector por peldano, el rival de los peldanos reducidos pasa a
        ser NUESTRO ejecutor con el macro que el CEM encontro para esa escala
        -que si juega ahi, y llega al techo medido- mientras el peldano de
        competicion conserva a v48.
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
        """Admite LISTA, un nivel por trabajador.

        Por que hace falta: el tope de peones gradua la DIFICULTAD del rival,
        pero los cuatro peldanos siguen siendo el mismo agente. Y los publicos
        no se parecen -medido: el que mas nos estorba es v16-rc5 (nos deja
        36.139 $) aunque v48 puntue mas (153.720)-. Entrenar contra un solo
        estilo arriesga aprender a batir a ESE, no a jugar.
        """
        vals = (list(level) if isinstance(level, (list, tuple))
                else [level] * self.n_procs)
        for k, c in enumerate(self.conns):
            c.send(("nivel", vals[k % len(vals)]))
        for c in self.conns:
            c.recv()

    def _media(self, v):
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    def forget_results(self):
        """Vacia el historial de victorias en todos los trabajadores y en el
        agregado del padre. Lo usa la promocion de rival: contra el rival nuevo
        la tasa tiene que medirse desde cero, no arrastrar la del anterior."""
        for c in self.conns:
            c.send(("olvida_resultados", None))
        for c in self.conns:
            c.recv()
        self._wr = [float("nan")] * self.n_procs

    def rival_per_rung(self):
        """Dinero del rival en CADA peldano. El dato ya se recibia por
        trabajador (`_riv[k]`) y no se exponia: sin el no se puede comprobar
        que la escalera este graduada ni que cada peldano aporte algo."""
        out = []
        for d in self._riv:
            try:
                out.append(float(d.get("dinero", float("nan"))))
            except Exception:
                out.append(float("nan"))
        return out

    def win_rate_per_rung(self):
        """Tasa de victoria de CADA trabajador, o sea de cada peldano.

        Cada trabajador juega un peldano distinto de la escalera, asi que su
        `_wr` es exactamente lo exprimido que esta ese peldano. Con la media
        global esto no se ve: un peldano ganado al 100 % y otro perdido al
        100 % dan lo mismo que dos al 50 %, y son situaciones opuestas.
        """
        return list(self._wr)

    def win_rate(self, ultimos=60):
        return self._media(self._wr)

    def mean_money(self, ultimos=50):
        return self._media(self._din)

    def mean_useful(self, ultimos=400):
        return self._media(self._util)

    def rival_stats(self, ultimos=50):
        vs = [r for r in self._riv if r]
        if not vs:
            return {"dinero": float("nan"), "cultivos": float("nan"),
                    "animales": float("nan"), "unidades": float("nan")}
        return {k: self._media([r[k] for r in vs]) for k in vs[0]}

    def mean_unsold(self, ultimos=50):
        return self._media(self._sinliq)

    def cerrar(self):
        for c in self.conns:
            try:
                c.send(("cerrar", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)
