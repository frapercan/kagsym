"""PAIRED fixed yardstick: does the full model really beat the state of the art?

The plain yardstick compares two independent MEANS. With an sd of $13,300 per
seed that does not distinguish +6.6% from zero (see the statistical power note).
Here the SAME seed is played by both checkpoints and the per-seed difference is
what is read: the scenario's variance cancels and far less sample is needed for
the same standard error.

The seeds are fixed BEFORE looking at anything, and they differ from the
yardstick's 601-608 so that nothing tuned on those is rewarded.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H, D, CASH = 24, 30, 3000
# BASE DE SEMILLAS CONFIGURABLE. La vara usaba 9001+ y `evalua.py` 7101+, sin
# forma de alinearlas: dos instrumentos sobre conjuntos distintos dan cambios
# distintos para los mismos dos checkpoints -medido el 2026-09-24: -2.197 $ en
# 9001+ contra -9.025 $ en 7101+, una discrepancia de 6.800-. Con esto se puede
# repetir el pareado sobre las MISMAS semillas que la evaluacion interna.
_BASE = int(os.environ.get("KAG_SEED0", "9001"))
SEEDS = list(range(_BASE, _BASE + 200))
SEEDS = SEEDS[:int(os.environ.get("KAG_NSEEDS", len(SEEDS)))]
# Contador POR PROCESO de fallos del rival. Igual que `_RIV_FALLOS` en
# `environment.py`: con fork cada worker lleva el suyo, y lo que importa no es
# el total exacto sino que deje de ser invisible.
_RIV_FALLOS = [0]
PROCS = int(os.environ.get("KAG_PROCS", "6"))                            # training uses 11; do not choke it
# la cadencia se pide por lado con el sufijo "@turno" en la ruta


def _one(args):
    ckpt, s = args
    # Sufijos, separados por coma tras "@":
    #   turno        -> llamar a la red CADA TURNO (si no, una vez al dia)
    #   macro=<ruta> -> los diales macro salen de OTRO checkpoint
    # El segundo existe porque la cabeza macro NO se supervisa en clonacion:
    # lee del tronco, asi que al soltar el tronco emite diales entrenados para
    # una representacion que ya no existe. Esto separa "el micro clonado es
    # malo" de "el micro clonado se llevo el macro por delante".
    ckpt, _, _suf = ckpt.partition("@")
    _fl = dict((x.split("=", 1) + [True])[:2] for x in _suf.split(",") if x)
    # "turno" mueve LAS DOS cabezas a cadencia por turno; "turnomicro" y
    # "turnomacro" mueven solo una. Hace falta separarlas porque reemitir el
    # MACRO cada turno es un paseo aleatorio sobre la estrategia, y eso ya
    # esta medido aqui: resamplear el macro a diario costo 40.972 -> 18.967 $
    # (-54%). Sin separar, "por turno pierde" mezcla ese efecto con el del
    # mapa micro, que es lo que la cinta dice que hay que refinar.
    turno_mi = ("turno" in _fl) or ("turnomicro" in _fl)
    turno_ma = ("turno" in _fl) or ("turnomacro" in _fl)
    ck_macro = _fl.get("macro")
    # "@hist0" reproduce el medidor viejo -historico a ceros y sin
    # _destinations- para poder medir el sesgo en semillas pareadas.
    _hist0 = "hist0" in _fl
    # "@ruido" evalua la politica MUESTREADA en vez de la media. Sirve para
    # separar "la politica es mala" de "la politica solo funciona con ruido":
    # si el determinista se hunde y el muestreado no, lo aprendido vive en la
    # exploracion y no en la media, que es justo lo que se desplegaria.
    _ruido = "ruido" in _fl
    import torch
    # ONE THREAD PER PROCESS. Torch opens threads on its own: measured, each
    # worker took 1.6 cores, and added to training's 11 processes that was ~21
    # threads on 12 cores. The thrashing made the measurement undefined AND
    # slowed training down. With one thread, the work is ~800 s of CPU spread
    # across the pool's processes.
    torch.set_num_threads(1)
    from kagsym import obs as O, spec
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.migrate_ckpt import load_strict
    from kagsym.nets import world as M
    from kagsym.nets.world import E2EAgent, WorldConfig
    import kagsym.macro as _M
    global _CACHE
    try:
        net, ops = _CACHE[ckpt]
    except Exception:
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        # LA CONFIG DEL CHECKPOINT, no los defaults de hoy. `main.py`:106 hace
        # `WorldConfig(**d["cfg"])` y evalua.py tambien; aqui se reconstruia
        # desde la dataclass arrastrando solo `con_ops`. Para los campos que
        # cambian formas el fallo es ruidoso (load_strict revienta), pero para
        # los que no lo es NO: un checkpoint sin `log_sigma_micro` se rellena
        # con sigma_micro=0,15 por defecto en vez de la suya, que es justo la
        # sigma del modo @ruido. Hoy es inocuo -los cuatro checkpoints vivos
        # traen cfg igual al default y log_sigma_micro-, pero deja de serlo en
        # cuanto se mida uno de otra arquitectura, que es cuando mas importa.
        _c = d.get("cfg")
        if isinstance(_c, dict):
            _c = dict(_c); _c["device"] = "cpu"
            cfg = WorldConfig(**_c)
        elif _c is not None:
            cfg = _c
        else:
            cfg = WorldConfig(device="cpu", con_ops=True)
        net = E2EAgent(cfg)
        ops = bool(getattr(cfg, "con_ops", True))
        # LA RAMPA. Vive en `ck["delta_rampa"]`, no en los pesos, asi que este
        # bucle la ignoraba por completo: comparar un checkpoint con rampa
        # contra su base daba +0 $ +- 0 con sd 0 -las dos partidas IDENTICAS-,
        # y eso se lee como "no hay efecto" cuando en realidad es "no lo estoy
        # midiendo". Casi tira un resultado bueno por el motivo equivocado.
        from kagsym.rampa import del_checkpoint as _delck
        net._rampa, net._rampa_vivos = _delck(d)
        load_strict(net, d["sd"], ckpt, macro_fields=d.get("macro_fields"))
        net.eval()
        try:
            _CACHE[ckpt] = (net, ops)
        except NameError:
            _CACHE = {ckpt: (net, ops)}
    net_macro = net
    if ck_macro:
        try:
            net_macro = _CACHE[("macro", ck_macro)]
        except Exception:
            dm = torch.load(ck_macro, map_location="cpu", weights_only=False)
            net_macro = E2EAgent(WorldConfig(device="cpu",
                                             con_ops=bool((dm.get("cfg") or {}).get("con_ops", True))))
            load_strict(net_macro, dm["sd"], ck_macro, macro_fields=dm.get("macro_fields"))
            net_macro.eval()
            _CACHE[("macro", ck_macro)] = net_macro
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.HAND_CAP = None
    # EL HISTORICO, que este medidor llevaba a CEROS. `forward` lo rellena con
    # ceros si no llega, asi que toda evaluacion hecha aqui media una politica
    # ciega al flujo del rival -N_HIST = 4 x N_PRODUCTS- mientras que el
    # entrenamiento se lo daba lleno (`env.encode()` devuelve g, b, hf). Lo
    # mismo con `_destinations`, que nuestro `encode_obs` si recibe.
    # KAG_HIST=0 reproduce el comportamiento viejo para poder medir la
    # diferencia en semillas pareadas.
    _HIST_REAL = (not _hist0) and os.environ.get("KAG_HIST", "1") != "0"
    HIST = torch.zeros(1, M.N_HIST)
    env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                 "startingMoney": CASH}, seed=s)
    o = env.reset()
    ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
    rv = public_with_cap("v48-fast-routes", 10 ** 6)
    day = None
    with torch.no_grad():
        while not env.done:
            ob = o[0]
            # CADENCIA, pedida por lado con sufijos en la ruta. La cadencia diaria fue decision nuestra de RL (30
            # pasos por episodio en vez de 720), no un limite del motor: el
            # forward cuesta 6,30 ms a un hilo -4,54 s por los 720 turnos-
            # contra 1000 ms de presupuesto POR TURNO. Medido en las
            # repeticiones top, agregar el dia destruye la etiqueta: 2,16
            # verbos distintos por casilla-dia, techo de acuerdo 64,1%; por
            # turno son 1,02 verbos y el techo sube al 98,9%.
            #
            # PERO medido: mover prod entero a por turno cuesta -13.346 $
            # (t -21,5). Se entreno a cadencia diaria y su mapa es un PLAN DEL
            # DIA; reemitirlo desde observaciones de media tarde que nunca vio
            # es otra distribucion. Por eso existen turnomicro/turnomacro.
            # Mismo fichero y mismo camino de codigo para todas, a proposito.
            _nuevo = day != ob["day"]
            if _nuevo or turno_mi or turno_ma:
                gr, b = O.encode_obs(
                    ob, getattr(ag, "_destinations", None) if _HIST_REAL else None)
                _h = (torch.from_numpy(
                          np.asarray(O.rival_flow(ob), dtype=np.float32)).unsqueeze(0)
                      if _HIST_REAL else HIST)
                out = net(torch.from_numpy(gr).unsqueeze(0),
                          torch.from_numpy(b).unsqueeze(0), _h)
                if _ruido and _nuevo:
                    _epsr = torch.randn_like(out["macro_mu"])
                    _epsu = torch.randn_like(out["micro"])
                    out = dict(out)
                    out["macro_mu"] = out["macro_mu"] + net.log_sigma.exp() * _epsr
                    out["micro"] = out["micro"] + net.log_sigma_micro.exp() * _epsu
                if _nuevo or turno_ma:
                    _om = (out if net_macro is net else
                           net_macro(torch.from_numpy(gr).unsqueeze(0),
                                     torch.from_numpy(b).unsqueeze(0), _h))
                    _mm = _om["macro_mu"][0]
                    _rp = getattr(net_macro, "_rampa", None)
                    if _rp is not None:
                        from kagsym.rampa import offset as _offr
                        from kagsym.macro import N_MACRO as _NMv
                        _mm = _mm + torch.from_numpy(_offr(
                            _rp, getattr(net_macro, "_rampa_vivos", []) or [],
                            float(ob["day"]) / max(1, D), _NMv))
                    ag.macro = Macro.from_vector(torch.sigmoid(_mm).numpy())
                # ESTA GUARDA FALTABA. Sin ella "turnomacro" reescribia
                # tambien el mapa cada turno y media exactamente lo mismo que
                # "turno": -13.346 $ identico a cuatro cifras, que es como se
                # detecto. "turnomicro" si era correcto porque el macro si
                # estaba guardado.
                if _nuevo or turno_mi:
                    # `_split_micro`, EL CANONICO, no una particion a mano.
                    #
                    # Aqui habia `(mp[0], mp[1:])`, que mete TODO lo que no es
                    # el canal de valor como logits de verbo. Pero el mapa es
                    # `1 + N_OPS + 2K`: con 28 canales y N_OPS=19 sobran 8 que
                    # son CLAVES y CONSULTAS de asignacion, y se le estaban
                    # pasando al ejecutor como si fueran verbos.
                    #
                    # MEDIDO el 2026-09-24: el mismo checkpoint sobre las
                    # MISMAS semillas daba 37.840 $ por `tools/evalua.py` -que
                    # usa `_split_micro`- y 31.209 $ por esta vara. 6.631 $ de
                    # diferencia, y la vara es el medidor principal del
                    # proyecto desde que existen los canales de clave.
                    from kagsym.environment import _split_micro as _sm
                    mp = out["micro"][0].numpy()
                    ag.micro = (lambda _o, m=_sm(mp): m)
                day = ob["day"]
            try:
                act_r = rv(o[1])
            except Exception:
                # NUNCA EN SILENCIO. `environment.py`:563 ya decidio esto:
                # un rival roto se vuelve un rival debil, indistinguible de uno
                # al que ganamos, y el pareado no lo cancela porque afecta a
                # los dos lados por igual solo si falla igual en los dos.
                # Medido hoy: v48 lanza CERO veces en 3.595 turnos, asi que
                # esto es un seguro, no un parche. Si salta, la medida NO vale.
                import traceback
                _RIV_FALLOS[0] += 1
                if _RIV_FALLOS[0] <= 3:
                    print(f"[vara] RIVAL LANZO (fallo {_RIV_FALLOS[0]}):", file=sys.stderr)
                    traceback.print_exc()
                act_r = {"farmer": ["PASS"], "hands": [], "market": []}
            o, _ = env.step([ag(ob), act_r])
    r = env.rewards()
    return float(r[0]), float(r[1])


if __name__ == "__main__":
    import multiprocessing as mp
    A, B = sys.argv[1], sys.argv[2]
    t0 = time.time()
    print(f"PAIRED yardstick: {H}h x {D}d, cash {CASH}, against an uncapped "
          f"v48, {len(SEEDS)} seeds fixed in advance")
    for _l, _c in (("A", A), ("B", B)):
        print(f"  {_l} cadencia: "
              f"{'POR TURNO' if 'turno' in _c else 'diaria'}"
              + (f", macro de {_c.split('macro=')[1]}" if 'macro=' in _c else ""))
    print(f"  A = {A}\n  B = {B}\n")
    # PROGRESS. Without it there is no way to tell 40% from 95% done, and that
    # already cost throwing away a 15-minute measurement for lack of an ETA.
    def sweep(ck, tag):
        out = []
        with mp.Pool(PROCS) as pool:
            for i, r in enumerate(pool.imap(_one, [(ck, s) for s in SEEDS]), 1):
                out.append(r)
                if i % 20 == 0 or i == len(SEEDS):
                    print(f"    {tag}: {i}/{len(SEEDS)}  ({time.time()-t0:.0f}s)",
                          flush=True)
        return out
    ra = sweep(A, "A")
    rb = sweep(B, "B")
    a = np.array([x[0] for x in ra]); va = np.array([x[1] for x in ra])
    b = np.array([x[0] for x in rb]); vb = np.array([x[1] for x in rb])
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  {'':<10}{'ours':>10}{'v48':>10}{'margin%':>10}{'wins':>8}")
    print(f"  {'A':<10}{a.mean():>10.0f}{va.mean():>10.0f}"
          f"{100*(a-va).mean()/va.mean():>9.1f}%{int((a>va).sum()):>5}/{len(SEEDS)}")
    print(f"  {'B':<10}{b.mean():>10.0f}{vb.mean():>10.0f}"
          f"{100*(b-vb).mean()/vb.mean():>9.1f}%{int((b>vb).sum()):>5}/{len(SEEDS)}")
    print(f"\n  PAIRED DIFFERENCE A-B: {d.mean():+.0f} $  "
          f"+- {se:.0f} (se)   t = {d.mean()/max(1e-9, se):+.2f}")
    print(f"  sd per seed: {d.std(ddof=1):.0f} $   "
          f"A beats B on {int((d>0).sum())}/{len(SEEDS)}")
    verdict = ("A IS BETTER" if d.mean() > 2*se else
               "B IS BETTER" if d.mean() < -2*se else
               "INDISTINGUISHABLE at 2 se")
    print(f"\n  -> {verdict}   ({time.time()-t0:.0f}s)")
