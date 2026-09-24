"""Busqueda CEM sobre el punto de operacion del macro.

POR QUE ESTO Y NO ENTRENAR. Medido esta noche: PPO no mejora nada. Desde
partida_v2 (51.193 $) lo degrada -6.787 (t -4,96); desde prod (41.903 $) da
+904 (t +0,75, ruido); y `ret`, su propio objetivo, lleva 420 updates plano.
Ademas el sesgo del macro tiene senal/ruido 0,074: CERO de 67 diales son
resolubles por gradiente. Lo que el gradiente no puede mover, una busqueda si.

QUE SE BUSCA. Un desplazamiento ADITIVO `delta` en espacio logit sobre lo que
emite la red: v = sigmoid(macro_mu + delta). Aditivo y no absoluto porque
condicionar el macro al estado vale 16.097 $ (t -10,3) y un vector constante
lo tiraria. delta=0 reproduce la politica actual exactamente, asi que la
busqueda arranca en la linea base conocida.

SOLO LOS VIVOS. `tools/diales_vivos.py` mide que 18 de los 67 dan episodios
identicos hasta el digito en ambos extremos (49 vivos; eran 48/19 cuando el
audit se hizo con el historico a ceros) -seis de ellos porque el mapa de
valor de la red sustituye el valor en dolares de la heuristica-. Una dimension
muerta en una busqueda no es neutra: es ruido que diluye a las que si operan.

SEMILLAS COMUNES. Todos los candidatos de una generacion se evaluan en LAS
MISMAS semillas. Sin esto la seleccion mide el tablero y no el candidato: con
sd 22.526 $/semilla, elegir sobre 12 semillas independientes tiene un sesgo de
ganador medido en +10.708 $. Emparejado, el efecto del tablero se cancela.

Y LA VALIDACION ES APARTE. Lo que salga se mide en las semillas reservadas
7101-7300, que la busqueda no ha visto. El optimo sobre las de busqueda no es
un resultado; es una hipotesis.

    .venv312/bin/python tools/cem_diales.py [checkpoint] [generaciones]
"""
import sys, os, json, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

CK   = sys.argv[1] if len(sys.argv) > 1 else "runs/partida_v2.pt"
GENS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
POP    = int(os.environ.get("CEM_POP", 32))
ELITE  = int(os.environ.get("CEM_ELITE", 8))
NSEM   = int(os.environ.get("CEM_SEM", 12))
SIGMA0 = float(os.environ.get("CEM_SIGMA", 0.6))
SALIDA = os.environ.get("CEM_OUT", "runs/delta_macro.npy")

VIVOS = [d for d, _ in json.load(open("runs/diales_vivos.json"))["vivos"]]
_RIVF = [0]   # fallos del rival, por proceso
_C = {}


def _red():
    import torch
    if "n" in _C:
        return _C["n"]
    from kagsym.nets.world import E2EAgent, WorldConfig
    from kagsym.migrate_ckpt import load_strict
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    c = ck.get("cfg") or {}
    net = E2EAgent(WorldConfig(device="cpu", con_ops=True,
                               n_keys=int(c.get("n_keys", 4)),
                               ctx_micro=str(c.get("ctx_micro", "3x3"))))
    load_strict(net, ck["sd"], CK, macro_fields=ck.get("macro_fields"))
    net.eval()
    _C["n"] = net
    return net


def _offset(delta, progreso, N_MACRO, torch, np):
    """Delega en `kagsym.rampa.offset`, que es la UNICA definicion.

    Estaba aqui duplicada. El agente que se sube no podia reproducirla, asi
    que la busqueda optimizaba algo indesplegable; y una segunda definicion es
    exactamente como nacio el fallo del historico a ceros. Ahora la busqueda,
    el validador y `submit_kagsym/main.py` importan la misma funcion.
    """
    from kagsym.rampa import offset as _off
    return torch.from_numpy(_off(delta, VIVOS, progreso, N_MACRO))


def episodio(seed, delta):
    """Un episodio con el desplazamiento aplicado. `delta` en indices VIVOS."""
    import torch; torch.set_num_threads(1)
    from kagsym import obs as O, spec
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap, _split_micro as _sm
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.nets import world as Mw
    H, D = 24, 30
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D)
    # HAND_CAP es un global POR PROCESO que se hereda por fork, y
    # `train.py` lo asigna. Hoy es None y esto es inerte, pero lanzar la
    # busqueda desde un proceso que lo traiga con cap haria jugar TODOS
    # los episodios con manos capadas sin que la salida lo dijese.
    import kagsym.macro as _MM; _MM.HAND_CAP = None
    net = _red()
    _dias = D
    env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                 "startingMoney": 3000}, seed=seed)
    o = env.reset()
    ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
    rv = public_with_cap("v48-fast-routes", 10 ** 6)
    day = None
    with torch.no_grad():
        while not env.done:
            ob = o[0]
            if day != ob["day"]:
                # `_destinations` TAMBIEN: el agente que se sube pasa
                # `getattr(ag,"_destinations",None)` y sin el la rejilla
                # no lleva a donde va cada unidad. Segunda diferencia
                # encontrada comparando este bucle con submit_kagsym.
                gr, b = O.encode_obs(ob, getattr(ag, "_destinations", None))
                # EL HISTORICO, REAL. Lo puse a ceros y costo una noche: el
                # entrenamiento (`environment.encode` -> `Hf[i]=rival_flow`) y
                # el agente que se sube (`submit_kagsym/main.py`) lo rellenan
                # los dos, asi que buscar con ceros optimiza una condicion que
                # no ocurre NUNCA. KAG_HIST=0 reproduce el fallo para medirlo.
                _hf = (torch.zeros(1, Mw.N_HIST) if os.environ.get("KAG_HIST") == "0"
                       else torch.from_numpy(np.asarray(O.rival_flow(ob), dtype=np.float32)).unsqueeze(0))
                out = net(torch.from_numpy(gr).unsqueeze(0),
                          torch.from_numpy(b).unsqueeze(0), _hf)
                off = (torch.zeros(N_MACRO) if delta is None else
                       _offset(delta, float(ob["day"]) / max(1, _dias), N_MACRO, torch, np))
                ag.macro = Macro.from_vector(
                    torch.sigmoid(out["macro_mu"][0] + off).numpy())
                ag.micro = (lambda _o, m=_sm(out["micro"][0].numpy()): m)
                day = ob["day"]
            a = ag(ob)
            try:
                a2 = rv(o[1])
            except Exception:
                # NUNCA EN SILENCIO: un rival roto es un rival debil. Medido
                # hoy: v48 lanza cero veces en 3.595 turnos.
                import traceback, sys as _s
                _RIVF[0] += 1
                if _RIVF[0] <= 3:
                    print(f"[cem] RIVAL LANZO ({_RIVF[0]}):", file=_s.stderr); traceback.print_exc()
                a2 = {"farmer": ["PASS"], "hands": [], "market": []}
            o, _ = env.step([a, a2])
    return float(env.rewards()[0])


def _t(t):
    i, seed, delta = t
    try:
        return i, seed, episodio(seed, delta)
    except Exception:
        return i, seed, float("nan")


if __name__ == "__main__":
    import multiprocessing as mp
    D = len(VIVOS) * (2 if os.environ.get("CEM_RAMPA") == "1" else 1)
    mu = np.zeros(D, dtype=np.float64)
    sd = np.full(D, SIGMA0)
    rng = np.random.default_rng(20260924)
    print(f"[cem] {CK}  {len(VIVOS)} diales vivos"
          + ("  RAMPA a+b*progreso (96 dim)" if D > len(VIVOS) else "")
          + f"  pop {POP}  elite {ELITE}  "
          f"{NSEM} semillas comunes  {GENS} generaciones", flush=True)
    mejor_mu, mejor_val = mu.copy(), -1e18
    hist = []
    with mp.Pool(12) as pool:
        for g in range(GENS):
            t0 = time.time()
            # semillas comunes, distintas cada generacion para no sobreajustar
            sem = [9000 + g * NSEM + k for k in range(NSEM)]
            # EL CENTRO COMO CANDIDATO. La media de la POBLACION es `mu` mas
            # ruido, y donde perturbar sale caro queda por debajo de la base
            # aunque `mu` este muy por encima: declare nulo un +6.524 $
            # (t +4,35) leyendo ese numero. Metiendo `mu` en la poblacion la
            # lectura intermedia mide el centro, que es lo que se valida.
            cand = [np.zeros(D), mu.copy()] + [rng.normal(mu, sd) for _ in range(POP - 2)]
            tareas = [(i, s, c) for i, c in enumerate(cand) for s in sem]
            res = pool.map(_t, tareas, chunksize=1)
            acc = collections.defaultdict(list)
            for i, s, v in res:
                acc[i].append(v)
            pun = np.array([np.nanmean(acc[i]) for i in range(len(cand))])
            orden = np.argsort(-pun)
            el = [cand[i] for i in orden[:ELITE]]
            mu = np.mean(el, axis=0)
            sd = np.std(el, axis=0) + 0.05
            base = pun[0]                      # delta=0 en las MISMAS semillas
            centro = pun[1]                    # `mu` en esas mismas semillas
            if pun[orden[0]] > mejor_val:
                mejor_val, mejor_mu = pun[orden[0]], cand[orden[0]].copy()
            hist.append((g, float(base), float(centro), float(pun[orden[0]]), float(pun.mean())))
            np.save(SALIDA, mejor_mu)
            np.save(SALIDA.replace(".npy", "_mu.npy"), mu)
            print(f"  gen {g:3d}  base {base:9,.0f}  CENTRO {centro:9,.0f} "
                  f"({centro-base:+8,.0f})  mejor {pun[orden[0]]:9,.0f}  "
                  f"media {pun.mean():9,.0f}  "
                  f"|sd| {sd.mean():.3f}  {time.time()-t0:.0f}s", flush=True)
    json.dump(hist, open(SALIDA.replace(".npy", "_hist.json"), "w"))
    print(f"\n  -> {SALIDA}   (validar en semillas reservadas, no en estas)")
