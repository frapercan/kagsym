#!/usr/bin/env python
"""Evaluacion DETERMINISTA de UN checkpoint, con semillas reservadas.

POR QUE EXISTE. El ancla que se imprime durante el entrenamiento mide la
politica MUESTREADA, sobre la distribucion de entrenamiento. Lo que se
despliega es la MEDIA, sin ruido. MEDIDO el 2026-09-24: ancla 33.279 $ contra
47.608 $ de la vara determinista -un 30% de diferencia- mientras que en otra
politica, la noche anterior, las dos coincidian. **La calibracion entre ambas
no es estable**, asi que guardar por el ancla es guardar por un proxy que se
descalibra.

Esto corre unos pocos episodios sin ruido, contra el agente publico sin capar,
en semillas FIJAS y distintas de las del entrenamiento, e imprime una linea
que el entrenador parsea. Es el mismo camino de codigo que la vara pareada,
pero para un solo checkpoint y barato.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H, D, CASH = 24, 30, 3000
SEMILLAS = list(range(7101, 7301))     # reservadas: nunca se entrena en ellas


def _uno(args):
    ckpt, s = args
    import torch
    torch.set_num_threads(1)
    from kagsym import obs as O, spec
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap, _split_micro
    from kagsym.reward import ProductionLedger
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.migrate_ckpt import load_tolerant
    from kagsym.nets.world import E2EAgent, WorldConfig
    import kagsym.macro as _M
    global _CACHE
    try:
        net = _CACHE[ckpt]
    except Exception:
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = WorldConfig(**d["cfg"]) if isinstance(d["cfg"], dict) else d["cfg"]
        net = E2EAgent(cfg)
        load_tolerant(net, d["sd"], ckpt, verbose=False,
                      macro_fields=d.get("macro_fields"))
        net.eval()
        _CACHE[ckpt] = net
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.HAND_CAP = None
    env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                 "startingMoney": CASH}, seed=s)
    o = env.reset()
    ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
    rv = public_with_cap("v48-fast-routes", 10 ** 6)
    st = {"dia": None, "mapa": None}
    ag.micro = lambda ob: _split_micro(st["mapa"])
    # CONTAR PRODUCTOS AQUI, con el MISMO ledger que usa el entrenamiento.
    # La metrica `6_producto/*` del panel sale de UN solo trabajador ancla:
    # con episodios de sd 22.526 $, un cambio de 65 unidades de leche puede
    # ser ruido. Aqui son 100 episodios deterministas, asi que convierte una
    # sospecha -"se esta volviendo una granja de melon"- en una medida.
    led = ProductionLedger()
    with torch.no_grad():
        while not env.done:
            ob = o[0]
            if st["dia"] != int(ob["day"]):
                g, b = O.encode_obs(ob, getattr(ag, "_destinations", None))
                hf = np.asarray(O.rival_flow(ob), dtype=np.float32)
                out = net(torch.from_numpy(g).unsqueeze(0),
                          torch.from_numpy(b).unsqueeze(0),
                          torch.from_numpy(hf).unsqueeze(0))
                # DETERMINISTA: la media, sin ruido. Es lo que se despliega.
                ag.macro = Macro.from_vector(
                    torch.sigmoid(out["macro_mu"])[0].numpy())
                st["mapa"] = out["micro"][0].numpy()
                st["dia"] = int(ob["day"])
            _a = ag(ob)
            led.harvested(ob, _a, 0)
            led.sold(ob, _a)
            o, _ = env.step([_a, rv(o[1])])
    m = env.rewards()
    return float(m[0]), float(m[1]), dict(led.vendidas), dict(led.ingreso)


_CACHE = {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=200)   # las 200 reservadas: el docstring lo decia y el default lo desmentia
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    import multiprocessing as mp
    ss = SEMILLAS[:a.n]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(_uno, [(a.ckpt, s) for s in ss])
    nos = np.array([r[0] for r in res]); riv = np.array([r[1] for r in res])
    _uds, _ing = {}, {}
    for r in res:
        for k, v in (r[2] or {}).items():
            _uds[k] = _uds.get(k, 0.0) + v / len(res)
        for k, v in (r[3] or {}).items():
            _ing[k] = _ing.get(k, 0.0) + v / len(res)
    # WIN RATE ADEMAS DEL DINERO. La clasificacion final es un ajuste
    # Bradley-Terry sobre victorias/derrotas, no sobre margen: un competidor
    # perdio 103.148 a 103.147 y se llevo la penalizacion entera. Medido por
    # un competidor en la banda 2250+: la mediana de diferencia entre los dos
    # jugadores son 177 $ sobre bancos de 98k, y el 40% de las partidas se
    # decide por menos de 100 $.
    #
    # NO es nuestro criterio TODAVIA -perdemos por ~70.000 $, asi que ganar y
    # ganar por mucho son lo mismo- pero se reporta desde ya para poder
    # cambiar cuando la diferencia mediana baje de ~1.000 $, que es la señal.
    _gana = (nos > riv).astype(float) + 0.5 * (nos == riv)
    _dif = nos - riv
    d = {"dinero": float(nos.mean()),
         "win": float(_gana.mean()),
         "dif_mediana": float(np.median(_dif)),
         "cerca_1000": float((np.abs(_dif) < 1000).mean()),
         "se": float(nos.std(ddof=1) / len(nos) ** 0.5) if len(nos) > 1 else 0.0,
         "rival": float(riv.mean()),
         "margen_pct": float(100.0 * (nos.mean() - riv.mean()) / riv.mean()),
         "n": len(ss), "seg": round(time.time() - t0, 1),
         "uds": {k: round(v, 1) for k, v in sorted(_uds.items(), key=lambda x: -x[1])},
         "ing": {k: round(v, 0) for k, v in sorted(_ing.items(), key=lambda x: -x[1])}}
    print("EVAL " + json.dumps(d), flush=True)
    print(f"     win {d['win']:.3f}  mediana de diferencia {d['dif_mediana']:+.0f} $  "
          f"partidas a menos de 1.000 $: {100*d['cerca_1000']:.0f}%", flush=True)
    _v48 = {"STRAWBERRY": 430, "MILK": 335, "FERTILIZER": 397, "WHEAT": 313,
            "WOOL": 179, "MELON": 72}
    print("     producto      unidades   ingreso   (v48)")
    for k, u in list(d["uds"].items())[:8]:
        print(f"     {k.lower():12s} {u:8.1f} {d['ing'].get(k,0):9.0f}"
              + (f"   ({_v48[k]})" if k in _v48 else ""), flush=True)
    if a.json:
        json.dump(d, open(a.json, "w"))


if __name__ == "__main__":
    main()
