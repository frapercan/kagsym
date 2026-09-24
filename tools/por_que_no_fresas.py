"""Que cultivo ELIGE el agente, y por que no es el que da el dinero.

Medido con v3 en las reservadas: fresa 18,4 unidades contra las 430 de v48,
trigo 38,2 contra 313, y zanahoria y tomate a CERO. La fresa sale a ~200 $ la
unidad, asi que esas 430 son 86.000 $: el hueco entero cabe en ese producto.

Esta herramienta no supone nada sobre por que. Registra, dia a dia, que
devuelve `target_crop` -la eleccion- y que se PLANTA de verdad, que no tienen
por que coincidir: la eleccion puede ser correcta y morir despues en la
legalidad, en el presupuesto de semilla o en las casillas sostenibles.

    .venv312/bin/python tools/por_que_no_fresas.py [checkpoint]
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CK = sys.argv[1] if len(sys.argv) > 1 else "runs/partida_v3.pt"
os.environ.setdefault("CEM_OUT", "/dev/null")
sys.argv = [sys.argv[0], CK]
import cem_diales as C
C.CK = CK


def traza(seed):
    import torch; torch.set_num_threads(1)
    from kagsym import obs as O, spec, macro as MA
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap, _split_micro as _sm
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.nets import world as Mw
    H, D = 24, 30
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D)
    net = C._red()
    env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                 "startingMoney": 3000}, seed=seed)
    o = env.reset()
    ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
    rv = public_with_cap("v48-fast-routes", 10 ** 6)
    day = None
    elegido = collections.Counter(); plantado = collections.Counter()
    tiles = []
    with torch.no_grad():
        while not env.done:
            ob = o[0]
            if day != ob["day"]:
                gr, b = O.encode_obs(ob)
                out = net(torch.from_numpy(gr).unsqueeze(0),
                          torch.from_numpy(b).unsqueeze(0),
                          torch.zeros(1, Mw.N_HIST))
                ag.macro = Macro.from_vector(torch.sigmoid(out["macro_mu"][0]).numpy())
                ag.micro = (lambda _o, m=_sm(out["micro"][0].numpy()): m)
                day = ob["day"]
                try:
                    tc = MA.target_crop(ob, ag.macro)
                    elegido[tc if isinstance(tc, str) else str(tc)] += 1
                except Exception as e:
                    elegido["ERROR:" + repr(e)[:40]] += 1
                try:
                    tiles.append(MA.target_tiles(ob, ag.macro))
                except Exception:
                    pass
            a = ag(ob)
            for op in ([a.get("farmer")] if a.get("farmer") else []) + list(a.get("hands") or []):
                if isinstance(op, (list, tuple)) and op and op[0] == "PLANT":
                    plantado[op[1] if len(op) > 1 else "?"] += 1
            try: a2 = rv(o[1])
            except Exception: a2 = {"farmer": ["PASS"], "hands": [], "market": []}
            o, _ = env.step([a, a2])
    return elegido, plantado, tiles


if __name__ == "__main__":
    import multiprocessing as mp, numpy as np
    SEM = [7101 + k for k in range(12)]
    with mp.Pool(12) as p:
        res = p.map(traza, SEM)
    E = collections.Counter(); P = collections.Counter(); T = []
    for e, pl, t in res:
        E.update(e); P.update(pl); T += t
    print(f"[{CK}]  {len(SEM)} episodios\n")
    print("  target_crop ELIGE (dias-episodio):")
    for k, v in E.most_common():
        print(f"    {str(k):16s} {v:5d}  ({100*v/max(1,sum(E.values())):4.1f}%)")
    print("\n  se PLANTA de verdad:")
    for k, v in P.most_common():
        print(f"    {str(k):16s} {v:5d}  ({100*v/max(1,sum(P.values())):4.1f}%)")
    print(f"\n  siembras por episodio: {sum(P.values())/len(SEM):.1f}")
    if T:
        print(f"  target_tiles: media {np.mean(T):.1f}  min {min(T)}  max {max(T)}")
