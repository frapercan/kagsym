#!/usr/bin/env python
"""Un checkpoint contra UNA BANDA de agentes publicos, no contra uno.

POR QUE. La escalera de entrenamiento usa v48 con distintos topes de mano, y
ya esta medido que un v48 capado es OTRO agente, no una version debil: no es
una banda, es un agente y sus mutilaciones. Aqui hay 77 publicos en
`agents_pub/` y se usaban dos.

Y hay dos razones medidas por terceros para mirar la banda entera:

  * NO SON TRANSITIVOS. Entre los publicos fuertes hay ciclos limpios: A gana
    a B el 100%, B a C el 85%, C a A el 100%. Una tasa agregada los esconde.
  * HAY QUE SOBREVIVIR LA SUBIDA. Una submission arranca cerca de 600 y tiene
    que batir al campo de camino; un agente fuerte solo en la banda alta se
    atasca antes de llegar.

Reporta dinero Y victorias por rival, porque la clasificacion final es un
ajuste Bradley-Terry sobre victorias.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H, D, CASH = 24, 30, 3000
SEMILLAS = list(range(7101, 7301))
_C = {}


def _uno(args):
    ckpt, rival, s, asiento = args
    import torch
    torch.set_num_threads(1)
    from kagsym import obs as O, spec
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import load_public, _split_micro
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.migrate_ckpt import load_tolerant
    from kagsym.nets.world import E2EAgent, WorldConfig
    import kagsym.macro as _M
    try:
        if ckpt not in _C:
            d = torch.load(ckpt, map_location="cpu", weights_only=False)
            n_ = E2EAgent(WorldConfig(**d["cfg"]) if isinstance(d["cfg"], dict) else d["cfg"])
            load_tolerant(n_, d["sd"], ckpt, verbose=False,
                          macro_fields=d.get("macro_fields"))
            n_.eval(); _C[ckpt] = n_
        net = _C[ckpt]
        spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.HAND_CAP = None
        rv = load_public(rival)
        env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                     "startingMoney": CASH}, seed=s)
        o = env.reset()
        ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
        st = {"dia": None, "mapa": None}
        ag.micro = lambda ob: _split_micro(st["mapa"])
        yo, otro = asiento, 1 - asiento
        with torch.no_grad():
            while not env.done:
                ob = o[yo]
                if st["dia"] != int(ob["day"]):
                    g, b = O.encode_obs(ob, getattr(ag, "_destinations", None))
                    hf = np.asarray(O.rival_flow(ob), dtype=np.float32)
                    out = net(torch.from_numpy(g).unsqueeze(0),
                              torch.from_numpy(b).unsqueeze(0),
                              torch.from_numpy(hf).unsqueeze(0))
                    ag.macro = Macro.from_vector(torch.sigmoid(out["macro_mu"])[0].numpy())
                    st["mapa"] = out["micro"][0].numpy(); st["dia"] = int(ob["day"])
                acts = [None, None]
                acts[yo] = ag(ob); acts[otro] = rv(o[otro])
                o, _ = env.step(acts)
        m = env.rewards()
        return rival, float(m[yo]), float(m[otro]), None
    except Exception as e:
        # DE QUIEN ES EL FALLO. Este `try` envuelve TODO -checkpoint, rival,
        # 720 turnos, encode_obs, red y ejecutor-, y su salida se imprimia
        # entera bajo "rivales no cargaron (se ignoran)": un fallo NUESTRO se
        # reetiquetaba como problema del rival y se caia del denominador. Esa
        # es exactamente la forma del fallo que costo una noche: la medida
        # sale, parece sana, y describe otra cosa.
        import traceback
        _nuestro = not isinstance(e, (ImportError, ModuleNotFoundError, FileNotFoundError, KeyError))
        _et = "NUESTRO" if _nuestro else "carga del rival"
        if _nuestro:
            print(f"[banda] FALLO {_et} con {rival}:", file=sys.stderr)
            traceback.print_exc()
        return rival, float("nan"), float("nan"), f"[{_et}] {type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=4, help="semillas por rival")
    p.add_argument("--rivales", type=int, default=0, help="0 = todos")
    p.add_argument("--procs", type=int, default=4)
    a = p.parse_args()
    import multiprocessing as mp
    dirp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "agents_pub")
    nombres = sorted(f[:-3] for f in os.listdir(dirp) if f.endswith(".py"))
    if a.rivales:
        nombres = nombres[:a.rivales]
    # LOS DOS ASIENTOS, porque parear cancela la suerte del tablero y el
    # asiento; medido por terceros que el asiento no mueve el resultado, pero
    # cuesta cero comprobarlo aqui.
    tareas = [(a.ckpt, r, SEMILLAS[i // 2], i % 2)
              for r in nombres for i in range(a.n)]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(_uno, tareas)
    por = {}
    fallos = {}
    for r, mio, suyo, err in res:
        if err:
            fallos[r] = err
            continue
        por.setdefault(r, []).append((mio, suyo))
    filas = []
    for r, vs in por.items():
        mio = np.array([v[0] for v in vs]); suyo = np.array([v[1] for v in vs])
        w = float(((mio > suyo) + 0.5 * (mio == suyo)).mean())
        filas.append((r, w, mio.mean(), suyo.mean(), len(vs)))
    filas.sort(key=lambda x: -x[1])
    print(f"{a.ckpt}   {len(filas)} rivales x {a.n} partidas   "
          f"{time.time()-t0:.0f}s\n")
    print(f"{'rival':46s} {'win':>6s} {'nuestro':>9s} {'suyo':>9s}")
    for r, w, mi, su, n in filas:
        print(f"  {r[:44]:44s} {w:6.2f} {mi:9.0f} {su:9.0f}")
    ws = np.array([f[1] for f in filas])
    print(f"\n  RESUMEN: win medio {ws.mean():.3f}   "
          f"rivales batidos (>0,5): {int((ws > 0.5).sum())} de {len(ws)}")
    print(f"  banda donde ganamos ~50%: "
          f"{int(((ws > 0.35) & (ws < 0.65)).sum())} rivales")
    if fallos:
        print(f"\n  {len(fallos)} rivales no cargaron (se ignoran):")
        for r, e in list(fallos.items())[:3]:
            print(f"    {r[:40]}: {e[:60]}")


if __name__ == "__main__":
    main()
