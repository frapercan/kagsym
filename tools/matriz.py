#!/usr/bin/env python
"""Matriz de enfrentamientos entre NUESTROS propios checkpoints.

POR QUE ESTOS Y NO LOS PUBLICOS. Los 77 publicos estan todos por encima:
batimos a 1-2 y de esos dos se caen solos. No hay banda donde ganemos el 50%,
asi que como termometro de progreso no sirven. Nuestras instantaneas SI estan
a nuestro nivel por construccion, y ademas son modelos bajo control: sabemos
de que linaje vienen y de que update.

QUE BUSCA. Dos cosas que un numero agregado esconde:

  * CICLOS. Entre los publicos fuertes hay tripletes no transitivos medidos por
    terceros -A gana a B el 100%, B a C el 85%, C a A el 100%-. Si los hay
    entre los nuestros, cualquier ranking escalar -Elo incluido- es una
    ficcion, y ya medimos que el Elo del pool es una cinta de correr.
  * QUIEN ENSEÑA. El rival util es aquel contra el que estas cerca del 50%:
    p(1-p) maxima. Con la matriz se elige por medida en vez de por Elo.

TODO DETERMINISTA y en los DOS ASIENTOS: parear cancela el tablero, y jugar
los dos lados cancela cualquier ventaja posicional (medida: -522 +- 535, o
sea ninguna, pero cuesta cero comprobarlo).
"""
import argparse, itertools, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H, D, CASH = 24, 30, 3000
SEMILLAS = list(range(7401, 7601))      # distintas de las de `evalua.py`
_C = {}


def _carga(ck):
    import torch
    if ck not in _C:
        from kagsym.migrate_ckpt import load_tolerant
        from kagsym.nets.world import E2EAgent, WorldConfig
        d = torch.load(ck, map_location="cpu", weights_only=False)
        n = E2EAgent(WorldConfig(**d["cfg"]) if isinstance(d["cfg"], dict) else d["cfg"])
        load_tolerant(n, d["sd"], ck, verbose=False,
                      macro_fields=d.get("macro_fields"))
        n.eval(); _C[ck] = n
    return _C[ck]


def _duelo(args):
    ca, cb, sem, asiento = args
    import torch
    torch.set_num_threads(1)
    from kagsym import obs as O, spec
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import _split_micro
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    import kagsym.macro as _M
    try:
        spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.HAND_CAP = None
        nets = [_carga(ca), _carga(cb)]
        env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                     "startingMoney": CASH}, seed=sem)
        o = env.reset()
        # asiento=0: A en el puesto 0. asiento=1: A en el puesto 1.
        quien = [0, 1] if asiento == 0 else [1, 0]   # puesto -> indice de net
        ags, sts = [], []
        for _ in range(2):
            st = {"dia": None, "mapa": None}
            ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
            ag.micro = (lambda s: (lambda ob: _split_micro(s["mapa"])))(st)
            ags.append(ag); sts.append(st)
        with torch.no_grad():
            while not env.done:
                acts = []
                for puesto in (0, 1):
                    ob = o[puesto]; ag = ags[puesto]; st = sts[puesto]
                    net = nets[quien[puesto]]
                    if st["dia"] != int(ob["day"]):
                        g, b = O.encode_obs(ob, getattr(ag, "_destinations", None))
                        hf = np.asarray(O.rival_flow(ob), dtype=np.float32)
                        out = net(torch.from_numpy(g).unsqueeze(0),
                                  torch.from_numpy(b).unsqueeze(0),
                                  torch.from_numpy(hf).unsqueeze(0))
                        ag.macro = Macro.from_vector(
                            torch.sigmoid(out["macro_mu"])[0].numpy())
                        st["mapa"] = out["micro"][0].numpy()
                        st["dia"] = int(ob["day"])
                    acts.append(ag(ob))
                o, _ = env.step(acts)
        m = env.rewards()
        # dinero de A y de B, independientemente del puesto
        ma = m[0] if asiento == 0 else m[1]
        mb = m[1] if asiento == 0 else m[0]
        return ca, cb, float(ma), float(mb), None
    except Exception as e:
        return ca, cb, float("nan"), float("nan"), f"{type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--n", type=int, default=4, help="partidas por par (mitad por asiento)")
    p.add_argument("--procs", type=int, default=4)
    a = p.parse_args()
    import multiprocessing as mp
    nom = [os.path.basename(c).replace(".pt", "")[:14] for c in a.ckpts]
    pares = list(itertools.combinations(range(len(a.ckpts)), 2))
    tareas = [(a.ckpts[i], a.ckpts[j], SEMILLAS[k // 2], k % 2)
              for i, j in pares for k in range(a.n)]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(_duelo, tareas)
    W = np.full((len(a.ckpts), len(a.ckpts)), np.nan)
    idx = {c: i for i, c in enumerate(a.ckpts)}
    acc = {}
    for ca, cb, ma, mb, err in res:
        if err:
            continue
        acc.setdefault((idx[ca], idx[cb]), []).append(
            1.0 if ma > mb else (0.5 if ma == mb else 0.0))
    for (i, j), v in acc.items():
        W[i, j] = float(np.mean(v)); W[j, i] = 1.0 - W[i, j]
    print(f"{len(a.ckpts)} checkpoints, {len(pares)} pares x {a.n} partidas, "
          f"{time.time()-t0:.0f}s\n")
    print("  tasa de victoria de la FILA contra la COLUMNA\n")
    print(" " * 16 + "".join(f"{n[:9]:>10s}" for n in nom))
    for i, n in enumerate(nom):
        fila = "".join("       -- " if i == j or W[i, j] != W[i, j]
                       else f"{W[i,j]:10.2f}" for j in range(len(nom)))
        print(f"  {n:14s}{fila}")
    # ranking por victorias, y CICLOS
    med = np.nanmean(np.where(np.eye(len(nom), dtype=bool), np.nan, W), axis=1)
    print("\n  ranking por tasa media:")
    for i in np.argsort(-med):
        print(f"    {nom[i]:16s} {med[i]:.3f}")
    ciclos = []
    for i, j, k in itertools.permutations(range(len(nom)), 3):
        if (W[i, j] > 0.5 and W[j, k] > 0.5 and W[k, i] > 0.5):
            ciclos.append((nom[i], nom[j], nom[k]))
    vistos = set(); unicos = []
    for c in ciclos:
        key = frozenset(c)
        if key not in vistos:
            vistos.add(key); unicos.append(c)
    print(f"\n  CICLOS no transitivos: {len(unicos)}")
    for c in unicos[:5]:
        print(f"    {c[0]} > {c[1]} > {c[2]} > {c[0]}")
    if not unicos:
        print("    ninguno -- el orden es transitivo y un ranking escalar vale")


if __name__ == "__main__":
    main()
