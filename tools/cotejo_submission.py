"""Conduce el agente REAL de la submission contra el motor y lo compara.

Esta es la comprobacion que hoy habria ahorrado una noche entera: la busqueda
media con un bucle propio que pasaba el historico a ceros y no pasaba
`_destinations`, y nadie lo noto porque nada cotejaba los dos caminos.

Aqui se ejecuta `submit_kagsym/main.py:agent` -el MISMO codigo que juega en
Kaggle- sobre `FastEnv`, y se compara con `tools/cem_diales.episodio`. Si los
dos no dan el MISMO dinero en la MISMA semilla, uno de los dos miente, y el
que importa es el de Kaggle.

    .venv312/bin/python tools/cotejo_submission.py [checkpoint] [n_semillas]
"""
import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

CK = sys.argv[1] if len(sys.argv) > 1 else "runs/partida_v4.pt"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.environ["KAGSYM_CKPT"] = os.path.abspath(CK)
os.environ.setdefault("CEM_OUT", "/dev/null")


def _submission():
    """Carga main.py como Kaggle: compile + exec, y coge el ultimo invocable."""
    ruta = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "submit_kagsym", "main.py")
    src = open(ruta).read()
    g = {}
    exec(compile(src, ruta, "exec"), g)
    ult = [v for v in g.values() if callable(v) and getattr(v, "__name__", "") == "agent"]
    if not ult:
        raise SystemExit("no se encontro `agent` en main.py")
    return ult[-1], g


def juega_submission(seed, ag_fn, g):
    import torch; torch.set_num_threads(1)
    from kagsym import spec
    from kagsym.environment import public_with_cap
    from kagsym.fastenv import FastEnv
    H, D = 24, 30
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D)
    g["_ESTADO"].update({"net": None, "ag": None, "dia": None, "mapa": None, "ep": None})
    conf = {"episodeSteps": H * D, "turnsPerDay": H, "startingMoney": 3000}
    env = FastEnv(configuration=conf, seed=seed)
    o = env.reset()
    rv = public_with_cap("v48-fast-routes", 10 ** 6)
    while not env.done:
        a = ag_fn(o[0], conf)
        try: a2 = rv(o[1])
        except Exception: a2 = {"farmer": ["PASS"], "hands": [], "market": []}
        o, _ = env.step([a, a2])
    return float(env.rewards()[0])


if __name__ == "__main__":
    sys.argv = [sys.argv[0], CK]
    import cem_diales as C
    C.CK = CK
    import torch
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    from kagsym.rampa import del_checkpoint
    delta, vivos = del_checkpoint(ck)
    print(f"[cotejo] {CK}   rampa en el checkpoint: "
          f"{'SI, %d dims' % len(delta) if delta is not None else 'no'}")
    ag_fn, g = _submission()
    iguales = 0
    for k in range(NS):
        s = 7101 + k
        a = juega_submission(s, ag_fn, g)
        b = C.episodio(s, delta if delta is not None else None)
        ok = abs(a - b) < 1e-6
        iguales += ok
        print(f"  semilla {s}   submission {a:10,.0f}   busqueda {b:10,.0f}   "
              f"{'IGUAL' if ok else 'DIFIEREN  dif %+.0f' % (a - b)}")
    print(f"\n  {iguales}/{NS} identicos")
    raise SystemExit(0 if iguales == NS else 1)
