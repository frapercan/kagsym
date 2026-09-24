"""Cual de los 67 diales del macro OPERA realmente sobre la partida.

Motivacion: el RL no puede mover el sesgo del macro (senal/ruido 0,074, CERO
de 67 diales resolubles), asi que si algo va a moverlos es una busqueda. Pero
antes de gastar horas buscando sobre 67 dimensiones hay que saber cuantas
estan vivas: ya han aparecido cinco mecanismos cableados que no hacian nada,
y una dimension muerta en una busqueda no es neutra, es ruido que diluye.

Metodo: para cada dial, se juega el MISMO tablero con el dial forzado a 0,05
y a 0,95, dejando los otros 66 como los emite la red. Si el dinero y el
recuento de operaciones salen identicos hasta el digito, el dial no existe.

    .venv312/bin/python tools/diales_vivos.py [checkpoint] [n_semillas]
"""
import sys, os, collections, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CK = sys.argv[1] if len(sys.argv) > 1 else "runs/partida_v2.pt"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SEMILLAS = [7401 + k for k in range(NS)]

_CACHE = {}


def _red():
    import torch
    if "net" in _CACHE:
        return _CACHE["net"]
    from kagsym.nets.world import E2EAgent, WorldConfig
    from kagsym.migrate_ckpt import load_strict
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    c = ck.get("cfg") or {}
    net = E2EAgent(WorldConfig(device="cpu", con_ops=True,
                               n_keys=int(c.get("n_keys", 4)),
                               ctx_micro=str(c.get("ctx_micro", "3x3"))))
    load_strict(net, ck["sd"], CK, macro_fields=ck.get("macro_fields"))
    net.eval()
    _CACHE["net"] = net
    return net


def juega(seed, dial=None, val=None):
    """Un episodio completo. `dial` fijado a `val` tras la salida de la red."""
    import torch; torch.set_num_threads(1)
    import numpy as np
    from kagsym import obs as O, spec
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap, _split_micro as _sm
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.nets import world as Mw
    H, D = 24, 30
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D)
    net = _red()
    env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                 "startingMoney": 3000}, seed=seed)
    o = env.reset()
    ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
    rv = public_with_cap("v48-fast-routes", 10 ** 6)
    day = None; ops = collections.Counter()
    with torch.no_grad():
        while not env.done:
            ob = o[0]
            if day != ob["day"]:
                gr, b = O.encode_obs(ob)
                out = net(torch.from_numpy(gr).unsqueeze(0),
                          torch.from_numpy(b).unsqueeze(0),
                          torch.zeros(1, Mw.N_HIST))
                v = torch.sigmoid(out["macro_mu"])[0].numpy().copy()
                if dial is not None:
                    v[dial] = val
                ag.macro = Macro.from_vector(v)
                ag.micro = (lambda _o, m=_sm(out["micro"][0].numpy()): m)
                day = ob["day"]
            a = ag(ob)
            for op in ([a.get("farmer")] if a.get("farmer") else []) + list(a.get("hands") or []):
                if isinstance(op, (list, tuple)) and op:
                    ops[op[0]] += 1
            try: a2 = rv(o[1])
            except Exception: a2 = {"farmer": ["PASS"], "hands": [], "market": []}
            o, _ = env.step([a, a2])
    return float(env.rewards()[0]), dict(ops)


def _tarea(t):
    d, v, s = t
    try:
        return (d, v, s) + juega(s, d, v)
    except Exception as e:
        return (d, v, s, None, {"ERROR": repr(e)[:120]})


if __name__ == "__main__":
    import multiprocessing as mp
    from kagsym.macro import N_MACRO
    tareas = [(d, v, s) for d in range(N_MACRO) for v in (0.05, 0.95) for s in SEMILLAS]
    print(f"[diales] {CK}  {N_MACRO} diales x 2 extremos x {NS} semillas = {len(tareas)} episodios")
    with mp.Pool(12) as p:
        res = p.map(_tarea, tareas, chunksize=1)
    por = collections.defaultdict(dict)
    for d, v, s, din, ops in res:
        por[d][(v, s)] = (din, ops)
    vivos, muertos = [], []
    for d in range(N_MACRO):
        dif = 0.0; iguales = 0
        for s in SEMILLAS:
            a = por[d].get((0.05, s)); b = por[d].get((0.95, s))
            if not a or not b or a[0] is None or b[0] is None:
                continue
            if a[0] == b[0] and a[1] == b[1]:
                iguales += 1
            dif += abs(a[0] - b[0])
        dif /= max(1, len(SEMILLAS))
        (muertos if iguales == len(SEMILLAS) else vivos).append((d, dif))
    print(f"\n  VIVOS  {len(vivos)}/{N_MACRO}")
    print(f"  MUERTOS {len(muertos)}/{N_MACRO}: {[d for d, _ in muertos]}")
    vivos.sort(key=lambda x: -x[1])
    print("\n  los que mas mueven el dinero (|0,05 - 0,95|, media por semilla):")
    for d, dif in vivos[:15]:
        print(f"    dial {d:2d}   {dif:10,.0f} $")
    print("\n  los mas flojos de los vivos:")
    for d, dif in vivos[-8:]:
        print(f"    dial {d:2d}   {dif:10,.0f} $")
    json.dump({"vivos": vivos, "muertos": muertos}, open("runs/diales_vivos.json", "w"))
    print("\n  -> runs/diales_vivos.json")
