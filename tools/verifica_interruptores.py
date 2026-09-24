"""¿Este interruptor OPERA? Comprobacion obligatoria al montar un mecanismo.

El 2026-09-23 cayeron CINCO mecanismos que existian, estaban cableados y no
hacian nada:

  jepa_proy / aux_rival     en su inicializacion en TODOS los checkpoints
  CHAIN_VALUE               constante 0,000416 -- el dial no existia en el macro
  _EZ (emparejamiento)      x0,94-1,15 contra un valor que llega a 4,8e8
  stickiness                x1,02, y su optimo documentado no se reproduce
  KAG_COORD=aditiva         era una IDENTIDAD ALGEBRAICA: row*(1+(EZ-1))

El ultimo lo cazo un episodio real porque dio dinero y operaciones IDENTICOS
hasta el digito. Eso es lo que hace esta herramienta, y por eso tiene que ser
parte del montaje y no un paso aparte: un interruptor que no cambia el juego
es peor que no tenerlo, porque se mide y se cree.

    .venv312/bin/python tools/verifica_interruptores.py
    .venv312/bin/python tools/verifica_interruptores.py KAG_COORD
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

# (variable, valor_a, valor_b, que hace, "juego" | "entrenamiento")
#
# OJO CON LA CATEGORIA. Esta herramienta prueba si cambia LA PARTIDA, asi que
# solo vale para interruptores que afectan a las ACCIONES. Los que afectan a la
# RECOMPENSA -- shaping, pesos de perdida -- salen "inertes" aunque funcionen
# perfectamente, porque en inferencia no hay recompensa. Marcarlos evita leer
# un falso positivo como un bug.
INTERRUPTORES = [
    ("KAG_COORD", "multiplicativa", "aditiva",
     "coordinacion sumando en escala del turno en vez de multiplicando", "juego"),
    ("KAG_CADENA", "", "1.0", "valor de los recados encadenados", "juego"),
    ("KAG_VALOR", "", "heuristica", "quien pone los dolares de la tarea", "juego"),
    ("KAG_POTENCIAL", "1", "0", "shaping potencial", "entrenamiento"),
    ("KAG_PHI_W", "1.0", "0.3", "peso del shaping", "entrenamiento"),
]
SEEDS = [int(x) for x in os.environ.get("KSEEDS", "9001,9002,9003").split(",")]
# El default apuntaba a `runs/prod.pt.ultimo`, de hace dos dias: un
# interruptor se declaraba inerte sobre una politica que ya no jugamos.
CK = os.environ.get("KCK", "runs/partida_v4.pt")


def _juega(seed):
    import torch; torch.set_num_threads(1)
    from kagsym import obs as O, spec
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap, _split_micro as _sm
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.nets.world import E2EAgent, WorldConfig
    from kagsym.migrate_ckpt import load_strict
    from kagsym.nets import world as Mw
    H, D = 24, 30
    spec.set_turns_per_day(H); spec.set_episode_steps(H*D)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    c = ck.get("cfg") or {}
    net = E2EAgent(WorldConfig(device="cpu", con_ops=True,
                               n_keys=int(c.get("n_keys", 4)),
                               ctx_micro=str(c.get("ctx_micro", "3x3"))))
    load_strict(net, ck["sd"], CK, macro_fields=ck.get("macro_fields")); net.eval()
    env = FastEnv(configuration={"episodeSteps": H*D, "turnsPerDay": H,
                                 "startingMoney": 3000}, seed=seed)
    o = env.reset()
    ag = Agent(episode_steps=H*D, macro=Macro.from_vector([0.5]*N_MACRO))
    rv = public_with_cap("v48-fast-routes", 10**6)
    day = None; ops = collections.Counter()
    with torch.no_grad():
        while not env.done:
            ob = o[0]
            if day != ob["day"]:
                gr, b = O.encode_obs(ob, getattr(ag, '_destinations', None))
                # HISTORICO REAL, como el agente que se sube. Con ceros el
                # mismo episodio pasa de 76.607 $ a 25.335: un interruptor
                # medido asi se prueba sobre un agente mutilado.
                _hf = torch.from_numpy(np.asarray(O.rival_flow(ob), dtype=np.float32)).unsqueeze(0)
                out = net(torch.from_numpy(gr).unsqueeze(0),
                          torch.from_numpy(b).unsqueeze(0), _hf)
                ag.macro = Macro.from_vector(torch.sigmoid(out["macro_mu"])[0].numpy())
                mp = out["micro"][0].numpy()
                ag.micro = (lambda _o, m=_sm(mp): m)
                day = ob["day"]
            a = ag(ob)
            for op in ([a.get("farmer")] if a.get("farmer") else []) + list(a.get("hands") or []):
                if isinstance(op, (list, tuple)) and op:
                    ops[op[0]] += 1
            try: a2 = rv(o[1])
            except Exception: a2 = {"farmer": ["PASS"], "hands": [], "market": []}
            o, _ = env.step([a, a2])
    return float(env.rewards()[0]), ops


def _lado(var, val):
    """Un subproceso por lado: las banderas se congelan al importar, asi que
    cambiarlas en el proceso vivo no sirve de nada."""
    import subprocess, json
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aqui = os.path.dirname(os.path.abspath(__file__))
    cod = (
        "import sys, json\n"
        "sys.path.insert(0, " + repr(raiz) + ")\n"
        "sys.path.insert(0, " + repr(aqui) + ")\n"
        "from verifica_interruptores import _juega\n"
        "r = [_juega(s) for s in " + repr(SEEDS) + "]\n"
        "print('@@' + json.dumps([[a, dict(b)] for a, b in r]))\n"
    )
    e = dict(os.environ)
    if val == "":
        e.pop(var, None)
    else:
        e[var] = val
    p_ = subprocess.run([sys.executable, "-c", cod], capture_output=True,
                        text=True, env=e)
    for ln in p_.stdout.splitlines():
        if ln.startswith("@@"):
            return json.loads(ln[2:])
    raise SystemExit(f"  {var}={val}: no devolvio nada\n{p_.stderr[-600:]}")


if __name__ == "__main__":
    solo = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"{CK}   {len(SEEDS)} semillas   contra v48 sin limite\n")
    malos = []
    for var, a, b, desc, ambito in INTERRUPTORES:
        if solo and var != solo:
            continue
        if ambito != "juego":
            print(f"  {var}={a!r} vs {b!r}   {desc}")
            print(f"     (afecta a la RECOMPENSA: no se puede comprobar jugando)")
            continue
        ra, rb = _lado(var, a), _lado(var, b)
        da = np.array([x[0] for x in ra]); db = np.array([x[0] for x in rb])
        oa = collections.Counter(); ob = collections.Counter()
        for _, o in ra: oa.update(o)
        for _, o in rb: ob.update(o)
        igual = bool(np.allclose(da, db)) and oa == ob
        marca = "INERTE  <-- no cambia el juego" if igual else "opera"
        print(f"  {var}={a!r} vs {b!r}   {desc}")
        print(f"     dinero {da.mean():>9,.0f} -> {db.mean():>9,.0f}   "
              f"operaciones distintas: {sum((oa-ob).values())+sum((ob-oa).values()):>5}   {marca}")
        if igual:
            malos.append(var)
    print()
    if malos:
        print(f"  INERTES: {malos}")
        raise SystemExit(1)
    print("  todos operan")
