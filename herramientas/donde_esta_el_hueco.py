"""Contra v48 sin tope: QUE vende cada uno, producto a producto.

La medida que motivo la cabeza de mercado decia que el 84 % de la diferencia
estaba en FRESAS (71.170 $) y LECHE (60.179 $). Esa medida es ANTERIOR a abrir
la cartera por cultivo -un verbo PLANT por cultivo- y a la regla de venta por
turno. Repetirla dice si el hueco se movio o sigue donde estaba.
"""
import sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

H, D, CAJA = 24, 30, 3000
SEM = [601, 602, 603, 604]


def instrumenta(ckpt):
    from kagsym import obs as O, spec
    from kagsym.symbolic import tasks as T
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.migrate_ckpt import load_strict
    from kagsym.nets import world as M
    from kagsym.nets.world import E2EAgent, WorldConfig
    import kagsym.macro as _M
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    ops = bool((d.get("cfg") or {}).get("con_ops", False))
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.HAND_CAP = None
    net = E2EAgent(WorldConfig(device="cpu", con_ops=ops))
    load_strict(net, d["sd"], ckpt); net.eval()
    HIST = torch.zeros(1, M.N_HIST)
    ven = [collections.Counter(), collections.Counter()]   # uds vendidas
    ing = [collections.Counter(), collections.Counter()]   # ingreso
    for s in SEM:
        env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                     "startingMoney": CAJA}, seed=s)
        o = env.reset()
        ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
        rv = public_with_cap("v48-fast-routes", 10 ** 6)
        dia = None
        with torch.no_grad():
            while not env.done:
                ob = o[0]
                if dia != ob["day"]:
                    gr, b = O.encode_obs(ob)
                    sal = net(torch.from_numpy(gr).unsqueeze(0),
                              torch.from_numpy(b).unsqueeze(0), HIST)
                    ag.macro = Macro.from_vector(torch.sigmoid(sal["macro_mu"])[0].numpy())
                    mp = sal["micro"][0].numpy()
                    ag.micro = (lambda _o, m=((mp[0], mp[1:]) if ops else mp): m)
                    dia = ob["day"]
                a0 = ag(ob)
                try:
                    a1 = rv(o[1])
                except Exception:
                    a1 = {"farmer": ["PASS"], "hands": [], "market": []}
                for j, acc in enumerate((a0, a1)):
                    pre = o[j]["market"]["prices"]
                    for orden in (acc.get("market") or []):
                        if orden and orden[0] == "SELL":
                            p, n = orden[1], int(orden[2])
                            ven[j][p] += n
                            ing[j][p] += n * float(pre.get(p, 0))
                o, _ = env.step([a0, a1])
    return ven, ing


if __name__ == "__main__":
    from kagsym import spec
    ven, ing = instrumenta(sys.argv[1] if len(sys.argv) > 1 else "runs/completo.pt")
    n = len(SEM)
    print(f"contra v48 sin tope, {n} semillas, 24h x 30d. Ingreso bruto por producto.\n")
    print(f"  {'producto':<14}{'nos uds':>9}{'nos $':>10}{'v48 uds':>9}{'v48 $':>10}{'hueco $':>10}")
    rows = []
    for p in spec.PRODUCTS:
        a, b = ing[0][p] / n, ing[1][p] / n
        rows.append((b - a, p, ven[0][p] / n, a, ven[1][p] / n, b))
    rows.sort(reverse=True)
    for h, p, ua, a, ub, b in rows:
        print(f"  {p:<14}{ua:>9.0f}{a:>10.0f}{ub:>9.0f}{b:>10.0f}{h:>+10.0f}")
    ta, tb = sum(ing[0].values()) / n, sum(ing[1].values()) / n
    print(f"  {'TOTAL':<14}{'':>9}{ta:>10.0f}{'':>9}{tb:>10.0f}{tb-ta:>+10.0f}")
    dos = sum(h for h, *_ in rows[:2])
    print(f"\n  los DOS peores concentran {100*dos/max(1e-9,(tb-ta)):.0f} % del hueco")
