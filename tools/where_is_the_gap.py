"""Against an uncapped v48: WHAT each side sells, product by product.

The measurement that motivated the market head said 84% of the difference was
in STRAWBERRIES ($71,170) and MILK ($60,179). That measurement PREDATES opening
the portfolio per crop -one PLANT verb per crop- and the per-turn selling rule.
Repeating it says whether the gap moved or is still where it was.
"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

H, D, CASH = 24, 30, 3000
SEEDS = [601, 602, 603, 604]


def instrument(ckpt):
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
    ops = bool((d.get("cfg") or {}).get("con_ops", True))
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.HAND_CAP = None
    net = E2EAgent(WorldConfig(device="cpu", con_ops=ops))
    load_strict(net, d["sd"], ckpt, macro_fields=d.get("macro_fields"))
    net.eval()
    HIST = torch.zeros(1, M.N_HIST)
    sold = [collections.Counter(), collections.Counter()]   # units sold
    income = [collections.Counter(), collections.Counter()]  # gross income
    for s in SEEDS:
        env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                     "startingMoney": CASH}, seed=s)
        o = env.reset()
        ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
        rv = public_with_cap("v48-fast-routes", 10 ** 6)
        day = None
        with torch.no_grad():
            while not env.done:
                ob = o[0]
                if day != ob["day"]:
                    gr, b = O.encode_obs(ob)
                    out = net(torch.from_numpy(gr).unsqueeze(0),
                              torch.from_numpy(b).unsqueeze(0), HIST)
                    ag.macro = Macro.from_vector(torch.sigmoid(out["macro_mu"])[0].numpy())
                    mp = out["micro"][0].numpy()
                    ag.micro = (lambda _o, m=((mp[0], mp[1:]) if ops else mp): m)
                    day = ob["day"]
                a0 = ag(ob)
                try:
                    a1 = rv(o[1])
                except Exception:
                    a1 = {"farmer": ["PASS"], "hands": [], "market": []}
                for j, action in enumerate((a0, a1)):
                    pre = o[j]["market"]["prices"]
                    for order in (action.get("market") or []):
                        if order and order[0] == "SELL":
                            p, n = order[1], int(order[2])
                            sold[j][p] += n
                            income[j][p] += n * float(pre.get(p, 0))
                o, _ = env.step([a0, a1])
    return sold, income


if __name__ == "__main__":
    from kagsym import spec
    sold, income = instrument(sys.argv[1] if len(sys.argv) > 1 else "runs/completo.pt")
    n = len(SEEDS)
    print(f"against an uncapped v48, {n} seeds, 24h x 30d. "
          f"Gross income per product.\n")
    print(f"  {'product':<14}{'our uds':>9}{'our $':>10}"
          f"{'v48 uds':>9}{'v48 $':>10}{'gap $':>10}")
    rows = []
    for p in spec.PRODUCTS:
        a, b = income[0][p] / n, income[1][p] / n
        rows.append((b - a, p, sold[0][p] / n, a, sold[1][p] / n, b))
    rows.sort(reverse=True)
    for h, p, ua, a, ub, b in rows:
        print(f"  {p:<14}{ua:>9.0f}{a:>10.0f}{ub:>9.0f}{b:>10.0f}{h:>+10.0f}")
    ta, tb = sum(income[0].values()) / n, sum(income[1].values()) / n
    print(f"  {'TOTAL':<14}{'':>9}{ta:>10.0f}{'':>9}{tb:>10.0f}{tb-ta:>+10.0f}")
    worst_two = sum(h for h, *_ in rows[:2])
    print(f"\n  the WORST TWO concentrate "
          f"{100*worst_two/max(1e-9,(tb-ta)):.0f}% of the gap")
