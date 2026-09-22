"""Why does a unit PASS? The number alone does not say, and the two causes
need opposite fixes.

Measured against an uncapped v48: 11% of our unit-turns are PASS against their
5%, which is about 480 wasted actions per episode -roughly the whole watering
deficit that leaves us 21.8 live plants against their 43.2. But a PASS can mean
the executor offered this unit no legal task, or that it offered several and
the network valued them all at <= 0. The first is an enumeration problem and
the network cannot fix it; the second is the network's own output.

Usage:  python tools/why_pass.py [checkpoint] [seeds...]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

H, D, CASH = 24, 30, 3000


def run(ck, seeds):
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
    torch.set_num_threads(1)
    d = torch.load(ck, map_location="cpu", weights_only=False)
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.HAND_CAP = None
    net = E2EAgent(WorldConfig(device="cpu", con_ops=True))
    load_strict(net, d["sd"], ck, macro_fields=d.get("macro_fields")); net.eval()
    HIST = torch.zeros(1, M.N_HIST)
    tot = None
    for s in seeds:
        T.enable_pass_stats()
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
                    ag.micro = (lambda _o, m=(mp[0], mp[1:]): m)
                    day = ob["day"]
                try: a1 = rv(o[1])
                except Exception: a1 = {"farmer": ["PASS"], "hands": [], "market": []}
                o, _ = env.step([ag(ob), a1])
        st = T.collect_pass_stats()
        tot = st if tot is None else {k: tot[k] + v for k, v in st.items()}
    return tot


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else "runs/esc2.pt.ultimo"
    seeds = [int(x) for x in sys.argv[2:]] or [701, 702]
    st = run(ck, seeds)
    n = max(1, st["units"])
    print(f"{ck}, {len(seeds)} episodios de {H}h x {D}d\n")
    print(f"  unidad-turnos              {st['units']:>8}")
    print(f"  PASS                       {st['pass']:>8}  ({100*st['pass']/n:.1f}%)")
    p = max(1, st["pass"])
    print(f"\n  de esos PASS:")
    for k, lab in (("no_task_at_all", "no habia NINGUNA tarea en el tablero"),
                   ("all_blocked_by_inventory", "las habia, pero ninguna legal para esa unidad"),
                   ("all_value_nonpositive", "las habia y legales, pero valor <= 0"),
                   ("taken_by_another", "habia valor positivo, otra unidad la cogio")):
        print(f"    {lab:<48}{st[k]:>8}  ({100*st[k]/p:>5.1f}%)")
    print(f"\n  pares (unidad,tarea) ofrecidos   {st['offers']:>8}")
    print(f"  pares bloqueados por inventario  {st['blocked']:>8}  "
          f"({100*st['blocked']/max(1,st['offers']+st['blocked']):.1f}% de los posibles)")
