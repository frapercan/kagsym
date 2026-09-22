"""EXTERNAL yardstick for the long championship run.

It is needed because the trainer computes `margin_pct` against the TRAINING
opponent, and under self-play that opponent is ourselves: the margin stops
rising as soon as the copy gets stronger, even while the agent keeps improving.
The code itself documents it -over that same stretch the margin against the
2945 agent improved from -98.7% to -82.8% while the return did not move-.

So progress is read against something that does NOT move: v48 in the real game.

Usage:  python tools/yardstick.py runs/long.pt.ultimo [more checkpoints...]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

H, D, CASH = 24, 30, 3000
SEEDS = list(range(601, 609))        # fixed: comparable across checkpoints
# The opponent is v48 at full power by default -- the external thermometer.
# With KAG_VARA_PASIVO=1 it is the passive agent instead, which measures
# ABSOLUTE production with the market uncontested. The references there are
# known: inaction $3,000, us $72,796, the median of 63 public agents $186,594
# and the relaxed upper bound $195,532.
PASSIVE = bool(int(os.environ.get("KAG_VARA_PASIVO", "0")))


def evaluate(ckpt):
    from kagsym import obs as O, spec
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.migrate_ckpt import load_strict
    from kagsym.nets import world as M
    from kagsym.nets.world import E2EAgent, WorldConfig
    import kagsym.macro as _M
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    # The verb head is always built now; older checkpoints record whether they
    # had one, and that is what decides the shape of their micro map.
    ops = bool((d.get("cfg") or {}).get("con_ops", True))
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.HAND_CAP = None
    net = E2EAgent(WorldConfig(device="cpu", con_ops=ops))
    load_strict(net, d["sd"], ckpt, macro_fields=d.get("macro_fields"))
    net.eval()
    HIST = torch.zeros(1, M.N_HIST)
    ours, theirs = [], []
    for s in SEEDS:
        env = FastEnv(configuration={"episodeSteps": H * D, "turnsPerDay": H,
                                     "startingMoney": CASH}, seed=s)
        o = env.reset()
        ag = Agent(episode_steps=H * D, macro=Macro.from_vector([0.5] * N_MACRO))
        if PASSIVE:
            from kaggle_environments.envs.kaggriculture import kaggriculture as _E
            rv = _E.pass_agent
        else:
            rv = public_with_cap("v48-fast-routes", 10**6)
        day = None
        with torch.no_grad():
            while not env.done:
                ob = o[0]
                if day != ob["day"]:
                    gr, b = O.encode_obs(ob)
                    out = net(torch.from_numpy(gr).unsqueeze(0),
                              torch.from_numpy(b).unsqueeze(0), HIST)
                    ag.macro = Macro.from_vector(
                        torch.sigmoid(out["macro_mu"])[0].numpy())
                    mp = out["micro"][0].numpy()
                    ag.micro = (lambda _o, m=((mp[0], mp[1:]) if ops else mp): m)
                    day = ob["day"]
                try:
                    act_r = rv(o[1])
                except Exception:
                    act_r = {"farmer": ["PASS"], "hands": [], "market": []}
                o, _ = env.step([ag(ob), act_r])
        r = env.rewards(); ours.append(float(r[0])); theirs.append(float(r[1]))
    m, v = np.array(ours), np.array(theirs)
    d_ = m - v
    return (m.mean(), v.mean(), d_.mean(),
            d_.std(ddof=1) / np.sqrt(len(d_)), int((m > v).sum()),
            int(d.get("upd", 0) or 0))


if __name__ == "__main__":
    print(f"external yardstick: {H}h x {D}d, cash {CASH}, against "
          f"{'the PASSIVE agent' if PASSIVE else 'v48'}, "
          f"{len(SEEDS)} fixed seeds\n")
    print(f"  {'checkpoint':>34} {'upd':>5} {'money':>9} {'v48':>9} "
          f"{'margin%':>8} {'wins':>6} {'s':>5}")
    for c in sys.argv[1:]:
        t1 = time.time()
        try:
            m, v, dm, se, g, upd = evaluate(c)
        except Exception as e:
            print(f"  {c[-34:]:>34}  failed: {str(e)[:50]}", flush=True); continue
        print(f"  {c[-34:]:>34} {upd:>5} {m:>9.0f} {v:>9.0f} "
              f"{100*dm/v:>+7.1f}% {g:>3}/{len(SEEDS)} {time.time()-t1:>5.0f}",
              flush=True)
