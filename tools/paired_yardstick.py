"""PAIRED fixed yardstick: does the full model really beat the state of the art?

The plain yardstick compares two independent MEANS. With an sd of $13,300 per
seed that does not distinguish +6.6% from zero (see the statistical power note).
Here the SAME seed is played by both checkpoints and the per-seed difference is
what is read: the scenario's variance cancels and far less sample is needed for
the same standard error.

The seeds are fixed BEFORE looking at anything, and they differ from the
yardstick's 601-608 so that nothing tuned on those is rewarded.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

H, D, CASH = 24, 30, 3000
SEEDS = list(range(9001, 9201))      # 200, fixed in advance
PROCS = 6                            # training uses 11; do not choke it


def _one(args):
    ckpt, s = args
    import torch
    # ONE THREAD PER PROCESS. Torch opens threads on its own: measured, each
    # worker took 1.6 cores, and added to training's 11 processes that was ~21
    # threads on 12 cores. The thrashing made the measurement undefined AND
    # slowed training down. With one thread, the work is ~800 s of CPU spread
    # across the pool's processes.
    torch.set_num_threads(1)
    from kagsym import obs as O, spec
    from kagsym.symbolic.executor import Agent
    from kagsym.environment import public_with_cap
    from kagsym.fastenv import FastEnv
    from kagsym.macro import Macro, N_MACRO
    from kagsym.migrate_ckpt import load_strict
    from kagsym.nets import world as M
    from kagsym.nets.world import E2EAgent, WorldConfig
    import kagsym.macro as _M
    global _CACHE
    try:
        net, ops = _CACHE[ckpt]
    except Exception:
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        ops = bool((d.get("cfg") or {}).get("con_ops", True))
        net = E2EAgent(WorldConfig(device="cpu", con_ops=ops))
        load_strict(net, d["sd"], ckpt, macro_fields=d.get("macro_fields"))
        net.eval()
        try:
            _CACHE[ckpt] = (net, ops)
        except NameError:
            _CACHE = {ckpt: (net, ops)}
    spec.set_turns_per_day(H); spec.set_episode_steps(H * D); _M.HAND_CAP = None
    HIST = torch.zeros(1, M.N_HIST)
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
            try:
                act_r = rv(o[1])
            except Exception:
                act_r = {"farmer": ["PASS"], "hands": [], "market": []}
            o, _ = env.step([ag(ob), act_r])
    r = env.rewards()
    return float(r[0]), float(r[1])


if __name__ == "__main__":
    import multiprocessing as mp
    A, B = sys.argv[1], sys.argv[2]
    t0 = time.time()
    print(f"PAIRED yardstick: {H}h x {D}d, cash {CASH}, against an uncapped "
          f"v48, {len(SEEDS)} seeds fixed in advance")
    print(f"  A = {A}\n  B = {B}\n")
    # PROGRESS. Without it there is no way to tell 40% from 95% done, and that
    # already cost throwing away a 15-minute measurement for lack of an ETA.
    def sweep(ck, tag):
        out = []
        with mp.Pool(PROCS) as pool:
            for i, r in enumerate(pool.imap(_one, [(ck, s) for s in SEEDS]), 1):
                out.append(r)
                if i % 20 == 0 or i == len(SEEDS):
                    print(f"    {tag}: {i}/{len(SEEDS)}  ({time.time()-t0:.0f}s)",
                          flush=True)
        return out
    ra = sweep(A, "A")
    rb = sweep(B, "B")
    a = np.array([x[0] for x in ra]); va = np.array([x[1] for x in ra])
    b = np.array([x[0] for x in rb]); vb = np.array([x[1] for x in rb])
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  {'':<10}{'ours':>10}{'v48':>10}{'margin%':>10}{'wins':>8}")
    print(f"  {'A':<10}{a.mean():>10.0f}{va.mean():>10.0f}"
          f"{100*(a-va).mean()/va.mean():>9.1f}%{int((a>va).sum()):>5}/{len(SEEDS)}")
    print(f"  {'B':<10}{b.mean():>10.0f}{vb.mean():>10.0f}"
          f"{100*(b-vb).mean()/vb.mean():>9.1f}%{int((b>vb).sum()):>5}/{len(SEEDS)}")
    print(f"\n  PAIRED DIFFERENCE A-B: {d.mean():+.0f} $  "
          f"+- {se:.0f} (se)   t = {d.mean()/max(1e-9, se):+.2f}")
    print(f"  sd per seed: {d.std(ddof=1):.0f} $   "
          f"A beats B on {int((d>0).sum())}/{len(SEEDS)}")
    verdict = ("A IS BETTER" if d.mean() > 2*se else
               "B IS BETTER" if d.mean() < -2*se else
               "INDISTINGUISHABLE at 2 se")
    print(f"\n  -> {verdict}   ({time.time()-t0:.0f}s)")
