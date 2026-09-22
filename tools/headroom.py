"""Which cell is worth studying learning on: the one with HEADROOM.

The previous pre-flight filtered by x_inaction -whether money is made above
doing nothing-. The 12h x 5d cell passes that easily (2.7x) and is nevertheless
DEAD: CEM over the 32 dims finds nothing above the incumbent (+1.3%) and six RL
runs with different rewards all land at 1085-1095. Two methods sharing nothing
agreeing = it is the ceiling, not the method.

The right criterion is HEADROOM: how much search improves on a competent
incumbent. Without headroom there is nothing to learn and any ablation measures
noise.

Cash is scaled per UNIT-TURN -H * D * (1+cap)- instead of by eye. The previous
ladder was set by eye and the pre-flight knocked out 6 of 9 rungs for lack of
capital.

The last rung is the championship with its REAL configuration, which breaks the
pattern on purpose: $3,000 for 8,640 unit-turns is $0.35/ut against the $1.67
of the toy cell. The real game is capital-starved compared with our small
worlds, and that is worth having written down.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kagsym import spec
from kagsym.symbolic import tasks as T
from kagsym.symbolic.executor import Agent
from kagsym.fastenv import FastEnv
from kagsym.macro import Macro, N_MACRO
import kagsym.macro as _M

BASE = np.load("runs/ligas/init32_l0.npy")
RATIO = 400.0 / (12 * 5 * 4)           # $/unit-turn of the anchor cell
POP, ELITE, ITERS = 24, 6, 8
SEARCH_SEEDS, VAL_SEEDS = list(range(101, 107)), list(range(301, 313))

#            H   D  cap  cash (None = scaled per unit-turn)
CELLS =   [( 12,  5,  3, None),
          ( 24,  5,  3, None),
          ( 12, 10,  3, None),
          ( 24, 10,  5, None),
          ( 24, 30, 11, 3000)]         # championship, REAL configuration

def play(vec, opp, seeds, H, D, CASH):
    ours = []
    for s in seeds:
        env = FastEnv(configuration={"episodeSteps": H*D, "turnsPerDay": H,
                                     "startingMoney": CASH}, seed=s)
        o = env.reset()
        a = Agent(episode_steps=H*D, macro=Macro.from_vector(list(vec)))
        r = Agent(episode_steps=H*D, macro=Macro.from_vector(list(opp)))
        while not env.done:
            o, _ = env.step([a(o[0]), r(o[1])])
        ours.append(float(env.rewards()[0]))
    return float(np.mean(ours))

if __name__ == "__main__":
    t0 = time.time()
    print("HEADROOM per cell: how much CEM improves on the incumbent, "
          "re-evaluated on fresh seeds")
    print(f"cash per unit-turn = {RATIO:.2f} $/ut (anchor: the toy cell)\n")
    print(f"  {'cell':>16} {'ut':>6} {'cash':>6} {'incumb':>8} {'CEM':>8} "
          f"{'headroom':>9} {'s':>5}")
    for H, D, CAP, cash in CELLS:
        t1 = time.time()
        ut = H * D * (1 + CAP)
        CASH = int(cash if cash else max(100, round(ut * RATIO / 50) * 50))
        spec.set_turns_per_day(H); spec.set_episode_steps(H*D)
        _M.HAND_CAP = CAP
        inc = play(BASE, BASE, VAL_SEEDS, H, D, CASH)
        mu, sg = BASE.astype(float).copy(), np.full(N_MACRO, 0.25)
        best, pts_max = None, -1e9
        for _ in range(ITERS):
            pop = np.clip(mu + sg*np.random.randn(POP, N_MACRO), 0.0, 1.0)
            pts = np.array([play(v, BASE, SEARCH_SEEDS, H, D, CASH) for v in pop])
            idx = np.argsort(pts)[-ELITE:]
            mu, sg = pop[idx].mean(0), pop[idx].std(0) + 0.02
            if pts.max() > pts_max:
                pts_max, best = float(pts.max()), pop[int(np.argmax(pts))].copy()
        cem = play(best, BASE, VAL_SEEDS, H, D, CASH)   # FRESH seeds
        headroom = 100.0 * (cem - inc) / max(inc, 1.0)
        np.save(f"runs/ligas/headroom_{H}h{D}d.npy", best)
        print(f"  {f'{H}h x {D}d c{CAP}':>16} {ut:>6} {CASH:>6} {inc:>8.0f} "
              f"{cem:>8.0f} {headroom:>+8.1f}% {time.time()-t1:>5.0f}", flush=True)
    print(f"\n  no headroom -> the cell is exhausted: any ablation measures noise.")
    print(f"  self-play goes on the SMALLEST cell with real headroom.")
    print(f"  ({time.time()-t0:.0f}s)")
