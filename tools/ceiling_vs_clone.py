"""How much headroom there is above the opponent IN THIS CELL.

It separates the two hypotheses that explain the grid's draw equally well:

  H1  the opponent is too strong -> there is a ceiling, but they do not let us
      reach it. Fixed by starting against a weaker opponent.
  H2  the opponent sits at the cell's ceiling -> there is no room above and no
      opponent fixes anything; the cell has to change.

The measurement that separates them: the BEST fixed vector against THIS same
opponent. It is the clean comparison because the opponent is exactly that, a
fixed unconditioned vector, so the search stays inside its own class.

  ceiling >> 1076   -> H1: there is headroom, the opponent covers it.
  ceiling ~ 1076    -> H2: no headroom. Change the cell, not the opponent.

It is re-evaluated on FRESH seeds: the maximum of a search is inflated by
selection -measured in this same competition, between 1.2x and 1.81x-.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "ligas"))
import numpy as np, torch
from kagsym import spec
from kagsym.symbolic import tasks as T
from kagsym.symbolic.executor import Agent
from kagsym.fastenv import FastEnv
from kagsym.macro import Macro, N_MACRO
import kagsym.macro as _M

H, D, CAP, CASH = 12, 5, 3, 400
INIT = "runs/ligas/init32_l0.npy"
SEARCH_SEEDS = list(range(101, 107))   # the ones the search sees
VAL_SEEDS = list(range(301, 313))      # fresh, the same ones that judge the grid
POP, ELITE, ITERS = 24, 6, 8

spec.set_turns_per_day(H); spec.set_episode_steps(H*D); _M.HAND_CAP = CAP
OPP = list(np.load(INIT))

def play(vec, seeds):
    ours, theirs = [], []
    for s in seeds:
        env = FastEnv(configuration={"episodeSteps": H*D, "turnsPerDay": H,
                                     "startingMoney": CASH}, seed=s)
        o = env.reset()
        ag = Agent(episode_steps=H*D, macro=Macro.from_vector(list(vec)))
        rv = Agent(episode_steps=H*D, macro=Macro.from_vector(OPP))
        while not env.done:
            o, _ = env.step([ag(o[0]), rv(o[1])])
        r = env.rewards(); ours.append(float(r[0])); theirs.append(float(r[1]))
    return float(np.mean(ours)), float(np.mean(theirs))

if __name__ == "__main__":
    t0 = time.time()
    print(f"ceiling of the FIXED VECTOR class against the clone, cell "
          f"{H}h x {D}d, cash {CASH}, cap {CAP}")
    b0, r0 = play(OPP, VAL_SEEDS)
    print(f"  references   inaction {CASH}   the opponent (= our init) {b0:.0f}")
    print(f"  CEM {POP}x{ITERS} over {N_MACRO} dims, {len(SEARCH_SEEDS)} seeds "
          f"per draw\n")
    mu, sg = np.array(OPP, float), np.full(N_MACRO, 0.25)
    best_v, best_s = None, -1e9
    for it in range(ITERS):
        t1 = time.time()
        pop = np.clip(mu + sg * np.random.randn(POP, N_MACRO), 0.0, 1.0)
        pts = np.array([play(v, SEARCH_SEEDS)[0] for v in pop])
        idx = np.argsort(pts)[-ELITE:]
        mu, sg = pop[idx].mean(0), pop[idx].std(0) + 0.02
        if pts.max() > best_s:
            best_s, best_v = float(pts.max()), pop[int(np.argmax(pts))].copy()
        if it == 0:
            print(f"  (one iteration = {time.time()-t1:.0f}s -> ETA "
                  f"{(time.time()-t1)*ITERS:.0f}s)\n", flush=True)
        print(f"   it {it+1:>2}  search {pts.max():>7.0f}  "
              f"mean sigma {sg.mean():.3f}", flush=True)
    m, v = play(best_v, VAL_SEEDS)
    print(f"\n  search (inflated by selection)   {best_s:>7.0f}")
    print(f"  RE-EVALUATED on fresh seeds      {m:>7.0f}   opponent {v:.0f}   "
          f"margin {m-v:+.0f}")
    print(f"  the opponent itself              {b0:>7.0f}")
    headroom = 100.0 * (m - b0) / b0
    print(f"\n  HEADROOM over the opponent: {headroom:+.1f}%")
    print(f"  the grid's plateau was at ~1090 ({100*(1090-b0)/b0:+.1f}%)")
    print("  -> " + ("H1: there is headroom, the opponent covers it. "
                     "Weaken the opponent."
                     if headroom > 8 else
                     "H2: no headroom in this cell. Change the cell, not the "
                     "opponent."))
    np.save("runs/ligas/ceiling_vs_clone.npy", best_v)
    print(f"  ({time.time()-t0:.0f}s)")
