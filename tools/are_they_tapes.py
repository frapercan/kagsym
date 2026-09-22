"""Are the public agents deterministic tapes or reactive policies?

It matters because it decides how much self-play is needed: a tape is
MEMORISED, and memorising 24 tapes is not learning to play. A reactive policy
is not.

Two independent tests:
  1. SAME configuration, TWO different seeds. The board, the market and the
     town change; a tape indexed by step plays exactly the same. The fraction
     of turns with an identical action is measured.
  2. ANOTHER SCALE (12 h/day instead of 24). A tape built for 719 steps loses
     sync and collapses; a reactive policy adapts.

Measured for v48 at the time: $60,426 at 24h x 30d and ~0 at any other scale.
This checks whether that holds for all of them or only for that one.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def signature(name_, seed, hours=24, days=30):
    from kagsym import spec
    from kagsym.environment import load_public
    from kagsym.fastenv import FastEnv
    spec.set_turns_per_day(hours); spec.set_episode_steps(hours * days)
    ag = load_public(name_)
    env = FastEnv(configuration={"episodeSteps": hours * days,
                                 "turnsPerDay": hours, "startingMoney": 3000},
                  seed=seed)
    o = env.reset()
    acts = []
    while not env.done:
        a = ag(o[0])
        acts.append(json.dumps([a.get("farmer"), a.get("hands"),
                                a.get("market")], sort_keys=True))
        o, _ = env.step([a, {"farmer": ["PASS"], "hands": [], "market": []}])
    return acts, float(env.rewards()[0])


def analyse(name_):
    a1, d1 = signature(name_, 4242)
    a2, d2 = signature(name_, 9999)
    n = min(len(a1), len(a2))
    same = sum(1 for i in range(n) if a1[i] == a2[i]) / max(1, n)
    _, d12 = signature(name_, 4242, hours=12, days=30)
    return same, (d1 + d2) / 2, d12


if __name__ == "__main__":
    import multiprocessing as mp
    def one(name_):
        try:
            return (name_, *analyse(name_))
        except Exception:
            return (name_, None, None, None)
    entries = json.load(open("data/valid_ladder.json"))
    names = [x["name"] for x in entries]
    if "--few" in sys.argv:
        names = names[::4]
    print(f"{len(names)} agents. 'same' = % of turns with the SAME action on "
          f"two different seeds\n")
    print(f"  {'agent':<46}{'same':>7}{'24h $':>9}{'12h $':>9}")
    with mp.Pool(8) as p:
        res = p.map(one, names)
    import numpy as np
    sames = []
    for name_, i, d24, d12 in res:
        if i is None:
            print(f"  {name_[:44]:<46}{'failed':>7}"); continue
        sames.append(i)
        print(f"  {name_[:44]:<46}{100*i:>6.0f}%{d24:>9.0f}{d12:>9.0f}")
    sames = np.array(sames)
    print(f"\n  TAPES (>90% identical):  {int((sames>0.9).sum())}/{len(sames)}")
    print(f"  REACTIVE (<50%):         {int((sames<0.5).sum())}/{len(sames)}")
    print(f"  median agreement:        {100*np.median(sames):.0f}%")
