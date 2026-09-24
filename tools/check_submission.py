#!/usr/bin/env python
"""Gate: the uploaded agent and the evaluator must agree to the dollar.

Loads `submit_kagsym/main.py` the way Kaggle does (compile + exec, last
callable), drives it against the engine, and plays the same seeds through
`kagsym.evaluate.play`. Any difference means one of the two lies, and the one
that matters is the one Kaggle runs. Exit code 1 on any mismatch.

    python tools/check_submission.py runs/model.pt [--n 4]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.version import game_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_like_kaggle():
    path = os.path.join(ROOT, "submit_kagsym", "main.py")
    with open(path) as f:
        src = f.read()
    ns = {}
    exec(compile(src, path, "exec"), ns)
    fns = [v for v in ns.values() if callable(v) and getattr(v, "__name__", "") == "agent"]
    if not fns:
        raise SystemExit("no `agent` callable in main.py")
    return fns[-1]


def play_submission(agent_fn, opp, seed, seat=0):
    from kagsym.fastenv import FastEnv
    config = {"episodeSteps": E.HOURS * E.DAYS, "turnsPerDay": E.HOURS,
              "startingMoney": E.CASH}
    env = FastEnv(configuration=config, seed=seed)
    obs = env.reset()
    rival = E._opponent_callable(opp)
    other = 1 - seat
    while not env.done:
        actions = [None, None]
        actions[seat] = agent_fn(obs[seat], config)
        try:
            actions[other] = rival(obs[other])
        except Exception:
            actions[other] = dict(E.PASS_ACTION)
        obs, _ = env.step(actions)
    return float(env.rewards()[seat])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--opponent", default=E.V48.name)
    a = p.parse_args()
    # Both sides run in this interpreter, so a KAG_* variable that changes how
    # the agent plays is invisible to the comparison: refuse to run with any set.
    if game_env():
        raise SystemExit(f"game environment variables are set ({game_env()}); the gate "
                         f"cannot see them and Kaggle has none: unset them first")
    os.environ["KAGSYM_CKPT"] = os.path.abspath(a.checkpoint)
    opp = E.public(a.opponent)
    spec = E.PolicySpec(os.path.abspath(a.checkpoint))
    agent_fn = load_like_kaggle()
    same = 0
    print(f"[check] {a.checkpoint} vs {opp.label}, {a.n} seeds, seats 0 and 1")
    for k, seed in enumerate(S.RESERVED.seeds(a.n)):
        seat = k % 2
        sub = play_submission(agent_fn, opp, seed, seat)
        ev = E.play(spec, opp, seed, seat).money
        ok = abs(sub - ev) < 1e-6
        same += ok
        print(f"  seed {seed} seat {seat}  submission {sub:10,.0f}  evaluator {ev:10,.0f}  "
              f"{'SAME' if ok else 'DIFFER %+.0f' % (sub - ev)}", flush=True)
    print(f"\n  {same}/{a.n} identical")
    sys.exit(0 if same == a.n else 1)


if __name__ == "__main__":
    main()
