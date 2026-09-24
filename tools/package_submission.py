#!/usr/bin/env python
"""Build the submission tarball and prove it plays under the real Kaggle runner.

The tarball holds `main.py`, `model.pt` and the runtime part of `kagsym/`
(no training modules, no tests). Then `kaggle_environments` loads the
extracted `main.py` exactly as the platform does (compile + exec in a fresh
namespace, last callable) and plays it for a few turns against a built-in
agent. An agent that only fails under exec (`__file__`, late imports) fails
here, not on the leaderboard.

    python tools/package_submission.py [--out runs/submission.tar.gz] [--turns 48]
"""
import argparse
import os
import shutil
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMIT = os.path.join(ROOT, "submit_kagsym")

RUNTIME_MODULES = [
    "__init__.py", "spec.py", "fastenv.py", "obs.py", "macro.py", "policy.py",
    "migrate_ckpt.py", "environment.py", "reward.py", "potential.py", "version.py",
    "symbolic/__init__.py", "symbolic/executor.py", "symbolic/tasks.py",
    "symbolic/market_ops.py", "symbolic/assignment.py",
    "nets/__init__.py", "nets/world.py", "nets/blocks.py",
]


def build(out: str) -> str:
    model = os.path.join(SUBMIT, "model.pt")
    if not os.path.exists(model):
        raise SystemExit(f"{model} missing: copy the validated checkpoint there first")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(os.path.join(SUBMIT, "main.py"), arcname="main.py")
        tar.add(model, arcname="model.pt")
        for rel in RUNTIME_MODULES:
            tar.add(os.path.join(ROOT, "kagsym", rel), arcname=os.path.join("kagsym", rel))
    return out


def play_with_kaggle_runner(tarball: str, turns: int) -> dict:
    from kaggle_environments import make
    tmp = tempfile.mkdtemp(prefix="kagsym_submission_")
    with tarfile.open(tarball) as tar:
        tar.extractall(tmp, filter="data")
    main_py = os.path.join(tmp, "main.py")
    # The platform runs the agent from its own directory; nothing from the
    # repository may leak in through sys.path or the environment.
    env_ckpt = os.environ.pop("KAGSYM_CKPT", None)
    saved_path = list(sys.path)
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != ROOT]
    for m in [m for m in sys.modules if m == "kagsym" or m.startswith("kagsym.")]:
        del sys.modules[m]
    try:
        env = make("kaggriculture", configuration={"episodeSteps": turns}, debug=True)
        env.run([main_py, "random"])
        statuses = [s.status for s in env.state]
        money = env.state[0].observation["farms"][0]["money"]
        return {"statuses": statuses, "money": money, "steps": len(env.steps), "dir": tmp}
    finally:
        sys.path = saved_path
        if env_ckpt is not None:
            os.environ["KAGSYM_CKPT"] = env_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "runs", "submission.tar.gz"))
    p.add_argument("--turns", type=int, default=48)
    p.add_argument("--no-play", action="store_true")
    a = p.parse_args()
    out = build(a.out)
    size = os.path.getsize(out) / 1e6
    print(f"[package] {out}  {size:.1f} MB  ({len(RUNTIME_MODULES)} modules + main.py + model.pt)")
    if a.no_play:
        return
    r = play_with_kaggle_runner(out, a.turns)
    ok = r["statuses"][0] == "DONE" and r["money"] != 3000
    print(f"[package] kaggle runner: statuses {r['statuses']}  our money after {a.turns} turns "
          f"{r['money']:,.0f}  -> {'OK' if ok else 'FAILED'}")
    if not ok:
        print(f"  extracted at {r['dir']}; run it by hand to see the traceback")
        sys.exit(1)
    shutil.rmtree(r["dir"], ignore_errors=True)


if __name__ == "__main__":
    main()
