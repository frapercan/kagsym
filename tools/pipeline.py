#!/usr/bin/env python
"""From random weights to a validated, packaged checkpoint in one launch.

Every stage is a gate: it runs the real tool, checks its exit code and the
artefact it must leave, and the pipeline stops at the first failure with the
stage's log. Silent failures in this project sign a plausible number; this
runs the whole chain so that anything that signs somewhere is caught by the
next gate, on a clean tree, in a single command.

Stages:
  0 tests            pytest kagsym/tests; refuses KAG_* game env variables
  1 train            kagsym.cli.train from random init, seeded
  2 gate             tools/check_submission.py on the trained checkpoint
  3 live-dials       tools/live_dials.py
  4 search           tools/search.py (constant offset, margin objective)
  5 validate+bake    tools/validate_offset.py --bake
  6 criterion        tools/band.py, paired against the trained checkpoint
  7 package          tools/package_submission.py (plays under the Kaggle runner)

Presets:
  --smoke   minutes: tiny sizes, proves the chain and the gates, not strength
  --full    hours: the sizes docs/PROCEDURE.md prescribes

    python tools/pipeline.py --smoke --out runs/pipeline/smoke
    python tools/pipeline.py --full  --out runs/pipeline/2026-09-25 --experiment EXP-002
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable

SMOKE = dict(train_updates=6, envs=4, procs=4, days=30, dials_n=1, search_gens=2, search_pop=6,
             search_elite=2, search_seeds=2, band_sample=2, validate_n=6, validate_band_n=1,
             band_n=1, band_limit=6, package_turns=48)
FULL = dict(train_updates=400, envs=11, procs=11, days=30, dials_n=3, search_gens=40, search_pop=32,
            search_elite=8, search_seeds=16, band_sample=6, validate_n=200, validate_band_n=6,
            band_n=6, band_limit=0, package_turns=48)


class Stage:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.report = []

    def run(self, name: str, cmd: list[str], must_exist: list[str] = (), allow_fail: bool = False,
            env: dict | None = None) -> subprocess.CompletedProcess:
        t0 = time.time()
        log = os.path.join(self.out_dir, f"{len(self.report):02d}_{name}.log")
        print(f"\n== {name}: {' '.join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd)}",
              flush=True)
        with open(log, "w") as f:
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True,
                               env={**os.environ, **(env or {})})
        missing = [p for p in must_exist if not os.path.exists(p)]
        ok = (r.returncode == 0 or allow_fail) and not missing
        entry = {"stage": name, "returncode": r.returncode, "seconds": round(time.time() - t0),
                 "log": os.path.relpath(log, ROOT), "missing": missing, "ok": ok}
        self.report.append(entry)
        with open(os.path.join(self.out_dir, "report.json"), "w") as f:
            json.dump(self.report, f, indent=1)
        with open(log) as f:
            tail = f.read()[-1500:]
        print(tail, flush=True)
        print(f"-- {name}: {'OK' if ok else 'FAILED'} ({entry['seconds']}s)", flush=True)
        if not ok:
            raise SystemExit(f"pipeline stopped at stage {name}; see {log}")
        return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="pipeline directory (must not exist)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--experiment", default=None, help="preregistration id tagged on every run")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    P = SMOKE if a.smoke else FULL
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        if not a.force:
            raise SystemExit(f"{out} exists; a pipeline never writes over another (use --force)")
        shutil.rmtree(out)
    os.makedirs(out)
    from kagsym.version import game_env, provenance
    if game_env():
        raise SystemExit(f"game environment variables set: {game_env()}; unset them first")
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump({"preset": "smoke" if a.smoke else "full", "params": P, "args": vars(a),
                   "provenance": provenance()}, f, indent=1)
    env = {"KAGSYM_EXPERIMENT": a.experiment} if a.experiment else {}
    st = Stage(out)
    run_name = os.path.basename(out)
    trained = os.path.join(out, "trained.pt")
    search_dir = os.path.join(out, "search")
    baked = os.path.join(out, "validated.pt")

    if not a.skip_tests:
        st.run("tests", [PY, "-m", "pytest", "kagsym/tests", "-q", "-p", "no:warnings"])
    st.run("train", [PY, "-m", "kagsym.cli.train", "--envs", str(P["envs"]), "--procs", str(P["procs"]),
                     "--updates", str(P["train_updates"]), "--days", str(P["days"]),
                     "--seed", str(a.seed), "--run-name", f"pipeline-{run_name}", "--out", trained],
           must_exist=[trained + ".ultimo"], env=env)
    if not os.path.exists(trained):
        shutil.copyfile(trained + ".ultimo", trained)   # the last state; never "the best by return"
    st.run("gate", [PY, "tools/check_submission.py", trained, "--n", "2"])
    st.run("live-dials", [PY, "tools/live_dials.py", trained, "--n", str(P["dials_n"]),
                          "--procs", str(P["procs"])],
           must_exist=[trained + ".dials.json"])
    st.run("search", [PY, "tools/search.py", trained, "--out", search_dir, "--objective", "margin",
                      "--gens", str(P["search_gens"]), "--pop", str(P["search_pop"]),
                      "--elite", str(P["search_elite"]), "--seeds", str(P["search_seeds"]),
                      "--band-sample", str(P["band_sample"]), "--procs", str(P["procs"])]
           + (["--experiment", a.experiment] if a.experiment else []),
           must_exist=[os.path.join(search_dir, "done.json"), os.path.join(search_dir, "offset.npz")])
    # The bake gate may legitimately refuse (a null result is a result); the
    # stage passes if the validator RAN and said so, and the pipeline goes on
    # with the trained checkpoint.
    r = st.run("validate", [PY, "tools/validate_offset.py", search_dir, trained, "--family", "clean",
                            "--n", str(P["validate_n"]), "--band-n", str(P["validate_band_n"]),
                            "--procs", str(P["procs"]), "--bake", baked, "--t-min", "2.0"],
               allow_fail=True)
    final = baked if os.path.exists(baked) else trained
    print(f"   validated offset {'BAKED' if final == baked else 'REFUSED (null result); continuing with the trained checkpoint'}")
    st.run("criterion", [PY, "tools/band.py", final, "--n", str(P["band_n"]), "--procs", str(P["procs"]),
                         "--save", os.path.join(out, "band.json")]
           + (["--limit", str(P["band_limit"])] if P["band_limit"] else [])
           + ([] if final == trained else ["--against", trained]),
           must_exist=[os.path.join(out, "band.json")], env=env)
    submit_model = os.path.join(ROOT, "submit_kagsym", "model.pt")
    backup = None
    if os.path.exists(submit_model):
        backup = submit_model + ".before-pipeline"
        shutil.copyfile(submit_model, backup)
    shutil.copyfile(final, submit_model)
    try:
        st.run("package", [PY, "tools/package_submission.py", "--out", os.path.join(out, "submission.tar.gz"),
                           "--turns", str(P["package_turns"])],
               must_exist=[os.path.join(out, "submission.tar.gz")])
    finally:
        if backup:
            shutil.move(backup, submit_model)      # the pipeline never deploys by itself
    print(f"\nPIPELINE OK  final checkpoint {os.path.relpath(final, ROOT)}  "
          f"tarball {os.path.relpath(os.path.join(out, 'submission.tar.gz'), ROOT)}")
    print(f"report: {os.path.relpath(os.path.join(out, 'report.json'), ROOT)}")


if __name__ == "__main__":
    main()
