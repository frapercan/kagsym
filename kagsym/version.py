"""Provenance: the code, the environment and the tree that produced a number.

The problem it solves cost a whole search once: a 15-minute run starts, the
executor keeps being edited while it runs, and the workers, which imported
their modules when they were spawned, measure a world that no longer exists.
The number is valid inside the wrong world and nothing in it looks wrong.

Three fingerprints and one environment listing:

  fingerprint()        the files that decide HOW THE GAME IS PLAYED
  model_fingerprint()  the files that decide HOW LEARNING HAPPENS
  game_env()           the KAG_* environment variables that change how the
                       agent plays; none of them is set on Kaggle
  provenance()         all of the above plus commit, dirty flag, argv, pid

`provenance()` goes into every ledger line, every search directory and every
MLflow run. `check()` warns when a stored fingerprint is not the current one.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GAME_FILES = [
    "kagsym/spec.py",
    "kagsym/fastenv.py",
    "kagsym/policy.py",
    "kagsym/symbolic/tasks.py",
    "kagsym/symbolic/market_ops.py",
    "kagsym/symbolic/executor.py",
    "kagsym/symbolic/assignment.py",
    "kagsym/macro.py",
    "kagsym/potential.py",
    "kagsym/reward.py",
    "kagsym/obs.py",
]
MODEL_FILES = [
    "kagsym/nets/world.py",
    "kagsym/nets/blocks.py",
    "kagsym/cli/train.py",
    "kagsym/environment.py",
    "kagsym/parallel_env.py",
    "kagsym/migrate_ckpt.py",
    "kagsym/memory.py",
]

# Environment variables read by the game layer. Any of them set means the
# agent being measured is not the agent that plays on Kaggle.
GAME_ENV = [
    "KAG_VALOR", "KAG_COORD", "KAG_CADENA", "KAG_COMMIT", "KAG_FLUJO_MERCADO",
    "KAG_CAJA", "KAG_GAMMA", "KAG_POTENCIAL", "KAG_DENSO", "KAG_ESCALA",
    "KAG_BONUS", "KAG_PESO_ILEGAL", "KAG_PESO_RIVAL", "KAG_PESO_WIN",
    "KAG_PHI_FIN", "KAG_PHI_RECOCIDO", "KAG_PHI_W",
]


def _digest(files) -> str:
    h = hashlib.sha256()
    for f in files:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            # A renamed file used to hash as "?" and the fingerprint stayed
            # plausible. It must fail: a fingerprint over the wrong file list
            # is worse than none.
            raise FileNotFoundError(f"fingerprint file missing: {f}")
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def fingerprint() -> str:
    return _digest(GAME_FILES)


def model_fingerprint() -> str:
    return _digest(MODEL_FILES)


def game_env() -> dict:
    """The KAG_* variables that are set, with their values."""
    return {k: os.environ[k] for k in GAME_ENV if k in os.environ}


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def provenance() -> dict:
    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "game_fingerprint": fingerprint(),
        "model_fingerprint": model_fingerprint(),
        "game_env": game_env(),
        "argv": sys.argv[:12],
        "pid": os.getpid(),
    }


def check(stored, what: str = "result") -> bool:
    """Warn when a stored game fingerprint is not the current code's."""
    current = fingerprint()
    if stored is None:
        print(f"WARNING: {what} has no code fingerprint; the comparison cannot be validated",
              flush=True)
        return False
    if stored != current:
        print(f"WARNING: {what} was produced with game code {stored}; the current code is "
              f"{current}. Numbers from the two are NOT comparable.", flush=True)
        return False
    return True
