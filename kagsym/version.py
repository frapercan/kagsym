"""Fingerprint of the code that produced a result.

The problem it solves cost a whole search: a 15-minute CEM run starts, and
while it runs the executor keeps being edited. The workers imported the module
when they were spawned, so they consistently measure **a world that no longer
exists**. The resulting vector is the optimum of a version of the code that is
gone, and comparing it against one obtained later compares two different games.

The same class of error shows up whenever a process outlives an edit: a
perfectly valid number inside the wrong world. Looking at the result does not
reveal it.

The fingerprint is stored next to every vector and every checkpoint. If it does
not match the current code, the comparison is not valid and must be flagged.
"""
from __future__ import annotations

import hashlib
import os

# The files that determine HOW THE GAME IS PLAYED. Changing any of them
# invalidates comparisons between results obtained before and after.
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
# Files that determine HOW LEARNING HAPPENS, not how the game is played. They
# are kept separate on purpose: a CEM vector stays comparable even if the
# network or the PPO loop is touched -the game has not changed- but two
# CHECKPOINTS do not, if the architecture or the training changed in between.
# Merging the two fingerprints would invalidate comparisons that are valid.
MODEL_FILES = [
    "kagsym/nets/world.py",
    "kagsym/nets/blocks.py",
    "kagsym/cli/train.py",
    "kagsym/environment.py",
    "kagsym/parallel_env.py",
    "kagsym/migrate_ckpt.py",
]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _digest(files) -> str:
    h = hashlib.sha256()
    for f in files:
        try:
            with open(os.path.join(ROOT, f), "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"?")
    return h.hexdigest()[:12]


def fingerprint() -> str:
    """12 hex digits summarising the executor and the observation encoding."""
    return _digest(GAME_FILES)


def model_fingerprint() -> str:
    """12 hex digits summarising the network and the training loop.

    Stored next to checkpoints. If it does not match, two checkpoints are not
    comparable even when the GAME fingerprint is.
    """
    return _digest(MODEL_FILES)


def check(stored, what: str = "result") -> bool:
    """Warn when a stored fingerprint is not the current code's."""
    current = fingerprint()
    if stored is None:
        print(f"WARNING: {what} has no code fingerprint; the comparison "
              f"cannot be validated")
        return False
    if stored != current:
        print(f"WARNING: {what} was produced with code {stored} and the "
              f"current one is {current}. The comparison is NOT valid.")
        return False
    return True
