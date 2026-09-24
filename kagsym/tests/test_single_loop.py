"""There is one game loop, and no instrument is blind.

Three failures on 2026-09-24 had the same shape: two definitions of the same
loop that diverged (a hand-split micro map, a history fed as zeros without
destinations, an evaluator that ignored the ramp in the checkpoint). The last
one is the dangerous kind: a blind instrument says "no effect" in the same
words as a real null.

These tests walk the syntax tree, not the text: three audits by inspection
said "nothing is left" and all three were wrong.
"""
from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Files allowed to read the macro head's output themselves. Every name here
# is one more definition of the loop and one more place to diverge; the list
# should shrink, never grow. The deployed loop is `kagsym/policy.py`.
MAY_EMIT_MACRO = {
    "kagsym/policy.py",            # the deployed policy: the reference
    "kagsym/environment.py",       # the training environment
    "kagsym/parallel_env.py",      # self-play copies inside training workers
    "kagsym/cli/train.py",         # the trainer
    "kagsym/nets/world.py",        # defines the output; does not play
    "tools/matriz.py",             # to be ported to kagsym.evaluate
    "tools/evalua.py",             # used by the trainer's --eval-cada; to be ported
}


def _python_files():
    for sub in ("tools", "kagsym", "kagsym/cli", "kagsym/nets", "kagsym/symbolic", "submit_kagsym"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if n.endswith(".py"):
                yield f"{sub}/{n}", os.path.join(d, n)


def _reads_macro_head(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and sl.value == "macro_mu":
                return True
    return False


def test_no_new_game_loops():
    intruders = []
    for rel, path in _python_files():
        if rel in MAY_EMIT_MACRO:
            continue
        with open(path, encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue
        if _reads_macro_head(tree):
            intruders.append(rel)
    assert not intruders, (
        "these files read the macro head themselves and are not declared in "
        "MAY_EMIT_MACRO: " + ", ".join(intruders) + ". Make them drive kagsym.policy.Policy.")


# Declared exceptions, each with its reason. Allocating zeros to fill on the
# next line is legitimate; handing them to the network is not.
ZEROS_THEN_FILLED = {
    "kagsym/environment.py",       # `Hf[i] = rival_flow(...)` follows the allocation
}


def test_history_is_never_zeros():
    """`zeros(..., N_HIST)` handed to the network is the signature of the bug
    that mutilated the agent ($76,607 -> $25,335 on the same board)."""
    culprits = []
    for rel, path in _python_files():
        if rel in ZEROS_THEN_FILLED:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if "N_HIST" in t and "zeros" in t and not t.startswith("#"):
                    culprits.append(f"{rel}: {t[:70]}")
    assert not culprits, "history fed as zeros: " + " | ".join(culprits)


def test_submission_wrapper_has_no_loop_of_its_own():
    """main.py may only delegate to Policy: no observation encoding, no network call."""
    with open(os.path.join(ROOT, "submit_kagsym", "main.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "encode_obs" not in names and "rival_flow" not in names
    assert not _reads_macro_head(tree)
