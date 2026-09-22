"""STRUCTURAL audit of the decision cascade.

Earlier audits looked for CONSTANTS (`grep '^[A-Z_]* ='`) and therefore let two
whole families through:

  * literals inside function bodies -4 found in one pass, two of them worth
    +-22% each-
  * fixed SHAPES: a `min` that is an impassable ceiling, a loop that splits a
    budget and discards the remainder, a hard-wired preference order. They are
    not numbers, so no grep for numbers sees them.

This enumerates every decision site of every function in the cascade and prints
it for manual classification. It decides nothing: it only guarantees nobody is
looking the other way.
"""
import ast, os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kagsym")
# EVERYTHING that models behaviour. `spec.py` and `fastenv.py` are deliberately
# left out: they are the ENGINE -facts, not policy- and their numbers are meant
# to be fixed. `nets/` is left out too: it is architecture, and its sizes are
# chosen by measurement (see the architecture table), not game decisions.
CASCADE = ["symbolic/executor.py", "symbolic/tasks.py", "symbolic/market_ops.py",
           "symbolic/assignment.py", "macro.py", "obs.py", "reward.py",
           "environment.py", "potential.py", "parallel_env.py"]

# what is NOT a decision: engine facts and numerical guards
EXEMPT = ("spec.", "1e-6", "1e-9", "1e-12", "0.0)", "1.0)", "len(", "range(",
          "BOARD", "TURNS_PER_DAY", "EPISODE_STEPS", "DEFAULT_CONFIG")


def sites(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines()
    out = []
    for node in ast.walk(tree):
        # thresholds and comparisons against a literal
        if isinstance(node, ast.Compare):
            for c in node.comparators:
                if (isinstance(c, ast.Constant)
                        and isinstance(c.value, (int, float))
                        and c.value not in (0, 1, -1, True, False)):
                    out.append(("threshold", node.lineno))
        # min/max/sorted: ceilings, floors and preference orders
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "sorted" or (
                    node.func.id in ("min", "max")
                    and not any(isinstance(x, ast.Constant) and x.value in (0, 1, 0.0, 1.0)
                                for x in node.args)):
                out.append((node.func.id, node.lineno))
        # arithmetic against a literal: scales
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and isinstance(side.value, float):
                    out.append(("scale", node.lineno))
    # dedup by line, and drop the exempt ones
    seen, res = set(), []
    for kind, ln in sorted(out, key=lambda x: x[1]):
        if ln in seen:
            continue
        txt = lines[ln - 1].strip()
        if any(e in txt for e in EXEMPT):
            continue
        seen.add(ln)
        res.append((kind, ln, txt[:96]))
    return res


if __name__ == "__main__":
    total = 0
    for f in CASCADE:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        s = sites(p)
        if not s:
            continue
        print(f"\n===== {f}  ({len(s)} sites) =====")
        for kind, ln, txt in s:
            print(f"  {kind:<9} {ln:>5}  {txt}")
        total += len(s)
    print(f"\n  TOTAL: {total} decision sites to classify")
