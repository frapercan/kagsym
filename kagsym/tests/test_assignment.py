"""Our Hungarian against scipy: the same optimum, always.

scipy is used ONLY here, as a test oracle. The agent never imports it -the
submission container may not ship it, and that is the whole reason for writing
the algorithm by hand-.

    python -m pytest kagsym/tests/test_assignment.py
"""
from __future__ import annotations

import random
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kagsym.symbolic.assignment import max_assignment


def _scipy_optimum(m: np.ndarray) -> float:
    rows, cols = linear_sum_assignment(m, maximize=True)
    return float(m[rows, cols].sum())


def _sum_values(m: np.ndarray, assignment: list[int]) -> float:
    return float(sum(m[i, j] for i, j in enumerate(assignment)))


def test_coincide_con_scipy(n_cases: int = 200, verbose: bool = True) -> bool:
    """Random rectangular matrices, with positive and negative values."""
    worst = 0.0
    for seed in range(n_cases):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 14))
        m = int(rng.integers(n, n + 40))
        mat = rng.normal(0, 50, size=(n, m))
        assignment = max_assignment(mat.tolist())
        assert len(set(assignment)) == n, f"columnas repetidas en semilla {seed}"
        assert all(0 <= j < m for j in assignment), f"columna fuera de rango ({seed})"
        ours, opt = _sum_values(mat, assignment), _scipy_optimum(mat)
        worst = max(worst, abs(ours - opt))
        assert abs(ours - opt) < 1e-6, f"suboptimal at seed {seed}: {ours} vs {opt}"
    if verbose:
        print(f"OK: {n_cases} random matrices, max deviation from the optimum {worst:.2e}")
    return True


def test_empates_y_ceros(n_cases: int = 100, verbose: bool = True) -> bool:
    """The REAL case: many empty tiles are worth exactly the same and the dummy
    columns are worth 0. Ties break naive implementations."""
    for seed in range(n_cases):
        rng = np.random.default_rng(10_000 + seed)
        n = int(rng.integers(1, 12))
        m = n + int(rng.integers(0, 20))
        mat = rng.choice([0.0, 0.0, 0.0, 1.0, 2.5, 2.5, -3.0], size=(n, m))
        assignment = max_assignment(mat.tolist())
        ours, opt = _sum_values(mat, assignment), _scipy_optimum(mat)
        assert abs(ours - opt) < 1e-6, f"suboptimal with ties at seed {seed}: {ours} vs {opt}"
    if verbose:
        print(f"OK: {n_cases} matrices with ties, zeros and negatives")
    return True


def test_bordes(verbose: bool = True) -> bool:
    assert max_assignment([]) == []
    assert max_assignment([[1.0, 9.0, 3.0]]) == [1]
    mat = [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]
    assert _sum_values(np.array(mat), max_assignment(mat)) == 9.0
    try:
        max_assignment([[1.0], [2.0]])           # fewer columns than rows
    except ValueError:
        pass
    else:
        raise AssertionError("deberia exigir n <= m")
    if verbose:
        print("OK: edge cases (empty, one row, square, n>m)")
    return True


def test_presupuesto(verbose: bool = True) -> bool:
    """Realistic worst case: 16 units x (100 tiles + 16 dummies).

    The hard limit is 1 s/turn with a 60 s pool for 720 turns (~83 ms on
    average), and the assignment is only one part of the turn.
    """
    rng = random.Random(7)
    n, m = 16, 116
    worst = 0.0
    for _ in range(100):
        mat = [[rng.uniform(0, 100) for _ in range(m)] for _ in range(n)]
        t0 = time.perf_counter()
        max_assignment(mat)
        worst = max(worst, time.perf_counter() - t0)
    assert worst < 0.020, f"demasiado lento: {worst*1000:.1f} ms"
    if verbose:
        print(f"OK: worst case {n}x{m} en {worst*1000:.2f} ms (budget ~83 ms per turn)")
    return True


if __name__ == "__main__":
    test_coincide_con_scipy()
    test_empates_y_ceros()
    test_bordes()
    test_presupuesto()
    print("\nALL OK")
