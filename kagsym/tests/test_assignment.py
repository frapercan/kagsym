"""Our Hungarian solver against scipy: the same optimum, always.

scipy is used ONLY here, as a test oracle. The agent never imports it: the
submission container may not ship it, which is the reason the algorithm is
written by hand.
"""
from __future__ import annotations

import os
import random
import sys
import time

import numpy as np
import pytest

scipy_opt = pytest.importorskip("scipy.optimize")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kagsym.symbolic.assignment import max_assignment  # noqa: E402


def _scipy_optimum(m: np.ndarray) -> float:
    rows, cols = scipy_opt.linear_sum_assignment(m, maximize=True)
    return float(m[rows, cols].sum())


def _sum_values(m: np.ndarray, assignment: list[int]) -> float:
    return float(sum(m[i, j] for i, j in enumerate(assignment)))


def test_matches_scipy(n_cases: int = 200):
    """Random rectangular matrices with positive and negative values."""
    for seed in range(n_cases):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 14))
        m = int(rng.integers(n, n + 40))
        mat = rng.normal(0, 50, size=(n, m))
        assignment = max_assignment(mat.tolist())
        assert len(set(assignment)) == n, f"repeated columns on seed {seed}"
        assert all(0 <= j < m for j in assignment), f"column out of range on seed {seed}"
        ours, opt = _sum_values(mat, assignment), _scipy_optimum(mat)
        assert abs(ours - opt) < 1e-6, f"suboptimal on seed {seed}: {ours} vs {opt}"


def test_ties_and_zeros(n_cases: int = 100):
    """The real case: many empty tiles worth exactly the same, dummies worth 0."""
    for seed in range(n_cases):
        rng = np.random.default_rng(10_000 + seed)
        n = int(rng.integers(1, 12))
        m = n + int(rng.integers(0, 20))
        mat = rng.choice([0.0, 0.0, 0.0, 1.0, 2.5, 2.5, -3.0], size=(n, m))
        assignment = max_assignment(mat.tolist())
        ours, opt = _sum_values(mat, assignment), _scipy_optimum(mat)
        assert abs(ours - opt) < 1e-6, f"suboptimal with ties on seed {seed}: {ours} vs {opt}"


def test_edge_cases():
    assert max_assignment([]) == []
    assert max_assignment([[1.0, 9.0, 3.0]]) == [1]
    mat = [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]
    assert _sum_values(np.array(mat), max_assignment(mat)) == 9.0
    with pytest.raises(ValueError):
        max_assignment([[1.0], [2.0]])           # fewer columns than rows


def test_turn_budget():
    """Worst realistic case, 16 units x (100 tiles + 16 dummies), in the turn budget.

    Wall-clock under load measures contention, not code, so the check is
    skipped when the machine is busy.
    """
    if os.getloadavg()[0] > (os.cpu_count() or 1) * 0.5:
        pytest.skip("machine under load; wall-clock timing would measure contention")
    rng = random.Random(7)
    n, m = 16, 116
    worst = 0.0
    for _ in range(100):
        mat = [[rng.uniform(0, 100) for _ in range(m)] for _ in range(n)]
        t0 = time.perf_counter()
        max_assignment(mat)
        worst = max(worst, time.perf_counter() - t0)
    assert worst < 0.020, f"too slow: {worst * 1000:.1f} ms (budget ~83 ms per turn)"
