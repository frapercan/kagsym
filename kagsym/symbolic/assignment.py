"""Optimal assignment of units to tasks: Hungarian method, pure Python.

Why not `scipy.optimize.linear_sum_assignment`: the submission environment may
not ship scipy, and the agent cannot fail on a dependency. The matrix is tiny
(<=16 units x ~100 tasks), so hand-writing it is cheap while risking an
ImportError costs the whole episode.

`tests/test_assignment.py` checks against scipy that the optimum agrees.

Implementation: Hungarian method via shortest augmenting paths with potentials
(Jonker-Volgenant), O(n^2 m) worst case. Because there are always free dummy
columns, the typical augmenting path has length 1 and the real cost is O(n m).
"""
from __future__ import annotations

INF = float("inf")


def max_assignment(value: list[list[float]]) -> list[int]:
    """Assign each row a DISTINCT column, maximising the total sum.

    `value` has n rows (units) and m columns (tasks), with n <= m. Returns a
    list of length n: the column assigned to each row.
    """
    n = len(value)
    if n == 0:
        return []
    m = len(value[0])
    if m < n:
        raise ValueError(f"need at least as many columns as rows ({n} > {m})")

    # The algorithm minimises; negate the value to maximise.
    cost = [[-x for x in row] for row in value]

    u = [0.0] * (n + 1)          # row potential
    v = [0.0] * (m + 1)          # column potential
    p = [0] * (m + 1)            # p[j] = row assigned to column j (1-indexed)
    way = [0] * (m + 1)          # previous column on the augmenting path

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            row = cost[i0 - 1]
            ui = u[i0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - ui - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Walk the path back, reassigning each column to the previous row.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    res = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            res[p[j] - 1] = j - 1
    return res
