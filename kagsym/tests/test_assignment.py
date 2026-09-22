"""El humgaro propio contra scipy: mismo optimo, siempre.

scipy se usa SOLO aqui, como oraculo de test. El agente no lo importa nunca
-el contenedor de submission puede no traerlo, y esa es toda la razon de
escribir el algoritmo a mano-.

    python tests/test_assign.py
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


def _optimo_scipy(m: np.ndarray) -> float:
    rows, cols = linear_sum_assignment(m, maximize=True)
    return float(m[rows, cols].sum())


def _suma(m: np.ndarray, assignment: list[int]) -> float:
    return float(sum(m[i, j] for i, j in enumerate(assignment)))


def test_coincide_con_scipy(n_casos: int = 200, verbose: bool = True) -> bool:
    """Matrices aleatorias, rectangulares, con valores positivos y negativos."""
    peor = 0.0
    for seed in range(n_casos):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 14))
        m = int(rng.integers(n, n + 40))
        mat = rng.normal(0, 50, size=(n, m))
        assignment = max_assignment(mat.tolist())
        assert len(set(assignment)) == n, f"columnas repetidas en semilla {seed}"
        assert all(0 <= j < m for j in assignment), f"columna fuera de rango ({seed})"
        mio, opt = _suma(mat, assignment), _optimo_scipy(mat)
        peor = max(peor, abs(mio - opt))
        assert abs(mio - opt) < 1e-6, f"subóptimo en semilla {seed}: {mio} vs {opt}"
    if verbose:
        print(f"OK: {n_casos} matrices aleatorias, desviacion maxima del optimo {peor:.2e}")
    return True


def test_empates_y_ceros(n_casos: int = 100, verbose: bool = True) -> bool:
    """El caso REAL: muchas casillas vacias valen exactamente lo mismo y las
    columnas ficticias valen 0. Los empates rompen implementaciones ingenuas."""
    for seed in range(n_casos):
        rng = np.random.default_rng(10_000 + seed)
        n = int(rng.integers(1, 12))
        m = n + int(rng.integers(0, 20))
        mat = rng.choice([0.0, 0.0, 0.0, 1.0, 2.5, 2.5, -3.0], size=(n, m))
        assignment = max_assignment(mat.tolist())
        mio, opt = _suma(mat, assignment), _optimo_scipy(mat)
        assert abs(mio - opt) < 1e-6, f"subóptimo con empates en semilla {seed}: {mio} vs {opt}"
    if verbose:
        print(f"OK: {n_casos} matrices con empates, ceros y negativos")
    return True


def test_bordes(verbose: bool = True) -> bool:
    assert max_assignment([]) == []
    assert max_assignment([[1.0, 9.0, 3.0]]) == [1]
    mat = [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]
    assert _suma(np.array(mat), max_assignment(mat)) == 9.0
    try:
        max_assignment([[1.0], [2.0]])           # menos columnas que filas
    except ValueError:
        pass
    else:
        raise AssertionError("deberia exigir n <= m")
    if verbose:
        print("OK: casos borde (vacio, una fila, cuadrada, n>m)")
    return True


def test_presupuesto(verbose: bool = True) -> bool:
    """Peor caso realista: 16 unidades x (100 casillas + 16 ficticias).

    El limite duro es 1 s/turno con bolsa de 60 s para 720 turnos (~83 ms de
    media), y la asignacion es solo una parte del turno.
    """
    rng = random.Random(7)
    n, m = 16, 116
    peor = 0.0
    for _ in range(100):
        mat = [[rng.uniform(0, 100) for _ in range(m)] for _ in range(n)]
        t0 = time.perf_counter()
        max_assignment(mat)
        peor = max(peor, time.perf_counter() - t0)
    assert peor < 0.020, f"demasiado lento: {peor*1000:.1f} ms"
    if verbose:
        print(f"OK: peor caso {n}x{m} en {peor*1000:.2f} ms (presupuesto ~83 ms de turno)")
    return True


if __name__ == "__main__":
    test_coincide_con_scipy()
    test_empates_y_ceros()
    test_bordes()
    test_presupuesto()
    print("\nTODOS OK")
