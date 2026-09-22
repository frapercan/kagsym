"""Asignacion optima de unidades a tareas: algoritmo humgaro en Python puro.

Por que no `scipy.optimize.linear_sum_assignment`: el entorno de submission
puede no traer scipy, y el agente no puede fallar por una dependencia -es el
mismo criterio que `policy.py` ya aplica a torch-. La matriz es minuscula
(<=16 unidades x ~100 tareas), asi que escribirlo a mano cuesta poco y
arriesgarse a un ImportError cuesta la partida entera.

`tests/test_assign.py` verifica contra scipy que el optimo coincide.

Implementacion: metodo humgaro por caminos de aumento mas cortos con
potenciales (Jonker-Volgenant), O(n^2 m) en el peor caso. Como siempre hay
columnas ficticias libres, el camino de aumento tipico tiene longitud 1 y el
coste real es O(n m).
"""
from __future__ import annotations

INF = float("inf")


def max_assignment(value: list[list[float]]) -> list[int]:
    """Asigna a cada fila una columna DISTINTA maximizando la suma total.

    `valor` tiene n filas (unidades) y m columnas (tareas), con n <= m.
    Devuelve una lista de longitud n: la columna asignada a cada fila.
    """
    n = len(value)
    if n == 0:
        return []
    m = len(value[0])
    if m < n:
        raise ValueError(f"hacen falta al menos tantas columnas como filas ({n} > {m})")

    # El algoritmo minimiza; se niega el valor para maximizar.
    cost = [[-x for x in row] for row in value]

    u = [0.0] * (n + 1)          # potencial de fila
    v = [0.0] * (m + 1)          # potencial de columna
    p = [0] * (m + 1)            # p[j] = fila asignada a la columna j (1-indexado)
    way = [0] * (m + 1)          # columna previa en el camino de aumento

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        usada = [False] * (m + 1)
        while True:
            usada[j0] = True
            i0 = p[j0]
            row = cost[i0 - 1]
            ui = u[i0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if usada[j]:
                    continue
                cur = row[j - 1] - ui - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if usada[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Deshace el camino, reasignando cada columna a la fila anterior.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    res = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            res[p[j] - 1] = j - 1
    return res
