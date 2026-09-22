"""Auditoria ESTRUCTURAL de la cascada de decision.

Las auditorias anteriores buscaban CONSTANTES (`grep '^[A-Z_]* ='`) y por eso
dejaron pasar dos familias enteras:

  * literales dentro del cuerpo de las funciones -4 encontrados el 2026-09-22,
    dos de ellos valian +-22 %-
  * FORMAS fijas: un `min` que es techo infranqueable, un bucle que reparte
    presupuesto y tira lo que sobra, un orden de preferencia cableado. No son
    numeros, asi que ningun grep de numeros los ve.

Esto enumera cada punto de decision de cada funcion de la cascada y lo saca
para clasificarlo a mano. No decide nada: solo garantiza que no se mira a otro
lado.
"""
import ast, os, sys

RAIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kagsym")
# TODO lo que modela conducta. `spec.py` y `fastenv.py` quedan fuera a
# proposito: son el MOTOR -hechos, no politica- y ahi los numeros deben estar
# fijos. `redes/` tampoco: es arquitectura, y sus tamanos se eligen por medida
# (ver la tabla de arquitecturas), no son decisiones de juego.
CASCADA = ["symbolic/executor.py", "symbolic/tasks.py", "symbolic/market_ops.py",
           "symbolic/assignment.py", "macro.py", "obs.py", "reward.py",
           "environment.py", "potential.py", "parallel_env.py"]

# lo que NO es decision: hechos del motor y guardas numericas
EXENTO = ("spec.", "1e-6", "1e-9", "1e-12", "0.0)", "1.0)", "len(", "range(",
          "BOARD", "TURNS_PER_DAY", "EPISODE_STEPS", "DEFAULT_CONFIG")


def sitios(path):
    src = open(path, encoding="utf-8").read()
    arbol = ast.parse(src)
    lineas = src.splitlines()
    out = []
    for nodo in ast.walk(arbol):
        # umbrales y comparaciones con literal
        if isinstance(nodo, ast.Compare):
            for c in nodo.comparators:
                if (isinstance(c, ast.Constant)
                        and isinstance(c.value, (int, float))
                        and c.value not in (0, 1, -1, True, False)):
                    out.append(("umbral", nodo.lineno))
        # min/max/sorted: techos, suelos y ordenes de preferencia
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name):
            if nodo.func.id == "sorted" or (
                    nodo.func.id in ("min", "max")
                    and not any(isinstance(x, ast.Constant) and x.value in (0, 1, 0.0, 1.0)
                                for x in nodo.args)):
                out.append((nodo.func.id, nodo.lineno))
        # aritmetica con literal: escalas
        if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, (ast.Mult, ast.Div)):
            for lado in (nodo.left, nodo.right):
                if isinstance(lado, ast.Constant) and isinstance(lado.value, float):
                    out.append(("escala", nodo.lineno))
    # dedup por linea, y fuera lo exento
    vistos, res = set(), []
    for tipo, ln in sorted(out, key=lambda x: x[1]):
        if ln in vistos:
            continue
        txt = lineas[ln - 1].strip()
        if any(e in txt for e in EXENTO):
            continue
        vistos.add(ln)
        res.append((tipo, ln, txt[:96]))
    return res


if __name__ == "__main__":
    total = 0
    for f in CASCADA:
        p = os.path.join(RAIZ, f)
        if not os.path.exists(p):
            continue
        s = sitios(p)
        if not s:
            continue
        print(f"\n===== {f}  ({len(s)} sitios) =====")
        for tipo, ln, txt in s:
            print(f"  {tipo:<7} {ln:>5}  {txt}")
        total += len(s)
    print(f"\n  TOTAL: {total} puntos de decision a clasificar")
