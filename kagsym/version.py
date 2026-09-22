"""Huella del codigo que produjo un resultado.

Problema que resuelve, y que costo una busqueda entera: un CEM de 15 minutos
arranca, y mientras corre se sigue tocando el ejecutor. Los trabajadores
importaron el modulo al nacer, asi que miden consistentemente... **un mundo que
ya no existe**. El vector resultante es el optimo de una version del codigo que
se ha ido, y compararlo con otro obtenido despues es comparar dos juegos.

Es la misma clase de error que las jaulas: un numero perfectamente valido dentro
de un mundo equivocado. Y no se detecta mirando el resultado.

La huella se guarda junto a cada vector y cada checkpoint. Si no coincide con la
del codigo actual, la comparacion no es valida y hay que decirlo.
"""
from __future__ import annotations

import hashlib
import os

# Los ficheros que determinan COMO se juega. Cambiar cualquiera invalida las
# comparaciones entre resultados obtenidos antes y despues.
FICHEROS = [
    "kagsym/exacto/tareas.py",
    "kagsym/exacto/mercado.py",
    "kagsym/exacto/ejecutor.py",
    "kagsym/exacto/asignacion.py",
    "kagsym/macro.py",
    "kagsym/potencial.py",
    "kagsym/recompensa.py",
    "kagsym/obs.py",
]
# Ficheros que determinan COMO SE APRENDE, no como se juega. Van aparte a
# proposito: un vector del CEM sigue siendo comparable aunque se toque la red o
# el lazo de PPO -el juego no ha cambiado-, pero dos CHECKPOINTS no lo son si la
# arquitectura o el entrenamiento cambiaron entre medias. Mezclar las dos
# huellas invalidaria comparaciones que si son validas.
FICHEROS_MODELO = [
    "kagsym/redes/mundo.py",
    "kagsym/redes/bloques.py",
    "kagsym/cli/entrenar_e2e.py",
    "kagsym/entorno.py",
    "kagsym/entorno_par.py",
    "kagsym/migrar_ckpt.py",
]
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def huella() -> str:
    """12 hex que resumen el ejecutor y la codificacion."""
    h = hashlib.sha256()
    for f in FICHEROS:
        p = os.path.join(RAIZ, f)
        try:
            with open(p, "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"?")
    return h.hexdigest()[:12]


def huella_modelo() -> str:
    """12 hex que resumen la red y el lazo de entrenamiento.

    Se guarda junto a los checkpoints. Si no coincide, dos checkpoints no son
    comparables aunque la huella del JUEGO si lo sea.
    """
    h = hashlib.sha256()
    for f in FICHEROS_MODELO:
        p = os.path.join(RAIZ, f)
        try:
            with open(p, "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"?")
    return h.hexdigest()[:12]


def comprueba(guardada, que: str = "resultado") -> bool:
    """Avisa si la huella guardada no es la del codigo actual."""
    actual = huella()
    if guardada is None:
        print(f"AVISO: {que} sin huella de codigo; no se puede validar la comparacion")
        return False
    if guardada != actual:
        print(f"AVISO: {que} se produjo con el codigo {guardada} y el actual es "
              f"{actual}. La comparacion NO es valida.")
        return False
    return True
