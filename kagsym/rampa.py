"""El desplazamiento del macro, definido UNA sola vez.

Existe porque hoy se perdio una noche entera midiendo con un bucle que no era
el del agente que se sube. La leccion no es "revisalo": es que no puede haber
dos definiciones. La busqueda (`tools/cem_diales.py`), el validador
(`tools/valida_delta.py`) y el agente desplegado (`submit_kagsym/main.py`)
importan esto y nada mas.

Dos formas, y la de arriba es un caso de la de abajo:

  CONSTANTE (len == n_vivos)   v = a
      Se hornea sumandolo al BIAS de `macro_mu`, que es un Linear(256,67):
      coste cero por turno y ni una linea en inferencia.

  RAMPA (len == 2*n_vivos)     v = a + b*progreso
      NO cabe en el bias, porque el bias es constante y esto depende del dia.
      Viaja en el checkpoint como `ck["delta_rampa"]` y se aplica aqui. El
      coste por turno es una suma de 67 floats una vez al dia.

`progreso` es dia/dias_totales, y los dias se derivan de la configuracion del
motor (`episodeSteps // turnsPerDay`), nunca de un 30 escrito a mano.
"""
import numpy as np


def offset(delta, vivos, progreso, n_macro):
    """Vector de `n_macro` para sumar a `macro_mu` ANTES del sigmoide."""
    off = np.zeros(int(n_macro), dtype=np.float32)
    if delta is None:
        return off
    d = np.asarray(delta, dtype=np.float32)
    n = len(vivos)
    v = d[:n] + d[n:2 * n] * float(progreso) if len(d) >= 2 * n else d[:n]
    off[list(vivos)] = v
    return off


def es_rampa(delta, vivos) -> bool:
    return delta is not None and len(np.asarray(delta)) >= 2 * len(vivos)


def del_checkpoint(ck):
    """(delta, vivos) si el checkpoint lleva una rampa; (None, None) si no.

    Un checkpoint sin el campo se comporta EXACTAMENTE como antes: el agente
    desplegado suma un vector de ceros, que es la identidad.
    """
    r = (ck or {}).get("delta_rampa")
    if not r:
        return None, None
    return np.asarray(r["delta"], dtype=np.float32), list(r["vivos"])
