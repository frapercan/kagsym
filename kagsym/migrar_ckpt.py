"""Migrar checkpoints entrenados antes de las features de horizonte absoluto.

El 2026-09-21 se anadieron dos dimensiones globales al final del bloque `time`
-`EPISODE_STEPS/720` y `tope_peones/PEONES_REF`- para que la red SEPA en que
liga juega. N_GLOBAL paso de 88 a 90, asi que la entrada del codificador global
(N_GLOBAL + N_HIST) paso de 124 a 126.

Consecuencia que costo media sesion encontrar: el patron habitual de carga

    ok = {k: v for k, v in d["sd"].items() if k in act and act[k].shape == v.shape}
    net.load_state_dict({**act, **ok})

descarta EN SILENCIO los tensores cuya forma no cuadra, y los deja como los
dejo la inicializacion: AL AZAR. Los dos descartados eran `mundo.glob_enc.0` y
`mundo.resumen.0`, o sea la modulacion FiLM de todo el tablero y el resumen que
alimenta macro, critico y cabeza auxiliar. Sintomas: dinero por semilla entre
19 $ y 40.667 $, y el MISMO checkpoint con la MISMA semilla dando resultados
distintos en cada proceso, porque el azar del init cambia.

La migracion es exacta. Las columnas nuevas van a CERO, asi que las features
nuevas no contribuyen nada y la funcion aprendida se recupera bit a bit; a
partir de ahi el entrenamiento puede darles peso si le sirven.
"""
from __future__ import annotations

import torch

from . import obs as O
from .redes.mundo import N_HIST

# Los rasgos absolutos nuevos van SIEMPRE al final del bloque `time`, y esta
# migracion lo da por hecho. Si alguna vez se anade uno en otra posicion, esto
# desplaza los pesos equivocados EN SILENCIO, que es exactamente el fallo que
# costo media sesion.
#
# 2026-09-21 (tarde): tercer rasgo absoluto, `TURNS_PER_DAY/24`. N_GLOBAL 90->91
# y la entrada 126->127. Ahora hay TRES anchos vivos, asi que el numero de
# columnas a insertar se deduce del ancho de origen en vez de estar fijo:
#     124 -> 127   inserta 3   (checkpoint anterior a todo)
#     126 -> 127   inserta 1   (checkpoint de anoche)
_FIN_TIME = O.GLOBAL_SLICES["time"].stop
NUEVO = O.N_GLOBAL + N_HIST
ANCHOS_CONOCIDOS = (NUEVO, NUEVO - 1, NUEVO - 3)


def migra_entrada(w: torch.Tensor) -> torch.Tensor:
    """(out, ancho viejo) -> (out, NUEVO), con ceros en las columnas nuevas."""
    ancho = w.shape[1]
    if ancho == NUEVO:
        return w
    if ancho not in ANCHOS_CONOCIDOS:
        raise ValueError(f"ancho inesperado {ancho}, esperaba uno de "
                         f"{ANCHOS_CONOCIDOS}")
    n_nuevas = NUEVO - ancho
    i = _FIN_TIME - n_nuevas
    out = w.new_zeros(w.shape[0], NUEVO)
    out[:, :i] = w[:, :i]
    out[:, i + n_nuevas:] = w[:, i:]
    return out


def migra_salida_macro(sd: dict) -> list:
    """Cabeza macro que CRECE, preservando la funcion aprendida.

    Cada vez que se expone una constante que estaba a ojo, el vector gana
    dimensiones y un checkpoint anterior deja de encajar. Extenderlo es
    correcto, pero solo de una forma:

      * peso  -> filas nuevas a CERO. Las dimensiones nuevas no dependen
                 todavia del estado, igual que cuando eran constantes.
      * sesgo -> logit(defecto del campo). El ejecutor hace sigmoid(macro_mu),
                 asi que eso devuelve EXACTAMENTE el valor que tenia la
                 constante, y la politica migrada se comporta igual.
      * sigma -> la MEDIANA de lo que el checkpoint ya habia aprendido, para
                 que lo nuevo explore a la misma escala que lo demas y no haya
                 que elegir un numero.

    El defecto se lee del propio `Macro`, no de una lista paralela: antes se
    daba por hecho que las dims nuevas eran el bloque `RANGOS_F` entero, y al
    anadir campos que no estan en esa tabla -los pesos de la regla por turno-
    el indice se salia. Leyendo los `dataclass` fields esto vale para
    cualquier ampliacion futura sin tocarlo.
    """
    import math
    from dataclasses import fields as _campos
    from .macro import Macro, N_MACRO
    tocadas = []
    porde = [float(f.default) for f in _campos(Macro)]
    lg = lambda x: math.log(max(1e-6, min(1 - 1e-6, x)) /
                            (1 - max(1e-6, min(1 - 1e-6, x))))
    for k, v in list(sd.items()):
        if k.endswith("macro_mu.weight") and v.shape[0] < N_MACRO:
            w = v.new_zeros(N_MACRO, v.shape[1]); w[: v.shape[0]] = v
            sd[k] = w; tocadas.append(f"{k}: {v.shape[0]} -> {N_MACRO} (a cero)")
        elif k.endswith("macro_mu.bias") and v.shape[0] < N_MACRO:
            b_ = v.new_empty(N_MACRO); b_[: v.shape[0]] = v
            for i in range(v.shape[0], N_MACRO):
                b_[i] = lg(porde[i])
            sd[k] = b_; tocadas.append(f"{k}: {v.shape[0]} -> {N_MACRO} (defecto del campo)")
        elif k.endswith("log_sigma") and v.shape[0] < N_MACRO:
            t = v.new_full((N_MACRO,), float(v.median()))
            t[: v.shape[0]] = v
            sd[k] = t; tocadas.append(f"{k}: {v.shape[0]} -> {N_MACRO} (mediana)")
    return tocadas


def migra_sd(sd: dict) -> tuple[dict, list[str]]:
    """Devuelve (state_dict migrado, lista de claves tocadas)."""
    out, tocadas = dict(sd), []
    tocadas += migra_salida_macro(out)
    for k in ("mundo.glob_enc.0.weight", "mundo.resumen.0.weight"):
        if k in out and out[k].shape[1] != NUEVO:
            out[k] = migra_entrada(out[k])
            tocadas.append(k)
    return out, tocadas


def carga_estricta(net, sd, nombre="checkpoint"):
    """Carga migrando, y REVIENTA si queda algo sin cargar.

    Deliberadamente ruidoso: el fallo original fue silencioso. Si un tensor no
    encaja, mejor una excepcion que una politica con partes al azar que parece
    funcionar y da numeros sin sentido.
    """
    sd, tocadas = migra_sd(sd)
    act = net.state_dict()
    # SIGMA DE LA CABEZA MICRO. Antes era constante del config; desde el
    # 2026-09-22 es un `nn.Parameter` que aprende PPO, como el del macro. Un
    # checkpoint anterior no lo trae, asi que se reconstruye con los valores
    # que ESE checkpoint usaba -canal 0 = sigma_micro, resto = sigma_ops-, y
    # entonces reproduce su conducta exacta en vez de arrancar en cualquier
    # sitio. Sin esto `carga_estricta` revienta, que es lo correcto pero
    # impediria reanudar nada anterior.
    if "log_sigma_micro" in act and "log_sigma_micro" not in sd:
        import math
        cfgd = getattr(net, "cfg", None)
        s_val = float(getattr(cfgd, "sigma_micro", 0.15) or 0.15)
        s_ops = float(getattr(cfgd, "sigma_ops", 0.03) or 0.03)
        t = act["log_sigma_micro"].clone()
        t[:] = math.log(s_ops)
        t[0] = math.log(s_val)
        sd = {**sd, "log_sigma_micro": t}
        tocadas = list(tocadas) + ["log_sigma_micro (reconstruido del cfg)"]
    mal = [(k, tuple(act[k].shape), tuple(sd[k].shape))
           for k in act if k in sd and act[k].shape != sd[k].shape]
    falta = [k for k in act if k not in sd]
    if mal or falta:
        raise RuntimeError(
            f"{nombre}: no encaja y se quedaria AL AZAR -> "
            f"formas distintas {mal}, ausentes {falta}")
    net.load_state_dict({**act, **sd})
    return tocadas


def carga_tolerante(net, sd, nombre="checkpoint", verbose=True):
    """Migra, carga lo que encaje, y DICE EN VOZ ALTA lo que queda al azar.

    La tolerancia es deliberada en algunos sitios (`--init-net` arranca de un
    preentreno con otra cabeza). Lo que no era deliberado es el silencio: el
    mensaje decia "56/58 tensores" sin nombrarlos, y los dos ausentes eran el
    codificador global y el resumen. Aqui se nombran, siempre.
    """
    sd, tocadas = migra_sd(sd)
    act = net.state_dict()
    # SIGMA DE LA CABEZA MICRO. Antes era constante del config; desde el
    # 2026-09-22 es un `nn.Parameter` que aprende PPO, como el del macro. Un
    # checkpoint anterior no lo trae, asi que se reconstruye con los valores
    # que ESE checkpoint usaba -canal 0 = sigma_micro, resto = sigma_ops-, y
    # entonces reproduce su conducta exacta en vez de arrancar en cualquier
    # sitio. Sin esto `carga_estricta` revienta, que es lo correcto pero
    # impediria reanudar nada anterior.
    if "log_sigma_micro" in act and "log_sigma_micro" not in sd:
        import math
        cfgd = getattr(net, "cfg", None)
        s_val = float(getattr(cfgd, "sigma_micro", 0.15) or 0.15)
        s_ops = float(getattr(cfgd, "sigma_ops", 0.03) or 0.03)
        t = act["log_sigma_micro"].clone()
        t[:] = math.log(s_ops)
        t[0] = math.log(s_val)
        sd = {**sd, "log_sigma_micro": t}
        tocadas = list(tocadas) + ["log_sigma_micro (reconstruido del cfg)"]
    ok = {k: v for k, v in sd.items() if k in act and act[k].shape == v.shape}
    azar = [k for k in act if k not in ok]
    net.load_state_dict({**act, **ok})
    if verbose:
        if tocadas:
            print(f"{nombre}: migrados a {NUEVO} entradas -> {', '.join(tocadas)}",
                  flush=True)
        if azar:
            print(f"AVISO {nombre}: {len(azar)} tensores NO cargados, quedan AL "
                  f"AZAR -> {', '.join(azar)}", flush=True)
        else:
            print(f"{nombre}: {len(ok)}/{len(act)} tensores, carga COMPLETA",
                  flush=True)
    return len(ok), len(act), azar
