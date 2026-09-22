"""Bloques compartidos por las redes.

`symlog` (Dreamer v3) en lugar de normalizacion ajustada: el dinero va de 0 a
10^5 y los rendimientos de 0 a 10, y una escala estimada sobre la poblacion
actual envejece en cuanto cambian los rivales. symlog no tiene estadisticas que
mantener.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x):
    """sign(x) * log(1+|x|). Comprime colas sin perder el signo ni el cero.

    Sustituye a normalizar con media y desviacion medidas del buffer. La
    diferencia importa: unas estadisticas ajustadas describen la poblacion de
    rivales del dia que se entreno, y si esa poblacion cambia quedan mal
    calibradas en silencio. symlog no estima nada, asi que no puede envejecer.
    Es lo que usa Dreamer v3 para funcionar en dominios dispares sin retocar
    hiperparametros.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(y):
    """Inversa de symlog."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(x + self.n2(self.c2(h)))


class BloqueSep(nn.Module):
    """ResBlock con la 3x3 SEPARADA en profundidad + punto.

    Por que. Medido el 2026-09-21 sobre 17.001 transiciones (predecir el verbo
    del experto, base 0,521): este bloque iguala EXACTAMENTE a `ResBlock`
    -0,977 los dos- con 26,8 MFLOPs frente a 182,7, o sea 6,8x menos. La
    convolucion 3x3 densa gasta c*c*9 = 147k MAC por pixel; separada en una 3x3
    por canal mas una 1x1 de mezcla son c*9 + c*c = 17,5k, 8,4x menos, con el
    MISMO campo receptivo. La CNN no era cara por ser CNN.

    En la misma tabla, un transformer sobre las 100 casillas saca 0,969 con
    90,5 MFLOPs: queda DOMINADO -mas caro y peor-. Y un agregado global tipo
    Deep Sets se queda en 0,904 frente al 0,896 de no mezclar nada, o sea que
    lo que falta no es contexto global sino geometria LOCAL. El sesgo inductivo
    de la convolucion era el correcto; solo estaba implementado caro.

    Idea de Sifre (2014), popularizada por Xception y MobileNet (2017).
    """

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.d1 = nn.Conv2d(c, c, k, padding=p, groups=c, bias=False)
        self.p1 = nn.Conv2d(c, c, 1, bias=False)
        self.n1 = nn.GroupNorm(8, c)
        self.d2 = nn.Conv2d(c, c, k, padding=p, groups=c, bias=False)
        self.p2 = nn.Conv2d(c, c, 1, bias=False)
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.gelu(self.n1(self.p1(self.d1(x))))
        return F.gelu(x + self.n2(self.p2(self.d2(h))))
