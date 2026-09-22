"""Blocks shared by the networks.

`symlog` (Dreamer v3) instead of fitted normalisation: money ranges from 0 to
10^5 and yields from 0 to 10, and a scale estimated over the current population
goes stale as soon as the opponents change. symlog keeps no statistics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x):
    """sign(x) * log(1+|x|). Compresses tails without losing sign or zero.

    It replaces normalising by a mean and standard deviation measured from the
    buffer. The difference matters: fitted statistics describe the population
    of opponents present on the day they were trained on, and when that
    population changes they are silently miscalibrated. symlog estimates
    nothing, so it cannot go stale. This is what Dreamer v3 uses to work across
    unrelated domains without retuning hyperparameters.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(y):
    """Inverse of symlog."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class ResBlock(nn.Module):
    """Residual block with two dense 3x3 convolutions and GroupNorm."""

    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(x + self.n2(self.c2(h)))


class SeparableBlock(nn.Module):
    """ResBlock with the 3x3 SPLIT into depthwise plus pointwise.

    Measured on 17,001 transitions (predicting the expert's verb, baseline
    0.521): this block matches `ResBlock` EXACTLY -0.977 both- at 26.8 MFLOPs
    against 182.7, i.e. 6.8x cheaper. A dense 3x3 costs c*c*9 = 147k MAC per
    pixel; split into a per-channel 3x3 plus a 1x1 mixer it is c*9 + c*c =
    17.5k, 8.4x less, with the SAME receptive field. The CNN was not expensive
    because it was a CNN.

    In the same table a transformer over the 100 tiles scores 0.969 at 90.5
    MFLOPs, so it is DOMINATED -more expensive and worse-. A global Deep
    Sets-style aggregate reaches 0.904 against 0.896 for mixing nothing at all,
    which says what is missing is not global context but LOCAL geometry. The
    inductive bias of convolution was the right one; it was just implemented
    expensively.

    NOT CURRENTLY IN USE: matching accuracy at 6.8x less compute did not
    translate into money in end-to-end training, and the dense block is what
    every measured checkpoint was trained with. Kept because the measurement
    stands and the trade-off may matter under a tighter time budget.

    Idea from Sifre (2014), popularised by Xception and MobileNet (2017).
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
