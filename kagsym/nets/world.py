"""End-to-end architecture: one world embedding, two policies, auxiliary heads.

               grid (50 channels: 25 mine + 25 the opponent's)
               glob (75: time, money, market, town, my private state)
               hist (the opponent's REALISED flow, solved exactly)
                                  |
                            WorldEncoder
                                  |
                              h_t  (shared embedding)
                    +-------------+-------------+-------------+
                    |             |             |             |
               MACRO head    MICRO head      critic      RIVAL head
               vector/day    10x10 map        V(s)     cumulative flow
                                                         (auxiliary)

Three decisions, each tied to something measured:

1. ONE EMBEDDING for macro and micro. Both policies read the same state; what
   changes is the resolution of the decision, not the information. Sharing the
   trunk is also the only way for the micro signal -dense, every turn- to help
   shape the representation the macro uses -sparse, once a day-.

2. THE OPPONENT ENTERS THROUGH THREE PLACES: their board is already in `grid`
   (it is public), the shared market in `glob`, and their realised flow in
   `hist`. The only genuinely hidden parts are their shed, their seeds and what
   their units carry, and those are solved EXACTLY from the market inventory
   one turn later.

3. THE AUXILIARY HEAD PREDICTS A CUMULATIVE LEVEL, NOT TIMING. Measured over
   8,360 held-out transitions: predicting the turn-by-turn flow is WORSE than
   not correcting (-26%), while the opponent's money level (+19%) and spending
   (+35%) are predicted well. It is asked for the latter. The labels are exact,
   so it is free supervision that shapes the embedding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import obs as O
from ..macro import N_MACRO
from .blocks import SeparableBlock, ResBlock, symlog

N_PRODUCTS = 9
N_HIST = 4 * N_PRODUCTS      # opponent flow over 4 backward windows


@dataclass
class WorldConfig:
    width: int = 128
    blocks: int = 6
    hidden: int = 256
    sigma_micro: float = 0.15   # se calibra midiendo cuantas asignaciones voltea
    # END-TO-END BOARD PLAY: besides the per-tile value, the network emits one
    # logit per LEGAL operation. Measured: picking the verb at random among the
    # legal ones yields $8,692 against $245 for doing nothing (35x the floor),
    # so there is gradient from scratch. A FIXED random ranking yields $397:
    # the right operation depends on the state, which is exactly what a
    # hand-written heuristic -a fixed ranking- cannot capture and a
    # state-conditioned network can.
    con_ops: bool = False
    # SEPARATE SIGMA for the verb channels. The 0.15 of `sigma_micro` was
    # calibrated by "measuring how many assignments it flips" on the VALUE
    # channel, in symlog dollars; inheriting it for the logits of a choice
    # among 19 verbs had no justification. Measured: at 0.15 the head had moved
    # 0.0059 in 200 steps, signal/noise 0.039 -the verb was still 96% noise
    # after 50 updates- and covering that would have taken ~4,000 steps.
    sigma_ops: float = 0.03
    # Trunk convolution shape. "denso" = the usual dense 3x3; "sep" = depthwise
    # plus pointwise, which MEASURES the same at 6.8x less cost (see
    # SeparableBlock). The trunk is 90.5% of the parameters -1,960,192 of
    # 2,165,558- and 51% of profiled play time, so 6.8x there is ~1.8x more
    # episodes per hour overall.
    #
    # Changing it INVALIDATES dense checkpoints: the shapes do not match. The
    # path that does not throw the training away is distillation.
    # Context shape of the MICRO HEAD. This is where a measured jump came from
    # ($1,397 -> $1,949 purely from moving 1x1 to 3x3), and it agrees with what
    # the architecture probe already said: what is missing is LOCAL GEOMETRY,
    # not global context. "1x1" = no context, as it used to be.
    ctx_micro: str = "3x3"
    conv: str = "denso"
    nucleo: int = 3          # kernel of the separable block; 7 widens the receptive field
    lr: float = 3e-4
    peso_aux: float = 0.1
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class CodificadorMundo(nn.Module):
    """(grid, glob, hist) -> spatial embedding h and global summary g."""

    def __init__(self, cfg: WorldConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.stem = nn.Sequential(
            nn.Conv2d(O.N_GRID_CH, w, 3, padding=1, bias=False),
            nn.GroupNorm(8, w), nn.GELU())
        if getattr(cfg, "conv", "denso") == "sep":
            self.blocks = nn.Sequential(*[SeparableBlock(w, cfg.nucleo)
                                          for _ in range(cfg.blocks)])
        else:
            self.blocks = nn.Sequential(*[ResBlock(w) for _ in range(cfg.blocks)])
        # The global vector modulates via FiLM instead of entering as a
        # channel: the day, prices and money affect ALL tiles at once.
        self.glob_enc = nn.Sequential(
            nn.Linear(O.N_GLOBAL + N_HIST, cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, 2 * w))
        self.resumen = nn.Sequential(
            nn.Linear(O.N_GLOBAL + N_HIST, cfg.hidden), nn.GELU())

    def forward(self, grid, glob, hist=None):
        if hist is None:
            hist = torch.zeros(glob.shape[0], N_HIST, device=glob.device, dtype=glob.dtype)
        g = symlog(torch.cat([glob, hist], dim=-1))
        h = self.stem(symlog(grid))
        scale, sesgo = self.glob_enc(g).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + sesgo[:, :, None, None]
        return self.blocks(h), self.resumen(g)


class _TileAttention(nn.Module):
    """Self-attention over the 100 board tiles."""

    def __init__(self, w: int, cabezas: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(w)
        self.att = nn.MultiheadAttention(w, cabezas, batch_first=True)
        self.norm2 = nn.LayerNorm(w)
        self.ffn = nn.Sequential(nn.Linear(w, 2 * w), nn.SiLU(), nn.Linear(2 * w, w))

    def forward(self, h):
        b, c, H, W = h.shape
        x = h.flatten(2).transpose(1, 2)          # (b, 100, c)
        y = self.norm(x)
        x = x + self.att(y, y, y, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x.transpose(1, 2).reshape(b, c, H, W)


class E2EAgent(nn.Module):
    """Everything wired: one trunk, two policies, a critic and auxiliary heads."""

    def __init__(self, cfg: WorldConfig):
        super().__init__()
        self.cfg = cfg
        w, hid = cfg.width, cfg.hidden
        self.world = CodificadorMundo(cfg)
        self.cuerpo = nn.Sequential(
            nn.Linear(2 * w + hid, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU())
        # Gaussian in LOGIT space, not Beta. Measured reason: resampling the
        # macro every day is a random walk over 30 days and destroys strategic
        # coherence -seed is bought that never gets planted, hands are hired
        # with no work-. The measured cost of that noise, with the SAME vector:
        # $40,972 fixed against $18,967 sampled (-54%).
        #
        # With a Gaussian in logit space the perturbation `eps` can be FIXED
        # PER EPISODE and applied to all 30 days: the mean still depends on the
        # state -the policy can raise the sell aggressiveness as the close
        # approaches- but the random jitter disappears. This is exploring in
        # POLICY space, not in action space.
        self.macro_mu = nn.Linear(hid, N_MACRO)
        self.log_sigma = nn.Parameter(torch.full((N_MACRO,), float(np.log(0.35))))

        # Channel 0 = value of acting on the tile (symexp -> $).
        # Channels 1.. = one logit per verb of OPS_VOCAB.
        # They live in ONE tensor on purpose: PPO already does
        # flatten(1).sum(-1) over the micro, so the loss needs no change.
        from ..symbolic.tasks import N_OPS
        self.n_ops = N_OPS if cfg.con_ops else 0
        # CAPACITY OF THE HEAD THAT DECIDES. Measured by decomposition: the
        # micro map contributes +$952 of 953 and the macro +1, and yet this
        # head was 2,064 parameters -0.1% of the model- against 1.96M in the
        # encoder. A LINEAR probe on the trunk for the only output that decides
        # anything.
        #
        # The 3x3 is not only capacity: it gives SPATIAL CONTEXT. On this board
        # a tile's decision depends on its neighbours -whom you water first,
        # where you plant so as not to scatter the units- and a 1x1 cannot see
        # that by construction.
        #
        # The last layer still starts at ZERO, so the measured property holds:
        # the residual starts neutral and does not throw away the exact
        # valuation, which is already worth $75,157.
        _c = getattr(cfg, "ctx_micro", "3x3")
        if _c == "1x1":
            self.micro_ctx = nn.Identity()
        elif _c == "3x3x2":
            # receptive field 5 from two 3x3s: cheaper than a 5x5 and with a
            # nonlinearity in between.
            self.micro_ctx = nn.Sequential(
                nn.Conv2d(w, w, 3, padding=1), nn.SiLU(),
                nn.Conv2d(w, w, 3, padding=1), nn.SiLU())
        elif _c == "attn":
            # ATTENTION OVER THE 100 TILES, parameter-matched to the 3x3
            # (~130k against 147k) so the comparison isolates GLOBAL MIXING
            # against LOCAL GEOMETRY and not capacity.
            #
            # Why here and not in the trunk: the old probe measured the trunk
            # and scored IMITATING the expert -which we now know is the
            # bottleneck- so it does not answer this question. And the micro
            # head is where attention makes sense: it decides tile by tile and
            # tiles interact -a unit does one task, and going to one is not
            # going to another-. The 1x1 sees no neighbours, the 3x3 sees
            # eight, this sees all hundred.
            self.micro_ctx = _TileAttention(w)
        elif _c == "5x5":
            self.micro_ctx = nn.Sequential(nn.Conv2d(w, w, 5, padding=2), nn.SiLU())
        else:
            self.micro_ctx = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.SiLU())
        self.micro = nn.Conv2d(w, 1 + self.n_ops, 1)
        # MICRO HEAD SIGMA, LEARNED. It used to be `cfg.sigma_micro` and
        # `cfg.sigma_verb`, two config constants: the head that contributes
        # NOTHING -the macro- explored adaptively, and the one contributing
        # +$952 of 953 did so with a hand-picked number. And that number was
        # calibrated back when it was believed to be secondary.
        #
        # It is initialised at EXACTLY the previous values, so behaviour at
        # startup is identical; from there PPO moves it, with the same
        # mechanism already used for the macro.
        _nc = 1 + self.n_ops
        _ini = torch.full((_nc, 1, 1), float(np.log(cfg.sigma_ops)))
        _ini[0] = float(np.log(cfg.sigma_micro))
        self.log_sigma_micro = nn.Parameter(_ini)
        # CRITIC. It used to be `nn.Linear(hid, 1)`: 257 parameters, another
        # linear probe on the trunk. And out of it comes the ADVANTAGE that
        # guides all of PPO, so its error turns into gradient noise in both
        # policies. At toy scale it gave R2 0.96 and was not urgent; at
        # championship scale, with a real opponent and a shared market, is
        # where it is expected to break -there the opponent's liquidation
        # accounts for 99.1% of the daily variance-.
        self.critico = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(),
                                     nn.Linear(hid, 1))
        # AUXILIARY HEAD, back with a NEW TARGET. The previous one predicted
        # the opponent's products, which are already in the input -their board
        # is encoded in full- so it was a trivial task that forced the encoder
        # to do nothing. And nobody wired its loss: it received no gradient and
        # stayed at its initialisation.
        #
        # It now predicts what the opponent will have READY at 1, 2, 3, 5, 8
        # and 13 days ahead, from TODAY's state. Their supply today is an
        # input, so getting tomorrow right requires modelling how their farm
        # grows -what is maturing, what was just planted- which is exactly what
        # the encoder did not represent.
        #
        # Why it matters: their liquidation accounts for 99.1% of the variance
        # of the daily reward, which is why putting it in the shaping sank the
        # critic to R2 -2.535. The lesson was "shaping can only carry what the
        # state predicts"; this attacks the other half, teaching the state to
        # predict it.
        # JEPA. The auxiliary head above predicts RAW opponent units, and part
        # of that is unpredictable from our observation -their shed is private,
        # we do not see their policy-. Forcing the encoder to predict noise
        # spends capacity; that is the flaw in choosing that target.
        #
        # Here the target is the FUTURE EMBEDDING of the encoder itself, with a
        # STOPPED gradient. What is unpredictable disappears from the target
        # because the embedding only retains what the encoder considers
        # relevant.
        #
        # No momentum encoder for now (predictor + stop-grad). The risk is
        # COLLAPSE: if the encoder always emits the same vector, predicting it
        # is trivial. It is watched through the standard deviation of the
        # projections, logged as `2_salud/jepa_sd`.
        self.d_jepa = 64
        self.jepa_proy = nn.Linear(hid, self.d_jepa)
        from ..obs import AUX_HORIZONS as _HZ
        self.n_hz = len(_HZ)
        self.jepa_pred = nn.Sequential(
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, self.d_jepa * self.n_hz))
        from ..obs import N_AUX_RIVAL as _NAUX
        self.n_aux = _NAUX
        self.aux_rival = nn.Linear(hid, _NAUX)
        # (historical note) The previous version was withdrawn: the trainer's
        # loss is `l_pi + value_weight * l_v` and nobody touches its output, so
        # it received gradient from nothing -it stayed at its initialisation
        # forever- and only cost time on every pass. `aux_weight: 0.1` also
        # remains in the config, the weight of a loss that was never wired: the
        # fossil of a half-finished design.
        # The micro starts at ZERO residual: the exact valuation is already
        # worth $75,157 and starting below that point would throw the search
        # away.
        nn.init.zeros_(self.micro.weight)
        nn.init.zeros_(self.micro.bias)
        if self.n_ops:
            # Zero is NOT neutral here: symexp(0) = 0, the Hungarian prefers
            # its dummy column and ALL units PASS. Measured: $225, which is
            # exactly the floor of doing nothing.
            # The correct start is "acting is worth something positive and
            # every verb is equally likely", which is the legal random policy:
            # $8,692 +- 4,003 over 6 seeds, 35x that floor. It is not a
            # heuristic: it does not say WHAT to do, only that doing something
            # legal beats standing still.
            with torch.no_grad():
                self.micro.bias[0] = 1.0

    def tronco(self, grid, glob, hist=None):
        h, g = self.world(grid, glob, hist)
        z = self.cuerpo(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3)), g], -1))
        return h, z

    def forward(self, grid, glob, hist=None):
        h, z = self.tronco(grid, glob, hist)
        return {
            "macro_mu": self.macro_mu(z),
            "rival": self.aux_rival(z),
            "jepa_p": self.jepa_pred(z).reshape(-1, self.n_hz, self.d_jepa),
            "jepa_z": self.jepa_proy(z),
            "micro": (self.micro(self.micro_ctx(h)) if self.n_ops
                      else self.micro(self.micro_ctx(h)).squeeze(1)),
            "valor": self.critico(z).squeeze(-1),
        }

    def macro_from(self, out, eps):
        """Macro action and its log-prob, for the given perturbation `eps`.

        `eps` is sampled ONCE per episode and reused across all 30 days. The
        log-probability is evaluated on each step's Gaussian marginal, which is
        what PPO needs; the temporal correlation only changes HOW trajectories
        are explored, not the distribution of each action.
        """
        mu = out["macro_mu"]
        sigma = self.log_sigma.exp()
        pre = mu + sigma * eps
        a = torch.sigmoid(pre)
        lp = (torch.distributions.Normal(mu, sigma).log_prob(pre)
              - (torch.log(a + 1e-8) + torch.log1p(-a + 1e-8))).sum(-1)
        return a, lp

    def macro_logprob(self, out, a):
        """Recompute the log-prob of an action already taken (update phase)."""
        mu = out["macro_mu"]
        sigma = self.log_sigma.exp()
        a = a.clamp(1e-6, 1 - 1e-6)
        pre = torch.log(a) - torch.log1p(-a)
        return (torch.distributions.Normal(mu, sigma).log_prob(pre)
                - (torch.log(a) + torch.log1p(-a))).sum(-1)

    def init_macro_at(self, vector, concentration: float = 6.0) -> None:
        """Centre the initial macro on a known vector (the one CEM found)."""
        with torch.no_grad():
            nn.init.zeros_(self.macro_mu.weight)
            for i, m in enumerate(list(vector)[:N_MACRO]):
                m = min(0.995, max(0.005, float(m)))
                self.macro_mu.bias[i] = float(np.log(m) - np.log1p(-m))
