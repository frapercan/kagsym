"""End-to-end PPO over the daily macro vector and the per-tile micro map.

One RL step is one DAY. Both actions are sampled ONCE PER DAY and held fixed
for all 24 hours:

    macro  60 numbers   the day's targets and every exposed parameter
    micro  1+N_OPS x 10 x 10   per-tile value plus one logit per legal verb

Their log-probabilities are summed: it is a single policy over one composite
action. The episode is 30 steps, which is what makes credit assignment
tractable.

What this loop adds over the best FIXED vector is dependence on the state:
changing plan according to what the opponent is doing. The CEM converged
(sigma 0.011) inside a 7-number space and still lost against most public
agents, so a fixed vector is not enough by construction.

The loop carries three mechanisms that are not standard PPO and each exists for
a measured reason:

  * a KL-targeted learning rate, per head. Without it up to 75% of each batch
    was clipped away (saturation 0.28, sd_logratio 11.7).
  * a factored importance ratio: one ratio per head instead of one over 1,632
    summed dimensions, plus a mask so dimensions that could not change any
    action do not enter the ratio.
  * an automatic curriculum with FIXED quotas for self-play and for the target
    rung, because p(1-p) is zero at both extremes.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from kagsym import obs as O
from kagsym.environment import LADDER
from kagsym.parallel_env import ParallelEnv
from kagsym.macro import N_MACRO
from kagsym.version import fingerprint, model_fingerprint
# Saved with every checkpoint: which macro field each row of the head is.
# Without it, a head of an unknown width has to be guessed by position, and
# a removed dimension shifts the meaning of every row after it in silence
# (see `migrate_ckpt.shrink_macro_head`).
from dataclasses import fields as _dc_fields
from kagsym.macro import Macro as _Macro
_MACRO_FIELDS = [f.name for f in _dc_fields(_Macro)]
from kagsym.nets.world import E2EAgent, WorldConfig, N_HIST


# Money each LADDER rung makes against a PASSIVE opponent, measured over 3
# seeds. It is the reference for the shortfall. Note it is not a ladder: the
# levels that work are worth almost the same (171-180k).
_BASE_RIVAL = [0.0, 0.0, 179514.0, 171878.0, 171392.0]

# What v48 makes against a PASSIVE opponent, by (days, level). Without this
# the shortfall cannot be computed with mixed horizons: dividing by the 30-day
# mark makes a 15-day game look like a thrashing by us when it is merely short.
# Measured over 2 seeds. level: 1=cap3 2=cap5 3=cap8 4=uncapped.
BASE_PER_HORIZON = {
    15: {1:  2918, 2:  7727, 3: 14837, 4:  17078},
    20: {1:  6106, 2: 24157, 3: 34184, 4:  61028},
    30: {1: 16049, 2: 43094, 3: 80932, 4: 176422},
}

# LEAGUES: (episode_steps, LADDER level). You climb by winning and DESCEND by
# losing -descending matters as much as climbing, because it is what prevents
# catastrophic forgetting-.
#
# Why leagues rather than picking the rung by hand: measured, the right rung
# depends on the horizon. At 30 days the heuristic wins 3/3 at level 2 and 0/3
# at level 3; at 15 days it wins 3/3 at 2 and 0.667 at 3. Picking it by eye
# each time is what left us training at win=0.000 (no gradient in the win term)
# and at win=1.000 (likewise).
#
# And a short horizon is NOT cheaper per sample -the cost per decision is 24
# turns either way- but it does give 1.61x more TERMINALS per second at equal
# batch size, which is the scarce resource: the per-seed sd is ~$9,000 and only
# the terminal reduces it.
# (steps, public level, rival macro path). If a path is given, the opponent is
# OUR executor with that macro -optimised by CEM for THAT horizon- and the
# level is ignored. Measured: at 5 days the 5-day specialist beats the 30-day
# specialist $3,843 against $2,920 (+31%, 100% wins, symmetric both ways) while
# the strongest public agent makes $723. Without our own opponent, short
# leagues give win=1.000, which carries no more gradient than win=0.000.
# FULL LADDER: (hours per day, days, opponent hand cap).
#
# A DIAGONAL through the fibonacci grid: first the game grows (L0-L3), then at
# competition scale the opponent grows (L4-L8). Both axes are measured
# separately:
#
#   hours  2 and 3 give x_inaction 0.73 -> acting DESTROYS value, excluded
#   days   below 5 the same happens
#   cap    1 and 2 are degenerate (the public agent makes $1-2) and 13 is
#          indistinguishable from uncapped (175,984 against 176,422), so
#          fibonacci samples this axis BADLY: all the gradient lives in 9-12
#          and it skips the lot. Hence 3/5/8/11 here, which are the measured
#          ones.
#
# At reduced scale the public agent collapses to ~$0, so L0-L3 are not won by
# "competing": they are a scale ramp and get crossed quickly. The criterion
# that matters there is `x_inaction`, not winning.
# (hours per day, days, opponent hand cap).
#
# THE SMALLEST GAME THAT TEACHES ANYTHING IS 13h x 13d. Measured with the
# heuristic and the best of 8 random vectors, 3 seeds, reachable ceiling as a
# multiple of inaction ($3,000):
#     5h x  5d  ->  0.99x   IMPOSSIBLE: nothing beats doing nothing in 25 turns
#     8h x  8d  ->  1.09x   marginal, indistinguishable from noise
#    13h x 13d  ->  4.98x   playable
#    21h x 21d  ->  9.21x   playable
# With a ceiling of 0.99 and a promotion threshold of 1.2 the league was a
# DEADLOCK: it could not be crossed even playing perfectly. The first two are
# discarded.
#
# On the cap axis: 1 and 2 are degenerate and 13 is indistinguishable from
# uncapped (175,984 against 176,422), so fibonacci samples that axis badly -all
# the gradient lives in 9-12- and the measured values are used instead.
# (hours per day, days, opponent cap, CEILING as a multiple of inaction).
#
# The ceiling is the BEST achievable in that league, measured with the
# best of 8 random vectors. It exists because a FIXED promotion threshold is a
# a deadlock: at 5h x 5d the ceiling is 0.99x -the optimal policy is TO DO
# NOTHING, there is no time for anything to mature or for a hire to pay back-
# and demanding 1.2x made the league impossible to cross even playing
# perfectly.
#
# That league is not redundant: it is the only one where the correct answer is
# restraint, which is exactly the mistake the agent makes at large scale
# (buying seed it does not plant, hiring hands that do not pay back).
#
# Promotion happens at 70% of the league ceiling, not at an absolute number.
# (hours, days, opponent cap, achievable margin, achievable x_inaction).
#
# THE SMALLEST VIABLE IS 36 TURNS (6h x 6d). Measured sweep,
# 3 seeds, best of the heuristic plus 6 random vectors:
#     5h x  5d   25 turns  x_inaction 0.99  <- NOTHING beats doing nothing
#     6h x  6d   36 turns             1.03  <- first profitable play
#     8h x  8d   64                    1,10
#    10h x 10d  100                    1,20
#    11h x 11d  121                    1,10
#    13h x 13d  169                    3,54  <- salto x3: discontinuidad
#
# The achievable MARGIN is the best demonstrated, not a theoretical maximum: a
# ceiling nobody has reached makes the league uncrossable. In the hard leagues
# the heuristic's margin is NEGATIVE (cap 8: -14,973; uncapped: -102,106), so
# there the objective is simply TO WIN -margin 0- which is achievable by
# definition because the opponent achieves it.
LEAGUES = [
    ( 6,  6,    3,   3088,  1.03),   # L0  minima viable
    ( 8,  8,    3,   3291,  1.10),   # L1
    (10, 10,    3,   3602,  1.20),   # L2
    (13, 13,    5,  10565,  3.54),   # L3  after the discontinuity
    (21, 21,    5,  27639,  9.21),   # L4
    (24, 30,    3,  60635, 21.40),   # L5  escala de COMPETICION
    (24, 30,    5,  22873, 16.70),   # L6
    (24, 30,    8,      0, 11.40),   # L7  la heuristica pierde -> objetivo: ganar
    (24, 30,   11,      0, 11.00),   # L8
    (24, 30, None,      0, 10.70),   # L9  competicion completa
]
CEILING_FRAC = 0.90
PROMOTION, RELEGATION, STAY = 0.60, 0.30, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=1,
                    help="chunks per epoch. 1 = full batch.")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--sigma", type=float, default=0.15, help="sigma of the VALUE channel")
    ap.add_argument("--sigma-verb", type=float, default=0.03,
                    help="sigma of the VERB channels (logits)")
    ap.add_argument("--level", type=int, default=2, help="LADDER rung")
    ap.add_argument("--selfplay-quota", type=int, default=2,
                    help="workers ALWAYS on self-play, outside the information split")
    ap.add_argument("--target-quota", type=int, default=3,
                    help="workers ALWAYS on the uncapped rungs (the "
                         "competition opponent)")
    ap.add_argument("--auto-curriculum", action="store_true",
                    help="split workers across rungs in proportion to the "
                         "Bernoulli information p(1-p), so the rung being "
                         "tested starts at p=0.5 -maximum information-")
    ap.add_argument("--rival-macros", default=None,
                    help="opponents PER WORKER: comma-separated .npy paths, '-' to keep the public agent")
    ap.add_argument("--slide", type=float, default=0.0,
                    help="win rate above which a rung is considered EXHAUSTED "
                         "and the ladder slides up. 0 = off")
    ap.add_argument("--levels", default=None,
                    help="LADDER rungs PER WORKER, comma-separated. Training "
                         "against a single opponent risks learning to beat "
                         "THAT one instead of learning to play")
    ap.add_argument("--force-macro", action="store_true",
                    help="re-apply --init AFTER --resume; without it the checkpoint silently overrides the requested macro")
    ap.add_argument("--rival-macro", default=None,
                    help=".npy path: the opponent is OUR executor with that macro")
    ap.add_argument("--own-rival", action="store_true",
                    help="on reduced rungs the opponent is OUR executor with "
                         "the macro CEM found for that scale, because the "
                         "public agents make ~$0 at any other scale")
    ap.add_argument("--grid", default=None,
                    help="MIXED grid of scales: 'h,d,cap;h,d,cap;...', one "
                         "rung per worker; cap 0 = uncapped opponent")
    ap.add_argument("--mix", default=None,
                    help="MIXED horizons in the same batch, in days: "
                         "'15,20,30'. One per worker; excludes --leagues")
    ap.add_argument("--leagues", action="store_true",
                    help="curriculum: promote on winning, relegate on losing")
    ap.add_argument("--league0", type=int, default=0, help="starting league")
    ap.add_argument("--hand-cap", type=int, default=None,
                    help="limit OUR hands (curriculum); None = uncapped")
    ap.add_argument("--init", default="runs/macro_vs_v48.npy")
    ap.add_argument("--init-net", default=None,
                    help="pretrained micro checkpoint")
    ap.add_argument("--promote-rival", type=float, default=0.0,
                    help="win rate above which the opponent is replaced by a frozen copy of the policy"
                         "0 = disabled. It attacks a measured failure"
                         "binaria en 0,09 de su maximo 0,25.")
    ap.add_argument("--resume", default=None,
                    help="continue from an RL checkpoint instead of restarting")
    ap.add_argument("--value-epochs", type=int, default=0,
                    help="extra epochs for the critic only (0 = none)")
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=None,
                    help="lr of the TRUNK; if omitted, the checkpoint's (or 3e-4)")
    ap.add_argument("--lr-heads", type=float, default=None,
                    help="lr of the heads; defaults to --lr")
    ap.add_argument("--kl-target", type=float, default=0.0,
                    help="if >0, the lr self-adjusts to hold this kl/dim")
    # 1.8e-4 was the threshold from when KL was measured WITHOUT normalising
    # by dimension. Normalised, the healthy KL of this problem lives at 1e-2 to
    # 4e-2, i.e. ninety times above: epochs ALWAYS aborted after the first one
    # -"[KL short 1]" on every update of the whole session- and at the same
    # time the controller kept lowering the lr to reach a target 111 times
    # larger than the abort threshold. Two dials pulling in opposite
    # directions: the trunk lr ended at 6.6e-6, 45 times below nominal, and
    # each batch gave ONE single gradient step.
    #
    # The default is now 2x the target, which is standard PPO practice: the
    # abort is a safety net for the odd batch, not the normal regime. A/B
    # abort is a safety net for the odd batch, not the normal regime. A/B
    # measured (11 updates, 8h x 13d + 13h x 13d, own calibrated opponent):
    # with 0.04 the return reaches 5.84 and aborts epochs on nearly every
    # update; with 0.12 it reaches 8.54 and aborts none. The reason is that
    # epoch 1 already measures kl/dim ~0.05 -there is a lag between the
    # rollout log-prob and the recomputed one- so a threshold of 2x the
    # target cuts before the first useful step is taken.
    # de dar el primer paso util.
    ap.add_argument("--jepa-warmup", type=int, default=0,
                    help="initial updates training representation only (policy frozen)")
    ap.add_argument("--freeze-trunk", type=int, default=0,
                    help="updates after warmup during which the TRUNK stays "
                         "frozen, so the heads do not learn against a moving "
                         "target")
    ap.add_argument("--jepa-weight", type=float, default=0.0,
                    help="weight of the JEPA loss: predict the future "
                         "EMBEDDING with a stopped gradient, so the "
                         "trunk keeps what is relevant. Watch "
                         "`2_health/jepa_sd`: if it falls to zero, "
                         "the representation has collapsed")
    ap.add_argument("--ctx-micro", default="3x3",
                    choices=("1x1", "3x3", "3x3x2", "5x5", "attn"),
                    help="spatial context shape of the micro head")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="weight of the AUXILIARY loss: predict the opponent's "
                         "supply at Fibonacci horizons, which forces the "
                         "encoder to model how their farm grows")
    ap.add_argument("--rival-flow", action="store_true",
                    help="feed the `hist` input with the opponent IMMINENT SUPPLY"
                         "ceros. Esa entrada existia (N_HIST = 4 x productos) y "
                         "nuestro ingreso en un mercado compartido.")
    ap.add_argument("--factored-ratio", action="store_true",
                    help="un cociente de importancia POR CABEZA (macro y micro) "
                         "macro -cuyo condicionamiento aporta +1 $ de 953- ")
    ap.add_argument("--sigma-floor-micro", type=float, default=0.0,
                    help="floor of the LEARNED micro sigma. 0 = no floor")
    ap.add_argument("--sigma-floor", type=float, default=0.12,
                    help="floor of the MACRO log_sigma: narrowing exploration measured -26 points")
    ap.add_argument("--kl-max", type=float, default=0.12,
                    help="abort the epoch if the policy moves further than this (kl/dim)")
    ap.add_argument("--selfplay", action="store_true",
                    help="opponent = a frozen snapshot of ourselves")
    ap.add_argument("--refresh", type=int, default=25,
                    help="how many updates between freezing a new version")
    ap.add_argument("--bank-seed", type=str, default=None,
                    help="comma-separated .npy paths to SEED the opponent bank")
    ap.add_argument("--bank", type=int, default=4,
                    help="how many old versions are kept as opponents")
    ap.add_argument("--out", default="runs/e2e.pt")
    ap.add_argument("--run-name", default="e2e-control")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WorldConfig(device=dev, sigma_micro=a.sigma,
                      sigma_ops=a.sigma_verb, ctx_micro=a.ctx_micro)
    net = E2EAgent(cfg).to(dev)
    vec0 = list(np.load(a.init))
    net.init_macro_at(vec0)
    if a.kl_target > 0 and a.kl_max < a.kl_target:
        raise SystemExit(f"--kl-max {a.kl_max:g} is SMALLER than --kl-target "
                         f"{a.kl_target:g}: the controller would raise KL to "
                         f"the target and the abort would cut it on every "
                         f"epoch. Use --kl-max >= 2x --kl-target.")
    if a.hand_cap is not None:
        from kagsym import macro as _M
        _M.HAND_CAP = a.hand_cap
        print(f"OUR hand cap: {a.hand_cap}", flush=True)
    # SCALE OF THE VALUE TARGET. `fret` is in raw units (mean ~20, sd ~8) and
    # the loss was smooth_l1 with beta=1.0: EVERY error larger than 1 unit fell
    # into the pure L1 regime, with a +-1 gradient carrying no magnitude of
    # error. Measured on a toy regression with perfect linear signal and this
    # same scale: R2 = -16.58 that way, against +0.826 normalised. It explains
    # a critic stuck at -0.405 without the trunk being at fault.
    #
    # The critic predicts NORMALISED and is denormalised for GAE, which needs
    # raw units because `ret = adv + Vn` bootstraps from V.
    _vmu, _vsd, _vn = 0.0, 1.0, 0
    _n_reused = _n_total = 0
    d0 = {}
    if a.resume:
        # CONTINUE, do not restart. Each run used to start from the supervised
        # pretraining and threw away all accumulated RL. It is only valid if
        # the architecture has not changed: N_GLOBAL went from 75 to 88 and
        # N_MACRO from 7 to 13, and with that the shapes do not fit. What fits
        # is loaded and the reuse is reported.
        d0 = torch.load(a.resume, map_location="cpu", weights_only=False)
        from kagsym.migrate_ckpt import load_tolerant
        _n_reused, _n_total, _random_ = load_tolerant(
            net, d0["sd"], a.resume, macro_fields=d0.get("macro_fields"))
        net.to(dev)
        print(f"resumed from {a.resume}: {_n_reused}/{_n_total} tensors "
              f"reused (update {d0.get('upd','?')}, "
              f"return {d0.get('ret', float('nan')):.2f})", flush=True)
    elif a.init_net:
        # Start from a micro head that ALREADY reconstructs the valuation by
        # supervision. TOLERANT, like `--resume`: the pretraining may have a
        # macro head of a different width, but what matters from it -the trunk
        # and the value map- does fit. Loading only what is compatible avoids
        # redoing the pretraining for every new dimension, and how much is
        # reused is reported so it cannot pass unnoticed.
        d0 = torch.load(a.init_net, map_location="cpu", weights_only=False)
        from kagsym.migrate_ckpt import load_tolerant
        current = net.state_dict()
        _nok, _ntot, _ = load_tolerant(
            net, d0["sd"], a.init_net, macro_fields=d0.get("macro_fields"))
        net.init_macro_at(vec0)        # the macro head, from the vector
        net.to(dev)
        print(f"micro pretrained from {a.init_net}: "
              f"{_nok}/{_ntot} tensors reused", flush=True)
    # TWO RATES. Measured starting from scratch: after 40 updates
    # `micro.weight` was 0.013 and the verb logits 0.008, against a sampling
    # sigma of 0.15 -noise crushed the learned signal 19 to 1- meaning the head
    # was limited by NUMBER OF STEPS, not by gradient. At the same time KL per
    # dimension grew from 0.036 to 0.363 because the critic, which shares the
    # trunk, was shaking the encoder: the policy moved enormously without
    # having learned anything, and money peaked at update 15 and fell from
    # there.
    #
    # A linear head has to TRAVEL from zero; the encoder only has to be
    # fine-tuned. A single lr cannot serve both.
    # FOUR GROUPS, not two. The total KL decomposes into the MACRO's plus the
    # MICRO's because the two policies are independent Gaussians, so each can
    # have its own control loop. With a single controller the macro -which
    # contributes +$1 of 953 and whose weights grow 4.3x against the micro's
    # 2.8x- spends the divergence budget and brakes the head that actually
    # decides: measured at championship scale, saturation rising to 0.16 while
    # the lr of EVERYTHING fell from 3e-4 to 3.6e-5.
    #
    # Group 3 (critic, auxiliaries) is not policy: it generates no KL and is
    # not controlled by KL.
    _g_macro, _g_micro, _g_otros, _trunk = [], [], [], []
    for _n, _p in net.named_parameters():
        _b = _n.split(".")[0]
        if _b in ("macro_mu", "log_sigma"):
            _g_macro.append(_p)
        elif _b in ("micro", "micro_ctx", "log_sigma_micro"):
            _g_micro.append(_p)
        elif _b in ("critic", "aux_rival", "jepa_proy", "jepa_pred"):
            _g_otros.append(_p)
        else:
            _trunk.append(_p)
    _heads = _g_macro + _g_micro + _g_otros
    _rm_now = None
    # SELF-PLAY RUNGS WITH THE FULL NETWORK. The opponent's identity lives in
    # the RUNG, not in the worker: the curriculum reassigns workers and without
    # this the frozen opponent would be lost on the first reallocation.
    _pool_net = set()
    _frozen_net = None
    _pool, _pool_p, _assign = [], [], []
    _levels = ([int(x) for x in a.levels.split(',')] if a.levels else None)
    if _levels:
        from kagsym.environment import LADDER as _LADDER
        print('opponents per worker: ' + ', '.join(
            str(_LADDER[min(n, len(_LADDER)-1)]) for n in _levels), flush=True)
    if a.lr is None:
        a.lr = 3e-4
        _lr_explicit = False
    else:
        _lr_explicit = True
    _lrc = a.lr_heads if a.lr_heads is not None else a.lr
    # `--lr-heads` ALONE also overrides. Without this, `_lr_explicit` was only
    # was activated by `--lr`, so on resume both lrs from the checkpoint were
    # restored and the requested change was lost SILENTLY -the same class of
    # bug as `--init` clobbered by `--resume`, which cost `--force-macro`-.
    # What matters here is the heads/trunk RATIO: the KL controller rescales
    # both groups together, so the ratio is the only thing that survives.
    if a.lr_heads is not None:
        _lr_explicit = True
    # 0 tronco | 1 macro | 2 micro | 3 critico+auxiliares
    opt = torch.optim.Adam([{"params": _trunk, "lr": a.lr},
                            {"params": _g_macro, "lr": _lrc},
                            {"params": _g_micro, "lr": _lrc},
                            {"params": _g_otros, "lr": _lrc}])
    print(f"lr tronco {a.lr:.1e} ({len(_trunk)} tensores) | "
          f"lr cabezas {_lrc:.1e} ({len(_heads)} tensores)", flush=True)
    # OPTIMIZER STATE, not just the weights. Without this every resume starts
    # with Adam's moments at zero and the factory lr, so the first step is
    # enormous: measured, a resume from $20,957 fell to $4,717 in five updates,
    # with kl/dim of 16.9 on the first. More was lost than recovered.
    #
    # Only when the ARCHITECTURE has not changed. The optimizer's
    # `load_state_dict` does not validate shapes: it accepts the old moments
    # and blows up later, inside the first step(), with "size of tensor a (124)
    # must match b (126)". If `--resume` could not reuse ALL tensors, the
    # network is a different one and the moments are worthless.
    _same_arch = (not a.resume) or (_n_reused == _n_total)
    if a.resume and "opt" in d0 and _same_arch:
        try:
            # MOMENTS PER TENSOR, not all-or-nothing. When a new constant is
            # exposed the macro vector grows, and with it the head:
            # `macro_mu.weight`, `macro_mu.bias` and `log_sigma` change shape.
            # The other 66 tensors are identical and their moments remain
            # valid -they are precisely what prevents the huge first step-.
            # Throwing them all away for three that changed is what cost
            # $20,957 -> $4,717 in five updates.
            #
            # The optimizer's `load_state_dict` does NOT validate shapes: it
            # accepts the old moments and blows up later, inside step(). So
            # they are filtered BEFORE, comparing against the parameter they
            # belong to. Whatever is dropped, Adam reinitialises on its first
            # step.
            _state = dict(d0["opt"])
            _ps = [q for g in opt.param_groups for q in g["params"]]
            _outside = []
            _new_state = {}
            for _i, _v in (_state.get("state") or {}).items():
                _j = int(_i)
                _ok = _j < len(_ps)
                if _ok:
                    for _c in ("exp_avg", "exp_avg_sq"):
                        _t = _v.get(_c)
                        if _t is not None and tuple(_t.shape) != tuple(_ps[_j].shape):
                            _ok = False
                if _ok:
                    _new_state[_i] = _v
                else:
                    _outside.append(_j)
            _state["state"] = _new_state
            opt.load_state_dict(_state)
            if _outside:
                print(f"  moments reset on {len(_outside)} of {len(_ps)} "
                      f"tensors (they changed shape); the rest keep Adam",
                      flush=True)
            # Adam's MOMENTS are always restored -they are what prevents the
            # huge first step- but an explicitly requested lr OVERRIDES the
            # checkpoint's. Without this, restoring the optimizer silently
            # clobbered the lr change the operator came to make.
            if _lr_explicit:
                opt.param_groups[0]["lr"] = a.lr
                opt.param_groups[1]["lr"] = _lrc
                print(f"  EXPLICIT lr ({a.lr:.1e} / {_lrc:.1e}, ratio "
                      f"{_lrc/a.lr:.1f}x), overrides the checkpoint's",
                      flush=True)
            print(f"  optimizer restored: trunk lr "
                  f"{opt.param_groups[0]['lr']:.2e}, heads "
                  f"{opt.param_groups[1]['lr']:.2e}", flush=True)
        except Exception as e:
            print(f"  WARNING: could not restore the optimizer ({e}); "
                  f"continuing with factory lrs", flush=True)
    elif a.resume and "opt" in d0:
        print(f"  optimizer NOT restored: the architecture changed "
              f"({_n_reused}/{_n_total} tensors). Moments zeroed.", flush=True)
    if a.resume and _n_reused < _n_total and getattr(net, "n_ops", 0):
        # INACTION RESCUE. If the architecture changed, the reinitialised
        # input layers send noise into the value head, which emits negatives;
        # the Hungarian prefers its dummy column and ALL units PASS. And
        # inaction is an ABSORBING STATE: if nobody acts there is no variation
        # to learn from. Measured: 70 updates pinned at exactly $3,000 -the
        # starting money- at 5 days, against the $3,746 the same macro makes
        # without a network.
        #
        # The value channel's bias is reset to 1.0 = "acting is worth something
        # positive", which is the only start you can escape from.
        with torch.no_grad():
            net.micro.bias[0] = 1.0
        print("  value bias reset to 1.0 (inaction rescue)", flush=True)
    print(f"code fingerprint: {fingerprint()}", flush=True)
    print(f"device: {dev} | opponent: {LADDER[a.level]} | init: {a.init}",
          flush=True)

    from kagsym import spec
    from kagsym.macro import Macro
    # Distributed rollouts: measured 632 steps/s serially against 10,069 in
    # parallel with 48 envs and 10 processes (15.9x). Startup is 1-3 s, once.
    if a.force_macro:
        # AFTER the resume on purpose: `init_macro_at` runs earlier and the
        # checkpoint overwrites it. Measured: the 5-day CEM macro was requested
        # ([0.121, 0.754, 0.057, ...]) and the network emitted the pretrained
        # one ([0.01, 0.24, 0.02, ...]), i.e. the 30-day vector. The
        # experiment was not testing what it was meant to test and nothing
        # said so.
        net.init_macro_at(vec0)
        print(f"macro FORCED to {a.init}", flush=True)
    _steps = a.steps
    if a.mix:
        # MIXED, not sequential. In sequence the policy trains at one horizon
        # only and forgets the previous one -measured twice: real margin from
        # -82.7% to -98.3%-. Very short horizons are avoided on purpose:
        # at 5 days doing nothing gives $3,000 and our heuristic $2,920, so
        # acting DESTROYS value and the lesson learned there is "do nothing".
        _days = [int(x) for x in a.mix.split(",")]
        _steps = [d * 24 for d in _days]
        a.days = max(_days)
        print(f"MIXED horizons: {_days} days -> {_steps} steps", flush=True)
    _hours = None
    if a.grid:
        # MIXED FIBONACCI GRID. All three axes at once, one rung per worker,
        # all in the same batch. What makes it possible: `spec` and `HAND_CAP`
        # are PER-PROCESS globals, and the observation carries two ABSOLUTE
        # dimensions -EPISODE_STEPS/720 and cap/HANDS_REF- so the network knows
        # which rung it is playing and can condition the policy instead of
        # averaging over the three.
        #
        # Mixed rather than sequential, again from measurement: a sequential
        # curriculum gave -98.3% twice because the policy forgets the previous
        # rung; mixing gave -75.4%, the best result of that session.
        #
        # And with the transfer matrix measured: upward 0-5%, downward 43-96%.
        # Training ONLY at the top does not transfer down; training mixed goes
        # up AND down (mixing 15/20/30 gave 1.22x at 10 days and 2.66x at 13,
        # neither seen in training, and also won at 30 days: 12.78x against
        # 10.65x for the one trained only there).
        _grid = []
        for t in a.grid.split(";"):
            h, d, tp = (int(x) for x in t.split(","))
            _grid.append((h, d, tp))
        _steps = [h * d for h, d, _ in _grid]
        _hours = [h for h, _, _ in _grid]
        # THIRD AXIS = THE OPPONENT'S CAP, not ours. `--hand-cap` limits OUR
        # hands (a constraint on our own action space); the ladder's difficulty
        # axis was always the opponent. That is also how the rungs were
        # calibrated, so they have to mean the same thing or the measured
        # ceilings do not apply. 0 = uncapped (opponent at full power).
        _rival_cap = [(tp if tp > 0 else None) for _, _, tp in _grid]
        a.days = max(d for _, d, _ in _grid)
        print(f"MIXED GRID, {len(_grid)} rungs:", flush=True)
        for h, d, tp in _grid:
            print(f"   {h:>3}h x {d:>3}d = {h*d:>4} turns, cap {tp}", flush=True)
    env = ParallelEnv(a.envs, n_procs=a.procs, steps=_steps,
                          macro=Macro.from_vector(vec0),
                          level=(_levels if _levels else a.level),
                          hand_cap=a.hand_cap, hours=_hours)
    if a.grid:
        env.set_rival_cap(_rival_cap)
        print(f"OPPONENT cap per rung: {_rival_cap}", flush=True)
        if a.own_rival:
            # A real opponent on every rung. Measured: v48 makes $60,426 at
            # 24h x 30d and $0-30 at any other scale, because its tape is
            # indexed by step for 719 steps at 24h/day. Without this, on ten of
            # the eleven rungs the win term is free and nobody drains the
            # shared market: you train to farm in an empty world and then
            # compete in a full one.
            #
            # It is NEVER substituted at competition scale: there the public
            # agent DOES play -$60,426 measured- and it is the only rung where
            # winning means anything. The substitution is for rungs where the
            # public agent is dead, not a way to avoid the real opponent.
            _COMPETITIVE = (24, 30)
            _riv = []
            for _h, _d, _ in _grid:
                _f = f"runs/ligas/macro_{_h}h{_d}d.npy"
                if (_h, _d) == _COMPETITIVE or not os.path.exists(_f):
                    _riv.append(None)
                else:
                    _riv.append(list(np.load(_f)))
            env.set_rival_macro(_riv)
            _n = sum(1 for v in _riv if v is not None)
            print(f"opponent = OUR calibrated executor on {_n}/{len(_grid)} "
                  f"rungs; public agent on the rest", flush=True)
            for (_h, _d, _), _v in zip(_grid, _riv):
                print(f"   {_h}h x {_d}d: "
                      + ("executor with calibrated macro" if _v is not None
                         else "v48-fast-routes"), flush=True)
    # Ranges of env indices belonging to each worker, i.e. to each rung of the
    # grid. The parallel env hands out envs in order, `per_proc[k]`
    # consecutively per worker.
    _GROUPS = None
    if a.grid and len(set(env.steps_proc)) > 1:
        _GROUPS, _o = [], 0
        for _m in env.envs_per_proc:
            _GROUPS.append((_o, _o + _m))
            _o += _m
        print(f"advantage normalised per rung: {len(_GROUPS)} groups "
              f"{[b - a_ for a_, b in _GROUPS]}", flush=True)
    if a.rival_macros:
        _rm = []
        for _t in a.rival_macros.split(","):
            _t = _t.strip()
            _rm.append(None if _t in ("-", "") else list(np.load(_t)))
        env.set_rival_macro(_rm)
        _rm_now = list(_rm)
        # The POOL of rungs: (level, cap, macro). It starts with the initial
        # assignment and the automatic curriculum redistributes workers among
        # them according to how much information each one gives.
        if _levels:
            _pool = [(_levels[i % len(_levels)],
                      _rival_cap[i % len(_rival_cap)],
                      _rm[i % len(_rm)]) for i in range(len(_rm))]
            _pool_p = [None] * len(_pool)
            _assign = list(range(len(_pool)))
        print("opponents per worker (macro): " + ", ".join(
            ("public" if x is None else "OURS") for x in _rm), flush=True)
    elif a.rival_macro:
        env.set_rival_macro(list(np.load(a.rival_macro)))
        print(f"opponent = our executor with {a.rival_macro}", flush=True)
    _league = a.league0
    _in_league = 0
    _RIV_ULT = [0.0]

    def _build_league(idx):
        """Build the environment for league `idx`: hours, days and opponent cap.

        Changing hours or days forces recreating the workers, because
        `spec.TURNS_PER_DAY` and `spec.EPISODE_STEPS` are PER-PROCESS globals.
        """
        hours, days, cap, _margin, _ceiling = LEAGUES[idx]
        spec.set_turns_per_day(hours)
        steps = days * hours
        spec.set_episode_steps(steps)
        a.steps, a.days = steps, days
        e = ParallelEnv(a.envs, n_procs=a.procs, steps=steps,
                            macro=Macro.from_vector(vec0), level=4,
                            hand_cap=a.hand_cap,
                            hours=hours)
        e.set_rival_cap(cap)
        print(f"LEAGUE {idx}/{len(LEAGUES)-1}: {hours}h x {days}d, v48 "
              f"capped at {cap if cap is not None else 'nothing'} "
              f"({steps} steps)", flush=True)
        return e

    if a.leagues:
        env.close_()
        env = _build_league(_league)

    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
        mlflow.set_experiment("kaggriculture-world-model")
        mlflow.start_run(run_name=a.run_name)
        mlflow.log_params(vars(a))
        # REFERENCES. A money curve cannot be read without its baselines.
        mlflow.log_params({
            "ref_inaction": 3000,            # measured, at any horizon
            "ref_legal_random": 8692,        # a random legal verb every turn
            "ref_heuristica_v48tope5": 30065,
            "ref_backbone6_v48tope5": 32645,
            "ref_termometro_backbone6": -82.7,
            "ref_v48_tope5": 34905,
            "ref_experto_2945_vs_v48": 82382,
        })
        use_mlflow = True
    except Exception:
        use_mlflow = False

    # PERTURBATION PER EPISODE, not per day. Measured: resampling every day
    # costs 54% of the return ($40,972 fixed -> $18,967 sampled), because 30
    # days of random jitter destroy the farm's coherence.
    eps = torch.randn(a.envs, N_MACRO, device=dev)
    _NC = 1 + getattr(net, "n_ops", 0)
    _U_SHAPE = (10, 10) if _NC == 1 else (_NC, 10, 10)
    # One sigma per channel: value and verbs live on different scales.
    # The MICRO head's sigma is no longer a config constant: it is an
    # `nn.Parameter` of the network learned by PPO, like the macro's. It is
    # recomputed at each use because after `opt.step()` the value changes;
    # caching it in a variable would leave a stale tensor and, during the
    # update, a graph that no longer corresponds.
    def SIG():
        return net.log_sigma_micro.exp()
    eps_u = torch.randn(a.envs, *_U_SHAPE, device=dev)
    best = -1e18
    # SAFETY NET against collapse. Measured: one update with kl 25.7 took the
    # cash from $34,900 to $3 in fifteen updates and the controller reacted too
    # late. It keeps the last GOOD state and restores it if the return
    # collapses, also halving the pace.
    _lifeline = {"ret": None, "sd": None, "opt": None, "rescates": 0}
    _grad_norms = []
    _last_promo = -10**9
    ret_ep = []          # returns of CLOSED EPISODES, not per-update sums
    accum = np.zeros(a.envs, dtype=np.float64)

    if a.selfplay:
        # Against a full-power public agent we lose 100%: the terminal is -1
        # ALWAYS and carries not one bit about the only thing that scores.
        # Against a copy of ourselves the rate sits around 0.59 with both
        # making the same money ($36,385 against $36,295), which is where a
        # binary signal has maximum information.
        env.set_selfplay(net)
        print("opponent: SELF-PLAY (a frozen snapshot of the policy)",
              flush=True)

    import copy
    bank = []
    # Bank seeds: DIVERSE opponents, not from our own lineage.
    _seed_vectors = []
    for _p in [x.strip() for x in (a.bank_seed or "").split(",") if x.strip()]:
        try:
            _v = np.load(_p)
            if _v.ndim != 1 or _v.shape[0] > N_MACRO:
                print(f"  seed ignored {_p}: dims {_v.shape}, expected "
                      f"({N_MACRO},)", flush=True)
                continue
            if _v.shape[0] < N_MACRO:
                # A vector from a version with fewer parameters. It is PADDED
                # with `Macro`'s defaults, which are exactly what the constant
                # held when it was hand-set: that way the old vector still
                # describes the same policy. Dropping it -what happened before-
                # emptied the bank silently and the run fell back to pure
                # self-play with no diversity.
                from kagsym.macro import Macro as _Mc
                _d = np.array(_Mc().to_vector(), dtype=np.float64)
                _d[: _v.shape[0]] = _v
                print(f"  seed {os.path.basename(_p)}: {_v.shape[0]} dims "
                      f"-> {N_MACRO}, padded with the defaults", flush=True)
                _v = _d
            _seed_vectors.append((os.path.basename(_p)[:-4], [float(x) for x in _v]))
        except Exception as _e:
            print(f"  seed ignored {_p}: {_e}", flush=True)
    if _seed_vectors:
        print(f"bank seeded with {len(_seed_vectors)} diverse opponents: "
              f"{', '.join(n for n, _ in _seed_vectors)}", flush=True)

    t0 = time.time()
    # AXIS CONTINUITY. On resume the counter went back to 1 and MLflow drew
    # the continuation OVERLAID on the previous stretch instead of after it:
    # the learning curve was split in two and could not be read as one. The
    # offset joins it back up.
    _upd0 = int(d0.get("upd", 0) or 0) if a.resume else 0
    if _upd0:
        print(f"the step axis continues from {_upd0}", flush=True)
    for upd in range(1, a.updates + 1):
        if a.selfplay and upd % a.refresh == 0:
            # A bank of older versions: without it the policy can forget how
            # to beat what it already knew how to beat, and cycle. It
            # alternates between the most recent version and one from the bank
            # chosen in turn.
            bank.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            bank[:] = bank[-a.bank:]
            # The population is diverse seeds PLUS our own snapshots. Rotating
            # only among our own snapshots is playing against the same lineage.
            # The seeds occupy ONE slot because they are all deployed at once
            # -one per worker-. With one slot per seed the rotation spent
            # identical turns.
            _pool = (([("macro", "diversos", None)] if _seed_vectors else [])
                     + [("foto", f"auto{i}", sd) for i, sd in enumerate(bank)])
            _kind, _name, _target = _pool[(upd // a.refresh) % len(_pool)]
            if _kind == "macro":
                # All diverse opponents AT ONCE, one per worker:
                # `set_rival_macro` accepts that natively. Rotating one at a
                # time makes the gradient see them in series, which is how it
                # forgets to beat what it already knew how to beat.
                env.set_rival_macro([v for _, v in _seed_vectors])
                _name = f"{len(_seed_vectors)} diverse ones at once"
            else:
                rival = E2EAgent(cfg)
                rival.load_state_dict(_target)
                env.set_selfplay(rival)
            print(f"  [upd {upd}] opponent -> {_name} ({_kind}), population "
                  f"of {len(_pool)}", flush=True)
        G, B, H, AM, AU, LP, V, R, D, MS = [], [], [], [], [], [], [], [], [], []
        LPMA, LPMI = [], []          # log-prob per head, for the factored ratio
        HF = []                      # opponent supply per day (auxiliary target)
        JZ = []                      # JEPA projections per day (target, stopped)
        for _ in range(a.days):
            g, b, hf = env.encode()
            # The `hist` input carried ZEROS from the start despite being
            # designed and wired into the encoder. With --rival-flow it gets
            # what belongs there: the opponent's imminent supply, which is the
            # mechanism by which they affect us -they dump produce, the
            # marginal price falls, our income drops-.
            h = (hf if a.rival_flow
                 else np.zeros((a.envs, N_HIST), dtype=np.float32))
            HF.append(torch.from_numpy(np.asarray(hf, dtype=np.float32)).to(dev))
            tg = torch.from_numpy(g).to(dev)
            tb = torch.from_numpy(b).to(dev)
            th = torch.from_numpy(h).to(dev)
            with torch.no_grad():
                s = net(tg, tb, th)
                am, lp_m = net.macro_from(s, eps)
                au = s["micro"] + SIG() * eps_u
                _lpu = torch.distributions.Normal(
                    s["micro"], SIG()).log_prob(au)
                lp = lp_m + _lpu.flatten(1).sum(-1)   # provisional; se corrige abajo
                _lp_mi = _lpu.flatten(1).sum(-1)
                if a.jepa_weight > 0:
                    JZ.append(s["jepa_z"].detach())
            rec, fin = env.step_day(maps=au.cpu().numpy(), macros=am.cpu().numpy())
            # The mask is only known AFTER playing the day: it says which
            # dimensions genuinely influenced some decision. The log-prob is
            # recomputed with it so PPO's ratio ignores the rest.
            _ms = env.masks()
            if _ms is not None:
                mt = torch.from_numpy(_ms).to(dev)
                lp = lp_m + (_lpu * mt).flatten(1).sum(-1)
                _lp_mi = (_lpu * mt).flatten(1).sum(-1)
                MS.append(mt)
            # return per closed episode, which IS comparable across updates
            accum += rec
            for k in np.nonzero(fin)[0]:
                ret_ep.append(float(accum[k])); accum[k] = 0.0
            if fin.any():
                idx = torch.from_numpy(np.nonzero(fin)[0]).to(dev)
                eps[idx] = torch.randn(len(idx), N_MACRO, device=dev)
                eps_u[idx] = torch.randn(len(idx), *_U_SHAPE, device=dev)
            G.append(g); B.append(b); H.append(h)
            AM.append(am); AU.append(au); LP.append(lp)
            LPMA.append(lp_m); LPMI.append(_lp_mi)
            V.append(s["value"] * _vsd + _vmu)
            R.append(rec); D.append(fin)
        with torch.no_grad():
            g, b, hf = env.encode()
            ult = (net(torch.from_numpy(g).to(dev),
                      torch.from_numpy(b).to(dev))["value"] * _vsd + _vmu)

        R = np.array(R); D = np.array(D); Vn = torch.stack(V).cpu().numpy()
        adv = np.zeros_like(R); acc = 0.0; u = ult.cpu().numpy()
        for t in reversed(range(a.days)):
            next_ = u if t == a.days - 1 else Vn[t + 1]
            delta = R[t] + a.gamma * next_ * (1 - D[t]) - Vn[t]
            acc = delta + a.gamma * a.lam * (1 - D[t]) * acc
            adv[t] = acc
        ret = adv + Vn
        _b_mu = float(ret.mean()); _b_sd = float(ret.std()) + 1e-6
        _vn += 1
        _w = 1.0 / min(_vn, 100)
        _vmu = (1 - _w) * _vmu + _w * _b_mu
        _vsd = (1 - _w) * _vsd + _w * _b_sd
        # CRITIC R2 in flight. Measured separately, its ceiling with enough
        # data is 0.900 and in flight it was at 0.675: the gap is training, not
        # noise. Without logging it there is no way to see whether it closes.
        _vr = ret.reshape(-1); _vp = Vn.reshape(-1)
        _sse = float(((_vr - _vp) ** 2).sum())
        _sst = float(((_vr - _vr.mean()) ** 2).sum())
        _r2 = 1.0 - _sse / _sst if _sst > 0 else 0.0
        # ADVANTAGE NORMALISED PER RUNG when there is a grid.
        #
        # The shaping `scale` is a constant ($2,000) that does NOT depend on
        # the rung, so the terminal potential is worth ~20 in a 720-turn
        # episode and ~2.5 in a 105-turn one. Normalising the advantage over
        # the WHOLE batch, samples from the large rung take almost all the
        # magnitude and those from the small ones are crushed against zero:
        # mixing would stop serving any purpose, which is the opposite of what
        # it is for.
        #
        # Normalising per rung puts all rungs on equal footing. It is standard
        # practice in multi-task RL and it does not change the sign of any
        # advantage, only its relative scale across tasks.
        if _GROUPS is not None:
            for _a, _b in _GROUPS:
                _sl = adv[:, _a:_b]
                if _sl.size:
                    adv[:, _a:_b] = (_sl - _sl.mean()) / (_sl.std() + 1e-8)
        else:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        fg = torch.from_numpy(np.concatenate(G)).to(dev)
        fb = torch.from_numpy(np.concatenate(B)).to(dev)
        fh = torch.from_numpy(np.concatenate(H)).to(dev)
        fam, fau, flp = torch.cat(AM), torch.cat(AU), torch.cat(LP)
        flpma, flpmi = torch.cat(LPMA), torch.cat(LPMI)
        # AUXILIARY TARGET, Fibonacci horizons. For the state of day t, what
        # the opponent will have READY at t+1, t+2, t+3, t+5, t+8 and t+13.
        # Beyond the last available day the last one is repeated: it invents no
        # information and keeps the lengths square.
        #
        # Predicting only t+1 would be nearly trivial -one day of growth is
        # deterministic-; the game's cycles live between 2 and 13 days and that
        # is where the encoder has to learn something.
        # JEPA TARGET: for day t, the projection of the encoder ITSELF at
        # t+k, detached and frozen from the rollout. If it were recomputed
        # during the epochs the target would move with the predictor, which is
        # a direct route to collapse.
        _fjz = None
        if a.jepa_weight > 0 and JZ:
            from kagsym.obs import AUX_HORIZONS as _HZ
            _njz = len(JZ)
            _fjz = torch.cat([
                torch.stack([JZ[min(t + k, _njz - 1)] for k in _HZ], dim=1)
                for t in range(_njz)])
        _fhf = None
        if HF:
            from kagsym.obs import AUX_HORIZONS, RIVAL_WINDOWS
            _nd = len(HF)
            # from (days, envs, 4*P) to the first window: what is READY today
            _ready = [x.reshape(x.shape[0], len(RIVAL_WINDOWS), -1)[:, 0, :]
                      for x in HF]
            _fhf = torch.cat([
                torch.cat([_ready[min(t + k, _nd - 1)] for k in AUX_HORIZONS],
                          dim=-1)
                for t in range(_nd)])
        fms = torch.cat(MS) if MS else None
        fadv = torch.from_numpy(adv.reshape(-1).astype(np.float32)).to(dev)
        fret = torch.from_numpy(ret.reshape(-1).astype(np.float32)).to(dev)
        kl_cuts = 0
        # (the safety net is initialised before the loop)
        # SAFETY NET. The KL guard limits how far the policy moves in one
        # update, but it does not repair what already broke: measured, a single
        # update with kl 25.7 took the cash from $34,900 to $3 in fifteen
        # updates, and the controller reacted too late. In an unattended run
        # that is hours lost without anyone noticing.
        #
        # The last GOOD state is kept and, if the return collapses below half
        # the recent best, it is restored and the pace is halved. It does not
        # prevent the bad update; it prevents it from taking the whole night.

        # MINIBATCHES. At full batch, 4 epochs are 4 optimizer STEPS over
        # ~2,160 samples. Splitting into N does not change the compute -the
        # same number of sample-gradients- but gives 4*N steps, each starting
        # from the weights the previous one updated. It shows up most in the
        # CRITIC, which is a regression: its R2 sits at 0.80 with a measured
        # ceiling of 0.900, and that gap is documented as training, not noise.
        _N = len(fadv)
        # The POLICY always at full batch: splitting it multiplies its KL by 24
        # (measured) and the controller would have to undo it by cutting the lr.
        # `--minibatches` affects ONLY the critic, further down.
        _nm, _size = 1, _N
        for _ep in range(a.epochs):
            _order = torch.randperm(_N, device=dev) if _nm > 1 else None
            for _j in range(_nm):
                if _nm > 1:
                    _sel = _order[_j * _size:(_j + 1) * _size]
                    if len(_sel) == 0:
                        continue
                    _fg, _fb, _fh = fg[_sel], fb[_sel], fh[_sel]
                    _fam, _fau, _flp = fam[_sel], fau[_sel], flp[_sel]
                    _flpma, _flpmi = flpma[_sel], flpmi[_sel]
                    _fadv, _fret = fadv[_sel], fret[_sel]
                    _fms = fms[_sel] if fms is not None else None
                else:
                    _fg, _fb, _fh = fg, fb, fh
                    _fam, _fau, _flp = fam, fau, flp
                    _flpma, _flpmi = flpma, flpmi
                    _fadv, _fret = fadv, fret
                    _fms = fms
                s = net(_fg, _fb, _fh)
                _lp2 = torch.distributions.Normal(
                    s["micro"], SIG()).log_prob(_fau)
                if _fms is not None:
                    _lp2 = _lp2 * _fms
                _lpma = net.macro_logprob(s, _fam)
                _lpmi = _lp2.flatten(1).sum(-1)
                lp = _lpma + _lpmi
                if a.factored_ratio:
                    # ONE RATIO PER HEAD. PPO sums the log-prob over ALL
                    # dimensions -60 from the macro plus 1,600 from the micro-
                    # into a single ratio, so the macro's exploration noise
                    # enters every sample's ratio and drags the micro with it:
                    # when the ratio leaves the band because of the macro, the
                    # micro's gradient is lost with it.
                    #
                    # And the macro does not deserve it: measured, its
                    # conditioning contributes +$1 of 953. Separating the
                    # ratios lets each head explore its own space without
                    # clipping the other.
                    #
                    # The `fms` mask already attacked this problem halfway
                    # -ignoring dimensions that decide nothing-; this is the
                    # complete version.
                    _rma = torch.exp((_lpma - _flpma).clamp(-10, 10))
                    _rmi = torch.exp((_lpmi - _flpmi).clamp(-10, 10))
                    l_pi = -0.5 * (
                        torch.min(_rma * _fadv,
                                  _rma.clamp(1 - a.clip, 1 + a.clip) * _fadv)
                        + torch.min(_rmi * _fadv,
                                    _rmi.clamp(1 - a.clip, 1 + a.clip) * _fadv)
                    ).mean()
                    ratio = _rmi
                else:
                    ratio = torch.exp((lp - _flp).clamp(-10, 10))
                    l_pi = -torch.min(ratio * _fadv,
                                      ratio.clamp(1 - a.clip, 1 + a.clip) * _fadv).mean()
                l_v = torch.nn.functional.smooth_l1_loss(
                    s["value"], (_fret - _vmu) / _vsd)
                l_aux = torch.zeros((), device=dev)
                if a.aux_weight > 0 and _fhf is not None:
                    # symlog: supply ranges from 0 to >100 units and an error
                    # of 80 cannot weigh 80 times one of 1.
                    _t = _fhf[_sel] if _nm > 1 else _fhf
                    l_aux = torch.nn.functional.smooth_l1_loss(
                        s["rival"], torch.sign(_t) * torch.log1p(_t.abs()))
                l_jepa = torch.zeros((), device=dev)
                if a.jepa_weight > 0 and _fjz is not None:
                    _tj = _fjz[_sel] if _nm > 1 else _fjz
                    # cosine: only the embedding's DIRECTION matters; the
                    # norm can be inflated by the encoder without learning
                    # anything.
                    _p = torch.nn.functional.normalize(s["jepa_p"], dim=-1)
                    _q = torch.nn.functional.normalize(_tj, dim=-1)
                    l_jepa = (1.0 - (_p * _q).sum(-1)).mean()
                if upd <= a.jepa_warmup:
                    # WARMUP: representation only. The policy does not move,
                    # so the encoder is shaped by a DENSE, low-variance
                    # gradient before anything depends on it. It attacks the
                    # startup race -encoder, critic and policy chasing each
                    # other- which is a candidate explanation for the takeoff
                    # lottery: the critic starts at R2 -1.7 to -4.1 and until
                    # it rises, the advantage is noise.
                    #
                    # The value head IS trained: it is what the policy will
                    # need on day one and costs nothing to have ready.
                    loss = a.value_weight * l_v + a.jepa_weight * l_jepa
                else:
                    loss = (l_pi + a.value_weight * l_v + a.aux_weight * l_aux
                               + a.jepa_weight * l_jepa)
                opt.zero_grad(); loss.backward()
                if a.jepa_warmup < upd <= a.jepa_warmup + a.freeze_trunk:
                    # MIND THE INTERVAL: it freezes AFTER the warmup, not
                    # during. During the warmup the trunk is precisely what has
                    # to learn; freezing it there would make the phase a no-op
                    # and we would then measure "JEPA does not work" when what
                    # did not work was the setup.
                    #
                    # The gradient is zeroed rather than removing the tensors
                    # from the optimizer, so Adam's state is not lost.
                    for _p in _trunk:
                        if _p.grad is not None:
                            _p.grad = None
                # The norm BEFORE clipping, which is what the function
                # returns. It is the direct diagnostic for the most expensive
                # bug we have had: with `log_prob` over the sample without
                # `detach`, mu cancels and this reads 0.000e+00 while
                # everything else looks normal -the indirect symptom was "the
                # rollout column does not move", which took 5,456 episodes to
                # read-.
                _gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                _grad_norms.append(float(_gn))
                opt.step()
                if a.sigma_floor > 0:
                    with torch.no_grad():
                        net.log_sigma.clamp_(min=float(np.log(a.sigma_floor)))
                        if a.sigma_floor_micro > 0:
                            net.log_sigma_micro.clamp_(
                                min=float(np.log(a.sigma_floor_micro)))

            # KL DIVERGENCE ABORT. Gradient clipping limits the step's
            # MAGNITUDE, not how far the POLICY moves. Measured: 80 stable
            # updates (win ~0.5, $44,000) and then a cliff -win 0.015,
            # $7,514- there is no recovering from. A single bad update
            # destroys the policy; this cuts it before that happens.
            with torch.no_grad():
                # Over the WHOLE BATCH, with its own forward. With
                # minibatches, `lp` is that of the LAST chunk and `flp` that of
                # everything: comparing them would compare different shapes.
                _s = net(fg, fb, fh)
                _l2 = torch.distributions.Normal(_s["micro"], SIG()).log_prob(fau)
                if fms is not None:
                    _l2 = _l2 * fms
                lp = net.macro_logprob(_s, fam) + _l2.flatten(1).sum(-1)
                # PER DIMENSION. The log-prob is a SUM over the sampled
                # dimensions, so the total KL scales with the head's size: when
                # the micro went from 10x10 to 16x10x10 the same real movement
                # gave 16x more KL and the guard cut at epoch 1 of 4, always.
                # Normalising makes it comparable across architectures.
                _nd = (N_MACRO + float(fms.flatten(1).sum(-1).mean())
                       if fms is not None
                       else N_MACRO + int(np.prod(fau.shape[1:])))
                kl = float((flp - lp).mean().clamp(min=0)) / _nd
                # KL PER HEAD. The two policies are independent Gaussians, so
                # the total KL decomposes into the sum and each half can drive
                # its own lr. Without this the macro -which contributes +$1 of
                # 953- spends the divergence budget and the controller brakes
                # EVERYTHING, including the micro, which contributes the
                # rest.
                _lpma2 = net.macro_logprob(_s, fam)
                _lpmi2 = _l2.flatten(1).sum(-1)
                _ndmi = (float(fms.flatten(1).sum(-1).mean()) if fms is not None
                         else int(np.prod(fau.shape[1:])))
                kl_ma = float((flpma - _lpma2).mean().clamp(min=0)) / N_MACRO
                kl_mi = float((flpmi - _lpmi2).mean().clamp(min=0)) / max(1.0, _ndmi)
                _d = (lp - flp)
                _sat = float((_d.abs() > 10).float().mean())
                _sd = float(_d.std())
            if upd % 5 == 0 or upd == 1:
                print(f"    [diag] kl/dim={kl:.2e}  sd(lp-flp)={_sd:.2f}  "
                      f"ratio saturation={_sat:.1%}  dims={_nd}", flush=True)
            # THE WORST HEAD, not the mean. Aggregate KL dilutes: measured,
            # `kl_micro` reached 0.112 while the aggregate stayed below the
            # 0.06 cut and the guard did not fire, with saturation rising to
            # 0.22. It is the same error as the mean win rate hiding the
            # per-rung state.
            #
            # It adds no new number: it uses the per-head KLs already computed
            # for the lr controllers.
            if max(kl, kl_ma, kl_mi) > a.kl_max:
                kl_cuts += 1
                break

        # STEP SIZE CONTROLLED BY MEASURED KL. Guessing the lr by hand failed
        # twice in a row: at 3e-4 kl/dim grew from 0.036 to 0.363 in 30 updates
        # -money peaked at 15 and fell- and at 3e-3 it jumped to 3.23 with 98%
        # saturation on the first update. KL is already measured every epoch,
        # so it closes the loop, not me.
        if a.kl_target > 0:
            def _factor(_k):
                if _k > 2.0 * a.kl_target:
                    return 0.7
                if _k < 0.5 * a.kl_target:
                    return 1.1
                return 1.0
            _fma, _fmi = _factor(kl_ma), _factor(kl_mi)
            # group 1 = macro, group 2 = micro: each with ITS own loop.
            for _ix, _f2 in ((1, _fma), (2, _fmi)):
                if _f2 != 1.0:
                    opt.param_groups[_ix]["lr"] = min(
                        1e-2, max(1e-6, opt.param_groups[_ix]["lr"] * _f2))
            # Group 3 -critic and auxiliaries- is not policy: it generates no
            # KL and is not touched.
            # THE MACRO NEVER FASTER THAN THE MICRO. Without this the loop
            # allocates speed in INVERSE proportion to impact: a head that does
            # not change behaviour generates little KL and is let run, while
            # the one that decides generates a lot and is braked. Measured at
            # championship scale: macro lr 6.7x in 15 updates, `micro_w_norm`
            # FLAT -the head contributing $952 of 953 stopped learning- and
            # money falling from $33,508 to $30,080.
            #
            # It is a RELATION between two measured quantities, not a chosen
            # number: the macro may go as fast as the micro, never faster.
            opt.param_groups[1]["lr"] = min(opt.param_groups[1]["lr"],
                                            opt.param_groups[2]["lr"])
            # THE TRUNK IS CONTROLLED BY THE TOTAL KL, which is what it
            # genuinely produces: it feeds both heads, so its effect on the
            # policy is the whole, not the minimum of two loops that belong to
            # someone else.
            #
            # It used to be `min(_fma, _fmi)` and that is a RATCHET: it brakes
            # if EITHER head brakes (0.7) but only accelerates if BOTH
            # accelerate at once (1.1). With two heads moving independently,
            # going down is far more likely than going up, so the trunk falls
            # to the floor even when the KL does not ask for it.
            #
            # Measured over 2,625 updates: trunk lr 0.18x while the heads rose
            # 1.35x -the heads/trunk ratio went from 1.1x to 8.3x- with total
            # KL at 0.007 against a target of 0.010. It was BELOW the target:
            # the loop should have been accelerating the trunk, not strangling
            # it. The shared representation was left nearly frozen and only the
            # heads on top of it learned, which coincides with money stalling
            # from update 1744 onwards.
            #
            # The safety property holds by itself: if a head blows up, the
            # total KL rises with it and the trunk brakes anyway.
            _ftr = _factor(kl)
            if _ftr != 1.0:
                opt.param_groups[0]["lr"] = min(
                    1e-2, max(1e-6, opt.param_groups[0]["lr"] * _ftr))
            if upd % 5 == 0 or upd == 1:
                print(f"    [lr] tronco {opt.param_groups[0]['lr']:.2e}  "
                      f"macro {opt.param_groups[1]['lr']:.2e} (kl {kl_ma:.3f})  "
                      f"micro {opt.param_groups[2]['lr']:.2e} (kl {kl_mi:.3f})  "
                      f"kl/dim {kl:.3f} -> objetivo {a.kl_target}", flush=True)

        # EXTRA epochs for the critic only. Measured: its ceiling with enough
        # data is R2 = 0.900 and in flight it sits at 0.675; the gap is
        # training, not noise. Sharing epochs with the policy makes the policy
        # pull on the trunk and the value never finishes fitting.
        # MINIBATCHES FOR THE CRITIC ONLY. Measured: splitting the POLICY into
        # 8 multiplies its KL by 24 and saturates the ratio at 100% -the
        # controller would have to cut the lr in the same proportion and the
        # net movement would be unchanged-. The critic does not have that
        # problem: it is a regression
        # and does not enter the importance ratio, so more small steps
        # only help it. Its R2 sits at 0.80 with a measured ceiling of 0.900.
        _nmv = max(1, int(a.minibatches))
        _size_v = max(1, _N // _nmv)
        for _ in range(a.value_epochs):
          _order_v = torch.randperm(_N, device=dev) if _nmv > 1 else None
          for _jv in range(_nmv):
            if _nmv > 1:
                _sv = _order_v[_jv * _size_v:(_jv + 1) * _size_v]
                if len(_sv) == 0:
                    continue
                _g, _b, _h, _r = fg[_sv], fb[_sv], fh[_sv], fret[_sv]
            else:
                _g, _b, _h, _r = fg, fb, fh, fret
            v_pred = net(_g, _b, _h)["value"]
            l = torch.nn.functional.smooth_l1_loss(v_pred, (_r - _vmu) / _vsd)
            opt.zero_grad(); l.backward()
            _gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            _grad_norms.append(float(_gn))
            opt.step()

        # SELF-PLAY BY MERIT, not by schedule. Each "ours" rung carries a
        # frozen version of the policy. The rotation is NOT round-robin: only
        # the one we already dominate is replaced.
        #
        # Why it matters: a version from a thousand updates ago that still
        # beats us 40% of the time is the best teacher we have, and
        # overwriting it because "it is its turn" throws it away. And since
        # this is a simulator, measuring against whoever costs us is cheap:
        # rotating blind is what you do when evaluation is expensive.
        #
        # The criterion is the same as for sliding -beating it consistently-
        # applied per rung, with its own rate.
        # ONE MEASUREMENT, SHARED. The merit relief and the automatic
        # curriculum run on the SAME update (both on `% refresh`), and the
        # relief calls `forget_results()`, which sets the whole history to nan.
        # The curriculum, which runs right after, asked again and ALWAYS found
        # nan: it never updated its estimate, so it split uniformly forever.
        #
        # Measured: 0 reallocations in 2,625 updates, share 1.00 on all eleven
        # rungs and an aggregate `win_rate` of nan. With 8 of the 11 workers on
        # rungs we won 88-100% of the time, i.e. 73% of the compute spent on
        # opponents that by the curriculum's own criterion teach nothing.
        _wpp_ref = None
        if upd % max(1, a.refresh) == 0:
            _wpp_ref = env.win_rate_per_rung()
        if _rm_now and upd % max(1, a.refresh) == 0:
            try:
                _wpp3 = list(_wpp_ref)
                _ours = [i for i, v in enumerate(_rm_now) if v is not None]
                # THRESHOLD 0.5, and it is not a hand-set constant: it is the
                # definition of "we are better than that version". It used to
                # use `a.slide`, which defaults to 0, so `0% >= 0` was true and
                # it relieved precisely the versions that were BEATING us
                # 100%.
                _dominated = [i for i in _ours
                              if i < len(_wpp3) and _wpp3[i] == _wpp3[i]
                              and _wpp3[i] > 0.5]
                if _dominated:
                    _new = fam.mean(0).detach().cpu().numpy().tolist()
                    # the most dominated of all gives up its slot
                    _k_ref = max(_dominated, key=lambda i: _wpp3[i])
                    _rm_now[_k_ref] = _new
                    env.set_rival_macro(_rm_now)
                    # THE WHOLE NETWORK, not the macro vector.
                    # `set_rival_macro` sends a vector, i.e. OUR EXECUTOR WITH
                    # THE BOARD HEURISTIC and no micro head; that removes
                    # precisely the component that carries the value.
                    #
                    # Measured with the SAME macro on both sides, 5 seeds of
                    # 24h x 30d: the network $64,762 against the vector
                    # opponent's $29,924 -+116%, 5 of 5-. That is why the
                    # relief replaced it 105 times in one night without it ever
                    # ceasing to lose: it could not win, it was missing half
                    # the agent.
                    _frozen_net = E2EAgent(cfg)
                    _frozen_net.load_state_dict(
                        {k: v.detach().cpu().clone()
                         for k, v in net.state_dict().items()})
                    _j_ref = _assign[_k_ref] if _assign and _k_ref < len(_assign) else _k_ref
                    _pool_net.add(_j_ref)
                    env.set_selfplay_on(
                        [k for k, j in enumerate(_assign or [])
                         if j == _j_ref] or [_k_ref], _frozen_net)
                    env.forget_results()
                    print(f"  [upd {upd}] rung {_k_ref} was being won at "
                          f"{_wpp3[_k_ref]:.0%}: relieved by OUR current policy. "
                          f"The rest are kept because they still teach",
                          flush=True)
            except Exception as _e:
                print(f"  warning: could not relieve self-play ({_e})",
                      flush=True)
        # ------- AUTOMATIC CURRICULUM -------
        if a.auto_curriculum and upd % max(1, a.refresh) == 0 and _levels:
            try:
                # the measurement from BEFORE the relief; see `_wpp_ref` above.
                _w4 = list(_wpp_ref) if _wpp_ref is not None else env.win_rate_per_rung()
                # the estimate lives in the RUNG, not in the worker: if a
                # worker changes rung, its history stays with the rung it was
                # playing.
                for _k4, _v4 in enumerate(_w4):
                    if _v4 == _v4 and _k4 < len(_assign):
                        _p4 = _pool_p[_assign[_k4]]
                        _pool_p[_assign[_k4]] = (0.7 * _p4 + 0.3 * _v4
                                               if _p4 is not None else _v4)
                _info = [( (p4 * (1.0 - p4)) if p4 is not None else 0.25 )
                         for p4 in _pool_p]          # unmeasured -> p=0.5
                # FIXED QUOTAS, outside the information split. `p(1-p)`
                # measures where the ESTIMATE is most uncertain, not where the
                # objective is, and that fails at both extremes:
                #
                #  * It is 0 when we ALWAYS lose. The uncapped rung is the
                #    competition opponent and we lose it 100%, so the criterion
                #    abandons precisely what measures us.
                #  * It is 0.25 -the MAXIMUM- for self-play, because a frozen
                #    copy of ourselves gives p = 0.5 by construction. Without a
                #    quota it would take everything and we would end up
                #    training to beat ourselves instead of to beat them.
                #
                # Measured against a passive opponent: us $72,796, median of 63
                # public agents $186,594. The exam distribution is them, not
                # our own copy.
                _n = len(_assign)
                _is_self = [j for j in range(len(_pool)) if _pool[j][2] is not None]
                _is_target = [j for j in range(len(_pool))
                           if _pool[j][1] is None and _pool[j][2] is None]
                _qa = min(max(0, a.selfplay_quota), len(_is_self), _n)
                _qo = min(max(0, a.target_quota), len(_is_target), _n - _qa)
                _fixed = ([_is_self[i % len(_is_self)] for i in range(_qa)]
                          + [_is_target[i % len(_is_target)] for i in range(_qo)])
                _free = _n - len(_fixed)
                # the rest by information, and ONLY over the rungs that do
                # not have a quota of their own
                _rest = [j for j in range(len(_pool))
                          if j not in set(_is_self) | set(_is_target)]
                _total = sum(_info[j] for j in _rest) if _rest else 0.0
                if _free > 0 and _total > 1e-9:
                    _new_assign = list(_fixed)
                    for _j4 in _rest:
                        _new_assign += [_j4] * max(0, int(round(_free * _info[_j4] / _total)))
                    _best = max(_rest, key=lambda j: _info[j])
                    _new_assign = (_new_assign[:len(_fixed)]
                                   + _new_assign[len(_fixed):][:_free])
                    _new_assign += [_best] * (_n - len(_new_assign))
                    _new_assign = _new_assign[:_n]
                elif _free > 0:
                    _new_assign = list(_fixed) + [_fixed[-1] if _fixed else 0] * _free
                else:
                    _new_assign = list(_fixed)[:_n]
                if True:
                    if _new_assign != _assign:
                        _assign = _new_assign
                        env.set_rival_cap([_pool[j][1] for j in _assign])
                        env.raise_level([_pool[j][0] for j in _assign])
                        env.set_rival_macro([_pool[j][2] for j in _assign])
                        # Self-play rungs carry a NETWORK, not a vector, and
                        # `set_rival_macro` has just overwritten it everywhere.
                        # It is restored to the workers that land on one.
                        if _pool_net and _frozen_net is not None:
                            _returning = [k for k, j in enumerate(_assign)
                                        if j in _pool_net]
                            if _returning:
                                env.set_selfplay_on(_returning, _frozen_net)
                        env.forget_results()
                        _res = {}
                        for j in _assign:
                            _res[j] = _res.get(j, 0) + 1
                        print(f"  [upd {upd}] CURRICULUM: split by information "
                              f"-> " + ", ".join(
                                  f"p{j}x{n}" for j, n in sorted(_res.items())),
                              flush=True)
            except Exception as _e:
                print(f"  aviso: curriculo automatico ({_e})", flush=True)
        if a.slide > 0 and upd % 25 == 0 and _levels:
            try:
                _wpp = env.win_rate_per_rung()
                # the easiest rung is the first in the list
                if _wpp and _wpp[0] == _wpp[0] and _wpp[0] >= a.slide:
                    # Adding at the top ROTATES THE STYLE instead of
                    # duplicating the last one: otherwise, after enough sliding
                    # all eleven rungs would end up being the same opponent and
                    # we would lose the variety that is the reason for having
                    # eleven.
                    _styles = sorted(set(_levels))
                    _sig = _styles[(_styles.index(_levels[-1]) + 1)
                                    % len(_styles)]
                    _rival_cap = _rival_cap[1:] + [_rival_cap[-1]]
                    _levels = _levels[1:] + [_sig]
                    env.set_rival_cap(_rival_cap)
                    env.raise_level(_levels)
                    env.forget_results()
                    print(f"  [upd {upd}] LADDER SLID: the easiest rung "
                          f"easiest was won at {_wpp[0]:.0%}; dropped from sampling. "
                          f"caps now {_rival_cap}", flush=True)
            except Exception as _e:
                print(f"  warning: could not slide the ladder ({_e})", flush=True)
        if (a.promote_rival > 0 and upd % 5 == 0
                and upd - _last_promo >= a.refresh):
            # The minimum separation is `--refresh`. Without it promotion
            # cascades: clearing the history is not enough because five updates
            # later the new window has very few episodes and, winning, gives
            # ~1.0 again. Measured: 9 promotions in 45 updates.
            _wr = env.win_rate()
            if _wr == _wr and _wr >= a.promote_rival:
                # The opponent has become too small: it is replaced by a
                # FROZEN copy of the current policy. The copy does not learn,
                # so the next stretch is played against an adversary that
                # already knows what we knew, and the binary signal regains
                # variance.
                _new = E2EAgent(cfg)
                _new.load_state_dict({k: v.detach().cpu().clone()
                                        for k, v in net.state_dict().items()})
                env.set_selfplay(_new)
                # Without this promotion cascades: the 60-episode window is
                # still full of wins against the old opponent.
                env.forget_results()
                _last_promo = upd
                print(f"  [upd {upd}] OPPONENT PROMOTED (win={_wr:.3f} >= "
                      f"{a.promote_rival}): it becomes a copy of the current "
                      f"policy", flush=True)
        if upd % 5 == 0 or upd == 1:
            wr = env.win_rate()
            rv = env.rival_stats()
            ret = float(np.mean(ret_ep[-80:])) if ret_ep else float("nan")
            print(f"upd {upd:4d}/{a.updates}  win={wr:.3f}  ret={ret:7.2f}  "
                  + (f"[KL cut {kl_cuts}] " if kl_cuts else "")
                  + f"$={env.mean_money():7.0f} vs {rv['money']:7.0f}  "
                  f"cult={rv['crops']:4.1f}r  uds={rv['units']:4.1f}r  "
                  f"macro={[round(float(x),2) for x in fam.mean(0)]}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            _ph = getattr(env, "money_by_horizon", lambda: {})()
            if len(_ph) > 1:
                print("        x_inaction per rung: " + "  ".join(
                    f"{k} {v/3000.0:.2f}" for k, v in sorted(
                        _ph.items(), key=lambda kv: kv[1])), flush=True)
            # PROMOTION AND DEMOTION. Demotion matters as much as promotion:
            # it is what stops the policy forgetting leagues already won.
            # `STAY` prevents churn: a league is judged over at least that
            # many updates inside it.
            if a.leagues:
                _in_league += 1
                if _in_league >= STAY:
                    # RELEGATION BY INACTION, not by defeat. Winning does not
                    # detect collapse: measured, a self-play league climbed
                    # four rungs at win=1.00 while money stayed at $3,000 -the
                    # starting cash- and the real margin fell to
                    # -98.3%. `x_inaction` does detect it, at ANY scale:
                    # 1.0 means the policy is doing nothing.
                    _inaction = env.mean_money() / 3000.0
                    _RIV_ULT[0] = float((env.rival_stats() or {}).get("money", 0.0) or 0.0)
                    # AND NO PROMOTION WHILE INERT. A rung is passed by
                    # PRODUCING value, not by outliving an even worse opponent:
                    # on L0 the public agent makes $171, so a 1.00 win rate is
                    # reached with x_inaction 0.99 -below doing nothing-.
                    # Promoting there is exactly the degenerate promotion that
                    # sank the earlier leagues, with no self-play involved.
                    # Threshold RELATIVE to the league ceiling, not absolute.
                    # PROMOTION = LEAGUE DOMINATED, not "I win more than I
                    # lose". Two conditions: the MARGIN reaches 90% of the best
                    # demonstrated in that league, and the policy is ALIVE.
                    # Promoting half-learned is what cascaded in earlier
                    # attempts.
                    #
                    # OVER INACTION, not over zero. This is the third time this
                    # rule breaks through the same class of bug: the bar was
                    # crossed without producing anything.
                    #   - by win rate:      the L0 opponent makes $4, so 1.00
                    #                       is won standing still
                    #   - by margin:        margin 2,973 just by keeping cash
                    #   - by raw x_inaction: the L0 ceiling is 1.03 and
                    #                       inaction 1.00, so the WHOLE
                    #                       learnable range is 3% and any
                    #                       threshold below 1.0 promotes an
                    #                       inert policy
                    # What must be demanded is 90% of the LEARNABLE margin,
                    # which is whatever sits above doing nothing.
                    _ceiling_l = LEAGUES[_league][4]
                    _target_inaction = 1.0 + CEILING_FRAC * max(0.0, _ceiling_l - 1.0)
                    _margin_now = _money - _RIV_ULT[0]
                    if _inaction >= _target_inaction and _league < len(LEAGUES) - 1:
                        _league += 1
                        print(f"  [upd {upd}] PROMOTED: x_inaction "
                              f"{_inaction:.3f} >= {_target_inaction:.3f} "
                              f"(ceiling {_ceiling_l:.2f}), margin "
                              f"{_margin_now:.0f} ->", flush=True)
                    elif _inaction < 0.35 * LEAGUES[_league][4] and _league > 0:
                        _league -= 1
                        print(f"  [upd {upd}] RELEGATED: inert policy "
                              f"(x_inaction {_inaction:.2f}) ->", flush=True)
                    else:
                        _in_league = STAY - 1   # keep measuring
                        _league = _league
                    if _in_league >= STAY:
                        env.close_(); env = _build_league(_league); _in_league = 0
                        accum[:] = 0.0; ret_ep.clear()
            if use_mlflow:
                try:
                    import mlflow
                    # WHAT IS LOGGED AND WHY. Through a whole debugging
                    # session, win_rate and money diagnosed NOTHING: they are
                    # the result, not the cause. Every real diagnosis came out
                    # of these others, which were only in the log:
                    #   kl/dim          grew from 0.036 to 0.363 -> divergence
                    #   saturation      99.4% -> PPO's ratio was noise
                    #   lr_*            the controller strangling the step
                    #   micro_w_norm    the head was not travelling (0.013 in
                    #                   40 updates)
                    #   active_dims     how many dimensions really decide
                    # INACTION GUARD. `startingMoney` is $3,000 and doing
                    # NOTHING ends exactly at 3,000. Measured: the self-play
                    # leagues converged to a flat 3,000 at EVERY horizon -the
                    # policy sat on the cash- because the frozen snapshot made
                    # 2,588, i.e. WORSE than inaction, and beating it required
                    # doing nothing. `win` said 1.00 and the real margin was
                    # -98.3%.
                    _money = env.mean_money()
                    _inertia = _money / float(
                        spec.DEFAULT_CONFIG.get("startingMoney", 3000) or 3000)
                    # Only when the horizon is SINGLE. When mixed, `league` is
                    # constant and redundant, and money averaged across
                    # 15/20/30 days means nothing: it is broken down instead.
                    _extra = {}
                    if a.leagues:
                        _extra["league"] = float(_league)
                    _by_horizon = getattr(env, "money_by_horizon", lambda: {})()
                    if len(_by_horizon) > 1:
                        # PER RUNG, and in multiples of inaction as well as in
                        # dollars. Inaction is $3,000 at ANY scale -not spending
                        # leaves the cash still- so x_inaction IS comparable
                        # across rungs while raw money is not: a 24-turn rung
                        # cannot earn what a 720-turn one does, and averaging
                        # them hides exactly what must be watched, which is
                        # whether SOME rung has gone inert.
                        for _d, _v in _by_horizon.items():
                            _extra[f"money_{_d}"] = float(_v)
                            _extra[f"x_inaction_{_d}"] = float(_v) / 3000.0
                        _xs = [v / 3000.0 for v in _by_horizon.values()]
                        _extra["x_inaction_min"] = float(min(_xs))
                        _extra["inert_rungs"] = float(
                            sum(1 for x in _xs if x < 1.02))
                    else:
                        _extra["over_inaction"] = float(_inertia)
                    if len(_by_horizon) > 1:
                        # With a grid the average is useless as a warning: it
                        # mixes 24-turn and 720-turn rungs. Warn per rung.
                        _inert = [k for k, v in _by_horizon.items()
                                    if v / 3000.0 < 1.02]
                        if _inert and upd > 20:
                            print(f"  WARNING upd {upd}: INERT {len(_inert)}/"
                                  f"{len(_by_horizon)} -> " + ", ".join(
                                      f"{k} {_by_horizon[k]/3000.0:.2f}x"
                                      for k in _inert), flush=True)
                    elif _inertia < 1.05 and upd > 20:
                        print(f"  WARNING upd {upd}: money {_money:.0f} ~ "
                              f"doing-nothing ({_inertia:.2f}x). Inert policy.",
                              flush=True)
                    # DASHBOARD IN FOUR GROUPS. It used to be 19 flat metrics
                    # mixing "am I winning" with "is the machinery healthy",
                    # with no reference lines and several broken by the mix of
                    # horizons. The numbered prefix groups them in the UI and
                    # fixes the reading order.
                    _RIV = float(rv.get("money", 0.0) or 0.0)
                    _m = {
                        # 1_RESULT: the only thing that says whether we are winning.
                        "1_result/win_rate": float(wr),
                        "1_result/margin_pct": (100.0 * (_money - _RIV) / _RIV
                                                   if _RIV > 0 else 0.0),
                        # x1 = doing nothing. MEASURED: inaction leaves the
                        # initial $3,000 at ANY horizon. (The "$245" quoted
                        # earlier was something else: units standing still but
                        # the market layer buying, which loses.)
                        "1_result/x_inaction": _money / 3000.0,
                        # 2_HEALTH: whether the machinery is learning.
                        "2_health/critic_r2": float(_r2),
                        "2_health/kl_per_dim": float(kl),
                        "2_health/saturation": float(_sat),
                        "2_health/epochs_run": float(a.epochs - kl_cuts),
                        # 3_CONTEXT: what we are being measured against.
                        "3_context/rival_money": _RIV,
                        # 4_DIAG: only looked at when something fails.
                        "4_diag/lr_trunk": float(opt.param_groups[0]["lr"]),
                        "4_diag/lr_heads": float(opt.param_groups[1]["lr"]),
                        "4_diag/active_dims": float(_nd),
                        "4_diag/sd_logratio": float(_sd),
                        "4_diag/reward": float(R.sum(0).mean()),
                    }
                    if not a.mix:
                        _m["1_result/money"] = _money
                    # DAMAGE PER HORIZON: how much we take from the opponent
                    # relative to what it makes against a PASSIVE player at THAT
                    # SAME horizon. It is the only competitive signal that moves
                    # while we lose every episode. The 30-day baseline applied
                    # to 15-day games used to lie: it read "thrashing" when all
                    # that happened was that the game was short.
                    _riv_by_horizon = getattr(env, "rival_by_horizon", lambda: {})()
                    for _d, _rv_d in _riv_by_horizon.items():
                        _b = BASE_PER_HORIZON.get(_d, {}).get(a.level, 0.0)
                        if _b > 0:
                            _m[f"3_context/damage_{_d}d_pct"] = 100.0 * (1.0 - _rv_d / _b)
                    # Scenario: always present, so the run can be followed.
                    _m["3_context/league"] = float(_league) if a.leagues else -1.0
                    _m["3_context/rival_level"] = float(a.level)
                    for _k, _v in _extra.items():
                        _m[("1_result/" if _k.startswith("money")
                            else "3_context/") + _k] = float(_v)
                    with torch.no_grad():
                        _m["4_diag/micro_w_norm"] = float(net.micro.weight.norm())
                        # MACRO SIGMA. Measured: narrowing the search costs 26
                        # points of win rate (3.3 se), so sigma falling is an
                        # ALARM, not a sign of convergence. It is watched
                        # together with its floor.
                        # GRADIENT NORM, before clipping. It is the direct
                        # diagnostic for the most expensive bug we have had:
                        # with log_prob over the sample without detach, mu
                        # cancels and this reads 0.000e+00 while the
                        # rest
                        # parece normal. El sintoma indirecto -"el despliegue
                        # no se mueve"- tardo 5.456 episodios en leerse.
                        if _grad_norms:
                            _gg = _grad_norms[-50:]
                            _m["2_health/grad_norm"] = float(np.mean(_gg))
                            _m["2_health/grad_zero_pct"] = 100.0 * float(
                                np.mean([g < 1e-9 for g in _gg]))
                        if a.jepa_weight > 0 and JZ:
                            # COLLAPSE WATCHDOG. If the encoder always emits
                            # the same vector, predicting it is trivial and
                            # nothing is learned. This falls to zero if that
                            # happens.
                            _z = torch.cat(JZ)
                            _m["2_health/jepa_sd"] = float(_z.std(0).mean())
                        try:
                            _m["2_health/kl_macro"] = float(kl_ma)
                            _m["2_health/kl_micro"] = float(kl_mi)
                            _m["4_diag/lr_macro"] = float(opt.param_groups[1]["lr"])
                            _m["4_diag/lr_micro"] = float(opt.param_groups[2]["lr"])
                        except Exception:
                            pass
                        # WIN RATE PER RUNG. Each worker plays one, so without
                        # this the sliding decides on a figure nobody can audit:
                        # the mean over eleven hides a rung won 100% of the time
                        # next to another lost 100% of the time.
                        try:
                            # INDEXED BY RUNG, not by worker. The curriculum
                            # reassigns workers to rungs, so `_wr[k]` is "what
                            # worker k plays NOW", not a fixed rung. Logging it
                            # by k left the labels frozen while the content
                            # changed: the table seemed to say we were drawing
                            # against the hardest opponent when that rung was
                            # not even being played.
                            _wpp2 = env.win_rate_per_rung()
                            _rpp2 = env.rival_per_rung()
                            _map = (_assign if _assign else list(range(len(_wpp2))))
                            for _i2, _v2 in enumerate(_wpp2):
                                if _v2 == _v2 and _i2 < len(_map):
                                    _m[f"5_rung/win_{_map[_i2]:02d}"] = float(_v2)
                            for _i2, _v2 in enumerate(_rpp2):
                                if _v2 == _v2 and _i2 < len(_map):
                                    _m[f"5_rung/rival_{_map[_i2]:02d}"] = float(_v2)
                            # how many workers sit on each rung
                            for _j2 in set(_map):
                                _m[f"5_rung/quota_{_j2:02d}"] = float(
                                    sum(1 for x in _map if x == _j2))
                        except Exception:
                            pass
                        _m["2_health/sigma_macro"] = float(net.log_sigma.exp().mean())
                        _sm = net.log_sigma_micro.detach().exp()
                        _m["2_health/sigma_micro_value"] = float(_sm[0].mean())
                        if _sm.shape[0] > 1:
                            _m["2_health/sigma_micro_verb"] = float(_sm[1:].mean())
                        _m["2_health/sigma_floor"] = float(a.sigma_floor)
                        # x_inaction: 1.0 = the policy is INERT. Four
                        # different collapses would have been visible at a
                        # glance with this on the dashboard.
                        _m["1_result/x_inaction"] = float(_money) / 3000.0
                        # how much of the macro depends on the STATE. If it is
                        # ~0 the head emits a constant and conditions nothing.
                        _m["2_health/macro_w_norm"] = float(net.macro_mu.weight.norm())
                        if getattr(net, "n_ops", 0):
                            _m["2_health/verb_signal_noise"] = float(
                                net.micro.bias[1:].abs().max()) / max(1e-9, cfg.sigma_ops)
                    mlflow.log_metrics(_m, step=_upd0 + upd)
                except Exception:
                    pass
        # Save the BEST by episode return, not the last. In an earlier run
        # update 40 was better than 55 and it got overwritten.
        # The LAST state, unconditionally. Saving only "the best by return"
        # left the checkpoint frozen at update 16 for 100 updates: in self-play
        # the return is RELATIVE and stops rising as soon as the opponent gets
        # stronger, even while the agent keeps improving -measured, the margin
        # against a strong public agent improved from -98.7% to -82.8% over
        # that same stretch-. Without this, an unattended night accumulates
        # nothing.
        if upd % 10 == 0 or upd == a.updates:
            torch.save({"sd": net.state_dict(), "cfg": vars(cfg), "init": vec0,
                        "macro_fields": _MACRO_FIELDS,
                        "upd": upd, "fingerprint": fingerprint(),
                        "model_fingerprint": model_fingerprint(),
                        "opt": opt.state_dict()}, a.out + ".ultimo")
        # --- safety net: check and, if due, rescue ---
        if ret_ep and len(ret_ep) >= 40:
            _r80 = float(np.mean(ret_ep[-80:]))
            _prev = _lifeline["ret"]
            # DROP RELATIVE TO MAGNITUDE, not a fraction of the value: with
            # NEGATIVE returns `r < 0.5*prev` inverts -from -0.8 to -0.7 is an
            # IMPROVEMENT and fired the rescue-. This works for both signs.
            _threshold = _prev - 0.5 * abs(_prev) if _prev is not None else None
            if _prev is not None and _r80 < _threshold and _lifeline["sd"] is not None:
                net.load_state_dict(_lifeline["sd"])
                try:
                    opt.load_state_dict(_lifeline["opt"])
                except Exception:
                    pass
                for _gr in opt.param_groups:
                    _gr["lr"] = max(1e-6, _gr["lr"] * 0.5)
                _lifeline["rescates"] += 1
                ret_ep.clear()
                print(f"  [upd {upd}] RESCATE {_lifeline['rescates']}: el retorno "
                      f"fell from {_prev:.1f} to {_r80:.1f}; restoring the last "
                      f"good state and halving the pace", flush=True)
            elif _prev is None or _r80 > _prev:
                _lifeline["ret"] = _r80
                _lifeline["sd"] = {k: v.detach().cpu().clone()
                                     for k, v in net.state_dict().items()}
                _lifeline["opt"] = opt.state_dict()

        if ret_ep and len(ret_ep) >= 20:
            r80 = float(np.mean(ret_ep[-80:]))
            if r80 > best:
                best = r80
                torch.save({"sd": net.state_dict(), "cfg": vars(cfg),
                            "init": vec0, "macro_fields": _MACRO_FIELDS, "ret": r80, "upd": upd,
                            "fingerprint": fingerprint(),
                            "model_fingerprint": model_fingerprint(),
                            "opt": opt.state_dict()}, a.out)


if __name__ == "__main__":
    main()
