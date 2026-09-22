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
# (horas_por_dia, dias, tope_de_peones_del_rival).
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
# On the cap axis: 1 and 2 are degenerate and 13 is indistinguishable from uncapped
# (175.984 vs 176.422), asi que fibonacci muestrea mal ese eje -todo el
# gradiente vive en 9-12- y van los valores medidos.
# (horas_por_dia, dias, tope_del_rival, TECHO en multiplos de la inaccion).
#
# The ceiling is the BEST achievable in that league, measured with the
# mejor de 8 vectores al azar. Existe porque un umbral de ascenso FIJO es un
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
# (horas, dias, tope_del_rival, margen_alcanzable, x_inaccion_alcanzable).
#
# THE SMALLEST VIABLE IS 36 TURNS (6h x 6d). Measured sweep,
# 3 semillas, mejor de heuristica + 6 vectores al azar:
#     5h x  5d   25 turns  x_inaction 0.99  <- NOTHING beats doing nothing
#     6h x  6d   36 turnos             1,03  <- primera jugada rentable
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
LIGAS = [
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
FRAC_TECHO = 0.90
ASCENSO, DESCENSO, PERMANENCIA = 0.60, 0.30, 12


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
                    help="sigma de los canales de VERBO (logits)")
    ap.add_argument("--level", type=int, default=2, help="peldano de ESCALERA")
    ap.add_argument("--selfplay-quota", type=int, default=2,
                    help="workers ALWAYS on self-play, outside the information split")
    ap.add_argument("--target-quota", type=int, default=3,
                    help="trabajadores SIEMPRE en los peldanos sin tope (el "
                         "rival de competicion)")
    ap.add_argument("--auto-curriculum", action="store_true",
                    help="split workers across rungs in proportion to Bernoulli information p(1-p)"
                         "probado arranca en p=0,5 -informacion maxima-, asi ")
    ap.add_argument("--rival-macros", default=None,
                    help="opponents PER WORKER: comma-separated .npy paths, '-' to keep the public agent")
    ap.add_argument("--slide", type=float, default=0.0,
                    help="win rate above which a rung is considered EXHAUSTED and the ladder slides up. 0 = off"
                         "saturacion, pero autoinfligido-.")
    ap.add_argument("--levels", default=None,
                    help="LADDER rungs PER WORKER, comma-separated"
                         "arriesga aprender a batir a ESE en vez de a jugar.")
    ap.add_argument("--force-macro", action="store_true",
                    help="re-apply --init AFTER --resume; without it the checkpoint silently overrides the requested macro")
    ap.add_argument("--rival-macro", default=None,
                    help=".npy path: the opponent is OUR executor with that macro")
    ap.add_argument("--own-rival", action="store_true",
                    help="on reduced rungs the opponent is OUR executor with the macro CEM found for that scale"
                         "hacen ~0 $ en cualquier otra escala.")
    ap.add_argument("--grid", default=None,
                    help="rejilla MEZCLADA de escalas: 'h,d,tope;h,d,tope;...'. ")
    ap.add_argument("--mix", default=None,
                    help="horizontes MEZCLADOS en el mismo lote, en dias: '15,20,30'. "
                         "Cada trabajador juega uno. Excluye --ligas.")
    ap.add_argument("--leagues", action="store_true",
                    help="currículo: asciende ganando, desciende perdiendo")
    ap.add_argument("--league0", type=int, default=0, help="liga inicial")
    ap.add_argument("--hand-cap", type=int, default=None,
                    help="limita NUESTROS peones (curriculo); None = sin tope")
    ap.add_argument("--init", default="runs/macro_vs_v48.npy")
    ap.add_argument("--init-net", default=None,
                    help="pretrained micro checkpoint")
    ap.add_argument("--promote-rival", type=float, default=0.0,
                    help="win rate above which the opponent is replaced by a frozen copy of the policy"
                         "actual. 0 = desactivado. Ataca un fallo medido: al "
                         "binaria en 0,09 de su maximo 0,25.")
    ap.add_argument("--verb-head", action="store_true",
                    help="give the network the VERB head"
                         "frente a 8,94 ofrecidas-. Sin esta cabeza esas ")
    ap.add_argument("--mode", default="residuo", choices=("residuo", "directo", "ops"))
    ap.add_argument("--resume", default=None,
                    help="continuar desde un checkpoint de RL en vez de reempezar")
    ap.add_argument("--value-epochs", type=int, default=0,
                    help="extra epochs for the critic only (0 = none)")
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=None,
                    help="lr del TRONCO; si se omite, el del checkpoint (o 3e-4)")
    ap.add_argument("--lr-heads", type=float, default=None,
                    help="lr de las cabezas; por defecto = --lr")
    ap.add_argument("--kl-target", type=float, default=0.0,
                    help="if >0, the lr self-adjusts to hold this kl/dim")
    # 1.8e-4 was the threshold from when KL was measured WITHOUT normalising
    # dimension. Al normalizarlo, el KL sano de este problema vive en 1e-2 a
    # 4e-2, i.e. ninety times above: epochs ALWAYS aborted after
    # la primera -"[KL corto 1]" en cada update de la sesion entera- y a la vez
    # the controller kept lowering the lr to reach a target 111 times larger
    # than the abort threshold. The two dials pulling in opposite directions:
    # the trunk lr ended at 6.6e-6, 45 times below the
    # nominal, y cada lote daba UN solo paso de gradiente.
    #
    # The default is now 2x the target, which is standard PPO practice: the
    # abort is a safety net for the odd batch, not the normal regime. A/B
    # measured (11 updates, 8h x 13d + 13h x 13d, own opponent
    # calibrado): con 0,04 el retorno llega a 5,84 y aborta epocas en casi todos
    # los updates; con 0,12 llega a 8,54 y ya no aborta ninguna. El motivo es
    # that epoch 1 already measures kl/dim ~0.05 -there is a lag between the
    # rollout y la recalculada-, asi que un umbral de 2x el objetivo corta antes
    # de dar el primer paso util.
    ap.add_argument("--jepa-warmup", type=int, default=0,
                    help="initial updates training representation only (policy frozen)")
    ap.add_argument("--freeze-trunk", type=int, default=0,
                    help="updates after warmup during which the TRUNK stays frozen"
                         "de sobre un objetivo movil.")
    ap.add_argument("--jepa-weight", type=float, default=0.0,
                    help="weight of the JEPA loss: predict the future EMBEDDING with a stopped gradient"
                         "relevante. Vigilar `2_salud/jepa_sd`: si cae a cero, "
                         "colapso.")
    ap.add_argument("--ctx-micro", default="3x3",
                    choices=("1x1", "3x3", "3x3x2", "5x5", "attn"),
                    help="spatial context shape of the micro head")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="weight of the AUXILIARY loss: predict the opponent supply at Fibonacci horizons"
                         "Obliga al codificador a modelar como crece su granja, ")
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
                    help="rival = instantanea congelada de uno mismo")
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
    cfg = WorldConfig(device=dev, sigma_micro=a.sigma, con_ops=(a.mode == "ops" or a.verb_head),
                      sigma_ops=a.sigma_verb, ctx_micro=a.ctx_micro)
    net = E2EAgent(cfg).to(dev)
    vec0 = list(np.load(a.init))
    net.init_macro_at(vec0)
    from kagsym.symbolic import tasks as _T
    _T.MICRO_MODE = a.mode
    if a.kl_target > 0 and a.kl_max < a.kl_target:
        raise SystemExit(f"--kl-max {a.kl_max:g} es MENOR que --kl-objetivo "
                         f"{a.kl_target:g}: the controller would raise KL "
                         f"to the target and the abort would cut it on every "
                         f"epoca. Usa --kl-max >= 2x --kl-objetivo.")
    if a.hand_cap is not None:
        from kagsym import macro as _M
        _M.HAND_CAP = a.hand_cap
        print(f"tope de NUESTROS peones: {a.hand_cap}", flush=True)
    # ESCALA DEL OBJETIVO DE VALOR. `fret` va en unidades crudas (media ~20,
    # sd ~8) y la perdida era smooth_l1 con beta=1.0: TODO error mayor de 1
    # unit fell into pure L1 regime, with a +-1 gradient carrying no magnitude
    # of error. Measured on a toy regression with perfect linear signal and
    # esta misma escala: R2 = -16,58 asi, contra +0,826 normalizando. Explica
    # el critico en -0,405 sin culpar al tronco.
    #
    # The critic predicts NORMALISED and is denormalised for GAE, which needs
    # unidades crudas porque `ret = adv + Vn` bootstrapea de V.
    _vmu, _vsd, _vn = 0.0, 1.0, 0
    _n_reusados = _n_total = 0
    d0 = {}
    if a.resume:
        # CONTINUE, do not restart. Each run used to start from the supervised
        # pretraining and threw away all accumulated RL. Only valid if the
        # architecture has not
        # cambiado: hoy N_GLOBAL paso de 75 a 88 y N_MACRO de 7 a 13, y con eso
        # the shapes do not fit. What fits is loaded and the reuse reported.
        d0 = torch.load(a.resume, map_location="cpu", weights_only=False)
        from kagsym.migrate_ckpt import load_tolerant
        _n_reusados, _n_total, _azar = load_tolerant(net, d0["sd"], a.resume)
        net.to(dev)
        print(f"reanudado desde {a.resume}: {_n_reusados}/{_n_total} tensores "
              f"reusados (update {d0.get('upd','?')}, "
              f"retorno {d0.get('ret', float('nan')):.2f})", flush=True)
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
        _nok, _ntot, _ = load_tolerant(net, d0["sd"], a.init_net)
        net.init_macro_at(vec0)        # the macro head, from the vector
        net.to(dev)
        print(f"micro preentrenado desde {a.init_net}: "
              f"{_nok}/{_ntot} tensores reusados", flush=True)
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
    _g_macro, _g_micro, _g_otros, _tronco = [], [], [], []
    for _n, _p in net.named_parameters():
        _b = _n.split(".")[0]
        if _b in ("macro_mu", "log_sigma"):
            _g_macro.append(_p)
        elif _b in ("micro", "micro_ctx", "log_sigma_micro"):
            _g_micro.append(_p)
        elif _b in ("critico", "aux_rival", "jepa_proy", "jepa_pred"):
            _g_otros.append(_p)
        else:
            _tronco.append(_p)
    _cabezas = _g_macro + _g_micro + _g_otros
    _rm_actual = None
    # SELF-PLAY RUNGS WITH THE FULL NETWORK. The opponent's identity lives in
    # the RUNG, not in the worker: the curriculum reassigns workers and without
    # this the frozen opponent would be lost on the first reallocation.
    _pool_red = set()
    _red_congelada = None
    _pool, _pool_p, _asig = [], [], []
    _niveles = ([int(x) for x in a.levels.split(',')] if a.levels else None)
    if _niveles:
        from kagsym.environment import LADDER as _ESC
        print('opponents per worker: ' + ', '.join(
            str(_ESC[min(n, len(_ESC)-1)]) for n in _niveles), flush=True)
    if a.lr is None:
        a.lr = 3e-4
        _lr_explicito = False
    else:
        _lr_explicito = True
    _lrc = a.lr_heads if a.lr_heads is not None else a.lr
    # `--lr-cabezas` A SOLAS tambien manda. Sin esto, `_lr_explicito` solo se
    # was activated by `--lr`, so on resume both lrs from the checkpoint were
    # restored and the requested change was lost SILENTLY -the same class of
    # bug as `--init` clobbered by `--resume`, which cost `--force-macro`-.
    # What matters here is the heads/trunk RATIO: the KL controller rescales
    # both groups together, so the ratio is the only thing that survives.
    if a.lr_heads is not None:
        _lr_explicito = True
    # 0 tronco | 1 macro | 2 micro | 3 critico+auxiliares
    opt = torch.optim.Adam([{"params": _tronco, "lr": a.lr},
                            {"params": _g_macro, "lr": _lrc},
                            {"params": _g_micro, "lr": _lrc},
                            {"params": _g_otros, "lr": _lrc}])
    print(f"lr tronco {a.lr:.1e} ({len(_tronco)} tensores) | "
          f"lr cabezas {_lrc:.1e} ({len(_cabezas)} tensores)", flush=True)
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
    _arq_igual = (not a.resume) or (_n_reusados == _n_total)
    if a.resume and "opt" in d0 and _arq_igual:
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
            _est = dict(d0["opt"])
            _ps = [q for g in opt.param_groups for q in g["params"]]
            _fuera = []
            _nuevo_est = {}
            for _i, _v in (_est.get("state") or {}).items():
                _j = int(_i)
                _ok = _j < len(_ps)
                if _ok:
                    for _c in ("exp_avg", "exp_avg_sq"):
                        _t = _v.get(_c)
                        if _t is not None and tuple(_t.shape) != tuple(_ps[_j].shape):
                            _ok = False
                if _ok:
                    _nuevo_est[_i] = _v
                else:
                    _fuera.append(_j)
            _est["state"] = _nuevo_est
            opt.load_state_dict(_est)
            if _fuera:
                print(f"  momentos reiniciados en {len(_fuera)} de {len(_ps)} "
                      f"tensores (cambiaron de forma); el resto conserva Adam",
                      flush=True)
            # Adam's MOMENTS are always restored -they are what prevents the
            # huge first step- but an explicitly requested lr OVERRIDES the
            # checkpoint's. Without this, restoring the optimizer silently
            # clobbered the lr change the operator came to make.
            if _lr_explicito:
                opt.param_groups[0]["lr"] = a.lr
                opt.param_groups[1]["lr"] = _lrc
                print(f"  lr EXPLICITO ({a.lr:.1e} / {_lrc:.1e}, ratio "
                      f"{_lrc/a.lr:.1f}x), overrides the checkpoint's",
                      flush=True)
            print(f"  optimizador restaurado: lr tronco "
                  f"{opt.param_groups[0]['lr']:.2e} cabezas "
                  f"{opt.param_groups[1]['lr']:.2e}", flush=True)
        except Exception as e:
            print(f"  WARNING: could not restore the optimizer ({e}); "
                  f"continuing with factory lrs", flush=True)
    elif a.resume and "opt" in d0:
        print(f"  optimizador NO restaurado: la arquitectura cambio "
              f"({_n_reusados}/{_n_total} tensores). Momentos a cero.", flush=True)
    if a.resume and _n_reusados < _n_total and getattr(net, "n_ops", 0):
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
        print("  sesgo de valor repuesto a 1.0 (rescate de la inaccion)",
              flush=True)
    print(f"code fingerprint: {fingerprint()}", flush=True)
    print(f"dispositivo: {dev} | rival: {LADDER[a.level]} | init: {a.init}", flush=True)

    from kagsym import spec
    from kagsym.macro import Macro
    # Distributed rollouts: measured 632 steps/s serially against 10,069 in
    # parallel with 48 envs and 10 processes (15.9x). Startup is 1-3 s, once.
    if a.force_macro:
        # AFTER the resume on purpose: `init_macro_at` runs earlier and the
        # checkpoint overwrites it. Measured: the 5-day CEM macro was requested
        # ([0.121, 0.754, 0.057, ...]) and the network emitted the pretrained
        # one ([0.01, 0.24, 0.02, ...]), i.e. the 30-day vector. The experiment
        # probaba lo que yo creia y nada avisaba.
        net.init_macro_at(vec0)
        print(f"macro FORZADO a {a.init}", flush=True)
    _pasos = a.steps
    if a.mix:
        # MIXED, not sequential. In sequence the policy trains at one horizon
        # only and forgets the previous one -measured twice: real margin from
        # -82.7% to -98.3%-. Very short horizons are avoided on purpose:
        # a 5 dias no-hacer-nada da 3.000 $ y nuestra heuristica 2.920, o sea
        # that acting DESTROYS value and the lesson learned there is "do
        # hagas nada".
        _dias = [int(x) for x in a.mix.split(",")]
        _pasos = [d * 24 for d in _dias]
        a.days = max(_dias)
        print(f"MEZCLA de horizontes: {_dias} dias -> {_pasos} pasos", flush=True)
    _horas = None
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
        _rej = []
        for t in a.grid.split(";"):
            h, d, tp = (int(x) for x in t.split(","))
            _rej.append((h, d, tp))
        _pasos = [h * d for h, d, _ in _rej]
        _horas = [h for h, _, _ in _rej]
        # THIRD AXIS = THE OPPONENT'S CAP, not ours. `--hand-cap` limits OUR
        # hands (a constraint on our own action space); the ladder's difficulty
        # axis was always the opponent. That is also how the rungs were
        # calibrated, so they have to mean the same thing or the measured
        # ceilings do not apply. 0 = uncapped (opponent at full power).
        _rivtope = [(tp if tp > 0 else None) for _, _, tp in _rej]
        a.days = max(d for _, d, _ in _rej)
        print(f"REJILLA mezclada, {len(_rej)} peldanos:", flush=True)
        for h, d, tp in _rej:
            print(f"   {h:>3}h x {d:>3}d = {h*d:>4} turnos, tope {tp}", flush=True)
    env = ParallelEnv(a.envs, n_procs=a.procs, steps=_pasos,
                          macro=Macro.from_vector(vec0),
                          level=(_niveles if _niveles else a.level),
                          mode=a.mode, tope_peones=a.hand_cap, hours=_horas)
    if a.grid:
        env.set_rival_cap(_rivtope)
        print(f"OPPONENT cap per rung: {_rivtope}", flush=True)
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
            _COMPET = (24, 30)
            _riv = []
            for _h, _d, _ in _rej:
                _f = f"runs/ligas/macro_{_h}h{_d}d.npy"
                if (_h, _d) == _COMPET or not os.path.exists(_f):
                    _riv.append(None)
                else:
                    _riv.append(list(np.load(_f)))
            env.set_rival_macro(_riv)
            _n = sum(1 for v in _riv if v is not None)
            print(f"rival = NUESTRO ejecutor calibrado en {_n}/{len(_rej)} "
                  f"peldanos; publico en los demas", flush=True)
            for (_h, _d, _), _v in zip(_rej, _riv):
                print(f"   {_h}h x {_d}d: "
                      + ("executor with calibrated macro" if _v is not None
                         else "v48-fast-routes"), flush=True)
    # Ranges of env indices belonging to each worker, i.e. to each rung of the
    # grid. The parallel env hands out envs in order, `per_proc[k]`
    # consecutively per worker.
    _GRUPOS = None
    if a.grid and len(set(env.steps_proc)) > 1:
        _GRUPOS, _o = [], 0
        for _m in env.por_proc:
            _GRUPOS.append((_o, _o + _m))
            _o += _m
        print(f"advantage normalised per rung: {len(_GRUPOS)} groups "
              f"{[b - a_ for a_, b in _GRUPOS]}", flush=True)
    if a.rival_macros:
        _rm = []
        for _t in a.rival_macros.split(","):
            _t = _t.strip()
            _rm.append(None if _t in ("-", "") else list(np.load(_t)))
        env.set_rival_macro(_rm)
        _rm_actual = list(_rm)
        # The POOL of rungs: (level, cap, macro). It starts with the initial
        # assignment and the automatic curriculum redistributes workers among
        # them according to how much information each one gives.
        if _niveles:
            _pool = [(_niveles[i % len(_niveles)],
                      _rivtope[i % len(_rivtope)],
                      _rm[i % len(_rm)]) for i in range(len(_rm))]
            _pool_p = [None] * len(_pool)
            _asig = list(range(len(_pool)))
        print("opponents per worker (macro): " + ", ".join(
            ("publico" if x is None else "NUESTRO") for x in _rm), flush=True)
    elif a.rival_macro:
        env.set_rival_macro(list(np.load(a.rival_macro)))
        print(f"opponent = our executor with {a.rival_macro}", flush=True)
    _liga = a.league0
    _en_liga = 0
    _RIV_ULT = [0.0]

    def _monta_liga(idx):
        """Build the environment for league `idx`: hours, days and opponent cap.

        Changing hours or days forces recreating the workers, because
        `spec.TURNS_PER_DAY` y `spec.EPISODE_STEPS` son globales POR PROCESO.
        """
        hours, days, cap, _marg, _techo = LIGAS[idx]
        spec.set_turns_per_day(hours)
        steps = days * hours
        spec.set_episode_steps(steps)
        a.steps, a.days = steps, days
        e = ParallelEnv(a.envs, n_procs=a.procs, steps=steps,
                            macro=Macro.from_vector(vec0), level=4,
                            mode=a.mode, tope_peones=a.hand_cap,
                            hours=hours)
        e.set_rival_cap(cap)
        print(f"LIGA {idx}/{len(LIGAS)-1}: {hours}h x {days}d, rival v48 "
              f"tope {cap if cap is not None else 'sin tope'} "
              f"({steps} pasos)", flush=True)
        return e

    if a.leagues:
        env.cerrar()
        env = _monta_liga(_liga)

    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
        mlflow.set_experiment("kaggriculture-world-model")
        mlflow.start_run(run_name=a.run_name)
        mlflow.log_params(vars(a))
        # REFERENCES. A money curve cannot be read without its baselines.
        mlflow.log_params({
            "ref_inaccion": 3000,            # medido, a cualquier horizonte
            "ref_azar_legal": 8692,          # verbo legal al azar cada turno
            "ref_heuristica_v48tope5": 30065,
            "ref_backbone6_v48tope5": 32645,
            "ref_termometro_backbone6": -82.7,
            "ref_v48_tope5": 34905,
            "ref_experto_2945_vs_v48": 82382,
        })
        usar_ml = True
    except Exception:
        usar_ml = False

    # PERTURBATION PER EPISODE, not per day. Measured: resampling every day
    # costs 54% of the return ($40,972 fixed -> $18,967 sampled), because 30
    # days of random jitter destroy the farm's coherence.
    eps = torch.randn(a.envs, N_MACRO, device=dev)
    _NC = 1 + getattr(net, "n_ops", 0)
    _FORMA_U = (10, 10) if _NC == 1 else (_NC, 10, 10)
    # One sigma per channel: value and verbs live on different scales.
    # The MICRO head's sigma is no longer a config constant: it is an
    # `nn.Parameter` of the network learned by PPO, like the macro's. It is
    # recomputed at each use because after `opt.step()` the value changes;
    # caching it in a variable would leave a stale tensor and, during the
    # update, a graph that no longer corresponds.
    def SIG():
        return net.log_sigma_micro.exp()
    eps_u = torch.randn(a.envs, *_FORMA_U, device=dev)
    best = -1e18
    # SAFETY NET against collapse. Measured: one update with kl 25.7 took the
    # cash from $34,900 to $3 in fifteen updates and the controller reacted too
    # late. It keeps the last GOOD state and restores it if the return
    # collapses, also halving the pace.
    _salvavidas = {"ret": None, "sd": None, "opt": None, "rescates": 0}
    _grad_normas = []
    _ultima_promo = -10**9
    ret_ep = []          # returns of CLOSED EPISODES, not per-update sums
    acum = np.zeros(a.envs, dtype=np.float64)

    if a.selfplay:
        # Against a full-power public agent we lose 100%: the terminal is -1
        # ALWAYS and carries not one bit about the only thing that scores.
        # Against a copy of ourselves the rate sits around 0.59 with both
        # making the same money ($36,385 against $36,295), which is where a
        # binary signal has maximum information.
        env.set_selfplay(net)
        print("rival: AUTO-JUEGO (instantanea congelada de la politica)", flush=True)

    import copy
    bank = []
    # Bank seeds: DIVERSE opponents, not from our own lineage.
    _seed_vectors = []
    for _p in [x.strip() for x in (a.bank_seed or "").split(",") if x.strip()]:
        try:
            _v = np.load(_p)
            if _v.ndim != 1 or _v.shape[0] > N_MACRO:
                print(f"  semilla ignorada {_p}: dims {_v.shape}, se esperaba "
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
                print(f"  semilla {os.path.basename(_p)}: {_v.shape[0]} dims "
                      f"-> {N_MACRO}, padded with the defaults", flush=True)
                _v = _d
            _seed_vectors.append((os.path.basename(_p)[:-4], [float(x) for x in _v]))
        except Exception as _e:
            print(f"  semilla ignorada {_p}: {_e}", flush=True)
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
        print(f"el eje de pasos continua desde {_upd0}", flush=True)
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
            _tipo, _nom, _obj = _pool[(upd // a.refresh) % len(_pool)]
            if _tipo == "macro":
                # All diverse opponents AT ONCE, one per worker:
                # `set_rival_macro` accepts that natively. Rotating one at a
                # time makes the gradient see them in series, which is how it
                # forgets to beat what it already knew how to beat.
                env.set_rival_macro([v for _, v in _seed_vectors])
                _nom = f"{len(_seed_vectors)} diversos a la vez"
            else:
                rival = E2EAgent(cfg)
                rival.load_state_dict(_obj)
                env.set_selfplay(rival)
            print(f"  [upd {upd}] rival -> {_nom} ({_tipo}), poblacion de "
                  f"{len(_pool)}", flush=True)
        G, B, H, AM, AU, LP, V, R, D, MS = [], [], [], [], [], [], [], [], [], []
        LPMA, LPMI = [], []          # log-prob per head, for the factored ratio
        HF = []                      # opponent supply per day (auxiliary target)
        JZ = []                      # proyecciones JEPA por dia (objetivo, detenido)
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
            rec, fin = env.step_day(mapas=au.cpu().numpy(), macros=am.cpu().numpy())
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
            acum += rec
            for k in np.nonzero(fin)[0]:
                ret_ep.append(float(acum[k])); acum[k] = 0.0
            if fin.any():
                idx = torch.from_numpy(np.nonzero(fin)[0]).to(dev)
                eps[idx] = torch.randn(len(idx), N_MACRO, device=dev)
                eps_u[idx] = torch.randn(len(idx), *_FORMA_U, device=dev)
            G.append(g); B.append(b); H.append(h)
            AM.append(am); AU.append(au); LP.append(lp)
            LPMA.append(lp_m); LPMI.append(_lp_mi)
            V.append(s["valor"] * _vsd + _vmu)
            R.append(rec); D.append(fin)
        with torch.no_grad():
            g, b, hf = env.encode()
            ult = (net(torch.from_numpy(g).to(dev),
                      torch.from_numpy(b).to(dev))["valor"] * _vsd + _vmu)

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
        if _GRUPOS is not None:
            for _a, _b in _GRUPOS:
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
            _listo = [x.reshape(x.shape[0], len(RIVAL_WINDOWS), -1)[:, 0, :]
                      for x in HF]
            _fhf = torch.cat([
                torch.cat([_listo[min(t + k, _nd - 1)] for k in AUX_HORIZONS],
                          dim=-1)
                for t in range(_nd)])
        fms = torch.cat(MS) if MS else None
        fadv = torch.from_numpy(adv.reshape(-1).astype(np.float32)).to(dev)
        fret = torch.from_numpy(ret.reshape(-1).astype(np.float32)).to(dev)
        kl_cortes = 0
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
        # `--minilotes` afecta SOLO al critico, mas abajo.
        _nm, _tam = 1, _N
        for _ep in range(a.epochs):
            _orden = torch.randperm(_N, device=dev) if _nm > 1 else None
            for _j in range(_nm):
                if _nm > 1:
                    _sel = _orden[_j * _tam:(_j + 1) * _tam]
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
                    s["valor"], (_fret - _vmu) / _vsd)
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
                    perdida = a.value_weight * l_v + a.jepa_weight * l_jepa
                else:
                    perdida = (l_pi + a.value_weight * l_v + a.aux_weight * l_aux
                               + a.jepa_weight * l_jepa)
                opt.zero_grad(); perdida.backward()
                if a.jepa_warmup < upd <= a.jepa_warmup + a.freeze_trunk:
                    # MIND THE INTERVAL: it freezes AFTER the warmup, not
                    # during. During the warmup the trunk is precisely what has
                    # to learn; freezing it there would make the phase a no-op
                    # and we would then measure "JEPA does not work" when what
                    # did not work was the setup.
                    #
                    # The gradient is zeroed rather than removing the tensors
                    # from the optimizer, so Adam's state is not lost.
                    for _p in _tronco:
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
                _grad_normas.append(float(_gn))
                opt.step()
                if a.sigma_floor > 0:
                    with torch.no_grad():
                        net.log_sigma.clamp_(min=float(np.log(a.sigma_floor)))
                        if a.sigma_floor_micro > 0:
                            net.log_sigma_micro.clamp_(
                                min=float(np.log(a.sigma_floor_micro)))

            # KL DIVERGENCE ABORT. Gradient clipping limits the step's
            # MAGNITUDE, not how far the POLICY moves. Measured: 80 stable
            # updates (win ~0.5, $44,000) and then a cliff
            # -win 0.015, 7 514 $- del que no se recupera. Un solo update malo
            # destroys the policy; this cuts it before that happens.
            with torch.no_grad():
                # Over the WHOLE BATCH, with its own forward. With minibatches,
                # `lp` es el del ULTIMO trozo y `flp` el de todo: compararlos
                # seria comparar formas distintas.
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
                kl_cortes += 1
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
        # y no entra en el cociente de importancia, asi que mas pasos pequenos
        # solo le hacen bien. Su R2 va por 0,80 con techo medido en 0,900.
        _nmv = max(1, int(a.minibatches))
        _tamv = max(1, _N // _nmv)
        for _ in range(a.value_epochs):
          _ordv = torch.randperm(_N, device=dev) if _nmv > 1 else None
          for _jv in range(_nmv):
            if _nmv > 1:
                _sv = _ordv[_jv * _tamv:(_jv + 1) * _tamv]
                if len(_sv) == 0:
                    continue
                _g, _b, _h, _r = fg[_sv], fb[_sv], fh[_sv], fret[_sv]
            else:
                _g, _b, _h, _r = fg, fb, fh, fret
            v_pred = net(_g, _b, _h)["valor"]
            l = torch.nn.functional.smooth_l1_loss(v_pred, (_r - _vmu) / _vsd)
            opt.zero_grad(); l.backward()
            _gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            _grad_normas.append(float(_gn))
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
        if _rm_actual and upd % max(1, a.refresh) == 0:
            try:
                _wpp3 = list(_wpp_ref)
                _mios = [i for i, v in enumerate(_rm_actual) if v is not None]
                # THRESHOLD 0.5, and it is not a hand-set constant: it is the
                # definition of "we are better than that version". It used to
                # use `a.slide`, which defaults to 0, so `0% >= 0` was true and
                # it relieved precisely the versions that were BEATING us
                # 100%.
                _dominados = [i for i in _mios
                              if i < len(_wpp3) and _wpp3[i] == _wpp3[i]
                              and _wpp3[i] > 0.5]
                if _dominados:
                    _nuevo = fam.mean(0).detach().cpu().numpy().tolist()
                    # the most dominated of all gives up its slot
                    _k_ref = max(_dominados, key=lambda i: _wpp3[i])
                    _rm_actual[_k_ref] = _nuevo
                    env.set_rival_macro(_rm_actual)
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
                    _red_congelada = E2EAgent(cfg)
                    _red_congelada.load_state_dict(
                        {k: v.detach().cpu().clone()
                         for k, v in net.state_dict().items()})
                    _j_ref = _asig[_k_ref] if _asig and _k_ref < len(_asig) else _k_ref
                    _pool_red.add(_j_ref)
                    env.set_selfplay_on(
                        [k for k, j in enumerate(_asig or [])
                         if j == _j_ref] or [_k_ref], _red_congelada)
                    env.forget_results()
                    print(f"  [upd {upd}] peldano {_k_ref} lo ganabamos al "
                          f"{_wpp3[_k_ref]:.0%}: relieved by OUR current policy. "
                          f"The rest are kept because they still teach",
                          flush=True)
            except Exception as _e:
                print(f"  warning: could not relieve self-play ({_e})",
                      flush=True)
        # ------- AUTOMATIC CURRICULUM -------
        if a.auto_curriculum and upd % max(1, a.refresh) == 0 and _niveles:
            try:
                # the measurement from BEFORE the relief; see `_wpp_ref` above.
                _w4 = list(_wpp_ref) if _wpp_ref is not None else env.win_rate_per_rung()
                # the estimate lives in the RUNG, not in the worker: if a
                # worker changes rung, its history stays with the rung it was
                # playing.
                for _k4, _v4 in enumerate(_w4):
                    if _v4 == _v4 and _k4 < len(_asig):
                        _p4 = _pool_p[_asig[_k4]]
                        _pool_p[_asig[_k4]] = (0.7 * _p4 + 0.3 * _v4
                                               if _p4 is not None else _v4)
                _info = [( (p4 * (1.0 - p4)) if p4 is not None else 0.25 )
                         for p4 in _pool_p]          # sin medir -> p=0.5
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
                _n = len(_asig)
                _es_auto = [j for j in range(len(_pool)) if _pool[j][2] is not None]
                _es_obj = [j for j in range(len(_pool))
                           if _pool[j][1] is None and _pool[j][2] is None]
                _qa = min(max(0, a.selfplay_quota), len(_es_auto), _n)
                _qo = min(max(0, a.target_quota), len(_es_obj), _n - _qa)
                _fijos = ([_es_auto[i % len(_es_auto)] for i in range(_qa)]
                          + [_es_obj[i % len(_es_obj)] for i in range(_qo)])
                _libres = _n - len(_fijos)
                # the rest by information, and ONLY over the rungs that do
                # not have a quota of their own
                _resto = [j for j in range(len(_pool))
                          if j not in set(_es_auto) | set(_es_obj)]
                _tot = sum(_info[j] for j in _resto) if _resto else 0.0
                if _libres > 0 and _tot > 1e-9:
                    _nuevo_asig = list(_fijos)
                    for _j4 in _resto:
                        _nuevo_asig += [_j4] * max(0, int(round(_libres * _info[_j4] / _tot)))
                    _mejor = max(_resto, key=lambda j: _info[j])
                    _nuevo_asig = (_nuevo_asig[:len(_fijos)]
                                   + _nuevo_asig[len(_fijos):][:_libres])
                    _nuevo_asig += [_mejor] * (_n - len(_nuevo_asig))
                    _nuevo_asig = _nuevo_asig[:_n]
                elif _libres > 0:
                    _nuevo_asig = list(_fijos) + [_fijos[-1] if _fijos else 0] * _libres
                else:
                    _nuevo_asig = list(_fijos)[:_n]
                if True:
                    if _nuevo_asig != _asig:
                        _asig = _nuevo_asig
                        env.set_rival_cap([_pool[j][1] for j in _asig])
                        env.raise_level([_pool[j][0] for j in _asig])
                        env.set_rival_macro([_pool[j][2] for j in _asig])
                        # Self-play rungs carry a NETWORK, not a vector, and
                        # `set_rival_macro` has just overwritten it everywhere.
                        # It is restored to the workers that land on one.
                        if _pool_red and _red_congelada is not None:
                            _vuelven = [k for k, j in enumerate(_asig)
                                        if j in _pool_red]
                            if _vuelven:
                                env.set_selfplay_on(_vuelven, _red_congelada)
                        env.forget_results()
                        _res = {}
                        for j in _asig:
                            _res[j] = _res.get(j, 0) + 1
                        print(f"  [upd {upd}] CURRICULUM: split by information "
                              f"-> " + ", ".join(
                                  f"p{j}x{n}" for j, n in sorted(_res.items())),
                              flush=True)
            except Exception as _e:
                print(f"  aviso: curriculo automatico ({_e})", flush=True)
        if a.slide > 0 and upd % 25 == 0 and _niveles:
            try:
                _wpp = env.win_rate_per_rung()
                # the easiest rung is the first in the list
                if _wpp and _wpp[0] == _wpp[0] and _wpp[0] >= a.slide:
                    # Adding at the top ROTATES THE STYLE instead of
                    # duplicating the last one: otherwise, after enough sliding
                    # all eleven rungs would end up being the same opponent and
                    # we would lose the variety that is the reason for having
                    # eleven.
                    _estilos = sorted(set(_niveles))
                    _sig = _estilos[(_estilos.index(_niveles[-1]) + 1)
                                    % len(_estilos)]
                    _rivtope = _rivtope[1:] + [_rivtope[-1]]
                    _niveles = _niveles[1:] + [_sig]
                    env.set_rival_cap(_rivtope)
                    env.raise_level(_niveles)
                    env.forget_results()
                    print(f"  [upd {upd}] ESCALERA DESLIZADA: el peldano mas "
                          f"easiest was won at {_wpp[0]:.0%}; dropped from sampling. "
                          f"topes ahora {_rivtope}", flush=True)
            except Exception as _e:
                print(f"  warning: could not slide the ladder ({_e})", flush=True)
        if (a.promote_rival > 0 and upd % 5 == 0
                and upd - _ultima_promo >= a.refresh):
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
                _nuevo = E2EAgent(cfg)
                _nuevo.load_state_dict({k: v.detach().cpu().clone()
                                        for k, v in net.state_dict().items()})
                env.set_selfplay(_nuevo)
                # Without this promotion cascades: the 60-episode window is
                # still full of wins against the old opponent.
                env.forget_results()
                _ultima_promo = upd
                print(f"  [upd {upd}] RIVAL PROMOCIONADO (win={_wr:.3f} >= "
                      f"{a.promote_rival}): pasa a ser una copia de la "
                      f"politica actual", flush=True)
        if upd % 5 == 0 or upd == 1:
            wr = env.win_rate()
            rv = env.rival_stats()
            ret = float(np.mean(ret_ep[-80:])) if ret_ep else float("nan")
            print(f"upd {upd:4d}/{a.updates}  win={wr:.3f}  ret={ret:7.2f}  "
                  + (f"[KL corto {kl_cortes}] " if kl_cortes else "")
                  + f"$={env.mean_money():7.0f} vs {rv['dinero']:7.0f}  "
                  f"cult={rv['cultivos']:4.1f}r  uds={rv['unidades']:4.1f}r  "
                  f"macro={[round(float(x),2) for x in fam.mean(0)]}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            _ph = getattr(env, "money_by_horizon", lambda: {})()
            if len(_ph) > 1:
                print("        x_inaction per rung: " + "  ".join(
                    f"{k} {v/3000.0:.2f}" for k, v in sorted(
                        _ph.items(), key=lambda kv: kv[1])), flush=True)
            # PROMOTION AND DEMOTION. Demotion matters as much as promotion:
            # it is what stops the policy forgetting leagues already won.
            # `TENURE` prevents churn: a league is judged over at least
            # ese numero de updates dentro.
            if a.leagues:
                _en_liga += 1
                if _en_liga >= PERMANENCIA:
                    # DESCENSO POR INACCION, no por victoria. La victoria no
                    # detects collapse: measured, a self-play league climbed
                    # four rungs at win=1.00 while money stayed
                    # en 3.000 $ -la caja inicial- y el margen real caia a
                    # -98,3 %. `x_inaccion` si lo detecta, y a CUALQUIER escala:
                    # 1.0 means the policy is doing nothing.
                    _inac = env.mean_money() / 3000.0
                    _RIV_ULT[0] = float((env.rival_stats() or {}).get("dinero", 0.0) or 0.0)
                    # AND NO PROMOTION WHILE INERT. A rung is passed
                    # PRODUCIENDO valor, no sobreviviendo a un rival aun peor:
                    # en L0 el publico hace 171 $, asi que se gana 1,00 con
                    # x_inaccion 0,99 -por debajo de no hacer nada-. Ascender
                    # there is exactly the degenerate promotion that sank the
                    # leagues
                    # anteriores, sin autojuego de por medio.
                    # Umbral RELATIVO al techo de la liga, no absoluto.
                    # ASCENSO = LIGA DOMINADA, no "gano mas de lo que pierdo".
                    # Two conditions: the MARGIN reaches 90% of the best
                    # demostrado en esa liga, y la politica esta VIVA. Promover
                    # half-learned is what cascaded in earlier attempts
                    # anteriores.
                    # SOBRE LA INACCION, no sobre cero. Tercera vez que esta
                    # rule breaks by the same class of bug: the bar is
                    # cruzaba sin producir nada.
                    #   - por victoria: el rival de L0 hace 4 $, se gana 1,00 quieto
                    #   - by margin:   margin 2,973 just by keeping the cash
                    #   - por x_inaccion bruto: el techo de L0 es 1,03 y la
                    #     inaction 1.00, so the WHOLE learnable range is 3%
                    #     and any threshold below 1.0 crosses it
                    #     una politica inerte.
                    # What must be demanded is 90% of the LEARNABLE margin,
                    # que es lo que hay por encima de no hacer nada.
                    _techo_l = LIGAS[_liga][4]
                    _obj_inac = 1.0 + FRAC_TECHO * max(0.0, _techo_l - 1.0)
                    _marg_actual = _din - _RIV_ULT[0]
                    if _inac >= _obj_inac and _liga < len(LIGAS) - 1:
                        _liga += 1
                        print(f"  [upd {upd}] ASCIENDE: x_inaccion {_inac:.3f} "
                              f">= {_obj_inac:.3f} (techo {_techo_l:.2f}), "
                              f"margen {_marg_actual:.0f} ->", flush=True)
                    elif _inac < 0.35 * LIGAS[_liga][4] and _liga > 0:
                        _liga -= 1
                        print(f"  [upd {upd}] DESCIENDE: politica inerte "
                              f"(x_inaccion {_inac:.2f}) ->", flush=True)
                    else:
                        _en_liga = PERMANENCIA - 1   # sigue midiendo
                        _liga = _liga
                    if _en_liga >= PERMANENCIA:
                        env.cerrar(); env = _monta_liga(_liga); _en_liga = 0
                        acum[:] = 0.0; ret_ep.clear()
            if usar_ml:
                try:
                    import mlflow
                    # WHAT IS LOGGED AND WHY. Through a whole debugging
                    # session, win_rate and money diagnosed NOTHING: they are
                    # el resultado, no la causa. Cada diagnostico real salio de
                    # estas otras, que estaban solo en el log:
                    #   kl/dim          crecio de 0,036 a 0,363 -> divergencia
                    #   saturacion      99,4% -> el cociente de PPO era ruido
                    #   lr_*            the controller strangling the step
                    #   micro_w_norm    the head was not travelling (0.013 in 40 updates)
                    #   dims_activas    cuantas dimensiones deciden de verdad
                    # GUARD DE INACCION. `startingMoney` son 3.000 $ y no
                    # hacer NADA acaba exactamente en 3.000. Medido: las ligas
                    # en autojuego convergieron a 3.000 clavados a TODO
                    # horizon -the policy sat on the cash- because
                    # la instantanea congelada hacia 2.588, o sea PEOR que la
                    # inaction, and beating it required doing nothing. `win` said
                    # 1,00 y el margen real era -98,3 %.
                    _din = env.mean_money()
                    _inercia = _din / float(
                        spec.DEFAULT_CONFIG.get("startingMoney", 3000) or 3000)
                    # Only when the horizon is SINGLE. When mixed, `league` is
                    # constante y sobra, y el dinero promediado entre 15/20/30
                    # days means nothing: it is broken down.
                    _extra = {}
                    if a.leagues:
                        _extra["liga"] = float(_liga)
                    _porh = getattr(env, "money_by_horizon", lambda: {})()
                    if len(_porh) > 1:
                        # POR PELDANO, y en multiplos de la inaccion ademas de
                        # en dolares. La inaccion son 3.000 $ a CUALQUIER
                        # scale -not spending leaves the cash still- so
                        # x_inaccion SI es comparable entre peldanos mientras
                        # that raw money is not: a 24-turn rung
                        # no puede ganar lo que uno de 720, y promediarlos
                        # esconde exactamente lo que hay que vigilar, que es si
                        # SOME rung has gone inert.
                        for _d, _v in _porh.items():
                            _extra[f"dinero_{_d}"] = float(_v)
                            _extra[f"x_inaccion_{_d}"] = float(_v) / 3000.0
                        _xs = [v / 3000.0 for v in _porh.values()]
                        _extra["x_inaccion_min"] = float(min(_xs))
                        _extra["peldanos_inertes"] = float(
                            sum(1 for x in _xs if x < 1.02))
                    else:
                        _extra["sobre_inaccion"] = float(_inercia)
                    if len(_porh) > 1:
                        # With a grid the average is useless as a warning: it
                        # mixes 24-turn and 720-turn rungs. Warn per rung.
                        _inertes = [k for k, v in _porh.items()
                                    if v / 3000.0 < 1.02]
                        if _inertes and upd > 20:
                            print(f"  AVISO upd {upd}: INERTES {len(_inertes)}/"
                                  f"{len(_porh)} -> " + ", ".join(
                                      f"{k} {_porh[k]/3000.0:.2f}x"
                                      for k in _inertes), flush=True)
                    elif _inercia < 1.05 and upd > 20:
                        print(f"  AVISO upd {upd}: dinero {_din:.0f} ~ "
                              f"no-hacer-nada ({_inercia:.2f}x). Politica inerte.",
                              flush=True)
                    # DASHBOARD EN CUATRO GRUPOS. Antes eran 19 metricas
                    # planas mezclando "voy ganando" con "la maquinaria esta
                    # healthy", with no reference lines and several broken by
                    # the mix of horizons. The numbered prefix groups them in
                    # the UI and fixes the reading order.
                    _RIV = float(rv.get("dinero", 0.0) or 0.0)
                    _m = {
                        # 1_RESULT: the only thing that says whether we are winning.
                        "1_resultado/win_rate": float(wr),
                        "1_resultado/margen_pct": (100.0 * (_din - _RIV) / _RIV
                                                   if _RIV > 0 else 0.0),
                        # x1 = no hacer nada. MEDIDO: la inaccion deja los
                        # 3.000 $ iniciales a CUALQUIER horizonte. (El "245 $"
                        # que se cito antes era otra cosa: unidades quietas
                        # but the market layer buying, which loses.)
                        "1_resultado/x_inaccion": _din / 3000.0,
                        # 2_SALUD: si la maquinaria aprende.
                        "2_salud/critico_r2": float(_r2),
                        "2_salud/kl_por_dim": float(kl),
                        "2_salud/saturacion": float(_sat),
                        "2_salud/epocas_corridas": float(a.epochs - kl_cortes),
                        # 3_CONTEXT: what we are being measured against.
                        "3_contexto/dinero_rival": _RIV,
                        # 4_DIAG: only looked at when something fails.
                        "4_diag/lr_tronco": float(opt.param_groups[0]["lr"]),
                        "4_diag/lr_cabezas": float(opt.param_groups[1]["lr"]),
                        "4_diag/dims_activas": float(_nd),
                        "4_diag/sd_logratio": float(_sd),
                        "4_diag/recompensa": float(R.sum(0).mean()),
                    }
                    if not a.mix:
                        _m["1_resultado/dinero"] = _din
                    # MERMA POR HORIZONTE: cuanto le quitamos al rival
                    # respecto a lo que hace contra un PASIVO en ESE MISMO
                    # horizon. It is the only competitive signal that moves
                    # while we lose every episode. With the 30-day base
                    # dias aplicada a partidas de 15 mentia: leia "paliza"
                    # cuando solo era que la partida era corta.
                    _rporh = getattr(env, "rival_by_horizon", lambda: {})()
                    for _d, _rv_d in _rporh.items():
                        _b = BASE_PER_HORIZON.get(_d, {}).get(a.level, 0.0)
                        if _b > 0:
                            _m[f"3_contexto/merma_{_d}d_pct"] = 100.0 * (1.0 - _rv_d / _b)
                    # Scenario: always present, so the run can be followed.
                    _m["3_contexto/liga"] = float(_liga) if a.leagues else -1.0
                    _m["3_contexto/nivel_rival"] = float(a.level)
                    for _k, _v in _extra.items():
                        _m[("1_resultado/" if _k.startswith("dinero")
                            else "3_contexto/") + _k] = float(_v)
                    with torch.no_grad():
                        _m["4_diag/micro_w_norma"] = float(net.micro.weight.norm())
                        # MACRO SIGMA. Measured: narrowing
                        # busqueda cuesta 26 puntos de tasa de acierto (3,3 ee),
                        # so sigma falling is an ALARM, not a sign of
                        # convergencia. Se vigila junto a su suelo.
                        # NORMA DEL GRADIENTE, antes de recortar. Es el
                        # the direct diagnostic for the most expensive bug we
                        # have had: with log_prob over the sample without
                        # detach, mu cancels and this reads 0.000e+00 while the
                        # rest
                        # parece normal. El sintoma indirecto -"el despliegue
                        # no se mueve"- tardo 5.456 episodios en leerse.
                        if _grad_normas:
                            _gg = _grad_normas[-50:]
                            _m["2_salud/grad_norma"] = float(np.mean(_gg))
                            _m["2_salud/grad_cero_pct"] = 100.0 * float(
                                np.mean([g < 1e-9 for g in _gg]))
                        if a.jepa_weight > 0 and JZ:
                            # VIGILANTE DE COLAPSO. Si el codificador emite
                            # always the same vector, predicting it is trivial and
                            # no se aprende nada. Esto cae a cero si pasa.
                            _z = torch.cat(JZ)
                            _m["2_salud/jepa_sd"] = float(_z.std(0).mean())
                        try:
                            _m["2_salud/kl_macro"] = float(kl_ma)
                            _m["2_salud/kl_micro"] = float(kl_mi)
                            _m["4_diag/lr_macro"] = float(opt.param_groups[1]["lr"])
                            _m["4_diag/lr_micro"] = float(opt.param_groups[2]["lr"])
                        except Exception:
                            pass
                        # WIN RATE POR PELDANO. Cada trabajador juega uno, asi
                        # that without this the sliding decides on a figure that
                        # nadie puede auditar: la media de los once esconde un
                        # peldano ganado al 100 % junto a otro perdido al 100 %.
                        try:
                            # INDEXADO POR PELDANO, no por trabajador. El
                            # currículo reasigna trabajadores a peldanos, asi
                            # that `_wr[k]` is "what worker k plays
                            # NOW", not a fixed rung. Logging it by k left the
                            # labels frozen while the
                            # contenido cambiaba: la tabla parecia decir que
                            # we were drawing against the hardest opponent
                            # when that rung was not even being played.
                            _wpp2 = env.win_rate_per_rung()
                            _rpp2 = env.rival_per_rung()
                            _map = (_asig if _asig else list(range(len(_wpp2))))
                            for _i2, _v2 in enumerate(_wpp2):
                                if _v2 == _v2 and _i2 < len(_map):
                                    _m[f"5_peldano/win_{_map[_i2]:02d}"] = float(_v2)
                            for _i2, _v2 in enumerate(_rpp2):
                                if _v2 == _v2 and _i2 < len(_map):
                                    _m[f"5_peldano/rival_{_map[_i2]:02d}"] = float(_v2)
                            # cuantos trabajadores hay en cada peldano
                            for _j2 in set(_map):
                                _m[f"5_peldano/cuota_{_j2:02d}"] = float(
                                    sum(1 for x in _map if x == _j2))
                        except Exception:
                            pass
                        _m["2_salud/sigma_macro"] = float(net.log_sigma.exp().mean())
                        _sm = net.log_sigma_micro.detach().exp()
                        _m["2_salud/sigma_micro_valor"] = float(_sm[0].mean())
                        if _sm.shape[0] > 1:
                            _m["2_salud/sigma_micro_verbo"] = float(_sm[1:].mean())
                        _m["2_salud/sigma_suelo"] = float(a.sigma_floor)
                        # x_inaccion: 1.0 = la politica esta INERTE. Cuatro
                        # different collapses would have been visible at a glance.
                        _m["1_resultado/x_inaccion"] = float(_din) / 3000.0
                        # how much of the macro depends on the STATE. If it is
                        # ~0 the head emits a constant and conditions nothing.
                        _m["2_salud/macro_w_norma"] = float(net.macro_mu.weight.norm())
                        if getattr(net, "n_ops", 0):
                            _m["2_salud/verbo_senal_ruido"] = float(
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
                        "upd": upd, "huella": fingerprint(), "huella_modelo": model_fingerprint(),
                        "opt": opt.state_dict()}, a.out + ".ultimo")
        # --- red de seguridad: comprobar y, si toca, rescatar ---
        if ret_ep and len(ret_ep) >= 40:
            _r80 = float(np.mean(ret_ep[-80:]))
            _prev = _salvavidas["ret"]
            # DROP RELATIVE TO MAGNITUDE, not a fraction of the value: with
            # retornos NEGATIVOS `r < 0.5*prev` se invierte -de -0,8 a -0,7 es
            # an IMPROVEMENT and fired the rescue-. This works for both signs.
            _umbral = _prev - 0.5 * abs(_prev) if _prev is not None else None
            if _prev is not None and _r80 < _umbral and _salvavidas["sd"] is not None:
                net.load_state_dict(_salvavidas["sd"])
                try:
                    opt.load_state_dict(_salvavidas["opt"])
                except Exception:
                    pass
                for _gr in opt.param_groups:
                    _gr["lr"] = max(1e-6, _gr["lr"] * 0.5)
                _salvavidas["rescates"] += 1
                ret_ep.clear()
                print(f"  [upd {upd}] RESCATE {_salvavidas['rescates']}: el retorno "
                      f"fell from {_prev:.1f} to {_r80:.1f}; restoring the last "
                      f"good state and halving the pace", flush=True)
            elif _prev is None or _r80 > _prev:
                _salvavidas["ret"] = _r80
                _salvavidas["sd"] = {k: v.detach().cpu().clone()
                                     for k, v in net.state_dict().items()}
                _salvavidas["opt"] = opt.state_dict()

        if ret_ep and len(ret_ep) >= 20:
            r80 = float(np.mean(ret_ep[-80:]))
            if r80 > best:
                best = r80
                torch.save({"sd": net.state_dict(), "cfg": vars(cfg),
                            "init": vec0, "ret": r80, "upd": upd,
                            "huella": fingerprint(), "huella_modelo": model_fingerprint(), "opt": opt.state_dict()}, a.out)


if __name__ == "__main__":
    main()
