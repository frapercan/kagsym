"""Migrate checkpoints trained before later observation and head changes.

Two global dimensions were appended to the end of the `time` block
-`EPISODE_STEPS/720` and `hand_cap/HANDS_REF`- so that the network KNOWS which
league it is playing. N_GLOBAL went from 88 to 90, so the global encoder input
(N_GLOBAL + N_HIST) went from 124 to 126.

The consequence took half a session to find: the usual loading pattern

    ok = {k: v for k, v in d["sd"].items() if k in act and act[k].shape == v.shape}
    net.load_state_dict({**act, **ok})

SILENTLY discards tensors whose shape does not match, leaving them as the
initialisation left them: RANDOM. The two discarded were the global encoder and
the summary, i.e. the FiLM modulation of the whole board and the vector that
feeds macro, critic and auxiliary head. Symptoms: per-seed money between $19
and $40,667, and the SAME checkpoint with the SAME seed giving different
results in each process, because the init randomness changes.

The migration is exact. New columns go to ZERO, so the new features contribute
nothing and the learned function is recovered bit for bit; from there training
can give them weight if they help.
"""
from __future__ import annotations

import torch

from . import obs as O
from .nets.world import N_HIST

# New absolute features ALWAYS go at the end of the `time` block, and this
# migration assumes it. If one is ever added elsewhere, this shifts the wrong
# weights SILENTLY, which is exactly the failure that cost half a session.
#
# Third absolute feature: `TURNS_PER_DAY/24`. N_GLOBAL 90->91 and the input
# 126->127. There are now THREE live widths, so the number of columns to
# insert is derived from the source width instead of being fixed:
#     124 -> 127   insert 3   (checkpoint older than everything)
#     126 -> 127   insert 1   (checkpoint from the previous change)
_TIME_END = O.GLOBAL_SLICES["time"].stop
NEW_WIDTH = O.N_GLOBAL + N_HIST
KNOWN_WIDTHS = (NEW_WIDTH, NEW_WIDTH - 1, NEW_WIDTH - 3)


def migrate_input(w: torch.Tensor) -> torch.Tensor:
    """(out, old width) -> (out, NEW_WIDTH), zeros in the new columns."""
    width = w.shape[1]
    if width == NEW_WIDTH:
        return w
    if width not in KNOWN_WIDTHS:
        raise ValueError(f"unexpected width {width}, expected one of "
                         f"{KNOWN_WIDTHS}")
    n_new = NEW_WIDTH - width
    i = _TIME_END - n_new
    out = w.new_zeros(w.shape[0], NEW_WIDTH)
    out[:, :i] = w[:, :i]
    out[:, i + n_new:] = w[:, i:]
    return out


def migrate_macro_head(sd: dict) -> list:
    """Macro head that GROWS, preserving the learned function.

    Every time a hand-set constant is exposed, the vector gains dimensions and
    an older checkpoint stops fitting. Extending it is correct, but only one
    way:

      * weight -> new rows to ZERO. The new dimensions do not depend on the
                  state yet, exactly as when they were constants.
      * bias   -> logit(the field's default). The executor applies
                  sigmoid(macro_mu), so this returns EXACTLY the value the
                  constant had, and the migrated policy behaves identically.
      * sigma  -> the MEDIAN of what the checkpoint had already learned, so the
                  new dimensions explore at the same scale as the rest and no
                  number has to be chosen.

    The default is read from `Macro` itself, not from a parallel list: it used
    to assume the new dims were the whole `PARAM_TABLE` block, and when fields
    that are not in that table were added -the per-turn rule weights- the index
    ran off the end. Reading the dataclass fields works for any future
    extension without being touched.
    """
    import math
    from dataclasses import fields as _fields
    from .macro import Macro, N_MACRO
    touched = []
    defaults = [float(f.default) for f in _fields(Macro)]
    lg = lambda x: math.log(max(1e-6, min(1 - 1e-6, x)) /
                            (1 - max(1e-6, min(1 - 1e-6, x))))
    for k, v in list(sd.items()):
        if k.endswith("macro_mu.weight") and v.shape[0] < N_MACRO:
            w = v.new_zeros(N_MACRO, v.shape[1]); w[: v.shape[0]] = v
            sd[k] = w; touched.append(f"{k}: {v.shape[0]} -> {N_MACRO} (zeroed)")
        elif k.endswith("macro_mu.bias") and v.shape[0] < N_MACRO:
            b_ = v.new_empty(N_MACRO); b_[: v.shape[0]] = v
            for i in range(v.shape[0], N_MACRO):
                b_[i] = lg(defaults[i])
            sd[k] = b_; touched.append(f"{k}: {v.shape[0]} -> {N_MACRO} (field default)")
        elif k.endswith("log_sigma") and v.shape[0] < N_MACRO:
            t = v.new_full((N_MACRO,), float(v.median()))
            t[: v.shape[0]] = v
            sd[k] = t; touched.append(f"{k}: {v.shape[0]} -> {N_MACRO} (median)")
    return touched


# DIMENSION REMOVED from the macro vector. `residual_cap` was written to
# `tasks.RESIDUAL_CAP` and nothing read it again once the residual mode
# disappeared: a learned dimension that decided nothing. It sat at index 30 of
# `Macro`, so EVERY checkpoint saved before its removal has one extra row
# there, and every row after it means one field further along than it does now.
#
# Growing such a head without deleting that row first is the silent
# misalignment this module exists to prevent: the head would still load, and 28
# dimensions would quietly decide the wrong thing.
#
# `LEGACY_MAX_WIDTH` is the widest head that ever carried it (60 = the vector
# just before the removal). New checkpoints carry `macro_fields`, so they are
# recognised by name and never touched by this path.
REMOVED_DIM_INDEX = 30
LEGACY_MAX_WIDTH = 60


def shrink_macro_head(sd: dict, macro_fields=None) -> list:
    """Delete the row of the removed dimension in a pre-removal checkpoint."""
    import torch as _t
    from .macro import N_MACRO
    if macro_fields is not None:
        # The checkpoint says which field each row is: nothing to guess.
        return []
    touched = []
    for k, v in list(sd.items()):
        if not (k.endswith("macro_mu.weight") or k.endswith("macro_mu.bias")
                or k.endswith("log_sigma")):
            continue
        n = v.shape[0]
        if not (REMOVED_DIM_INDEX < n <= LEGACY_MAX_WIDTH) or n == N_MACRO:
            continue
        keep = [i for i in range(n) if i != REMOVED_DIM_INDEX]
        sd[k] = v[_t.tensor(keep, device=v.device)]
        touched.append(f"{k}: {n} -> {sd[k].shape[0]} "
                       f"(dim {REMOVED_DIM_INDEX}, residual_cap, removed)")
    return touched


def migrate_stem(sd: dict, n_in: int) -> list:
    """Stem that GROWS when a channel is appended to the grid.

    The commitment channel is the 51st, appended AFTER both farm blocks
    precisely so this migration is a zero column at the end and not a shift of
    half the stem's weights. Zero means the network ignores it until training
    gives it weight, so an older checkpoint keeps its exact function.
    """
    touched = []
    for k, v in list(sd.items()):
        if k.endswith("stem.0.weight") and v.dim() == 4 and v.shape[1] < n_in:
            w = v.new_zeros(v.shape[0], n_in, *v.shape[2:])
            w[:, : v.shape[1]] = v
            sd[k] = w
            touched.append(f"{k}: {v.shape[1]} -> {n_in} input channels (zeroed)")
    return touched


def migrate_micro_head(sd: dict, n_out: int) -> list:
    """Micro head that GROWS when channels are added to the per-tile map.

    The assignment keys and queries are 2K extra channels of the same 1x1
    conv. An older checkpoint has 1+N_OPS and has to reach 1+N_OPS+2K without
    changing what it does, which is the same contract as the macro head:

      * weight -> new rows to ZERO
      * bias   -> ZERO, because the score multiplies by exp(<q,k>) and the
                  neutral element of a product is 1 = exp(0), not 0
      * sigma  -> the value the verb channels carry, so the new dimensions
                  explore at the scale of the other logits

    Growing them to anything else would silently change every assignment made
    by every checkpoint trained before the keys existed.
    """
    touched = []
    for k, v in list(sd.items()):
        if k.endswith("micro.weight") and v.dim() == 4 and v.shape[0] < n_out:
            w = v.new_zeros(n_out, *v.shape[1:]); w[: v.shape[0]] = v
            sd[k] = w; touched.append(f"{k}: {v.shape[0]} -> {n_out} (zeroed)")
        elif k.endswith("micro.bias") and v.dim() == 1 and v.shape[0] < n_out:
            b_ = v.new_zeros(n_out); b_[: v.shape[0]] = v
            sd[k] = b_; touched.append(f"{k}: {v.shape[0]} -> {n_out} (zeroed)")
        elif k.endswith("log_sigma_micro") and v.shape[0] < n_out:
            t = v.new_full((n_out,) + tuple(v.shape[1:]),
                           float(v[-1].reshape(-1)[0]))
            t[: v.shape[0]] = v
            sd[k] = t; touched.append(f"{k}: {v.shape[0]} -> {n_out} (verb sigma)")
    return touched


# Attributes renamed when the code moved to English. `state_dict` keys are
# ATTRIBUTE paths, so renaming a submodule invalidates every earlier
# checkpoint. They are translated on load, the same way growing heads are.
RENAMES = {"mundo.": "world.", "critico.": "critic."}


def migrate_keys(sd: dict) -> list[str]:
    """Translate keys of checkpoints predating the ES->EN rename."""
    touched = []
    for old, new in RENAMES.items():
        keys = [k for k in sd if k.startswith(old)]
        for k in keys:
            sd[new + k[len(old):]] = sd.pop(k)
        if keys:
            touched.append(f"{old}* -> {new}* ({len(keys)} tensors)")
    return touched


def migrate_sd(sd: dict, macro_fields=None, micro_out=None) -> tuple[dict, list[str]]:
    """Returns (migrated state_dict, list of touched keys).

    `macro_fields` is the field list the checkpoint was saved with. When it is
    there, the head is recognised by name; when it is not -every checkpoint
    older than that field- the legacy fix-up below applies.
    """
    out, touched = dict(sd), []
    touched += migrate_keys(out)        # first of all: the old names
    touched += shrink_macro_head(out, macro_fields)
    touched += migrate_macro_head(out)
    if micro_out:
        touched += migrate_micro_head(out, int(micro_out))
    from . import obs as _O
    touched += migrate_stem(out, _O.N_GRID_CH)
    for k in ("world.glob_enc.0.weight", "world.resumen.0.weight"):
        if k in out and out[k].shape[1] != NEW_WIDTH:
            out[k] = migrate_input(out[k])
            touched.append(k)
    return out, touched


def load_strict(net, sd, name_="checkpoint", macro_fields=None):
    """Load with migration, and RAISE if anything is left unloaded.

    Deliberately loud: the original failure was silent. If a tensor does not
    fit, an exception beats a policy with random parts that looks like it
    works and produces meaningless numbers.
    """
    sd, touched = migrate_sd(
        sd, macro_fields,
        micro_out=(net.micro.bias.shape[0]
                   if hasattr(net, 'micro') and hasattr(net.micro, 'bias') else None))
    act = net.state_dict()
    # MICRO HEAD SIGMA. It used to be a config constant; it is now an
    # `nn.Parameter` learned by PPO, like the macro one. An older checkpoint
    # does not carry it, so it is rebuilt with the values THAT checkpoint used
    # -channel 0 = sigma_micro, the rest = sigma_verb- and then reproduces its
    # exact behaviour instead of starting somewhere arbitrary. Without this,
    # `load_strict` raises, which is correct but would make it impossible to
    # resume anything older.
    # CABEZA NUEVA SIN PASADO. `espacial` se añadio el 2026-09-23 y ningun
    # checkpoint anterior la lleva, asi que `load_strict` se niega -- con
    # razon, porque su trabajo es que nada quede al azar en silencio.
    #
    # Pero una cabeza NUEVA no tiene nada que preservar: su inicializacion
    # fresca ES lo correcto, a diferencia de una capa del tronco que quedara
    # aleatoria. Se permite SOLO para cabezas declaradas aqui, y se anuncia;
    # la guarda sigue estricta para todo lo demas.
    _NUEVAS = ("espacial",)
    _frescas = [k for k in act if k.split(".")[0] in _NUEVAS and k not in sd]
    if _frescas:
        print(f"  {name_}: cabeza nueva sin pasado, inicializada fresca: "
              f"{sorted({k.split('.')[0] for k in _frescas})}", flush=True)
        sd = {**sd, **{k: act[k].clone() for k in _frescas}}

    if "log_sigma_micro" in act and "log_sigma_micro" not in sd:
        import math
        cfgd = getattr(net, "cfg", None)
        s_val = float(getattr(cfgd, "sigma_micro", 0.15) or 0.15)
        s_ops = float(getattr(cfgd, "sigma_ops", 0.03) or 0.03)
        t = act["log_sigma_micro"].clone()
        t[:] = math.log(s_ops)
        t[0] = math.log(s_val)
        sd = {**sd, "log_sigma_micro": t}
        touched = list(touched) + ["log_sigma_micro (rebuilt from cfg)"]
    bad = [(k, tuple(act[k].shape), tuple(sd[k].shape))
           for k in act if k in sd and act[k].shape != sd[k].shape]
    missing = [k for k in act if k not in sd]
    if bad or missing:
        raise RuntimeError(
            f"{name_}: does not fit and would be left RANDOM -> "
            f"shape mismatch {bad}, missing {missing}")
    net.load_state_dict({**act, **sd})
    return touched


def load_tolerant(net, sd, name_="checkpoint", verbose=True, macro_fields=None):
    """Migrate, load what fits, and SAY OUT LOUD what is left random.

    Tolerance is deliberate in some places (`--init-net` starts from a
    pretraining run with a different head). What was not deliberate is the
    silence: the message said "56/58 tensors" without naming them, and the two
    missing ones were the global encoder and the summary. Here they are always
    named.
    """
    sd, touched = migrate_sd(
        sd, macro_fields,
        micro_out=(net.micro.bias.shape[0]
                   if hasattr(net, 'micro') and hasattr(net.micro, 'bias') else None))
    act = net.state_dict()
    # MICRO HEAD SIGMA. It used to be a config constant; it is now an
    # `nn.Parameter` learned by PPO, like the macro one. An older checkpoint
    # does not carry it, so it is rebuilt with the values THAT checkpoint used
    # -channel 0 = sigma_micro, the rest = sigma_verb- and then reproduces its
    # exact behaviour instead of starting somewhere arbitrary. Without this,
    # `load_strict` raises, which is correct but would make it impossible to
    # resume anything older.
    if "log_sigma_micro" in act and "log_sigma_micro" not in sd:
        import math
        cfgd = getattr(net, "cfg", None)
        s_val = float(getattr(cfgd, "sigma_micro", 0.15) or 0.15)
        s_ops = float(getattr(cfgd, "sigma_ops", 0.03) or 0.03)
        t = act["log_sigma_micro"].clone()
        t[:] = math.log(s_ops)
        t[0] = math.log(s_val)
        sd = {**sd, "log_sigma_micro": t}
        touched = list(touched) + ["log_sigma_micro (rebuilt from cfg)"]
    ok = {k: v for k, v in sd.items() if k in act and act[k].shape == v.shape}
    random_ = [k for k in act if k not in ok]
    net.load_state_dict({**act, **ok})
    if verbose:
        if touched:
            print(f"{name_}: migrated to {NEW_WIDTH} inputs -> {', '.join(touched)}",
                  flush=True)
        if random_:
            print(f"WARNING {name_}: {len(random_)} tensors NOT loaded, left "
                  f"RANDOM -> {', '.join(random_)}", flush=True)
        else:
            print(f"{name_}: {len(ok)}/{len(act)} tensors, load COMPLETE",
                  flush=True)
    return len(ok), len(act), random_
