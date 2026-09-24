# Known debt

What is known to be wrong or oversized, kept because fixing it now would
change the game under a running measurement or costs more than it returns
today. Each item says what it is, why it waits, and what unblocks it.

## Game files (changing them changes every number)

- **`f_turns_init` and `f_turns_min` are dead by wiring.** The executor
  reads `TURNS_PER_TILE_INIT/MIN` in `Agent.__init__`, and `apply_params`
  writes them on the first turn, after construction. The dials exist, are
  emitted, and decide nothing. Fix: read them lazily on the first
  `_recalibrate`. Waits for: no search or validation in flight (it changes
  the game fingerprint and the live-dial list).
- **`tools/live_dials.py` calls a dial dead** when both extremes give the
  same money and opponent money on 3 seeds. A dial could change operations
  without changing money; the previous tool also compared action counts.
  Cheap to add back through `Episode` if it ever matters.
- **`macro.apply_params` writes 37 module globals** in three modules on every
  turn and never restores them. Safe today because every evaluator plays
  episodes serially and re-applies before acting; a threaded evaluator would
  cross-contaminate candidates. The fix is threading the macro through the
  signatures, a game-file change.
- **Nine `KAG_*` variables are read at import time** by the game layer.
  They are now listed in every ledger line and refused by the gate; removing
  them (making the switches explicit arguments) is a game-file change.
- **Sales revenue is paid twice in the training reward** (dense term plus
  terminal potential) and `--gamma` does not reach the shaping, which reads
  `KAG_GAMMA`. Training is only used to leave random initialisation, so it
  waits for the trainer rewrite.

## Trainer (`kagsym/cli/train.py`, 2,700 lines, 64 flags)

- Five opponent-replacement mechanisms and five curricula coexist; only
  two-seat self-play plus an anchor worker are used. The league
  (`kagsym/liga.py`, `--rivales-liga`, `--liga-dir`, `--linaje`) never
  measured a snapshot (`min_partidas` unreachable) and is ceremony; its
  driver `tools/liga.py` is archived. Removing the league from the trainer is
  a 300-line surgery that waits for a trainer rewrite around
  `kagsym.evaluate`.
- `--eval-cada` shells out to `tools/quick_eval.py` on the first 12 reserved
  seeds: a winner's-curse selector (short evaluations are anticorrelated
  with the truth, -0.46). Keep it as a log line, never as the save criterion.
- The PPO ratio is joint over ~450 dimensions and saturates the clip after
  the first epoch (`ratio saturation=100%` in the diag lines).
  `--factored-ratio` exists and is off. The macro head's learning rate is
  tied to the micro head's by the KL controller. Both are why PPO does not
  move the macro; neither is worth fixing until training is worth running.
- `LADDER_CAPS` changed shape in the last uncommitted work
  (`[None,3,5,8,None]` to `[None,3,5,6,7,8,None]`): `--level 3/4` in old
  logs mean different opponents than today.
- **Worker processes are not owned:** `parallel_env` never closes the child
  end of its pipes and `cerrar()` cannot kill a worker stuck in a public
  agent; a dead trainer can leave orphans. Kill by recorded PID.
- **The `.mejor.json` bar is keyed by `--out` only** (no digest, no commit).
  `--out` now defaults to the run name so two trainers no longer share it.

## Evaluation

- The band with 6 seeds per opponent resolves win-rate differences of about
  0.1. Deciding on a +0.02 needs ~50 seeds per opponent (7,400 episodes,
  ~40 minutes on 11 cores). Budget it when a candidate gets close.
- The RESERVED family (200 seeds) has been used to validate several rounds;
  each reuse adds false-positive risk at t >= 2. `CLEAN` (30000+) is the
  fresh instrument; when it is spent, register a new family.
- `god-s-mode-hacked-stores` does not load (`base_agent` module missing
  from its package) and `shop-router-0909` throws every turn. Both are
  excluded and listed; on Kaggle they may behave differently.

## Repository

- `kagsym/nets/world.py` still computes `aux_rival`, `jepa_pred/proy` and
  `espacial` on every forward; `aux_rival` and `jepa_proy` have never
  received a gradient in any live checkpoint. Removing them changes the
  checkpoint format (`migrate_ckpt` would need a rule) and the model
  fingerprint; cost at inference is small. Do it with the trainer rewrite.
- MLflow readers `tools/compare_runs.py` and `tools/diagnostico.py` keep
  their Spanish names and read the training database only.
- `docs/env_src/` is a copy of the engine sources used to derive
  `SPECIFICATION.md`; not versioned.

## Stage consistency (tools/regret.py, 2026-09-24)

- **Hands were hired where nothing could pay** (3/9/22 in 2/3/5-day
  solitaires). Fixed by valuing a unit-turn with the standing work
  (`market_ops._value_per_action`), +345 $ (t 1.8) at full scale. Two other
  fixes measured and rejected: a demand cap under the macro (-39,212 $,
  t -22) and removing the value floor alone (-3,509 $, t -11.5).
- **The farmer wanders and digs in an idle world** (WEST, NORTH, DIG with
  no crop to come). Free actions, inconsistent with the stage; `DIG_VALUE`
  is a dial and the weeding value ignores whether anything will be planted.
- **A 5-day world is not idle** (carrot 3 days, wheat 4): partida_v5 hires
  13 hands and plants nothing there. The yardstick for k >= 5 is a planner's
  optimum on the exact engine, not yet built (the archived receding-horizon
  search is the candidate).
