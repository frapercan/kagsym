# Research direction: one learner, three clocks

*A prompt for an agent taking over the research. Everything below is
measured in this repository on 2026-09-24 unless marked as a design choice.
Read docs/PROCEDURE.md, docs/DEBT.md and docs/experiments/ before touching
anything.*

## The question you are answering

Can the model converge, cleanly and measurably, to a near-optimal strategy
of its own game, and can its actions be shown to be good at every rhythm of
the game (turn, day, cycle, episode)? Opponents and the leaderboard are NOT
the question yet. The space is reduced on purpose so that convergence can be
seen in minutes and judged against an oracle:

```
world      24h x 8 days, passive opponent (solitaire), 3,000 $ starting cash,
           the executor values the honest 8-day horizon; carrot (3d), wheat (4d)
           pay, goose almost, nothing else does
seeds      SEARCH / TRAINING for anything that optimises; RESERVED (20+) for
           every number reported; never the same family on both sides
criterion  money on reserved seeds, deterministic policy; and REGRET against
           the rollout oracle (tools/regret.py --scenarios oracle-8 --k 32),
           per day; and the audit of the plays (tools/audit_play.py)
```

## What is already measured in this world (partida_v5 weights; 20 reserved seeds)

```
v5 as it is (30-day prior)                      2,854 $   loses money: hires, plants nothing
CEM on v5's ramp, 15 generations, centre        4,061 $   the best fixed schedule so far
PPO from v5, 150 updates (75 s)                 4,129 $   regret vs local oracle 554
PPO from RANDOM weights, 300 updates (3 min)    5,788 $   regret vs local oracle 468; oracle 6,385
```

The audit of the from-scratch policy on seed 7101: 50 tiles planted (of 100,
one quadrant bought on day 0), 2-3 plantings per tile, 32% tile-day
occupancy, 7 hands every day with 160/517/641 unit-turns pass/move/work,
17,095 $-days of cash idle while tiles were free, 17 seeds never planted,
nothing unsold at the end. That is where the remaining regret lives.

The learning machinery works here: the same trainer that could not move
the policy in the full game (KL controller tying the macro head to the micro
head, a joint ratio over ~450 dims saturating the clip, 11 samples per
update against 80,000 $ of noise) learns from nothing to +2,788 $ of profit
in three minutes when the signal is clean.

## The design to build: expert iteration with an evolutionary outer loop

Three mechanisms exist in the repo and answer three different questions.
They are not alternatives; they run on three clocks of one learner.

1. **Per decision: the exact rollout oracle** (`Policy.macro_candidates`,
   `tools/regret.py`). At each day boundary it tries K candidates of the
   policy's own Gaussian on the exact engine and keeps the best. It is not
   deployable (1 s per turn on Kaggle) but it is the teacher: its choices
   are improved targets, and its gain over the policy is the regret.
2. **Per batch: the network** trained by
   - imitation of the oracle's daily macro choices (supervised, dense,
     low noise; keep the micro map's own targets), and
   - PPO on the environment reward, so it can go beyond what the oracle
     explored around the current mean.
   Convergence is when the regret against the oracle is inside the
   oracle's own noise on every day AND the oracle stops finding gains
   (self-consistency). Distilling a K=16 oracle once failed (the choices
   were noise around the mean); use K >= 32, several seeds per day, and
   the oracle's *gain* as the sample weight.
3. **Per generation: CEM / evolution** on what the gradient cannot reach:
   the executor's dials with thresholds (`runs/<ckpt>.dials.json`), the
   windowed offsets (`kagsym.policy.Offset`), and the trainer's
   hyperparameters. It is also the cheap bar: the best fixed schedule.
   (value of state-dependence = PPO - CEM; value of search = oracle - PPO)

Suggested order: (a) run the from-scratch PPO to convergence in this world
and plot regret per day; (b) add the imitation term from oracle targets and
show it lowers the regret faster; (c) grow the world (8 -> 14 -> 30 days,
still passive) and show the same convergence; (d) only then the shared
market with an opponent. Each step is a preregistration in
docs/experiments/ with its decision rule written before launch.

## What each rhythm must show (the quality of the plays)

- turn: unit-turns in PASS/move against work (`audit_play.py`)
- day: plant + water the same day; hires sized to work; sell into the
  marginal price, not in bursts (the closing failure of the full game:
  strawberry 156 -> 49 $ in two days)
- cycle: plantings per tile, tile-day occupancy, units ready and waiting
- episode: nothing unsold, no seed unplanted, no cash idle while tiles are
  free, land bought when it pays; regret vs oracle per day

## Non-negotiables (each cost a night when broken)

- One policy (`kagsym/policy.py`), one evaluator (`kagsym/evaluate.py`),
  seed families (`kagsym/seeds.py`), the gate `tools/check_submission.py`.
- Every number on reserved seeds, deterministic, paired when comparing.
  Fewer than 20 paired seeds decide nothing.
- Preregister; the decision rule does not change while the run is alive.
  Stopping early is allowed only for a power reading or an instrument
  bug, and it is written down as such.
- A reduced world is valid for convergence and mechanics. It is NOT
  evidence about the full game against an opponent: dial directions
  invert there (measured twice). Do not claim transfer.
- Long runs start from a launcher script with a recorded PID; never
  `pkill -f`. Modules freeze at import; the code on disk is not the run.
- MLflow (`docs/MLFLOW.md`): kind, commit, dirty flag, fingerprints,
  checkpoint digest, seed family, experiment id on every run.

## Deliverables

1. A regret curve (per day, reserved seeds) for PPO-from-scratch to
   convergence in the 8-day solitaire, with the audit before and after.
2. The imitation term and its effect on the curve, same seeds.
3. The same in the 14-day and 30-day solitaires.
4. A written verdict: which rhythms converge, which do not, and what the
   oracle still finds; in docs/experiments/, numbers first.
