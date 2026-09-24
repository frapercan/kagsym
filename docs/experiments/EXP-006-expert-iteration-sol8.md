# EXP-006: expert iteration in the 8-day solitaire

Written before launch, 2026-09-24 23:45. The decision rule does not change
while the run is alive.

## The question, in two halves

1. **Can the executor's current interface express a near-optimal plan?** A
   global per-day oracle (a small CEM at every day boundary: 3 generations x
   32 candidates around the policy's mean with a wide sigma, exact rollouts
   to the end, passive opponent) decides each day's macro. Its money on 20
   reserved seeds is the bar of what this executor can do with perfect daily
   decisions. Known points: policy 5,788 (best PPO), local oracle 6,385,
   hand bound 8,000-10,000.
2. **Can the network learn it?** The teacher's daily choices become
   supervised targets for the macro head (weighted by the teacher's gain
   over the policy's own choice), on top of the policy-gradient term of
   `kagsym/cli/train_solitaire.py --heads-only`. Regret per day against the
   teacher on reserved seeds is the convergence measure.

## Decision rule

- Gate between the halves: if the global oracle's mean on 20 reserved seeds
  is >= 7,000 $, the interface can express the plan and half 2 runs. If it
  is < 6,500 $, the ceiling is the interface: half 2 is NOT run, and the
  next step is the plan interface (docs/RESEARCH_DIRECTION.md, mechanism 2).
  Between the two, half 2 runs and the verdict says so.
- Half 2 converges when the mean policy's regret against the teacher is
  within the teacher's own noise (the spread between K=32 and K=96) on
  every day, with the mean policy's reserved-seed money non-decreasing over
  the last 100 updates. Nothing is deployed from this experiment.

## Baselines (measured, 20 reserved seeds, deterministic)

```
partida_v5                                 2,854
CEM schedule on v5 (baked, exp005_v5_cem8) 4,471
PPO from scratch, best point               5,788   (then oscillates 3,098-5,753)
minimal PG from the CEM point, heads only  4,39-4,46 plateau (300 updates)
local rollout oracle around the PPO point  6,385
```
