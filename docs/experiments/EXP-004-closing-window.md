# EXP-004: the closing, searched as a window from day 18 on the full game

Written before launch, 2026-09-24 15:20. The decision rule does not change
while the run is alive.

## Why

EXP-002's opening lifts the day-8 position (+4,801 $, t 13.3) and is ahead
at day 20 (19,700 $ vs 17,056 $ on seed 7101), then loses 9,000 $ in the
last nine days: the larger strawberry harvest is dumped and the price falls
from 156 to 49 in two days. With the correct windowed semantics the full-game
loss is -9,718 +- 1,355 $ (t -7.2). The closing logic does not scale with the
harvest. This is the second rung of the ladder by phases: fix the closing
given the opening.

## Setup

- Base: `runs/partida_v5_open8.pt` = partida_v5's weights and ramp with the
  EXP-002 opening as a schedule window (day < 8). An EXPERIMENT ARTEFACT,
  not a validated checkpoint; it exists so the closing search can start from
  the opening it must fix.
- Candidate: a constant offset on the live dials applied from day 18
  (`--from-day 18`), replacing v5's ramp inside that window only.
- Objective `margin` on the full game against v48 and the 2945 expert, both
  seats, 4 common SEARCH seeds per generation (512 full episodes each).
- `tools/search.py runs/partida_v5_open8.pt --out runs/search/exp004_closing18
  --objective margin --from-day 18
  --opponents v48-fast-routes,the-2945-farm-96-vs-the-top-10-public-bots
  --seeds 4 --pop 32 --elite 8 --gens 30 --sigma 0.4 --experiment EXP-004`
  (sigma 0.4, not 0.6: the base is already a tuned point and the searches
  from tuned points spent their first generations recovering it).

## Decision rule

1. Not stopped before generation 30 unless the process dies, or a power
   reading at generation 10 shows centre - base cannot resolve 3,000 $ of
   margin by generation 30.
2. Validate the closing window against the base `partida_v5_open8.pt` at
   full scale (`tools/validate_offset.py ... --family clean --n 200 --band-n 6`),
   then the RESULTING two-window schedule against the real `partida_v5` on
   the same clean seeds and the band (`tools/band.py --against`). The
   number that decides is the second: does opening + closing beat v5 on
   money (t >= 2) without lowering the band win rate?
3. If the closing recovers the opening's loss but not more, the phases
   cancel and the offset space is closed for both; the next step is the
   executor's selling logic itself (sell steadily, not in bursts).

## Baseline (same instrument)

```
partida_v5              band win 0.064, beaten 5 of 74; money vs v48 (200 clean) reference
partida_v5 + opening    money vs v48 (200 clean, paired vs v5): -9,718 +- 1,355 (t -7.2)
```

## Result (2026-09-24 17:05)

Search: 30 generations of ~120 s; centre - base +1,192 +- 165 on the
search seeds (t 7.2), last 15 generations +1,399; sigma 0.35 -> 0.38 (no
contraction: the elite never agreed on a direction).

Validation (a), the closing window against its base (v5 + opening), 200
CLEAN seeds against v48, paired: **+235 +- 161 $ (t 1.46), better on 50%
of boards.** Gate refused; nothing baked; step (b) not reached.

Verdict (rule 3): the closing window recovers a fraction of the opening's
-9,718 $, not the loss and not more. The offset space of the macro is closed
around partida_v5 for the opening (EXP-002/003), the closing (EXP-004) and
the whole game (EXP-001). What the dials cannot express is how the harvest
is converted into cash: the executor sells a large harvest in bursts and
crashes its own price. That is executor logic (`market_ops.sell_orders`),
and it is the next experiment: sell steadily against the marginal price
with the opponent's supply forecast, measured on the day 18-30 window of
the full game, paired, with the opening of EXP-002 as the stress case.
