# EXP-003: the opening offset, searched on the full game

Written before launch, 2026-09-24. The decision rule does not change while
the run is alive.

## Why, after EXP-002

EXP-002 searched the opening (day < 8) in the 8-day universe with the
position value at day 8 as objective and passed its rung gate (+4,801 $,
t 13.3). At full scale it LOST: -13,297 +- 1,445 $ of final money against
v48 on 200 clean seeds (t -9.2). The trace shows why the proxy failed: the
searched opening is ahead until day 20 (19,700 $ vs 17,056 $) and loses
9,000 $ in the last nine days; the unchanged closing policy converts a
larger farm (11 animals, 12 hands, 22-35 plants) worse than a small one.
The day-8 position at marginal prices predicts day 20, not day 30.

A reduced universe costs 43 s per generation; the full game about 90 s.
The proxy saved a factor of two and did not transfer. So the opening is
searched again with the true objective.

## Question

Does an opening offset (day < 8) searched on the FULL game with the margin
against v48 and the 2945 expert as objective transfer to the criterion?

## Setup

- Start: `runs/partida_v5.pt` (its ramp as generation-0 centre); the offset
  applies while day < 8; dials from `runs/partida_v5.pt.dials.json`.
- Objective `margin` at full scale (ours minus theirs at day 30), fixed
  opponents v48 and 2945, both seats, 4 common SEARCH seeds per generation
  (32 x 2 x 4 x 2 = 512 full episodes per generation).
- `tools/search.py runs/partida_v5.pt --out runs/search/exp003_opening_full
  --ramp --objective margin --until-day 8
  --opponents v48-fast-routes,the-2945-farm-96-vs-the-top-10-public-bots
  --seeds 4 --pop 32 --elite 8 --gens 30 --experiment EXP-003`

## Decision rule

1. Not stopped before generation 30 unless the process dies, or a power
   reading at generation 10 shows centre - base cannot resolve 3,000 $ of
   margin by generation 30.
2. Validate: `tools/validate_offset.py runs/search/exp003_opening_full
   runs/partida_v5.pt --family clean --n 200 --band-n 6 --bake runs/partida_v6.pt`.
   Bake only if money t >= 2 on the 200 clean seeds AND the band win-rate
   difference is >= 0.
3. If money passes and the band does not move (within +-0.02), the result
   is "money, not wins" again, and the offset space is closed for the
   opening as well: the next step is the closing logic of the executor
   (selling and feeding with a large farm), measured on the day 20 to 30
   window.
