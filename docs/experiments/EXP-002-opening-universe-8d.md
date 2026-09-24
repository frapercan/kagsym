# EXP-002: the opening, searched in the 8-day universe

Written before launch, 2026-09-24 13:35. The decision rule does not change
while the run is alive.

## Question

The first rung of the ladder of universes: playing the first 8 days of a
30-day game, can a search over an OPENING offset (applied while day < 8)
close the gap between our position at day 8 and v48's, and does the gain
transfer to the full game and the criterion (band win rate)?

## Measured baseline (same instrument, partida_v5, agent horizon 30 days)

```
day 8, position value with the 30-day horizon (cash + shed + what pays in time)
                          ours       theirs     ratio
vs v48        seed 7101  11,106     24,692      0.45
vs v48        seed 7102  12,535     28,231      0.44
vs 2945       seed 7101  10,540     33,262      0.32
```

## Setup

- Universe: 24h x 8d played, our policy values a 30-day game
  (`--days 8 --agent-horizon 30`); the offset applies while day < 8.
- Start: `runs/partida_v5.pt`, its stored ramp as generation-0 centre; live
  dials from `runs/partida_v5.pt.dials.json`.
- Objective `value`: our position value minus the opponent's at the cut.
- Opponents fixed: `v48-fast-routes` and `the-2945-farm-96-vs-the-top-10-public-bots`,
  both seats; 8 common seeds per generation from the SEARCH family, rotating.
  Per generation: 32 candidates x 2 opponents x 8 seeds x 2 seats = 1,024
  episodes of 192 turns (about a quarter of a full episode each).
- `tools/search.py runs/partida_v5.pt --out runs/search/exp002_opening8 --ramp
  --objective value --days 8 --agent-horizon 30 --until-day 8
  --opponents v48-fast-routes,the-2945-farm-96-vs-the-top-10-public-bots
  --seeds 8 --pop 32 --elite 8 --gens 30 --experiment EXP-002`

## Decision rule

1. Not stopped before generation 30 unless the process dies, or unless a
   power reading at generation 10 shows the standard error of centre - base
   cannot resolve 2,000 $ of position value by generation 30.
2. Intermediate reading: CENTRE minus BASE (paired). Informative only.
3. Rung gate (this universe): the centre's position value against v48 on 30
   RESERVED seeds, paired against the base, must gain >= +3,000 $ at t >= 3
   (a quarter of the gap). If it does not, the offset space cannot fix the
   opening and the next step is the executor's opening logic, not a search.
4. Transfer: `tools/validate_offset.py runs/search/exp002_opening8
   runs/partida_v5.pt --family clean --n 200 --band-n 6` at FULL scale.
   Bake only if the band win-rate difference is >= 0 and money t >= 2.
5. A gain at day 8 that does not survive the full game is recorded as
   "opening improved, game not": the next rung (14 days) then searches from
   the baked opening, not from v5.

## Result, rung gate (2026-09-24 14:10)

Search: 30 generations of 43 s; centre - base +3,580 +- 368 over the run
(t +9.7), +4,985 +- 314 over the last 15; sigma 0.61 -> 0.30; no candidate
disqualified.

Rung gate on 30 RESERVED seeds, both opponents, both seats, paired against
the base (partida_v5 with its own ramp), position value at day 8 with the
30-day horizon:

```
                 our value            theirs     ratio          paired relative gain
vs v48        13,292 -> 19,255       30,605   0.43 -> 0.63    +4,244 +- 515   t +8.2   better 87%
vs 2945       13,732 -> 19,257       33,854   0.41 -> 0.57    +5,357 +- 503   t +10.7  better 90%
both (n 120)  13,512 -> 19,256       32,230   0.42 -> 0.60    +4,801 +- 362   t +13.3  better 88%
```

Rule 3 (>= +3,000 at t >= 3) passes. The opening offset closes about a
third of the day-8 gap. Rule 4 (transfer at full scale) runs next:
`runs/search/exp002_validate.log`.

## Result, transfer (2026-09-24 14:25): opening improved, game not

Full game, 200 CLEAN seeds against v48, paired against the base:
**-13,297 +- 1,445 $ (t -9.20), better on 25% of boards.** Band result
appended below when the validation finishes; the bake is refused regardless.

Why the proxy failed, from the trace on seed 7101 (base -> searched opening):

```
day   money base / opening    plants     animals    hands    strawberry price
 10      3,042 /    146        7 / 22      7 / 10    9 /  9
 20     17,056 / 19,700       34 / 31      8 / 11   11 / 12     185 / 156
 22     32,444 / 31,718       21 / 27      8 / 11   10 / 12     159 /  49
 24     39,426 / 32,519       17 / 11      8 / 11   10 / 13     135 /  39
 29     48,387 / 39,327
```

The searched opening is ahead until day 20 and then dumps a larger
strawberry harvest into the market at once: the price falls from 156 to 49
in two days (the base keeps it above 135). The closing logic does not scale
with the harvest it is given. The day-8 position at marginal prices predicts
day 20, not day 30; a rung yardstick must be predictive of the final, and
this one is not. Next: EXP-003 searches the opening on the full game (the
proxy saved a factor of two and did not transfer), and the closing becomes
its own rung.
