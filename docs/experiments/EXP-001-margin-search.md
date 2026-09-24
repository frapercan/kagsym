# EXP-001: CEM on the macro ramp with the margin objective

Written before launch, 2026-09-24. The decision rule does not change while
the run is alive.

## Question

Does searching the macro offset against a sample of the public band, with
margin (ours minus theirs) as objective, move the criterion (band win rate),
which the money objective against v48 did not?

## Setup

- Start: `runs/partida_v5.pt`, its stored ramp as generation-0 centre.
- Live dials: `runs/live_dials.json` from `tools/live_dials.py` on v5 (3 seeds).
- Search: `tools/search.py --ramp --objective margin --band-sample 6 --seeds 4
  --pop 32 --elite 8 --gens 40`, seeds from the SEARCH family, rotating.
  Per generation: 32 candidates x 6 opponents x 4 seeds x 2 seats.
- Gate before launch: `tools/check_submission.py runs/partida_v5.pt --n 6` = 6/6.

## Decision rule

1. The run is not stopped before generation 40 unless the process dies.
2. Intermediate reading: CENTRE minus BASE per generation. Informative only.
3. At the end, `tools/validate_offset.py <mu> runs/partida_v5.pt --family clean
   --n 200 --band-n 6`: money paired on CLEAN seeds against v48, and the band
   paired on RESERVED seeds against v5.
4. Bake only if the band win-rate difference is >= +0.02 (about two
   opponents of 74) AND the money instrument is not negative at t <= -2.
   The band is the criterion; money is a sanity check, not a requirement.
5. If the band difference is within +-0.02 the result is a null: the
   offset space cannot buy wins at this scale, and the next step is the
   executor (tiles under crop, the opening), not another search.

## Baseline (measured, same instrument)

```
partida_v5 with ramp    win 0.064   beaten 5 of 74   money 51,304 +- 614
```

## Result

(filled in after the run)
