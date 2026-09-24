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

## Result: stopped at generation 7 of 40 (deviation from rule 1, 2026-09-24 13:15)

```
generation   base      centre - base   population mean - base
    0      -80,409         +689             -18,595
    1      -15,581       -2,918             -14,432
    2      -37,991         -965             -12,090
    3      -52,323          -31              -8,185
    4      -81,724       +5,608              -7,411
    5      -67,391      -12,636             -12,676
    6      -48,863      -12,221              -9,846
mean centre - base   -3,210 +- 2,574   (t -1.25)
```

Why it was stopped, against the preregistration's rule 1: a power reading,
not a result reading. The base varies by 22,000 $ between generations from
the opponent sample alone (6 opponents x 4 seeds), the population mean sits
12,000 $ below the base in every generation (the initial sigma of 0.6 is too
wide for a checkpoint already at a validated optimum), and after 40
generations the standard error on centre - base would be about 1,000 $ of
margin. The decision rule needs +0.02 of band win rate; with a 60,000-80,000 $
gap per board, a margin gain of a few thousand dollars cannot buy it. The run
also predates the minesweeper fixes (candidates with failed episodes were
averaged, artefacts had no identity), so it would have needed re-validation
in any case. Verdict: null at the power available; the offset space cannot
buy wins at this scale. Next: the executor (tiles under crop, the opening),
via the reduced-calendar self-play pipeline.

Artefacts: `runs/search/exp001_margin_history.json` (imported to MLflow as
`kagsym/search / exp001_margin (stopped)`).
