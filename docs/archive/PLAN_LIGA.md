# Experimental plan: a league with anti-progressive relaxation

2026-09-21. Every number here is measured, not assumed. Sources at the bottom.

## The design in one paragraph

The game never changes — full 24h x 30d, same board, same economy. What changes
is **how much the opponent is handicapped**, and the handicap is progressively
withdrawn as the policy climbs. The denominator stays fixed across the whole
ladder: the relaxed upper bound of **$195,532**, which relaxes *our* action
budget and is therefore independent of the opponent. That gives a single
comparable scale from bottom to top, which is exactly what a win-rate league
lacks.

## The rungs, measured

Ordered by opponent money **unimpeded** (playing against inaction, so our own
behaviour doesn't move their score through the shared market), as a fraction of
the bound:

| # | opponent | cap | their $ | % of bound | we win (CEM vector) |
|---|---|---|---|---|---|
| — | v16-rc5 | 3 | 2,216 | 1.1% | **EXCLUDED — below inaction** |
| 1 | v48-fast-routes | 3 | 21,665 | 11.1% | 8/8 |
| 2 | v16-rc5 | 5 | 30,220 | 15.5% | 8/8 |
| 3 | v48-fast-routes | 5 | 36,648 | 18.7% | 8/8 |
| 4 | v16-rc5 | 8 | 57,337 | 29.3% | 7/8 |
| 5 | v48-fast-routes | 8 | 77,220 | 39.5% | 4/8 ← **crossover** |
| 6 | v48-fast-routes | 11 | 129,092 | 66.0% | 0/8 |
| 7 | v16-rc5 | 11 | 129,167 | 66.1% | 0/8 |
| 8 | v48-fast-routes | none | 135,895 | 69.5% | 0/8 |
| 9 | v16-rc5 | none | 158,576 | 81.1% | 0/8 |
| 10 | the-2945-farm-96 | none | 180,186 | 92.1% | 0/8 |

Reference points on the same scale: inaction 1.5%, our best fixed vector 28.3%.

**Rung 0 is excluded and that exclusion is the whole point.** `v16 @ cap 3`
scores $2,216, below the $3,000 you get for doing nothing. Beating it requires
inaction. That is precisely the failure that promoted an inert policy through
four leagues before (Trick 8), and this time it was caught by a cheap
pre-flight check rather than by a collapse three days in.

**Two agents survive the handicap, not four.** `shop-router-0909` cannot load.
`the-2945-farm-96` throws `IndexError` at every cap and only runs uncapped — so
capping does *not* universally work, it works for agents whose logic happens to
resize. The pool is therefore thin at the bottom (one or two opponents) and
widest at the top (three), which is backwards from what a curriculum wants.

## The two targets at each rung ("variable targets")

Beating the opponent is **necessary but not sufficient** — rung 0 above proves
why. So each rung carries two, and both must hold:

```
T1  win rate against the rung's opponent  >  50%
T2  our money  >  ref(r)                             <- the ablation that matters
```

where `ref(r)` is what the **best fixed strategy vector** achieves at that same
rung. T2 is the target that varies per rung, and it is the one that measures the
project's actual thesis: anything above `ref(r)` is, by construction, the value
of conditioning on state. A policy that only clears T1 has learned to exploit a
handicap; one that clears T2 has learned something a fixed vector cannot express.

`ref(r)` is known for v48 (62,556 / 52,174 / 49,106 / 39,990 / 37,673 at caps
3/5/8/11/none) and **still has to be measured for v16** — that is step 0.

## Promotion criterion

```
promote from rung r when, over N seeds fixed in advance:
    T1 holds  AND  T2 holds  AND  achieved/bound has stopped climbing
```

The third clause is what distinguishes *exhausted* from *stuck*, and it only
became possible once an upper bound existed. Against a lower bound, "I found
nothing better" is what both look like.

Demotion: if `achieved/bound` at rung r falls below the value it had on
promotion, drop back. Ratchets are how leagues end up celebrating collapse.

## Sample sizes, fixed before looking

Paired sd is ~13,300 $/seed at full scale, so:

```
effect to resolve     seeds
        20%            ~15
        10%            ~55
         5%           ~220
```

T2 margins near a crossover will be small, so **N = 60 per promotion decision**,
and it is chosen now, not after seeing a result. The two false positives in this
project (+18.3%, then +10.3%, both actually zero) came from choosing N after
looking.

## Steps

**Step 0 — finish the calibration.** Measure `ref(r)` for v16 at each cap (8
seeds each, ~2 min). Without it, rungs 2/4/7/9 have no T2. Also re-check whether
`shop-router-0909`'s missing file is recoverable — a third capped opponent would
double the bottom-half pool.

**Step 1 — pre-flight every rung.** For each rung, assert the opponent beats
inaction. Any rung that fails is excluded, automatically, before training. This
is a five-line check and it is the guardrail that Trick 8 lacked.

**Step 2 — instrument before training.** Log per rung, every evaluation:
`achieved/bound`, `achieved/ref(r)`, win rate, and `x_inaction`. An inert policy
reads 1.5% / far below 1 / high / 1.00 — unmistakable at a glance.

**Step 3 — start at rung 1 and climb.** Success criterion written here, now:
*the run is a success if the policy clears T2 at any rung where the fixed vector
does not.* That is a single, falsifiable statement, and it is deliberately not
"beats v48", because beating a handicapped v48 is already done and proves
nothing about the thesis.

**Step 4 — the honest failure branch.** If the policy clears T1 everywhere and
T2 nowhere, the result is: state-dependence buys nothing on this ladder either,
and the neurosymbolic split should move — more of the problem should go to
search, less to learning. That is a publishable answer and it should be reported
as one, not retried until it goes away.

## What this design does NOT do

- **It does not use reduced horizons as rungs.** Measured: transplanting one
  scenario's optimum into another leaves you *below inaction* in 8 of 12 cases.
  Reduced horizons are a laboratory for seeing mechanisms cheaply (20 ms/episode,
  ~150x lower variance), not a curriculum.
- **It does not promote on win rate alone.** See rung 0.
- **It does not treat the cap as a calibrated difficulty dial.** At cap 3, v48
  scores 21,665 and v16 scores 2,216 — 10x apart. Rung difficulty is measured.

## Sources

`runs/ligas/pool.py` (the rung table), `runs/ligas/vs_topes.py` (`ref(r)` for
v48), `runs/ligas/cota_superior.py` (the bound), `runs/ligas/calibra_rejilla.py`
(achievable ceilings), `runs/ligas/dos_sillas.py` (full-scale reference points).
