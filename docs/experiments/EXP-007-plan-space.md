# EXP-007: an explicit plan space, searched on the exact engine

2026-09-25. World: 24h x 8 days, passive opponent, 3,000 $ cash, deterministic.
Seeds: RESERVED 7101-7105 for the search, 7106-7120 unseen for the report.

## Why

EXP-005/006 showed the dial interface (67 executor constants) is a coarse
model of the problem: PPO reaches 5,788 $ and oscillates; a minimal policy
gradient plateaus at 4,408 $; the per-day global oracle over the dials
reaches 6,537 $ (EXP-006 half 1). None of the dials says "25 tiles of
carrot, 7 hands on harvest day, sell now". A `Plan` does (`kagsym/plan.py`):
crop, tiles, hands, land, animals, selling -- each a number or a per-day
schedule -- executed literally by the symbolic layer (routes, assignment and
legality unchanged). `tools/plan_search.py` searches it exhaustively: a grid
of constant plans, then coordinate descent over the per-day schedules from
three starting points.

## Results (mean money, reserved seeds)

```
                                                  5 search seeds   15 unseen   20 seeds
best constant plan (carrot 25 tiles, 6 hands)          5,383
+ hands per day                                        5,610
+ selling per day                                      5,759
+ executor leaks fixed (below)                         5,776
+ crop per day  (coordinate descent, 3 starts)         5,947          5,982      5,973 +- 26
PPO from scratch, best point (the bar)                 5,744          5,803      5,788
paired plan - PPO                                                                 +185, se 16, 19/20
global per-day oracle over the dials (EXP-006)                                    6,537
```

The best plan, in words: 20 carrot tiles on day 0 with 2 hands, wheat on
day 1 with none, 7 hands on day 3 to harvest and replant, a second wheat
sowing on day 4 sold with a half-day horizon, and no land (in 8 days no
land plan pays: the best of 200 lost money).

## Three executor leaks the trace exposed (fixed, tests in kagsym/tests/test_plan.py)

1. **Seed for a cycle that cannot finish.** `target_crop` accepted a crop
   whose cycle ended ON the last day; `plantable` (strict) refused to plant
   it. 50 wheat seeds ($500) idle from day 4. Plan mode now asks `plantable`.
2. **The last day was lost entirely.** `MIN_HAND_DAYS` refused every hire
   with one day left, so the farmer alone faced 24 tiles to harvest and sell.
   Plan mode hires what the plan says on every day.
3. **DROP valued only by the price it saves.** The nightly flush discards
   what does not fit in the 100-unit shed (116 carried, 100 kept) and on the
   last day what is still carried is worth nothing (the score is cash).
   `_shed_task` now values DROP by both margins. It is not plan-only in the
   code, but it cannot reach the deployed policy: the network's value map
   REPLACES the heuristic value on every tile, the shed access included.
   Measured: v5 vs v48, 100 reserved seeds x 2 seats, old code (a git
   worktree at HEAD) against new, 200 of 200 episodes identical to the
   dollar. The leak is therefore the network's to price in dial mode, and
   fixed for any agent that plays without a map.

Fixing the three moved the wheat-then-carrot plan on seed 7101 from 2,911
to 4,252 $ and the best plan from 5,759 to 5,776.

## Verdict

The plan space is a finer model of the problem than the dials: a search of
under six minutes on five seeds found a plan that beats the best learned
dial policy on fifteen seeds it never saw, by 3 %, with 19 of 20 seeds
better. It is still below the per-day dial oracle (6,537 $): coordinate
descent stops at a local optimum, and the plan does not yet say when to
harvest (green or ripe) or how to stagger sowings inside a day.

## Next (EXP-008, to preregister)

Expert iteration on the plan: (1) widen the search (CEM over schedules, or
day-by-day search with rollouts as the oracle does) until it reaches or
passes 6,537; (2) a network whose heads emit the day's plan (discrete tiles,
hands, crop, land; continuous selling) trained by imitation of the searched
plan and regression of its money; (3) the network's plan as the search's
starting point, so each iteration needs fewer rollouts. Gate: the network's
regret against the searched plan, per day, on reserved seeds.

## Part 2 (2026-09-25, early morning): the ladder from the minimum horizon

Rule adopted: never straight to the full game. Start where one decision
matters and climb: 4 days (one carrot cycle) -> 5 (wheat) -> 6 -> 8 -> 11
(tomato) -> 14 -> 30, always 24 hours, closing each rung before the next.

### Rung 1: 4 days, 25 carrot tiles

The strategy is trivial (plant everything on day 0, sell on day 3) and the
exhaustive grid confirmed it (1,125 constant plans: tiles 25, hands 3,
selling irrelevant). The money was not: 3,348 $ against a hand bound of
~4,600 (75 carrots sold in one day at the day-0 price curve gross 2,130).
The loss was tactical, on the last day:

```
hands   final   harvested   sold   carried at close   left on tiles
  3     3,348      51        27          24                17
  6     3,194      75        24          51                 0
  8     2,876      75        18          57                 0
```

More hands harvested everything and sold LESS. Four mechanisms, each exact
in the engine, each now in `kagsym/symbolic/tasks.py`:

1. **One DROP column for the whole crew.** The shed has one access tile per
   unlocked quadrant; the Hungarian assignment gives each column to one
   unit, so one unit per turn went to the shed and the rest passed with
   full hands. Now every carrying unit has its own column, worth what IT
   carries (the column carried the crew's maximum, so the unit with 3
   units looked like the one with 12).
2. **Three of the four shed-access tiles were never offered** because they
   are LOCKED; the engine resolves DROP before the LOCKED guard and lets
   units stand there. Offered now, but only to agents without a value map:
   with a map the heuristic dollar figure dominated the matrix (v5 vs
   passive, seed 7102: 89,117 -> 74,176 alone, 43,754 with mechanism 1).
3. **The last-day deadline.** A harvest reaches the score only through walk
   + HARVEST + walk + DROP + a SELL order the turn after; what cannot make
   the trip is worth 0, and so is a watering whose harvest cannot. Nothing
   is carried home now (test).
4. **`load`, a tactical plan field**: how many units a hand carries before
   a shed trip (the deadline overrides it). Without it a hand made a trip
   for every 3 units and 73 % of its turns were moves.

```
                                         4 days, 5 reserved seeds
one column, no deadline (hands 3)              3,348
per-unit columns + 4 tiles + deadline (8)      3,661
+ load 12 (8 hands)                            4,379
+ per-day schedule (hands 3,0,3,8; load 9)     4,533     bound ~4,600
```

Mechanisms 1 and 3 sit inside the assignment, so at first they reached the
deployed policy too. Measured paired, v5 vs v48, 100 reserved seeds x 2
seats, HEAD against the working tree: -5,356 $ (se 1,115, t -4.8, worse on
127 of 200) -- while the same code gave +4,059 on one seed against a
passive opponent. The network's map was trained with one DROP column and
no deadline; changing the semantics under a frozen policy is not a
measurement of the mechanism (docs/PROCEDURE.md: a widened space is only
judged by retraining). All four mechanisms are therefore active only for
agents WITHOUT a value map: the plan executor and the heuristic agent. The
deployed policy is byte-identical (200 of 200 paired episodes equal,
rechecked after the gate). Teaching them to the network is retraining
work, recorded in docs/DEBT.md.

## Part 3: the ladder one day at a time (2026-09-25, 00:26-)

`tools/plan_search.py` per rung, 5 reserved seeds, no opponent, pure search
(grid of constant plans, then coordinate descent over the per-day schedule
of hands, load, water_last, tiles, crop, selling). A day's crop may be a
50/50 mix (from rung 8 on in this pass; rungs 3-7 re-run with mixes).

```
days   money   plan                                                   the decision that appears
 1     3,000   nothing                                                do not spend (the idle floor holds)
 2     3,000   nothing
 3     3,802   20-25 carrot (a 50/50 mix on day 0), 6 hands last day  the green harvest: carrot yields 2-3 at age 2
 4     4,533   25 carrot; hands 3,0,3,8; load 9                       the full cycle; last-day logistics
 5     4,893   25 wheat; hands 3,0,3,2,8; load 15                     wheat replaces carrot
 6     5,478   land + 50 wheat; hands 4-7                             land pays with one cycle of 50 wheat
 7     6,360   land + 50 wheat/carrot alternating; hands 2-8          alternating crops across days
 8     7,443   land + 50 mixed (carrot/wheat 50/50) then carrot        the mix inside the day
```

Rung 8 on 20 reserved seeds: 7,379 +- 94 (15 unseen: 7,357), against PPO
5,788 and the per-day dial oracle 6,537: +27 % and +13 %. The old bar test
is re-pinned to this plan (7,388 on seed 7101).

`tools/plan_audit.py` says how a plan is executed. Rungs 4/6/8: units sold
72 of 75, 187 of ~206 (16 discarded by shed overflow), 313 of 317; nothing
carried home; the crew spends 51-58 % of its turns moving and 5-13 %
passing. That, and the shed overflow at rung 6, is what remains between
the plan and its bound: tactical, not strategic.

Coordinate descent is already local at rung 3 (3,752 found, 3,790 by hand
with 8 hands); the schedule search is to be strengthened before the high
rungs (CEM over schedules, or day-by-day search with rollouts).

### Rungs 9-11 and the limit of local search (01:07-01:21)

```
 9     8,298   land, alternating carrot/wheat, a day without sowing
10     9,334   land, mixes on days 0 and 6, wheat
11    26,463   20 MELON tiles, no land, 2 hands then 6 on the last day       melon: yields on day 10 at 250 $/unit
```

Melon is the rung-11 decision, and it is three coordinated changes away
from the 10-day plan (crop, 20 tiles, no land); each intermediate step is
worse (9,449 -> 6,170 -> 3,175 -> 26,463). A day-by-day search with exact
rollouts (`tools/plan_daysearch.py`: every single-field and twelve random
two-field variations of the day's plan, played to the end) refines a plan
in 20 s per seed at 11 days but cannot leave its basin: from the 10-day
plan it returns 9,573. The constant-plan grid finds melon (24,990). The
search is therefore staged: grid (global over constant plans) -> day-by-day
refinement with the state in hand, which records (state, day plan, money)
for the plan head. The ladder's coordinate descent was stopped after rung
11 (12-15 min per rung and rising).
