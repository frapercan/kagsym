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
