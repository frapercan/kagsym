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

## Part 4: the protocol climb, 1 to 30 (2026-09-25, 02:42-04:58, commit 3ba93b9)

`tools/ladder.py`, code frozen; unseen seeds 7106-7110 (the search saw
7101-7105). "prev+1" is the previous rung's schedule padded by one day on
the same unseen seeds: what the extra day buys.

```
days  unseen   prev+1   what changed              days  unseen   prev+1   what changed
  1    3,000       -    nothing to do               16   32,985   32,197   tiles, hands, selling
  2    3,000    3,000   nothing to do               17   32,716   33,097   crop, tiles, hands
  3    3,914    3,000   green carrot                18   32,472   32,478   hands
  4    4,557    4,391   hands (last day)            19   34,877   34,265   crop, hands, selling
  5    4,696    4,578   tiles, hands                20   35,795   35,135   crop, hands
  6    5,410    4,371   land + wheat                21   35,567   36,439   tiles, hands, load
  7    6,014    5,406   crop, hands                 22   36,867   36,055   tiles, hands
  8    6,887    6,756   hands                       23   37,343   34,988   crop, tiles, hands
  9    6,937    6,886   tiles, hands                24   37,661   36,717   crop, tiles, hands, selling
 10    8,786    6,495   mixes, land                 25   39,093   38,504   crop, hands, selling
 11   26,284    9,402   MELON                       26   41,589   39,855   crop, tiles, hands, selling
 12   29,925   27,222   tiles, hands                27   43,891   41,231   crop, tiles, hands
 13   30,077   29,921   tiles, hands                28   48,619   44,423   crop, tiles, hands, water, selling
 14   30,603   30,100   crop, tiles, hands          29   49,814   36,747   crop, hands, selling
 15   31,762   30,851   crop, tiles, hands          30   50,861   50,254   hands, selling
```

Times: 1-10 s per rung below 10 days, 60-120 s at 10-16, 330-735 s at
17-30. The 30-day plan never buys land, sows melon twice on 25 tiles and
strawberry in between, sells 300 units, carries nothing home.

### The verdict at 30 days

50,861 $ against a passive opponent. The deployed policy v5, dial
interface and network, makes 104,307 on seed 7106 in the same world, and
sells seven products: 218 milk, 185 fertiliser, 102 melon, 92 strawberry,
56 eggs, 39 wool, 17 wheat, with 14 animals bought. The plan space as
searched is missing the levers that matter at 30 days, measured by hand on
the unseen seeds:

```
searched 30-day plan                                         50,861
5-way mix on 25 tiles, 3 hands                               25,159
5-way mix, land on day 5, 50 tiles, 6 hands                  35,224
5-way mix, 25 tiles, 4 hands, 3 animals                      45,364
5-way mix, 25 tiles, 7 hands, 4 animals                      56,380   above the searched optimum, by hand
hand-crafted melon then 3 quadrants of 100 melons            16,235   the crew cannot water 100 tiles
```

Animals were a plan field the search never moved; portfolios beyond a
pair were not offered. Both are now in the search (`animals` in
{0,1,2,3,4,6,8}; three-way and all-crop mixes) and in the plan head, and
the ladder is climbed again from 1 under the protocol. Fertiliser is the
next lever v5 uses and the plan does not express.

### Second climb, stopped at rung 20 (2026-09-25 10:04-10:30)

Animals and portfolios in the day-by-day search only: identical to the
first climb to the dollar below 11 days, +1,100 at 11 (animals taken),
and 34,491 at 20 against the first climb's 35,795. The refinement cannot
reach a portfolio-with-animals plan from a monoculture in one or two
moves; the global stage (the constant-plan grid) had neither lever. The
grid now spans crops (singles, pairs, the all-crop mix), tiles, hands,
land and animals (0-6). Third climb from rung 1.

Fertiliser is not a separate lever: v5's 185 units sold are the
by-product of its 14 animals (each animal yields fertiliser daily), and
its 182 BUY_PRODUCT orders are wheat to feed them.

## Part 5: the blocks (2026-09-25, midday)

The 30-day game is not a prefix of its rungs: v5's strategy makes 2,793 at
10 days, 3,120 at 15, 24,675 at 20 and 95,656 at 30. It is livestock, which
pays late. Measured in isolation (nothing sown, plan-mode executor with the
kind forced), from 3,000 $:

```
block                 capital    14 days    30 days   idle
4 geese, 1 hand         1,200     8,658     17,434    41 %
8 geese, 2 hands        2,400    11,448     25,368    13 %
12 geese, 3 hands       3,600    12,028     25,212     8 %    eggs saturate
4 cows, 1 hand          1,600    13,174     34,323    51 %
7 cows, 2 hands         2,800    14,062     34,540    35 %    milk saturates
4 sheep, 1 hand         2,000    15,064     28,236    49 %
7 sheep, 2 hands        3,500    23,092     59,794    36 %
```

Seven sheep and two hands, nothing sown, beat the whole 30-day crop search
(50,861). A goose (300 $) returns ~150 $ a day in eggs and fertiliser (the
engine gives one fertiliser per animal per day when fed and cared for) and
pays back in 2-3 days; the executor's `animal_net_value` books manure as a
watering credit and refuses every animal below 14 days, which is why the
ladder never saw livestock. Each product saturates its own market, so
blocks are sub-additive within a product and additive across products:
the 30-day game is a PORTFOLIO of blocks, scheduled by cash. The plan now
says the kinds (`animals={kind: count}` per day) and the executor obeys.
