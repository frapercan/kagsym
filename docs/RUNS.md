# Run registry

Names that are not ours: **v48** is `v48-fast-routes`, a public agent from
the ladder (its author's version number), the strongest rival we have on
disk. **prod** is our deployed Kaggle policy, the file `runs/partida_v5.pt`
(dials + network; "v5" is its old lineage number).

Every search, dataset, tape and ML policy gets an id here the day it is
made: `P` portfolio searches, `D` datasets, `T` rival tapes, `H` learned
heads, `R` retrieval policies. "live20" is the yardstick: money against
the live v48 on the 20 reserved seeds 7101-7120, seat 0.

## Portfolio searches (`tools/plan_blocks.py`)

| id | file (runs/blocks/) | date | rival in the search | executor | live20 |
|---|---|---|---|---|---|
| P1-passive | portfolio_v1 | 2026-09-25 11:00 | passive | before the expansion ladder | 55,281 |
| P2-vs-v48 | portfolio_v48 | 11:22 | v48 live | same | 55,281 (same winner) |
| P3-demand | portfolio_demand | 12:34 | T-v48 tape, racing | + Plan.crop DEMAND | 54,583 |
| P4-refined | portfolio_v2 | 12:51 | T-v48 tape, racing | + ranked shed, feed chain, pens, feed cash reserve | 60,711 |
| P5-discount-zones | portfolio_v3 | 13:01 | T-v48 tape, racing | + Plan.discount, Plan.zones in the space | 64,660 (se 4,179) |
| P6-harvest-care | P6 | 13:47 | T-v48 tape, racing | + harvest before fertilise, care worth a unit (part 10) | pending |

Bars on the same yardstick: prod 62,309; v48 makes 120-190 k in those games.

## Datasets (`tools/plan_daysearch.py --opponent replay:`)

| id | files | seeds | rival in the rollouts | executor |
|---|---|---|---|---|
| D1 | runs/dataset/duel_lane0-2.jsonl | 7101-7120 | T-v48 | before the expansion ladder |

## Rival tapes (`tools/record_rival.py`)

| id | file | seeds | recorded against |
|---|---|---|---|
| T-v48 | runs/rivals/v48-fast-routes.json | 7101-7120 | P2's best plan |

## Learned and retrieval policies (EXP-008)

| id | file | trained on | live20 |
|---|---|---|---|
| H1 | runs/plan_head_duel.pt | D1, imitation, holdout 7109-7112 | 38,540 |
| R1 | runs/plan_knn_duel.pt | D1, nearest state | 43,115 |
| R2 | runs/plan_knn_own.pt | D1, rival-blind distance | 39,675 |

## Executor states referred to above

- "before the expansion ladder": commit 85fd055 (animal kinds, housing cap, placement valued).
- "expansion ladder": commits up to the Plan.discount one (ranked shed, feed chain and pickup valued by the animals, feed cash reserve, pens and placement worth the animal, housing by kind, zones, discount). EXP-007 part 9.

## Wins panel (2026-09-25 13:15; 20 seeds, seat 0, live public agents)

| rival (ladder position) | P5 wins | P5 money | prod wins | prod money | rival's money vs P5 |
|---|---|---|---|---|---|
| finding-conditiona (143) | 0/20 | 55,167 | 0/20 | 48,027 | 124,726 |
| testkaggriculture-hamburger (154) | 4/20 | 70,394 | 4/20 | 68,901 | 72,837 |
| v53-opening-signature | 0/20 | 60,247 | 0/20 | 43,347 | 147,862 |
| v48-fast-routes (top) | 0/20 | 64,660 | 0/20 | 62,309 | 135,692 |

Same duel (seed 7106), P5 78,347 against v48 153,374: v48 has 3 quadrants,
60 tiles and 15 animals by day 12 with 12 hands (P5: 2 quadrants, 25
tiles, 10 animals, 10 hands, 19 % of its unit-turns idle); its revenue is
strawberry 88 k and milk 76 k against P5's 14 k and 42 k.
