# EXP-008: a plan head that amortises the search

Written before launch, 2026-09-25 01:35. The decision rule does not change
while the run is alive.

## The question

The staged search (constant-plan grid, then day-by-day refinement with
exact rollouts; `tools/plan_search.py`, `tools/plan_daysearch.py`) finds
plans that beat every learned policy in the reduced worlds (EXP-007). It is
offline: minutes per game against 1 s per turn on Kaggle. Can a small
network reproduce the searched plan of the day from the state at the day
boundary, so that the deployed agent is the symbolic executor plus that
head, with no rollouts at play time?

## Data

Every day-boundary decision the day search records
(`runs/ladder/daysearch_<d>d.jsonl`): the state (day, cash, hands,
quadrants, tiles by crop and age, seeds, shed, prices) and the day's plan
(hands, load, water_last, tiles, crop or mix, selling, land) with the
rollout's money. Horizons 3-30 days on 24 hours, 5 SEARCH-side seeds each
(RESERVED 7101-7105 are the search's; the report uses 7106+ only). The
horizon enters the state as days left.

## Model

Inputs: the state features above, normalised. Outputs: one softmax head per
plan field over that field's value list (hands 9, load 6, water_last 2,
tiles 9, crop 5 singles + 10 mixes, selling 5, land 3). Loss: the sum of
cross-entropies against the searched value. No policy gradient, no critic.
A two-layer MLP of 256 units; Adam 1e-3; early stopping on held-out
horizons.

## Measures and decision rule

1. **Agreement** per field on held-out days (horizons not in training and
   unseen seeds).
2. **Regret**: the network's plan executed by the same executor on the
   reserved seeds 7106-7120 at each held-out horizon, against the searched
   plan's money on the same seeds. Reported per horizon.
3. Baselines: the searched plan (0 regret by construction), the previous
   rung's plan padded (what "no learning" gives), and the best constant
   plan of the horizon.

The head converges when the regret on held-out horizons is below 5 % of
the searched money at every rung up to the largest trained, and it does
not exceed the "previous rung padded" baseline anywhere. If it fails on
the melon rungs (11+) but not below, the state lacks what decides the
crop (days left against the crop's first yield): fix the features, not
the model. Nothing is deployed from this experiment.
