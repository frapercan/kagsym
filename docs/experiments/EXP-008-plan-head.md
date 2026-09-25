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

## Result of the first climb (2026-09-25 04:59; dataset of the 1-30 protocol run, 2,325 records: 30 horizons x 5 seeds x every day)

Training agreement after 2,000 epochs: hands 0.94, the rest 0.98-1.00.
Held-out horizons (6, 9, 12, 16, 24): hands 0.58, tiles 0.85, crop 0.94,
selling 0.97, load/water/land 0.99-1.00. Regret of the head's plan against
the searched plan on unseen seeds 7106-7110:

```
held-out   6 d  +8.7 %     9 d  +0.4 %    12 d  +12.1 %    16 d  +1.8 %    24 d  +2.5 %
train      0 to +2 % on most rungs; the head BEATS the search at 18-22 and 25 days
           (-2.5 to -7.6 %) and loses 10-17 % at 26 and 28
```

Per the rule (under 5 % on every held-out rung) the head has not
converged: it fails at 6 (it plays the 5-day plan) and at 12 (it plays
the 11-day plan, missing the day-12 refinement). That the head beats the
search on several trained rungs says the search is local there, not that
the head is good. The first climb's dataset lacked animals and
portfolios; the second climb's will not. Not deployed.

## The deployable policy (2026-09-25, 12:00)

`kagsym/plan_head/policy.py`: the symbolic executor driven by the head,
with the lifecycle of `kagsym.policy.Policy`; `Policy.from_checkpoint`
returns it for a checkpoint of kind `plan_head`, so the Kaggle wrapper
does not change. The state the head reads (`plan_head/state.py`) carries
the farm, the shops, the market inventory and the rival's visible farm.
Cost at play time: 6 ms a turn on average, 49 ms on the worst turn,
against a budget of 1,000 ms. The plan is set for the duration of one
`act` call only, so a paired evaluation with another policy in the same
process never sees it (test).

The second dataset is the duel one: the day search with the recorded
v48 inside its rollouts on seeds 7101-7120 (`runs/dataset/`), holdout
seeds 7116-7120, regret measured against the recorded rival and then
live.

## Result on the duel dataset (2026-09-25 12:26; 12 seeds, 360 records, holdout 7109-7112)

Training agreement 0.98-1.00 on every field; held-out: hands 0.47, tiles
0.73, crop 0.36. Regret against the searched schedule, with the recorded
rival: +30 to +74 % on EVERY seed, the training ones included (7108:
searched 93,175, head 24,010). Compounding error: one wrong day leads to
a state the head never saw. Live validation against v48 on 20 seeds
(12 searched + 8 never searched):

```
                                   mean money   wins vs v48
searched state-aware plan (12)        80,723       0/12
v5                                    62,309       0/20
fixed portfolio (consolidated)        55,281       0/20
plan head (this run)                  43,543       0/20
```

Verdict: the search direction holds (the state-aware plan beats v5 in
the live duel by +18 k, +30 %), the imitation does not amortise it with
360 records, and nobody wins a game against v48, which makes 150-190 k in
these duels. Per the rule, not deployed. Next: amortisation by retrieval
(the nearest recorded state's plan, no training) and more seeds; DAgger
if retrieval also fails.

## Why neither amortisation transfers (2026-09-25, 12:45)

Two deployable policies were validated live against v48 on 20 seeds after
a bug was fixed (land taken from day 0 alone): the trained head, 38,540;
retrieval of the nearest recorded state's plan, 43,115 (39,675 with a
distance blind to the rival's features). Both far below the searched
schedules played as fixed plans on the same seeds (80,723). Yet the
retrieval policy replays its own seed to the cent against the recorded
rival (test).

The mechanism, traced on seed 7108 live: the shop draw depends on our
own actions (part 6), so with a different rival the shops differ from
day 3 (FARMERS_MARKET instead of SMOOTHIE_SHOP); the nearest state is
then another seed's (7103, 7101) and the plan flips to another world's.
A state-conditioned policy built from 12 trajectories jumps between them
the moment the trajectory leaves them; a fixed schedule cannot jump and
keeps 71-99 k. "Lack of data" means lack of COVERAGE of the states the
deployed policy visits, not the count of records: that is what DAgger
collects, and what a search over a policy class with the state
conditioning built in (mix follows the observed demand) avoids.
