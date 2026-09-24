# EXP-005: can the model converge to a near-perfect strategy in a reduced space, with no opponent?

Written before launch, 2026-09-24 22:30. The question is about the learning
machinery, not about transfer: the reduced world is the whole game here.

## The space

24h x 8 days, passive opponent (solitaire), starting cash 3,000 $, the
executor values the honest 8-day horizon. Carrot (3 days), wheat (4) and
goose (first egg on day 4) can pay; melon, strawberry, tomato, cow and sheep
cannot. Evaluation seeds: RESERVED (held out from every search and every
training run, which use SEARCH / TRAINING).

## The bar (measured before learning)

```
partida_v5 alone, 10 reserved seeds       2,853 $   (loses money: hires, plants nothing that pays)
local rollout oracle (K=32 around v5)     3,193 $   regret 340 $, search wins on 6 of 8 days
global bar: CEM over the macro ramp in this world (runs/search/exp005_sol8_cem), then the
rollout oracle on top of the best schedule -> filled in below
```

## Mechanisms, cheapest first, each measured as regret on reserved seeds

1. **Search (CEM, ramp a + b*progress on the live dials)** in this world:
   the best fixed strategy the executor admits. Its regret against the
   rollout oracle is the value of state-dependence.
2. **PPO from v5** in this world (`--level 0`, no self-play, 150 updates,
   seeded): does the gradient close the gap where the search finds money?
   Health (heads with gradient, sigmas), progress paired against update 1.
3. **PPO from random weights** in this world: convergence from nothing.
4. **Imitation of the oracle**: the rollout oracle's daily choices as
   supervised targets for the macro head; regret after distillation.

## Convergence, defined

Regret per day against the best available oracle on 20 reserved seeds,
paired; a mechanism has converged when its regret is within the oracle's
own noise (the difference between K=32 and K=64) on every day. The quality
of actions across the rhythms is read from the day-by-day regret and from
`tools/stage_map.py --days 8` on the trained checkpoint.

## Decision rule

Every mechanism runs to its planned length. What is reported is the regret
curve, the day profile and the health of the network; nothing is baked or
deployed from this experiment.
