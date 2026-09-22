# kagsym

Neurosymbolic agent for [**Kaggriculture**](https://www.kaggle.com/competitions/kaggriculture)
(Kaggle × Google): two players, 720 turns —30 days of 24 hours—, a 10×10 board
and **a shared market** where selling sinks the price for both.

Read [`SPECIFICATION.md`](SPECIFICATION.md) first: it is the engine derived
exactly —action space with preconditions, reward, and the exactness traps
already verified—.

---

## The idea

The split between symbolic and learned is not a matter of taste, it is a matter
of competence:

| | who decides | why |
|---|---|---|
| **legality** | the engine | it is a fact, not an opinion |
| **unit assignment** | Hungarian | the exact optimum, 0.23 ms for 16×116 |
| **what to do and what it is worth** | the network | it is preference, and preference is learned |

The network emits, **once a day**, two things: a `macro` vector of 64
dimensions and a 10×10 `micro` map with one value channel and one per verb (19
verbs). The symbolic layer plays the 24 hours with that.

**One RL step is a day.** It is the decision that makes credit assignment
tractable, and it conditions everything else: calling the network once per turn
would multiply the cost by 24 in a competition with one second per turn.

## The principle: nothing guessed

Every value not derived from the engine is **learnable**. It is not an
aspiration; it is what has paid most in this project:

```
exposing 14 module constants               cell ceiling:  $939 → $1,258  (+34%)
exposing 4 literals inside functions       two of them were worth ±22% each
exposing 8 structural decisions            four live, between +23% and +71%
exposing 5 valuations inside `tile_task`   what a feeding, a trip or a watering is worth
```

Finding them with `grep` is not enough: that catches `SAT_HIGH = 0.85`, but not
a `min` that is an impassable ceiling, nor a loop that splits a budget and
**throws away the remainder**. That is what
[`tools/audit_decisions.py`](tools/audit_decisions.py) is for: it walks the
syntax tree and pulls out **every** decision site.

And once exposed, **perturb and measure**: reading the code does not tell a
live constant from a dead one. Multiplying the value of clearing weeds by a
thousand does not move a single dollar; moving the hiring cut-off hour gives
+38%.

A parameter can also be dead the other way round. `residual_cap` was emitted by
the network, written into a module global and never read again: a learned
dimension that decided nothing. It is gone, and `migrate_ckpt` deletes its row
from any older checkpoint before growing the head, because dropping a dimension
shifts the meaning of every row after it.

### Borderless parameterization

```python
positive   value = default · exp(logit(f))             → (0, +∞)
fraction   value = sigmoid(logit(default) + logit(f))  → (0, 1)
additive   weight = logit(f)                           → (−∞, +∞)
```

With `f = 0.5` the exact default comes out and the extremes reach the whole
domain. No scale constant remains: since the network emits `macro_mu` and
everywhere `f = sigmoid(macro_mu)`, it follows that `logit(f) == macro_mu`, so
the factor that would multiply is 1 **because sigmoid and logit cancel**, not
because somebody chose it.

## Full learning freedom, one path only

There used to be three modes —`residual`, `direct`, `ops`— so they could be
measured against each other. They are measured: the one where the network emits
value **and** verb is the only one with complete learning freedom, and it is
the only one left. Removing the other two, the greedy assigner, the route
assigner and the verb-sampling path leaves the same result **to the dollar** on
a 30-day, two-rung probe: it was dead code.

## Where we stand, measured

Against a passive opponent, 24h × 30d, $3,000 starting cash, six seeds:

```
symbolic layer, neutral macro          $21,545
symbolic layer, CEM macro              $51,905
the e2e network                        $72,796     +40% over CEM
median of 63 public agents            $186,594     2.6× over us
```

Learning works. The architecture's ceiling does not, yet: **we are at 39% of the
field**, and the gap is production, not timing —they harvest 2×, water 2.3× and
keep 44 live plants against our 15.4 while sowing the same amount—.

Against an uncapped v48 the last checkpoint makes $48,035 to its $115,305
(-58.3%, 0 of 8 fixed seeds). Broken down by product
([`tools/where_is_the_gap.py`](tools/where_is_the_gap.py)), 77% of the $76,203
difference in gross sales is strawberries and wool; milk, which used to be a
third of the gap, is now $9,028 of it, and we out-sell them on eggs.

## Training

```bash
python -m kagsym.cli.train \
  --updates 4000 --envs 44 --procs 11 \
  --grid "24,30,9;24,30,7;24,30,5;24,30,0;24,30,0;24,30,5" \
  --levels "6,7,8,12,15,1" \
  --rival-macros="-,-,-,-,-,vector.npy" \
  --auto-curriculum --selfplay-quota 2 --target-quota 3 \
  --ctx-micro 3x3 --kl-target 0.01 --kl-max 0.06
```

**The curriculum** splits the workers by Bernoulli information `p·(1−p)`, with
two fixed quotas **outside** that split:

- **target**: `p(1−p)` is 0 when you always lose, so the criterion would
  abandon precisely the opponent that measures us.
- **self-play**: a frozen copy gives `p ≈ 0.5` by construction, i.e. maximum
  information, and without a cap it would take the whole budget.

**Self-play** carries the complete network —macro and micro—, not a vector:
with the same macro on both sides, the network makes $75,009 and the vector
$33,164.

## Verifying

```bash
pytest kagsym/tests -q               # Hungarian == scipy; FastEnv == the real engine
python tools/yardstick.py ckpt.pt    # fixed yardstick: uncapped v48, 8 seeds
python tools/audit_decisions.py      # every decision site, for classification
```

`FastEnv` has to be **indistinguishable** from the engine turn by turn. If that
test fails, everything above it lies.

## Layout

```
kagsym/
  spec.py  fastenv.py      the engine and its exact clone
  obs.py                   observation encoding
  macro.py                 the 64-dimension vector and its 36 parameters
  reward.py                the objective (fixed) and the shaping
  potential.py
  environment.py           one step = one day
  parallel_env.py          N games in parallel
  symbolic/                the symbolic layer: tasks, market, Hungarian
  nets/                    the world model and the heads
  migrate_ckpt.py          loud checkpoint migration
  cli/train.py             the only trainer
  tests/
tools/                     measurement and ladder construction
```

`archivo/` keeps 84 historical measurement scripts and 16 inherited tools. They
are not production and do not enter the package, but they are the trace of why
each decision is what it is.

## Licence

MIT. It includes no third-party agents: the opponent ladder is built by
downloading them with [`tools/download_ladder.py`](tools/download_ladder.py).
