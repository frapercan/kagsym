# Exact specification of Kaggriculture

Everything here is **derived from the engine**
(`kaggle_environments.envs.kaggriculture`), not assumed. Every statement is
verifiable by running the interpreter.

---

## 1. The reward

**Only the cash in hand on the final turn scores.** Nothing else.

```
reward = farm["money"]   on the last step
score  = win / draw / loss               (the MARGIN does not score)
```

Exact consequences, not interpretations:

- The shed is worth **0**. Unit inventories are worth **0**. Standing crops are
  worth **0**. Animals are worth **0**. Bought land is worth **0**. Anything
  that is not cash at the close is a total loss.
- The 2945 expert finishes with shed 0, hands 0, crops 0: it liquidates exactly.
- Measured over 224 real ladder games: **50% are decided by <0.5%**, **26% by
  less than $41** (the value of ONE useful action) and **16.5% are exact
  draws**. Exactness is not hygiene: it is the decision variable.

---

## 2. Action space

### 2.1 Per unit — one action per unit per turn

Units = 1 farmer + N hands. **Hands last ONE day**: at the close
`farm["hands"] = []` and `hires_today = 0`.

| op | exact precondition | effect |
|---|---|---|
| `PASS` | always | nothing |
| `NORTH/SOUTH/EAST/WEST` | destination on the board (LOCKED is walkable) | moves 1 |
| `PLANT <crop>` | tile `None`, not LOCKED, `seeds[crop] > 0` | −1 seed, a plant is born |
| `WATER` | PLANT tile, `not watered_today` | waters; inside the window, +1 (+2 if fertilised) |
| `HARVEST` | dict with `yield_units > 0`; if PLANT, `day − planted_day >= first_yield_day` | moves to the unit's inventory |
| `FERTILIZE` | PLANT tile and **1 FERTILIZER in THE UNIT's inventory** | activates days `day..day+2` |
| `BUILD_COOP` / `BUILD_PASTURE` | tile `None`, not LOCKED | creates the structure |
| `PLACE <animal>` | on a compatible empty structure **and the animal in the unit** | places the animal |
| `PLACE <item> [n]` | on a shed access tile | leaves n in the shed |
| `PICKUP <item> [n]` | on shed access, `shed[item] > 0` | takes n |
| `DROP` | on shed access | dumps the WHOLE inventory |
| `FEED` | animal `not fed_today` and **1 WHEAT in THE UNIT's inventory** | marks it fed |
| `CARE` | animal `not cared_today` | deferred production bonus |
| `COLLECT_FERTILIZER` | animal with `fertilizer_available` | +1 FERTILIZER to the unit |
| `DIG` | non-empty tile and **no animal** | leaves the tile at `None` |

Every illegal action is a **silent no-op**: the engine returns the same state.
Measured: without a mask, 99% of our network's `PLANT`s were no-ops.

### 2.2 Market — up to `maxMarketOrdersPerTurn` orders per turn

`SELL`, `BUY_PRODUCT`, `BUY_SEED`, `BUY_ANIMAL` (by units, resolved step by step
alternating players) and `HIRE`, `BUY_LAND` (atomic, one per order).

- `HIRE`: cost `mult · fib(hires_today)` with **mult = 1**. The first 10 hands
  of the day cost **$143 in total**; the 15th alone already costs $610.
- `BUY_ANIMAL` leaves the animal **in the shed**, not on the board.
- A buy/sell round trip within the same turn yields **exactly 0**: the engine
  quotes `BUY_PRODUCT` at `inventory − 1` precisely to prevent the arbitrage.
  *(verified: $3,000 → $3,000)*

---

## 3. Exactness mechanics: where you lose by one

1. **Planting without watering the same day = a dead plant.**
   `consecutive_unwatered` is born at **1**; at the close it goes to 2 and the
   tile becomes `WEED`. *(verified: unwatered → WEED; watered → PLANT)*.
   `PLANT` and `WATER` are inseparable within the day.
2. **Two days unwatered = WEED.** Seed, tile and work are all lost.
3. **Two days unfed = the animal escapes**, the structure stays. Each animal is
   worth ~$1,700 net. `FEED` consumes 1 wheat **from the unit's inventory**: it
   has to be carried there. *(measured: selling the wheat cost us 17 of 25
   animals)*.
4. **Shed: 100 units.** At the day's close inventories are flushed and **what
   does not fit is discarded**.
5. **Decay:** past `max_lifespan_step`, −1 `yield_units` every 2 steps down to
   0, and then `WEED`. Harvesting late loses units.
6. **Watering window of non-`ongoing` crops:** they only add yield if watered
   between `(max_yield_day+1)//2` and `max_yield_day`. Watering outside only
   keeps the plant alive.
7. **The fertiliser bonus only counts on watered days**; the `CARE` one is only
   consumed on a day that is both producing **and** fed.
8. **The town drains on fixed steps:** shops every 4, the centre every 24. The
   price is an integer with a floor of 1.
9. **Final liquidation:** the shed does not score, so there is an exact instant
   past which any holding is a loss.

---

## 4. Derivable versus decidable

Project principle: *predict only what cannot be derived or sampled*. Applied to
the **action space**, not just to the state.

**Derivable — the executor, exact, no learning:**
minimum route (Manhattan; there are no obstacles), unit→task assignment,
legality mask, pairing PLANT+WATER within the day, wheat reserve = animals ×
days, not overflowing the shed, the liquidation calendar, viability by date (do
not plant what will not mature before the end).

**Decidable — what depends on the opponent and on risk, and only this:**
how much area, what crop mix, how many animals, how many hands, when to expand,
and above all **hold or dump into the shared market**.

The world model enters exactly there: its measured signal is about **level**
(opponent money +19%, their spending +35%), not timing (flow per turn: worse
than not correcting). And these decisions are about level.

---

## 5. The MDP as implemented

**State:** the encoded observation plus the world model's prediction (the
opponent's flow and their derived money) and one's own production history.

**Action — one macro vector per day, plus a per-tile value and verb map**, not
17 ops × 24 units × 720 turns. The macro carries what used to be written by
hand:

```
target area · crop mix · target animals · target hands
selling aggressiveness (hold <-> dump) · when to expand
```

**Reward.** The earlier measurement explains why net worth fails: with net
worth, *doing nothing* preserves $3,000 and is a local optimum — PPO converged
to $2,794, below not playing at all. The reward has to give **0 to passivity**:

```
dense     : income from selling OWN produce
explore   : +bonus the FIRST time each product is sold (once per game)
terminal  : win/loss, which is the only thing that scores
shaping   : potential-based, with Phi(s_T) = cash exactly (policy-invariant)
```

"Own produce" is not a detail: it excludes playing the market against oneself,
which is exactly where the engine already guarantees zero profit. The
per-product bonus is what breaks the monoculture local minimum.

A dense term for the opponent's money was tried and dropped: it sank the
critic, because the opponent's liquidation accounts for 99.1% of the daily
variance and no state the agent sees predicts it. The opponent belongs in the
objective (win/loss), not in the shaping.
