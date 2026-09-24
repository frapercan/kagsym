# Neurosymbolic RL for Kaggriculture: what actually moved the needle [WIP-NOTES]

**Update.** The code is now public: **https://github.com/frapercan/kagsym** (MIT).
This revision splits the post in two — **Part I is what the agent is**, **Part II is
the eleven-plus tricks**, which is how it should have been written the first time —
and brings the architecture section up to date, because the answer to "how many
numbers does the policy emit" changed from 14 to 64 since the first version, and
that change *is* the main result of the last stretch.

---

## In one paragraph, without the jargon

A commenter said this post uses a lot of fancy words, and that is fair. So, plainly:
the game gives you a farm, 720 turns and an opponent selling into the same market.
Some of the decisions have a *right answer you can compute* — the shortest path to a
tile, which unit should go where, what the 12th unit of wheat will sell for, whether
a crop still has time to ripen. Those I calculate exactly and never let a neural
network guess. The rest — *how much* land, *which* crop, *how many* animals, *when*
to dump produce on the market — has no right answer you can write down, because it
depends on what the other player does. Those a network learns. The whole project is
an attempt to find, **by measurement rather than by taste**, where the line between
the two sits. Most of what follows is the measurements that told me I had put the
line in the wrong place.

---

## A thank-you that isn't a formality

Every single number in this post is measured against agents somebody else published
and explained — `v48-fast-routes`, `v16-rc5-high-score-8c-4s-premium-market-lead`,
`the-2945-farm-96-vs-the-top-10-public-bots`, `shop-router-0909`, and the
rank-your-agent notebook that made head-to-head evaluation trivial. I did not just
use them as a leaderboard to climb. I used them as **opponents strong enough to make
a measurement mean something**, as **teachers** to clone from, and as the **external
thermometer** that caught my own training loop lying to me (Trick 8). When I needed
a difficulty ladder and there wasn't one, I built it by handicapping *their* agents,
because their internal logic is good enough to resize itself sensibly when you
constrain it (Trick 7).

The discussion threads did something different and just as necessary: they framed
the problem before I had any data of my own. The thread asking how the top agents
*minimise time and maximise profit at the same time* is what made me stop treating
this as an economics problem and start treating it as an action-budget problem —
which is the frame that eventually produced the crop-density result in
Part II. Someone
publishing that their JAX pipeline ran at ~10k steps/s is why I knew my own
throughput number was worth chasing rather than being quietly satisfied with it. And
the general consensus in the threads that RL is hard in this environment turned out
to be correct for reasons nobody had written down yet — most of Trick 6 and Trick 8
are me discovering *why*, the expensive way.

Reading those threads before writing code saved me from at least two of the dead
ends below, and pointed me straight at a third.

A self-play loop can convince you that you're winning while your real strength
collapses. The only thing that reliably stopped that here was a fixed, strong,
publicly-shared opponent sitting outside my training loop. So: thank you to everyone
who published a working agent and wrote up how it worked. This post is downstream of
that, and several of its results only exist because you did.

With that said — what follows is mostly **negative** results from building an
end-to-end learned agent. I think they're more useful than another "here's my
architecture" post, and they're certainly more honest about how the time was spent.

---

# Part I — the project

---

## The setup

**Exact engine clone.** Instead of reimplementing the rules, I call the real
`interpreter` on a duck-typed state object. It's exact by construction and never
desyncs when the engine updates. Runs at ~38,500 steps/s versus the full
`kaggle_environments` wrapper.

**Neurosymbolic split.** The network decides *strategy*; exact arithmetic handles
*mechanics*.

- **Symbolic** (derivable from the engine, stays hand-written): legality of
  operations, routing distances, Hungarian assignment of units to tiles,
  marginal-price arithmetic, final liquidation value.
- **Neural**: a daily macro vector (how much of each thing), and a per-tile head
  emitting a value plus one logit per legal verb.

The key idea: never let the network guess something you can compute exactly, and
never hand-code something that depends on the state in a way you can't write down.

---

## What this is actually an experiment about

Eleven tricks with no thesis is a list. Here is the thesis, and what each moving
part is for — because two of them are instruments, not competitors, and reading
them as competitors is how I misread my own results for a week.

**The question.** Not "can I win this competition". It is: *where exactly is the
line between what should be computed and what must be learned, and how do you
find that line by measurement instead of by taste?* I have an exact clone of the
engine, so anything derivable — legality, routing, assignment, marginal prices,
liquidation — I can compute perfectly. Everything else is a candidate for
learning. The interesting claim is that the line is discoverable, and that most
people (me included) put it in the wrong place by intuition.

**What CEM is for: it is the control, not a rival.** CEM searches for the best
*fixed* 14-number vector — a strategy that cannot look at the state at all. So
whatever it achieves is a measurement of **how much of this game is solvable
without conditioning on anything**. That makes it the denominator for the only
question that matters here: anything a learned policy gains over CEM is, by
construction, the value of state-dependence. Nothing else.

This is why "PPO does not beat CEM" is the central result of the project rather
than a disappointment. It is a clean ablation reporting that the state-dependence
I am paying for — the whole network — is currently earning nothing. A negative
result from a well-built instrument is still a measurement.

**What PPO is for: the hypothesis CEM cannot test.** The bet is that the optimal
strategy is *not* one fixed vector — that it should shift with the day, with
prices, with what the opponent has already dumped on the shared market. A fixed
vector cannot express that by construction. PPO is the candidate for that part.
The bet is currently unpaid, and I want to be exact about the status: not
refuted, **unmeasured in its favour**.

**What the public agents are for — three jobs, and conflating them is an error
I made.**

1. *External thermometer.* Fixed, strong, entirely outside the training loop.
   This is the only thing that ever caught my self-play league congratulating
   itself while sitting on its starting cash (Trick 8).
2. *Difficulty ladder.* Not by throttling them — that shatters them (Trick 7) —
   but by capping hiring, because their own logic resizes to the unit count.
3. *Tightness check on the upper bound.* See below. This is the job I only
   discovered at the end, and it is the most valuable.

What they are **not**: teachers (cloning failed twice at ~99.5% agreement,
Trick 4), and opponents at any scale but the full game (they are 719-step tapes
with persistent internal state; shrink the horizon and they score $0–30).

**The natural law I am actually hunting.** The cost of the three transitions:
inaction -> minimum viable movement -> the global maximum, even a conditional
one. That gets its own section, next.

**The four numbers that make any of this sayable.** This framing is the part I
would keep if I threw away everything else:

```
inaction                 $3,000    policy-independent floor, any horizon
CEM (best fixed vector)  $55,350   LOWER bound: best found without state
a real public agent     $145,080   where a good policy actually lands
relaxed problem         $195,532   UPPER bound: no policy can exceed it
```

(Full game. The relaxed bound ignores distances and inter-unit coordination, so
nothing can beat it, though nothing need reach it.)

Two things only exist because of that fourth number. First, **the bound is
tight**: a real published agent reaches **74%** of it, so the relaxation is not
fantasy. Second, we are at **28%** — which means the CEM ceiling I had been
treating as *the* target all along is 38% of what a real agent already achieves
in the same game. There is 2.6x of demonstrated headroom, not conjectured.

That also resolves something I could not answer before: how do you tell
"this scenario is exhausted" from "my search is stuck"? You cannot, against a
lower bound — "I found nothing better" is what both look like. Against an upper
bound you can, because `achieved / bound` is a fraction with a meaning.

---

## The natural law: what it costs to start moving

### Inaction is an absorbing state with a known, horizon-independent value

Doing nothing keeps **exactly the starting cash — $3,000, standard deviation
zero, at every horizon I have tested.** Not approximately. The strategy vector of
all zeros returns 3,000 on every seed.

That is a gift and I did not appreciate it for weeks. It means there is a free,
exact, *policy-independent* baseline at every scale, which is what lets you say
"this policy is inert" in one glance instead of inferring it from a win rate. It
is also why every number in this post is quoted as a multiple of it.

But it is an absorbing state. Every action in this game costs up front and repays
later: hiring costs Fibonacci money, seed costs money, feed costs money, and
movement costs *actions*, which are the genuinely scarce resource. You must go
down before you can go up. If the horizon does not let the investment repay, the
optimal policy is literally to sit still.

### The phase diagram, measured

Best achievable money by scenario, as a multiple of inaction. Found by search,
not asserted:

```
 2h x 13d   1.000      3h x  8d   1.000      3h x 13d   1.000
 3h x 21d   1.000      5h x  5d   1.000      5h x  8d   1.000     <- IMPOSSIBLE
 5h x 13d   1.073      6h x  6d   1.080      8h x  8d   1.248     <- thin
 5h x 21d   1.815      8h x 13d   5.113      8h x 21d   6.304     <- comfortable
13h x 13d   8.838     13h x 21d  10.109     24h x 30d  18.450     <- the real game
```

The top block is not "hard". It is **impossible**: no policy in the search space
beats sitting still, because there is no horizon in which the first hire pays for
itself. That is a fact about the economy, not about the optimizer.

And the axis is **not total turns**, which surprised me. Compare equal-length
games:

```
 3h x 21d =  63 turns  ->  1.000          5h x 21d = 105 turns  ->  1.815
 5h x 13d =  65 turns  ->  1.073          8h x 13d = 104 turns  ->  5.113
 8h x  8d =  64 turns  ->  1.248
```

At a fixed number of turns, hours-per-day dominates. The mechanism is clean once
you see it: a worker is hired *per day* at a fixed cost, but works *hours* per
day. So hours-per-day is the **productivity** of the investment and days is the
**time to repay it**, and they are not interchangeable. Sixty-four turns as
8x8 is a different economy from sixty-three turns as 3x21.

### The three transitions, and what each costs

This is the actual research programme, and I state it as three costs because that
is how it is measurable:

**1. Inaction -> first profitable action.** What is the smallest bundle that
repays? It is not a single lever. At 8h x 13d you need enough workers *and* the
right crop, and getting the crop wrong scores **0.39x** — 61% *worse* than doing
nothing. So the first viable move is a conjunction, and most of its neighbourhood
is worse than the floor. That is the barrier.

**2. First profitable action -> the best fixed strategy.** This is what CEM
measures, and it is a lot: 5.113x at 8h x 13d, 18.450x at full scale. All of it
achieved by a strategy that never looks at the state.

**3. The best fixed strategy -> the global maximum, which may be conditional.**
This is the open one, and the relaxed upper bound says it is the biggest of the
three: we are at 28% of the bound, a real agent is at 74%. Whether that last leg
*requires* conditioning on state is exactly the unproven premise of the project.

Cost 1 is a threshold. Cost 2 is search. Cost 3 is the bet. I had been conflating
all three under the word "training".

---

## The ladder, and the thing that is not a ladder

Two structures, two purposes. Conflating them cost me most of a week.

### Reduced horizons are a laboratory, not a curriculum

I assumed shorter games were an easier curriculum you could graduate from.
Trick 9 already said they are a different game. Measured directly, it is worse
than that — transplanting one scenario's optimum into another, as a percentage
of the destination's own ceiling:

```
   origin    ->  destination     money   % of destination ceiling
 8h x  8d    ->   8h x 13d         886        5.8%
 8h x  8d    ->   5h x 21d          18        0.3%
 8h x 13d    ->  24h x 30d         727        1.3%
24h x 30d    ->   8h x 13d       1,608       10.5%
 5h x 21d    ->  24h x 30d      21,398       38.7%
```

Inaction is 3,000. **In eight of the twelve transplants I ran, carrying over a
neighbouring scenario's optimum leaves you worse off than doing nothing.** Not
weak transfer — actively harmful. There is no sense in which solving the small
game moves you toward the big one.

So what *are* they for? They are where the mechanism is visible. At 8h x 13d an
episode costs 20 ms against 1 s at full scale, and the paired standard deviation
is roughly 150x smaller, so you can resolve effects there that are permanently
invisible in the real game. Trick 11 exists only because a full axis sweep is
seconds of compute at that scale. **Use them to see, not to graduate.**

### The actual ladder is the opponent, at full scale

The game stays identical — same horizon, same board, same economy — and the only
thing that varies is the opponent's hire cap, which works because their logic
resizes itself to its unit count (Trick 7). Same search vector throughout, no
training:

```
hire cap     ours      v48     margin    wins
       3   62,556    4,229  +1379.2%   12/12
       5   52,174   27,882     +87.1%   12/12
       8   49,106   58,765     -16.4%    5/12    <- the crossover
      11   39,990  110,859     -63.9%    0/12
    none   37,673  124,276     -69.7%    0/12
```

This is a curriculum in the way the horizon axis is not: monotone in difficulty,
a genuine 50/50 crossover point, and — crucially — **nothing learned at one rung
is invalidated at the next**, because it is the same game throughout.

### When to promote

The question I could not answer before the relaxed bound existed: how do you
distinguish *"this rung is exhausted"* from *"my search is stuck"*? Against a
lower bound you cannot — "I found nothing better" is what both look like.

Against the **upper** bound you can, because `achieved / bound` is a fraction
with a meaning. Promote when that fraction stops climbing, not when you win.
Winning is what promoted a policy that sat on its starting cash through four
leagues (Trick 8); a fraction of an upper bound cannot do that, because inaction
scores 1.5% of the bound and says so.

The honest caveat: the relaxed bound is verified tight **only at full scale**,
where a real agent reaches 74% of it. At reduced scales there is no live agent to
check it against, and an unvalidated bound is exactly the mistake I made once
already — an analytical bound ranked 2h x 13d as promising, and 145 updates of
training there drove the policy to 0.82x, destroying value.

---

## The interface, concretely

A commenter asked what actually connects purchases, pickup, routing, placement
and maintenance, and whether the executor keeps persistent jobs. Writing the
answer was more useful than anything else I did that week, so it goes in the
post rather than the thread.

**Everything is recomputed every turn. No jobs, no reservations.**

```python
n_units = max(len(hands), target_workers(macro))    # PLANNED, never the live count
free    = sustainable_tiles(n_units) - already_planted
values, op_logits = micro_head(obs)                 # 10x10, and 15x10x10
assign  = assign_units(obs, free, values, op_logits,
                       previous=yesterdays_destinations,   # the only surviving state
                       macro=macro)
orders  = concat(cat() for cat in order_by(macro.priorities))[:10]
```

The only state crossing the turn boundary is yesterday's assignment, used as a
**soft stickiness bonus** whose weight the policy sets. Measured, with the search
optimum fixed: stickiness 0.00 gives $23,793, 0.25 gives $27,177 (+14%), 1.00
gives $22,755. Interior optimum, so it cannot be reasoned to a constant.

Persistent jobs would lock in an arbitrary choice — unit ordering is meaningless
and greedy assignment strands units. But recomputing means nothing enforces
multi-turn commitment either, and one scalar is a blunt instrument for it. That
is an unresolved tension in my design, not a solved problem.

### The 14 numbers the policy actually emits

This is the part I never spelled out, and spelling it out changed what I think
the problem is. The network emits **14 reals in [0,1] per day**, plus the
per-tile map. The executor turns those 14 into exact quantities:

| # | name | resolver | what it really is |
|---|---|---|---|
| 0 | `tiles` | `int(v * min(arable, n_units * hours * 0.5))` | **integer** |
| 1 | `animals` | `int(v * (n_units * hours - planted) / 3)` | **integer** |
| 2 | `workers` | `int(round(v * 15))` | **integer** |
| 3 | `sell_horizon` | `max(1, int(round(v * 2 * hours)))` | **integer** |
| 4 | `crop` | `viable[int(v * len(viable))]`, sorted by profit/day | **categorical** |
| 5 | `expand` | float threshold | continuous |
| 6 | `stickiness` | `2.0 * v` | continuous |
| 7 | `fertilize` | `3.0 * v` | continuous |
| 8–13 | six priorities | `softmax(3v)`, **only the ordering is read** | **one of 6! = 720 orderings** |

Three things fall out of writing that table honestly.

**Twelve of fourteen outputs are discrete, and I gave all fourteen the same
Gaussian head.** That is a parameterization mismatch, not a tuning problem.

**The priorities are the most quantized part of the vector, not the least.** I
had assumed the softmax made them continuous. It doesn't: softmax is strictly
monotone, so sorting by softmax is identical to sorting by the raw value, and the
only consumer reads the ordering. Six reals collapse to one of 720 orderings. The
function's own docstring claims it returns a budget split between categories;
grep says nothing in the codebase consumes those fractions.

**The crop index is ambiguous across time.** The viable list shrinks as days run
out, so the same value means different crops on different days.

### The three boundaries

| boundary | symbolic (exact) | neural |
|---|---|---|
| movement | routes and distances | — |
| assignment | Hungarian over the matrix | the **value map** that feeds it |
| market | marginal prices, legality | how much, and in what order |

The head decides **what** and **how much**. Never **how** or **where to go**.
Movement is not a verb the network can emit — routing is exact, so asking for
`NORTH`/`SOUTH` would be guessing something computable. `PASS` is not a logit
either: the executor emits it when a unit has no legal options.

The verb vocabulary was 15 operations at the time of writing (it is 19 now, one PLANT per crop — see the update below):

```
PLANT  WATER  HARVEST  FERTILIZE  DIG  FEED  CARE  COLLECT_FERTILIZER
PLACE  BUILD_COOP  BUILD_PASTURE  PICKUP_ANIMAL  PICKUP_WHEAT
PICKUP_FERTILIZER  DROP
```

`PICKUP` is split by object because **the quantity depends on the object**.
Treating it as an argument-less verb emitted `1` every time, where the heuristic
takes `min(hungry_animals, wheat_in_shed)`. Livestock ate at half speed. Fixing
it raised that mode's ceiling from $26,536 to $40,763.

---

### What the policy emits today: 64 numbers, not 14

The table above is what the first version of this post described, and its three
conclusions are exactly what drove the redesign. Here is the current vector:

```
  8   levels        how much of each thing (tiles, animals, hands, selling,
                    crop bias, expansion, stickiness, fertilising)
  6   priorities    what gets sacrificed when cash does not cover everything
 36   exposed       the constants that used to be hand-written, borderless
  9   market        one multiplicative factor per product over its sale value
  5   turn rule     coefficients of a rule EVALUATED EVERY TURN (see below)
 --
 64   reals in [0,1], emitted once per day
```

plus the per-tile map, which is now `1 + 19` channels over 10x10 = **2,000**
Gaussian dimensions: one value channel and one logit per verb, with **one PLANT verb
per crop** rather than a single `PLANT` whose crop a hand-written function chose.

That last change is not cosmetic. With a single `PLANT` verb the network could
choose *whether* to plant and never *what*, and no downstream head could repair it:
the market head cannot sell what was never produced — measured, moving the
strawberry factor with a single plant verb does not change a single dollar.

**The daily/per-turn split, which is how you get per-turn reactivity for one
forward pass a day.** One RL step is a day: the network is called once and the
symbolic layer plays 24 turns. That makes credit assignment tractable and it fits a
competition with one second per turn. The cost is that everything learned is
*constant within the day* — including when to sell, which is the one decision that
genuinely depends on the turn.

So the policy emits a **rule**, not a level:

```
factor_p(t) = level_p * exp( w_price*x_price(t) + w_rival*x_rival(t)
                           + w_shed*x_shed(t)   + w_season*x_season(t)
                           + w_cash*x_cash(t) )
```

Five signals, dimensionless and centred on zero (expected price drop, share of the
imminent flow that is *theirs*, storage pressure, how far the season has run, how
much wealth is tied up in goods). The daily pass emits the five weights; the rule is
evaluated against each turn's state. Weights default to 0, so at `f = 0.5` the
factor collapses to the level and the behaviour is exactly the previous one.

### One mode, on purpose

There used to be three ways of combining the map with the heuristic valuation
(`residual`, `direct`, `ops`) so they could be measured against each other. They are
measured: the one where the network emits **value and verb** is the only one with
complete learning freedom, and it is the only one left. Deleting the other two, the
greedy assigner, the route assigner and a verb-sampling path left the result
identical **to the dollar** on a 30-day two-rung probe. Dead code that runs is worse
than dead code that does not: it is surface for a bug to hide in, and it is what
made a league opponent silently weaker than it looked (see the meta-lesson).

---

## Nothing guessed: the part that changed since the first version

The first version of this post said the policy emits **14 numbers**. It emits
**64**, and the extra 50 are not a bigger network — they are constants that used to
be written by hand in the symbolic layer and are now learned. That turned out to be
the highest-paying single thing in the project, so it deserves its own section.

**The rule.** Any value that cannot be derived from the engine is a candidate for
learning. Not "any value that looks important" — *any value*. A threshold, a
multiplier, a reserve, a floor, a softmax temperature. If the engine does not force
it, somebody chose it, and "somebody chose it" is the definition of a guess.

**What it paid, measured in three passes:**

```
exposing 14 module constants               cell ceiling:  $939 -> $1,258   (+34%)
exposing 4 literals inside function bodies two of them worth +-22% each
exposing 8 structural decisions            four live, between +23% and +71%
exposing 5 valuations inside `tile_task`   what a feeding, a trip, a watering is worth
```

**Why `grep` is not enough, and what to use instead.** Searching for `^[A-Z_]* =`
finds `SAT_HIGH = 0.85`. It does not find a `min()` that is an impassable ceiling, a
loop that splits a budget by weight and **throws away the remainder**, or a
hand-written preference order. Those are *shapes*, not numbers. So the audit walks
the AST and prints every decision site — thresholds, `min`/`max`/`sorted`, and any
arithmetic against a float literal — for manual classification. On the current tree
it reports **65 sites** across ten modules, and engine facts (`spec.*`, numerical
guards) are exempted by name so the list stays readable.
[`tools/audit_decisions.py`](https://github.com/frapercan/kagsym/blob/main/tools/audit_decisions.py)

**Borderless parameterization, which is the part I would steal.** The obvious
encoding for a learned constant is `value = lo + (hi - lo) * f`. That is two more
hand-set numbers per parameter — 72 of them, for the 36 I have — and, worse, a hard
floor
and ceiling: if another league or another opponent needs a value outside, the model
cannot even express it. Measured: one parameter pinned itself to the floor in 5 of 8
learned vectors, which is the signature of a wall, not of an optimum.

```python
positive   value  = default * exp(logit(f))             # (0, +inf)
fraction   value  = sigmoid(logit(default) + logit(f))  # (0, 1)
additive   weight = logit(f)                            # (-inf, +inf)
```

`f = 0.5` returns the *exact* previous default, so exposing a constant is
behaviour-preserving by construction and any change is attributable. And there is no
scale constant left over: since the head emits `macro_mu` and everywhere
`f = sigmoid(macro_mu)`, `logit(f) == macro_mu` exactly, so the multiplier that
would sit in front is 1 **because sigmoid and logit cancel**, not because somebody
picked it.

**Two failure modes, in opposite directions, both found by audit and not by
reading.**

*A live constant nobody is learning.* Two parameters were exposed in the dataclass,
given a slot in the vector and emitted by the network every single day — and the
function that writes parameters into the symbolic layer never wrote those two. The
softmax temperature over the priorities stayed pinned at 3.0 and the seed floor at
2.0 for the whole project. The network was paying to emit two dimensions that
decided nothing, and two hand-set constants survived the "nothing guessed" rule by
being *almost* exposed.

*A learned dimension nobody is reading.* The mirror image: `residual_cap` was
emitted, resolved, written to a module global — and nothing read that global any
more, since the mode it belonged to had been removed. A dead dial costs a dimension
of policy entropy and a row of the KL budget, forever, invisibly.

The general form is worth stating once: **exposing a constant is not the same as
wiring it.** Check both ends. Reading the code will not tell you — perturb the value
and measure whether a dollar moves.

---

## Where it stood at the first version

Honest numbers as of the first version of this post, so nobody has to infer
them. Deterministic, 24 matched
seeds, both seats, full game, against `v48-fast-routes` uncapped:

```
                        seat     ours       sd      v48    margin   wins
net (PPO)                  0   36,695   16,099  123,032   -70.2%   0/24
net (PPO)                  1   38,489   14,769  123,333   -68.8%   0/24
executor + search macro    0   38,011   18,560  123,037   -69.1%   0/24
executor + search macro    1   41,226   19,086  125,474   -67.1%   0/24
inaction                   0    3,000        0  145,080   -97.9%   0/24
```

Three things I have to own.

**The learned policy does not beat the derivative-free search** over the same 14
numbers — 36.7k/38.5k against 38.0k/41.2k, inside the noise. Conditioning on
state bought nothing measurable over one fixed vector. The entire premise of the
project is that it should. That premise is still unmeasured in its favour.

**Against the uncapped public agents, zero wins.** Against the handicapped ladder
in Trick 7, comfortable wins at caps 3 and 5 and a true crossover at 8.

**Seat makes no difference**, and I had never checked until someone asked.

---

---

## Where it stands now

Two measurements, both current, both against opponents outside the training loop.

**Against an uncapped `v48-fast-routes`**, full game, 8 fixed seeds, the external
yardstick that exists precisely because self-play margin is not comparable across
time:

```
checkpoint          upd     ours      v48    margin   wins
best by return      340   48,035  115,305   -58.3%    0/8
latest              900   46,342  139,972   -66.9%    0/8
```

560 updates of training moved neither number outside the noise on our side, and the
opponent's own money moved more than ours did — which is the shared-market effect
again: what changes their score most is how much product *we* keep off the market.

**Where the money is missing**, same opponent, 4 seeds, gross sales per product:

```
product        ours $    v48 $      gap
STRAWBERRY     13,853   53,023  +39,170
WOOL           11,482   31,110  +19,628
MILK           21,786   30,814   +9,028
FERTILIZER      9,904   18,336   +8,432
MELON          10,556   16,646   +6,091
WHEAT          11,269   13,099   +1,830
EGG             9,099        0   -9,099   <- we out-sell them
TOTAL          89,022  165,226  +76,203
```

**77% of the gap is two products.** That is the same shape as the measurement that
motivated the market head in the first version — then it was strawberries and milk
at 84% — and milk has closed from a third of the gap to $9,028 since the portfolio
stopped being monoculture. Strawberries have not moved.

It is worth being exact about what this says: the gap is **production**, not
timing. We plant about as much as they do and keep a third of the live plants,
so the market head cannot fix it — there is nothing in the shed to sell.

---

# Part II — the tricks

Eleven of these were in the first version; 12-14 are new. The numbering is kept so the comments below still line up.

---

## Trick 1: marginal prices, not nominal

Three separate times I lost 40-66% of final money to the same bug class: valuing a
batch at the nominal price. Each unit you sell moves the price of the next one.

Valuing 26 eggs at $50 each made animals look profitable when they weren't. With
nominal pricing the agent bought animals on day 14 and money went from $4,310 to
$345. The same bug bit the fertilizer bonus and the DROP action.

If you value anything in bulk, sum the marginal prices.

---

## Trick 2: the hour-0 trap

`len(farm["hands"])` is **always 0 at hour 0** — workers are cleared overnight and
rehired in hours 0-3. If you plan your day at hour 0 (which is natural), every
capacity calculation that uses your current worker count is computing with 1 unit.

This bit me **eight separate times** in different files. Symptoms: a farm that
couldn't grow past 12 tiles no matter what; rival statistics reporting "1 unit" for
an agent hiring nine; and an observation encoder where **31 of 88 global features
had zero variance**, because they are all zero at that exact moment.

Measured at hour 0 versus every hour:

```
hour   workers   items carried
  0      0.00        0.00
  1      7.90        3.87
 12      9.33       33.17
```

If you sample state, declare which hour you sampled at.

---

## Trick 3: paired comparison, or don't bother

Per-seed standard deviation of final money is about $9,000. Almost every A/B you
run is under-powered.

Clone the state, run both branches with the **same seed and the same opponent**,
and difference them. Measured variance reduction: **2.2x in sd**, roughly 4.8x
fewer samples for the same precision.

This is how I found the biggest single result in the project — and one I did not
expect.

**And there is a stronger version I only found at the end.** When both things you
want to compare can live in the *same episode*, put them there. A commenter asked
whether I had evaluated on both seats — i.e. as `player 0` and as `player 1` —
and I hadn't. My first attempt ran my agent in each seat against a fixed opponent
in the other: two separate experiments, seed variance in both, paired standard
error ~3,300 on an effect I was hoping was ~2,000. Useless.

The right design puts the **same policy in both seats of one episode**. Now the
seed, the board and the market are literally shared, not merely matched:

```
                           paired sd     what it can resolve
two experiments, matched      ~15,000     nothing under ~30,000
same episode, both seats        2,566     down to ~1,000
```

Same question, 6x more resolving power, same compute. The answer, over 60 seeds:
`+228 +- 331`. No seat advantage. Thirteen of the sixty tie **to the dollar**,
which is what tells you the asymmetry is divergence rather than bias.

One more thing, and it is the reason this trick exists. My first twelve seeds
said **+1,497**. Sixty fresh seeds said **+228**. The small sample I looked at
first inflated the effect **6.5x**. This happened to me while writing the reply
to that comment, after having written this section.

---

## The result that reframed everything

I swept my crop-density lever with paired comparison over 8 seeds, against the same
opponent throughout:

```
density   my money   opponent   plantings   paired delta
 0.004     62,416     13,852        0            —
 0.100     20,780     33,460       55       -41,637 +- 4,941   (8.4 sigma)
 0.500     12,202     33,008       59       -50,214 +- 4,695   (10.7 sigma)
```

**Farming cost me 41-50k, at 8-11 sigma.** Not a bug — economics. Crops displace
the animal economy: without farming I sold 831 units of product
(fertilizer/eggs/milk/wool); with farming, 316. Moving, watering and planting eat
roughly 68% of actions to produce less.

So my CEM search wasn't broken when it drove crop density to zero. It found the
true optimum *of my executor*. Every number I had measured before that was
ceilinged by a strategy that doesn't farm.

---

## Trick 4: imitation accuracy does not predict performance

I tried cloning the top public agents. Twice, with two different architectures.

- **Behavior cloning**, per-tile verb: **99.4%** held-out accuracy (52.7%
  baseline), two minutes of GPU. Resulting agent: **$3,036-6,727** — worse than
  picking a random legal verb.
- **DAgger**, per-unit policy, 103k samples, 12 rounds, beta annealed to 0 so the
  *policy* drives and the expert only labels: **99.67%** agreement. That is
  textbook Ross-Bagnell against distribution shift. Resulting agent:
  **-99% margin, 0 wins in 24 games.**

  **Correction, added after a comment asked the right question.** That 99.67% is
  measured on the **aggregated DAgger buffer** — every round pooled, including
  round 0, which is pure expert — not on fresh autonomous rollouts of the final
  policy. I originally wrote "on its own states," and that is misleading.

  Worse, it is the accuracy of **one head**. The loss supervises four targets
  (`op_target`, `crop_target`, `item_target`, `market_target`) and the only
  accuracy I ever logged was `acc_op`, the per-unit operation. Market loss was
  logged as a *loss* and never as an accuracy, so **hiring and purchasing
  accuracy were never measured at all** — which is exactly where other people
  report their first divergences. Given that this section's whole argument is
  that agreement measures the wrong thing, breaking it down per head was the
  obvious next step and I skipped it.

Near-perfect imitation, near-zero money, twice, by independent routes.

Diagnosis of the first one: 86% of actions were PASS, because the network chose
PLACE on 95% of tiles and nobody was carrying the animal. Agreement-with-expert
measures the wrong thing — you nail the marginal label and lose the chain.

---

## Trick 5: calibrate your floor before training anything

Three baselines are worth having at every horizon. **Measure them against the
opponent you will actually train against** — see the note below, because I got this
wrong myself.

```
do nothing (keep starting cash)          $3,000
fixed random verb ranking                  $397
random LEGAL verb, re-picked each turn   $8,692
hand-written heuristic                  $30,065
```

Two things fall out. First, there is real gradient from scratch: random-legal is
2.9x doing nothing, so RL is not starting blind. Second, and more useful: **a fixed
ranking scores $397 and re-picking each turn scores $8,692**, 22x apart. The correct
operation depends on state in a way no fixed ordering captures — which is exactly
what a hand-written heuristic *is*. That's a measured argument for learning the
verb, not an aesthetic one.

> **Caveat, and it's my own mistake.** The middle two rows were measured against a
> weak scripted opponent and the last row against a strong public agent. That makes
> the column not strictly comparable — the exact error this post warns about at the
> end. The 22x contrast between the two random policies is internally valid because
> both were measured the same way; the absolute levels across rows are not.

---

## Trick 6: three bugs that made PPO not work at all

These cost me most of a night, and all three were invisible in win-rate and money
curves.

**Importance ratios were pure noise.** The micro head emits 1+15 channels over 100
tiles = 1,614 Gaussian dims, but only ~40 influence any action. All 1,614 entered
`exp(logp - logp_old)`. Measured: **99.4% of ratios saturated at the +-10 clamp
after one gradient step.** Masking to action-relevant dims: **0.1%**. Dimensions
that cannot change the action must contribute exactly zero — that is not a design
choice, it is a bug.

*How the mask is built*, since a commenter asked and it is the part that makes it
practical: **the symbolic layer already knows**. It is the executor that enumerates
the legal operations of each tile, so while it plays the day it records two things
per tile — whether the value channel was read at all, and which verb logits were
actually *compared against each other*. A verb's logit only matters if there was
something to compare it with, so a tile with one legal option contributes its value
channel and no verb dimension. The mask comes back with the rollout and multiplies
the per-dimension log-probs before they are summed. No heuristic, no threshold: the
mask is a by-product of the legality enumeration that had to happen anyway.

**The critic was mis-scaled, not mis-trained.** `smooth_l1_loss(value, returns)`
with `beta=1.0`, on returns with mean ~20 and sd ~8. Every error above 1 unit sits
in the pure-L1 regime: gradient +-1, carrying no magnitude information. Toy
regression with a perfectly linear signal at that scale: **R2 = -16.6** raw versus
**+0.83** with a normalized target. Normalize your value target.

**The critic and the policy fought over a shared trunk.** I lowered the trunk LR to
stop the critic yanking the policy — and the critic's R2 went from 0.675 to
**-0.405**, because it lives on those same now-frozen features. Then I raised it and
the policy collapsed below do-nothing in 25 updates. The fix was a KL-targeted LR
controller: measure KL per dimension each epoch, cut LR 30% if over target, raise
10% if under. Stop hand-tuning a number you are already measuring.

After all three, critic R2 went **-0.405 to +0.698**, and money passed the
hand-written heuristic for the first time.

That last clause needs qualifying, and the qualification is the interesting
part. It passed the *hand-written* macro, which plants — and by this post's own
headline result, planting costs 41–50k. Against a derivative-free search over
the same 14 numbers, it did not pass. See "Where it ended up" below.

**A fourth instance of the same bug, found much later.** The masking fix above
applies to the per-tile head. The 14-number macro head has the identical problem
and I never noticed, because I was looking at the head that had already burned
me. Sweeping each macro axis independently at one rung, **8 of the 14 dimensions
are exactly flat** — they cannot change the action, they carry no gradient, and
every one of them was entering the importance ratio as noise.

The general form, which is the part worth taking away: **any dimension that
cannot change the action must contribute exactly zero to the ratio.** Having
fixed that once in one head does not mean you have fixed it. Check every head.

---

## Trick 7: build a ladder, because there isn't one

The public agents are all roughly equal strength, so you train at either 0% or 100%
win rate — no gradient either way.

Throttling them does not work: making an agent PASS on 20% of turns drops it from
**$179,514 to $312**. These are tightly-coupled plan executors, not reactive
policies — drop one turn and the unit is in the wrong place, the plant dies, the
animal starves.

Capping how many workers they can hire *does* work, because their own logic sizes
itself to the unit count:

```
hire cap    money vs passive
    3            16,049
    5            43,094
    8            80,932
 none           179,514
```

That gives a real ladder with a genuine crossover point.

**And I never ran my own best agent against it**, which I only noticed when a
commenter asked about final evaluation. The ladder had been sitting there for
weeks. Using the search-optimum vector, training nothing, 12 seeds:

```
hire cap     ours      v48     margin    wins
       3   62,556    4,229  +1379.2%   12/12
       5   52,174   27,882     +87.1%   12/12
       8   49,106   58,765     -16.4%    5/12    <- the crossover
      11   39,990  110,859     -63.9%    0/12
    none   37,673  124,276     -69.7%    0/12
```

So the ladder works, and cap 8 is a real 50/50. Note also that **my** money
barely moves across it (62.5k to 37.7k) while the opponent's moves 30x. Capping
them both weakens them and frees market for me — which is the shared-market
effect again, and a reminder that a handicap changes both sides of a shared
economy, not one.

---

## Trick 8: self-play can converge on doing nothing

This one is worth the whole post. I built a league that promoted on win rate against
a frozen snapshot of itself. It ascended through four leagues with a win rate near
1.00.

Actual money at every league: **$3,000.** Starting cash. Exactly.

The frozen opponent was making $2,588 — *worse than inaction*. So beating it
required doing nothing, and automatic promotion propagated that upward, league after
league. An external check against the fixed public agents showed real margin going
**-82.7% to -98.3%** while self-play reported winning.

Two guardrails I would now consider mandatory:

1. **Log money relative to doing nothing.** It is free to compute, and 1.0 means
   your policy is inert. Four separate collapses would have been caught in one
   glance.
2. **Keep a fixed external opponent entirely out of the training loop** and measure
   against it on a timer. Self-play money is relative; it tells you nothing
   absolute. Mine caught the collapse in 20 minutes.

---

## Trick 9: horizon is not a difficulty axis

I assumed shorter games were an easier curriculum. They are a *different game*. The
CEM-optimal worker count by horizon:

```
 5 days   0.057
 8 days   0.610
30 days   0.378
```

Non-monotonic. At 5 days a hire never repays its Fibonacci cost; at 8 it does and
you hire hard; at 30 the optimum rebalances toward livestock. A horizon specialist
beats a transplanted 30-day policy by **+31% at 5 days and +70% at 8**.

Worse, my time features were `day / N_DAYS` and `step / EPISODE_STEPS` — both
normalized by the current episode length. A 5-day game and a 30-day game produce the
*same* 0-to-1 signal. Sequential horizon training just averages incompatible
strategies over identical inputs.

Two changes are needed, and I want to be precise about which does what. Adding an
**absolute** horizon feature (`EPISODE_STEPS / 720`) makes the horizon
*identifiable*. Mixing horizons **within the same batch** rather than in sequence
makes it *worth using*. I tested the first one alone, keeping the curriculum
sequential, and got the identical -98.3% — because sequential training forgets
regardless of what the policy can see. Nothing in the gradient asks it to remember.

---

## Trick 10: your sample yield is probably terrible

```
per episode:  17,280 engine steps  ->  30 policy decisions
per update:   4 full-batch gradient steps, then the data is discarded
```

The bottleneck is not steps per second, it is how many *independent terminal
outcomes* you extract per unit of compute. Shorter episodes do not reduce engine
cost per decision — 24 turns either way — but they do double your terminals per
second, and with per-seed sd around $9,000, terminals are the scarce resource.

---

## Trick 11: sweep one axis at a time — the diagonal lies

I swept the straight line in strategy space from "do nothing" to the search
optimum, 48 seeds per point, and got a clean and dramatic picture: a flat plateau
for 95% of the path, then 1.13x -> 5.11x in a single step. A cliff. I was one
paragraph away from concluding that the reward landscape is piecewise-constant
and gradient methods are structurally blind here.

Then I swept **one axis at a time** from the same optimum, and the picture was
completely different (values are multiples of doing nothing):

```
workers    1.00 1.50 1.37 2.29 2.75 3.16 3.59 3.59 4.83 4.83 5.10 ...  10 improving steps
crop       1.00 0.39 0.39 0.39 1.04 1.04 1.08 1.08 1.03 ... 1.03 5.10  the actual cliff
tiles      0.70 1.16 4.30 5.10 4.98 ... 2.37 0.00 0.00 0.00            a cliff downward
animals    5.10 2.73 1.85 1.73 1.53 ...                                monotone worse
sell_horizon, fertilize, all 6 priorities: 5.10 flat, ONE level
```

A diagonal moves every coordinate at once, so what it draws is **a curve, not the
landscape**. Worker count has ten improving steps — there is plenty of gradient.
The cliff is one variable, the crop choice, and it is a *categorical* decision
encoded as a scalar: `viable[int(v * len(viable))]`. Crossing 0.75 flips one crop
to another and the money goes 1.139x -> 4.841x. Planting the *wrong* crop scores
0.39x, which is 61% worse than doing nothing.

My first write-up of this said the cliff was the worker count going 8 to 9,
because that is what changed a couple of grid points later. It contributes 0.26x
of the 3.97x jump. I attributed 93% of an effect to the wrong variable because I
read a one-dimensional slice of a fourteen-dimensional object.

If you take one thing from this post, consider taking this one: it is cheap, it
is mechanical, and it would have saved me a day.

---

## Trick 12: dead dials, in both directions

Covered above under "nothing guessed", stated here as a trick because it is the
cheapest check in this list:

- for every learned parameter, `grep` the global it writes and confirm something
  *reads* it;
- for every hand-set constant, confirm it is actually written by the resolver;
- then perturb it and confirm a dollar moves.

Reading the code catches neither direction. In this project one dead dial survived
months because the mode it served was deleted around it, and two live constants
survived because they had a dataclass field and no wiring, which *looks* like being
learned in every diff you read.

---

## Trick 13: a default argument is how a rename becomes a silent no-op

The per-turn selling rule reads its five weights like this:

```python
return {k: _logit(getattr(macro, "w_" + k)) for k in TURN_WEIGHTS}
```

It used to be `getattr(macro, "w_" + k, 0.5)`. With the default, renaming a field
and forgetting the key tuple makes the lookup fall back to the neutral value: the
whole per-turn rule switches itself off, **no error, no log line**, and the training
curve keeps looking plausible because 0.5 is exactly the value that reproduces the
old behaviour. It cost a bisection down to a single turn (`SELL CARROT 19` against
`23`) to find.

The fix is one character: drop the default. An `AttributeError` is the correct
outcome of a rename you did not finish.

---

## Trick 14: the code that is running is not the code on disk

A 15-minute search starts; you keep editing the executor while it runs. The workers
imported their modules when they were spawned, so they are measuring **a world that
no longer exists**, and the vector that comes out is the optimum of a version of the
code that is gone. Nothing in the output looks wrong.

Two cheap defences, both in the repo:

1. **A fingerprint of the files that decide how the game is played**, stored next to
   every vector and every checkpoint, printed at startup. Two of them, actually:
   one over the *game* files and one over the *learning* files, kept separate so a
   CEM vector stays comparable when the network changes — the game has not — while
   two checkpoints do not.
2. **Loud checkpoint loading.** The usual pattern —
   `{k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}` — silently
   drops every tensor whose shape moved and leaves it **randomly initialised**. That
   cost half a session once: the two dropped tensors were the global encoder and the
   summary vector, so per-seed money ranged from $19 to $40,667 and the *same*
   checkpoint gave different answers in different processes. Migration should be
   explicit and it should raise.

There is a corollary I hit while removing the dead dial above. If you ever **remove**
a dimension from a vector the network emits, every row after it in older checkpoints
now means something different. Growing a head is routine; shrinking one has to
delete that specific row *before* growing, or the load silently reinterprets 28
dimensions. New checkpoints now carry the field list itself, so no width is ever
guessed by position again.

---

## The meta-lesson

Of the confident diagnoses I made in one session, **five were wrong**, and
measurements killed all five:

- capping units does not speed anything up (cost scales with tiles, not units)
- shrinking exploration sigma cancels out under a KL budget
- cloning experts does not transfer
- short horizons are not easier
- my "validated" league opponent was silently crippled by a global mode flag that
  the training loop had set
- the reward landscape is a flat plateau with one needle (it is not — that was a
  diagonal slice; see Trick 11)
- I had never beaten a public agent (I had, comfortably, under the handicap I
  built myself and never used)

The habits that actually paid:

- Assert your patch matched before writing it.
- Verify by **symmetry** — run the same policy on both sides and check the wiring
  is live. I wrote here that mine tied "to the dollar"; measured properly, only
  **13 of 60 seeds tie exactly**. Identical strategies still diverge, because the
  two farms occupy different positions and the market couples them. An exact tie
  proves the wiring; a *failure* to tie proves nothing on its own.
- Log the thing that would *explain* a failure, not just the thing that *is* the
  failure.
- Never compare money measured against different opponents. The same agent varied
  2x for me depending on who it played. **In a shared-market game that is not a
  strong enough rule**: measured, `v48` earns 123k against my active policies,
  145k against inaction and 158k against the farming one. My own behaviour moves
  my *opponent's* score by 28%, because selling less product keeps their marginal
  prices high. So margin-vs-opponent is not comparable across my own arms either.
  Report absolute money against a policy-independent baseline — I use multiples
  of doing nothing.
- Write the success criterion down **before** looking at the result.

---

## Glossary

**Neurosymbolic split.** Exact arithmetic for anything derivable from the engine
(legality, routing, assignment, prices, liquidation); learning for anything that
depends on state in a way you can't write down. The test is not "is this
complicated" but "can I compute this exactly?"

**Borderless parameterization.** Encoding a learned constant as
`default * exp(logit(f))` (or the sigmoid form for fractions) instead of
`lo + (hi-lo)*f`. `f = 0.5` reproduces the default exactly and the extremes reach
the whole domain, so exposing a constant cannot change behaviour by itself and no
hand-set range is introduced along with it.

**Decision site.** Anything in the code that picks a number or an order: a
threshold, a `min` used as a ceiling, a `sorted` used as a preference, an
arithmetic scale. The unit the AST audit enumerates, because `grep` only finds the
ones that happen to be named constants.

**The per-turn rule.** The policy is called once a day, so anything it emits is
constant within the day. For selling — the one decision that really depends on the
turn — it emits the *coefficients of a rule* instead of a level, and the rule is
evaluated against each turn's state. Reactivity without 24x the forward passes.

**Macro / micro.** Macro: a daily strategy vector — 64 reals: how much of each
thing to aim for, plus the constants that used to be hand-written. Micro: a per-tile head emitting a value plus one logit per legal verb. Macro
says *how much*, micro says *what, where*.

**Hungarian assignment.** Optimal one-to-one matching of units to tiles given a
value matrix. Greedy fails in a specific way: the first unit takes a task another
was equally close to, stranding it — and unit ordering is meaningless, so that loss
is pure arbitrariness.

**Marginal price.** The price of the *k*-th unit sold, not the first. Any bulk
valuation must sum marginal prices. Getting this wrong cost me 40-66% of final
money, three separate times.

**CEM (Cross-Entropy Method).** Derivative-free search: sample N candidates, keep
the best K, refit a Gaussian to them, repeat. Good for the 14-number macro vector,
where no gradient reaches. Its ceiling is that it finds one *fixed* vector and
cannot condition on state.

**PPO.** On-policy gradient method: reuse a batch for a few epochs, then discard it,
because once the policy moves the old data is invalid. That is why sample yield
matters so much here.

**Importance ratio.** `exp(logp_new - logp_old)`. It is a *product* over sampled
dimensions, so dimensions that cannot affect the action do not cancel — they
multiply the noise.

**Critic / R2.** The value function predicting returns; its accuracy is what turns
raw returns into low-variance advantages. Negative R2 means it is worse than
predicting the mean, and PPO degenerates to REINFORCE with no baseline.

**Potential-based shaping.** Adding `gamma*Phi(s') - Phi(s)` leaves the optimal
policy unchanged for *any* Phi — but only over full episodes. Under truncation Phi
becomes the terminal value, and an optimistic Phi teaches the wrong thing.

**BC / DAgger.** Behavior cloning is supervised learning of expert actions. DAgger
is the standard fix for distribution shift: your policy drives, the expert only
labels. Both hit ~99.5% agreement here; both produced near-zero money.

**Common random numbers (paired comparison).** Run both branches from the same
cloned state, same seed, same opponent, and difference them. Seed variance cancels
instead of being averaged over — 2.2x less sd, roughly 4.8x fewer samples.

**Terminal.** One completed episode: one unbiased outcome. With per-seed sd around
$9,000, terminals are the scarce resource — not steps, not transitions.

**Self-play with a frozen snapshot.** The opponent is a copy of you from K updates
ago, so "winning" is meaningful rather than 0.5 by construction. Only if the
snapshot is better than doing nothing — mine wasn't, and the league happily promoted
a policy that sat on its starting cash.

---

Happy to go deeper on any of these in the comments.

---

## References

The techniques this post leans on, in the order they appear.

**Hungarian assignment.** Kuhn, H.W. (1955), *The Hungarian method for the
assignment problem*, Naval Research Logistics Quarterly 2(1-2).

**Common random numbers / paired simulation.** Standard variance-reduction
technique in discrete-event simulation; see Law & Kelton, *Simulation Modeling and
Analysis*, ch. 11. The idea is older than RL and underused in it.

**Cross-Entropy Method.** Rubinstein, R.Y. (1997), *Optimization of computer
simulation models with rare events*, EJOR 99(1); and De Boer, Kroese, Mannor &
Rubinstein (2005), *A tutorial on the cross-entropy method*, Annals of OR 134.

**Behavior cloning and covariate shift.** Ross, S. & Bagnell, J.A. (2010),
*Efficient reductions for imitation learning*, AISTATS. The quadratic-in-horizon
error compounding result is here.

**DAgger.** Ross, S., Gordon, G. & Bagnell, J.A. (2011), *A reduction of imitation
learning and structured prediction to no-regret online learning*, AISTATS.

**PPO.** Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017),
*Proximal policy optimization algorithms*, arXiv:1707.06347.

**Generalized advantage estimation.** Schulman, J., Moritz, P., Levine, S., Jordan,
M. & Abbeel, P. (2015), *High-dimensional continuous control using generalized
advantage estimation*, arXiv:1506.02438.

**Potential-based reward shaping.** Ng, A.Y., Harada, D. & Russell, S. (1999),
*Policy invariance under reward transformations*, ICML. The policy-invariance
guarantee — and the full-episode assumption that quietly breaks under truncation.

**League training and frozen opponents.** Vinyals, O. et al. (2019),
*Grandmaster level in StarCraft II using multi-agent reinforcement learning*,
Nature 575. The main/exploiter/league structure, and why a single frozen snapshot
is not enough.

**Catastrophic forgetting.** McCloskey, M. & Cohen, N.J. (1989), *Catastrophic
interference in connectionist networks*, Psychology of Learning and Motivation 24;
French, R.M. (1999), *Catastrophic forgetting in connectionist networks*, Trends in
Cognitive Sciences 3(4).

### Further reading, informed the design but isn't in the post

These came out of a literature pass while debugging the critic and the imitation
failures. I have not independently verified every one, so treat them as pointers
rather than endorsements.

- **Value target scaling.** van Hasselt, H., Guez, A., Hessel, M., Mnih, V. &
  Silver, D. (2016), *Learning values across many orders of magnitude* (PopArt),
  NeurIPS. Relevant because bootstrapped targets make naive normalization unsafe.
- **Categorical value heads.** Farebrother, J. et al. (2024), *Stop regressing:
  training value functions via classification for scalable deep RL*, ICML.
- **Decoupling value and policy.** Raileanu, R. & Fergus, R. (2021), *Decoupling
  value and policy for generalization in RL* (IDAAC), ICML; Cobbe, K., Hilton, J.,
  Klimov, O. & Schulman, J. (2021), *Phasic policy gradient*, ICML. Both are about
  the shared-trunk interference described in Trick 6.
- **Value-based imitation.** Garg, D., Chakraborty, S., Cai, C., Zhang, B. &
  Ermon, S. (2021), *IQ-Learn: inverse soft-Q learning for imitation*, NeurIPS;
  Al-Hafez, F. et al. (2023), *LS-IQ*, ICLR, which documents and corrects its
  reward bias.
- **Search with an exact simulator.** Silver, D. et al. (2018), *A general
  reinforcement learning algorithm that masters chess, shogi and Go through
  self-play*, Science 362; Danihelka, I., Guez, A., Schrittwieser, J. & Silver, D.
  (2022), *Policy improvement by planning with Gumbel*, ICLR. Worth reading before
  reaching for a learned world model when you already have the real one.
- **Entity-based architectures.** Vinyals et al. 2019 (above) for the transformer
  over units; Lee, J. et al. (2019), *Set Transformer*, ICML.
