# Update (2026-09-24): the score is wins, not money, and what one measurement loop found

*Plain-English update to "Neurosymbolic RL for Kaggriculture: what actually moved
the needle". A commenter said the original was hard to read; this one leads with
the numbers and keeps the vocabulary to what the game itself uses. Code:
https://github.com/frapercan/kagsym (MIT, `main`).*

## The one number that matters

The leaderboard fits a Bradley-Terry model over **wins**. Money is not the score.
I measured what money buys in wins by playing my agent against every public agent
I could download (74 that load), on both seats, on fixed seeds:

```
                           win rate   opponents beaten   money vs v48
current agent (v5)           0.064        5 of 74          51,300 $
same weights, no ramp        0.071        5 of 74          48,000 $
```

The "ramp" is a per-day offset on the strategy vector that a search found and that
is worth **+3,350 $ on 900 paired boards (t = 6.0)**. On the score it is worth
**-0.007**. Two days of money optimisation, measured on the criterion, bought
nothing. Against about 65 real agents the win rate is 0.00 on every board: they
finish at 115,000-135,000 $ and I finish at 40,000-67,000 $. That gap is the
research problem. Dial tuning is not.

## Trick 15: build one loop, and make the upload drive it

Every number in the original post came from a measurement loop written for the
tool that needed it: the trainer had one, the yardstick another, the search a
third, the submission a fourth. They diverged three times in one day, and each
time the number looked healthy and described a different agent:

- the search fed zeros where the agent feeds the opponent's supply forecast
  (same board: 76,607 $ became 25,335 $);
- a yardstick split the network's output by hand instead of with the shared
  function;
- an evaluator ignored the ramp stored in the checkpoint and reported the ramp
  as worth `+0 +- 0`. A blind instrument says "no effect" in the same words as
  a real null.

The fix is structural, not a review: there is now **one class that turns a
checkpoint into actions** (`kagsym/policy.py`), the Kaggle wrapper is twenty
lines that call it, and every evaluator drives the same class. A gate
(`tools/check_submission.py`) loads `main.py` the way Kaggle does (compile +
exec, last callable) and requires identity **to the dollar** with the evaluator
before any long measurement. A test walks the syntax tree of the repository and
fails if any file reads the network's output on its own.

The gate paid for itself on its first run. Same seed, same weights:

```
torch with 12 threads     42,242 $
torch with 1 thread       48,387 $
```

Floating-point summation order moves the strategy vector by about 1e-7 and the
symbolic executor thresholds it. The thread count is now pinned in the loader.
If your agent has any threshold downstream of a network, check this.

## Trick 16: a broken opponent is a weak opponent

`shop-router-0909` throws an exception on 719 of 720 turns in my harness (a
missing data file), so the engine plays PASS for it and it finishes at 3,000 $.
My old evaluator counted that as a 1.00 win and it was one of the "6 opponents
beaten". The new one counts opponent exceptions per turn, excludes any opponent
that fails on most turns, and lists it. Exceptions on my side raise; they are
never relabelled as "the opponent did not load".

## Trick 17: seeds have families, or your reserved set is not reserved

Three evaluators used three seed ranges and disagreed by 6,800 $ on the same two
checkpoints. Training seeds started at 1 and advanced into the "reserved"
evaluation range after about 645 updates, so late checkpoints were selected on
boards they had trained on. A search ran on seeds 9000-9959 while its
"independent" contrast defaulted to 9001-9200.

Now every seed comes from a named family (`kagsym/seeds.py`): reserved
7101-7300, search 9000-9999, clean 30000-30999, training 100000+. They are
disjoint by construction, a number is only compared with a number from the same
family, and every evaluation is appended to a ledger with the commit, the
checkpoint digest and the family it used.

## Why PPO did not learn here, measured

The original post said PPO did not improve any checkpoint. Now I can say why,
from the training curves and the code, and none of it is a bug in the PPO
mechanics:

1. **The strategy head could not move.** Its learning rate was tied to the
   tile-map head's by the KL controller ("the macro may go as fast as the micro,
   never faster"); the tile map overshot the KL target and both were throttled
   eight-fold. The strategy head ran at a KL of 5e-4 per update in 67 dimensions,
   about 0.001 per update in logit space, while the search finds value in
   offsets of 0.3. Its exploration width never left its initial value in 2,600
   updates.
2. **The PPO ratio was joint over ~450 dimensions.** Its log had a standard
   deviation of 10-25 against a clip of 0.2, so after the first epoch nearly
   every sample was clipped and contributed nothing.
3. **Eleven samples per update.** One exploration draw per episode, eleven
   episodes per update: the 67-dimensional strategy head got eleven samples per
   step. A CEM generation sees 512.

The auxiliary heads (opponent model, JEPA predictor) never received a gradient
in any checkpoint. The test for that does not need a fresh initialisation: a
parameter with no Adam state in the optimiser has never had a gradient.

## What is next

The gap is production (10 tiles under crop against 42) and the opening (day 0-1
capital goes to staff and livestock instead of seed, which compounds).

I tried the cheap thing first: a search on the strategy offset with the
opponent's **margin** as objective (ours minus theirs against samples of the
public band, both seats), preregistered as EXP-001. Seven generations in, the
centre of the search sat 3,200 +- 2,600 $ below its starting point and the
population 12,000 $ below it in every generation; the opponent sample alone
moved the baseline by 22,000 $ between generations. With a 60,000-80,000 $ gap
per board, a margin gain of a few thousand dollars cannot buy a win, so I
stopped it on a power reading and wrote that down as a deviation from its own
rule. The offset space cannot buy wins at this scale. The next experiments go
to the executor, trained fast on a reduced calendar (24 hours a day, fewer
days) against itself, and measured at full scale against the band.

Repository state: `main` is the rebuilt tree (one policy, one evaluator, seed
families, English docs, tests). The earlier Spanish state documents are in
`docs/archive/`, and `docs/DEBT.md` lists what is known to be wrong and why it
waits.
