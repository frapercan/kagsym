# kagsym

Neurosymbolic agent for [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
(Kaggle x Google): two players, 720 turns (30 days of 24 hours), a 10x10 board
and a shared market where selling sinks the price for both.

`SPECIFICATION.md` is the engine derived exactly: action space, preconditions,
reward, and the exactness traps already verified. `docs/PROCEDURE.md` is how a
result is produced and validated. `docs/archive/` holds the earlier state
documents (Spanish); they are history, not instructions.

## The criterion

The ladder fits Bradley-Terry over **wins** against the other submissions.
Money is not the score. Measured: 25% more money against the strongest public
agent bought two points of win rate.

```
python tools/band.py runs/model.pt --n 6      # the criterion
python tools/evaluate.py runs/A.pt runs/B.pt  # money vs one opponent, paired (diagnostic)
```

`tools/band.py` plays the checkpoint against every public agent in
`agents_pub/`, both seats, on reserved seeds, and reports the mean win rate
and how many opponents it beats. Opponents that fail to load or that throw on
most turns are excluded and listed: a broken opponent is a weak opponent and
a 1.00 against it is not a win the ladder would give.

State of the deployed policy (`runs/partida_v5.pt`, 2026-09-24, 6 seeds x 2
seats x 74 opponents):

```
win rate mean        0.064
opponents beaten     5 of 74
money vs v48         ~51,000 $     (they make 115,000-135,000 against us)
```

We lose every board against ~65 real agents by 60,000-80,000 $. That gap,
not dial tuning, is the research problem.

## One policy, one evaluator

Two nights were lost to measurement loops that were not the agent being
uploaded. There is now one definition of each thing:

| what | where | used by |
|---|---|---|
| checkpoint -> actions | `kagsym/policy.py` | the submission wrapper and every evaluator |
| play episodes, summarise, record | `kagsym/evaluate.py` | every tool that reports a number |
| seed families | `kagsym/seeds.py` | every tool that picks a seed |
| the gate | `tools/check_submission.py` | before any long measurement |

The gate loads `submit_kagsym/main.py` the way Kaggle does (compile + exec,
last callable) and requires identity to the dollar with the evaluator. It
found, on its first run, that the network is sensitive to the torch thread
count: the same seed gave $42,242 with 12 threads and $48,387 with one.

Seed families are disjoint by construction and a number is only comparable
with a number from the same family:

| family | range | purpose |
|---|---|---|
| reserved | 7101-7300 | validate what a search produced |
| search | 9000-9999 | what optimisers see; never a result |
| clean | 30000-30999 | second, independent instrument |
| matrix / dials | 7401-7600 | own-checkpoint duels, live-dial probes |
| training | 100000+ | what the trainer consumes |

Every evaluation appends a line to `runs/ledger.jsonl` (commit, checkpoint
digest, offset, seed family, summary). `python tools/ledger.py` prints it.

## What decides what

| | who decides | why |
|---|---|---|
| legality | the engine clone (`kagsym/fastenv.py`) | it is a fact |
| unit assignment | Hungarian (`kagsym/symbolic/assignment.py`) | the exact optimum in 0.3 ms |
| what a task is worth, how much of each thing | the network, once a day | it is preference |
| the operating point of the macro | search (`tools/search.py`) | the gradient cannot move it |

The network emits, once a day, a macro vector (targets and priorities, read by
the executor through `kagsym/macro.py`) and a 10x10 micro map (a value channel
and verb logits per tile). The symbolic layer plays the 24 hours with that.

The macro **offset** is an additive vector in logit space on the live dials,
constant or a ramp over the episode (`a + b * progress`). A constant is baked
into the bias of the macro head; a ramp travels in the checkpoint as `offset`
and the policy applies it. `tools/live_dials.py` decides which dials are live:
18 of the 67 produce identical episodes at both extremes.

## Why search and not PPO

PPO is mechanically correct here and does not improve any checkpoint
(-$6,787 from the best one, +$904 from one $10,000 worse). Three measured
mechanisms explain it: the macro head moves at a KL of 5e-4 per update
because its learning rate is tied to the micro head's and the KL controller
throttles both; the PPO ratio is joint over ~450 dimensions and saturates the
clip after the first epoch; and one exploration draw per episode with 11
episodes per update gives the 67-dim macro 11 samples per step. A CEM
generation sees 512. Training remains for leaving random initialisation
(`kagsym/cli/train.py`); everything after that is search and validation.

## Layout

```
kagsym/
  spec.py, fastenv.py        engine facts and the exact clone (38,000 steps/s)
  symbolic/                  executor, tasks, market operations, assignment
  macro.py, obs.py           the macro vector and the observation encoding
  nets/                      the network (encoder + heads)
  policy.py                  the deployed policy, defined once
  plan.py                    an explicit per-day plan the executor obeys literally
  evaluate.py, seeds.py      the evaluator and the seed families
  migrate_ckpt.py, version.py checkpoint migration and code fingerprints
  environment.py, parallel_env.py, reward.py, cli/train.py   training
  tests/                     engine equivalence, assignment, seeds, policy identity
tools/
  band.py, evaluate.py       the criterion and the money diagnostic
  check_submission.py        the gate
  search.py, live_dials.py, validate_offset.py   the search loop
  matrix.py, quick_eval.py   own-checkpoint duels; the trainer's log line
  ledger.py                  the record of every evaluation
  audit_decisions.py         AST walk over every decision site in the executor
  plan_search.py, regret.py  brute-force plan search; rollout oracles and regret
  download_ladder.py, public_ladder.py           ladder data
docs/
  PROCEDURE.md, DEBT.md      how a result is produced; what is known to be wrong
  experiments/               one preregistration per experiment, written before launch
  archive/                   earlier state documents (Spanish), history only
submit_kagsym/main.py        the Kaggle wrapper (plus model.pt, not versioned)
```

Working disk (not versioned): `runs/` (checkpoints, ledger, search outputs),
`agents_pub/` and `data/` (public agents and ladder data), `archivo/` (earlier
iterations).

## One launch, from random weights to a packaged checkpoint

```
python tools/pipeline.py --smoke --out runs/pipeline/smoke      # ~7 minutes
python tools/pipeline.py --full  --out runs/pipeline/<name> --experiment EXP-00x
```

The pipeline is the tutorial and the minesweeper at once: tests, training
from random weights on a reduced calendar (24 hours a day, 8 or 14 days)
against itself, the submission gate, live dials, search, validation with a
positive gate, the band criterion, and packaging played under the real
Kaggle runner. Every stage is a gate; the run stops at the first failure with
the stage's log, and its report is `report.json` in the output directory.
Training is reduced because the public agents die outside 24h x 30d and a
number measured in the reduced world describes another game: measurement
stages always run at full scale.

## Setup

```
python3.12 -m venv .venv312 && .venv312/bin/pip install -r requirements.txt
.venv312/bin/python -m pytest kagsym/tests -q
```

The engine equivalence test needs `kaggle-environments==1.32.7`, the version
that reproduces the ladder to the cent.
