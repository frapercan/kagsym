# Procedure: from a checkpoint to a validated, deployable one

Every number here is measured. Anything that did not survive a re-measurement
is marked. The instrument is `kagsym/evaluate.py`; the criterion is
`tools/band.py`; the seed families are `kagsym/seeds.py`.

## 0. Gates, before any long run (two minutes)

```
python tools/check_submission.py runs/model.pt --n 6     # 6/6 identical, or stop
python -m pytest kagsym/tests -q                          # engine, assignment, policy identity
```

If the submission wrapper and the evaluator differ on any seed, stop: the
measurement loop is not measuring the agent that plays, and every number it
produces describes another agent. This gate has been missing twice at a cost
of a night each.

## 1. Where the deployed policy stands (2026-09-24)

`runs/partida_v5.pt` (ramp offset baked, 49 live dials), band on reserved
seeds, 6 seeds x 2 seats x 74 opponents:

```
win rate mean            0.064
opponents beaten         5 of 74   (harvest-pulse, adaptive-land, melon-first, rank-your-agent, same-file)
money vs v48             51,304 +- 614
```

Against ~65 real agents the win rate is 0.00 on every board; they finish at
115,000-135,000 $ and we finish at 40,000-67,000 $.

Note on the previous bar: "win 0.077, 6 of 74" was measured with a loop that
ignored the ramp stored in the checkpoint and counted `shop-router-0909`
(throws on 719 of 720 turns, plays PASS, finishes at $3,000) as a win.

## 2. The loop

Every stage is a separate tool; `tools/pipeline.py` runs the whole chain in
one launch with a gate between stages (`--smoke` in minutes to prove the
chain, `--full` for the sizes below).

**Step 1 - live dials.** Which macro dials change the game for this
checkpoint. The list is written next to the checkpoint with its digest and
refused for any other checkpoint.

```
python tools/live_dials.py runs/model.pt --n 3      # -> runs/model.pt.dials.json
```

**Step 2 - search.** CEM over the offset of the live dials, starting from the
offset the checkpoint carries, into a directory that must not exist. The
search freezes a copy of the checkpoint, writes `meta.json` first (digest,
dials, provenance) and `done.json` only at the end; `offset.npz` carries the
vector with its dial map, ramp flag and digest.

```
python tools/search.py runs/model.pt --out runs/search/name --gens 40 --experiment EXP-00x
python tools/search.py runs/model.pt --out runs/search/name_ramp --ramp --gens 60
```

Objective `margin` (ours minus theirs against a sample of the band, both
seats) is the default: a win rate of 0.00 everywhere has no gradient, a
margin does, and it rewards sinking their market as much as growing ours.
`money` against v48 is the old objective; `win` is the criterion itself.

Reading the log: CENTRE minus BASE on the same seeds is the only informative
column. "best" is the expected maximum of ~30 noisy draws and is never a
result. The mean of the population is not the centre. A candidate with any
failed episode is disqualified (`dq` column), never averaged.

**Step 3 - validate and bake.** The optimum on the search seeds is a
hypothesis. It is measured paired with two instruments, and baked only if
the gate passes: finite t, every board played, non-degenerate, money t >= 2
and the band win-rate difference not negative. The gate is written
positively: NaN refuses.

```
python tools/validate_offset.py runs/search/name runs/model.pt --family clean --n 200 --band-n 6 --bake runs/new.pt
```

**Step 4 - the criterion.** The new checkpoint against the band, paired
against the previous one.

```
python tools/band.py runs/new.pt --against runs/model.pt --n 6
```

A checkpoint that gains money and not wins has not gained anything.

**Step 5 - deploy.** Copy to `submit_kagsym/model.pt`, run the gate again,
package and play under the real Kaggle runner:

```
python tools/check_submission.py submit_kagsym/model.pt --n 6
python tools/package_submission.py
```

Every measurement lands in `runs/ledger.jsonl` (`python tools/ledger.py`)
and in MLflow (`docs/MLFLOW.md`) with the commit, whether the tree was dirty,
the game fingerprint, the checkpoint digest and the seed family.

## 3. Sample sizes, fixed before looking

Money per seed has sd 15,000-22,000 $; paired differences ~9,000-13,000 $.

```
effect to resolve (paired money)     seeds
        20%                          ~15
        10%                          ~55
         5%                         ~220
```

Fewer than 100 paired seeds decide nothing about money. The band with 6
seeds per opponent resolves win-rate differences of ~0.1 and nothing finer;
it is the criterion, not a fine instrument. Never compare numbers from
different seed families. A reserved family used to validate a round is
partly spent; a new round validates on a different offset within the family
or on `clean`.

## 4. Traps, each measured and each paid for

- **The opponent supply history.** Feeding zeros instead of `rival_flow`
  mutilates the agent: same board, $76,607 became $25,335. The policy feeds it.
- **`_destinations`.** The observation encoder needs the executor's
  destinations; without them it is a different agent. The policy passes them.
- **Torch threads.** Same seed, $42,242 with 12 threads and $48,387 with one.
  `load_network` pins one thread; nothing else may change it.
- **`hands` is always 0 at hour 0.** Never sample or decide there.
- **Modules freeze at import.** A live run does not execute the code on disk.
  The ledger records the commit; compare with `git log`.
- **`pkill -f <pattern>` matches its own command line** and kills the shell.
  Use explicit PIDs.
- **The population mean is not the centre** of a CEM. `mu` is evaluated as a
  candidate so the intermediate reading measures what is validated.
- **A blind instrument says "no effect" in the same words as a real null.**
  An evaluator that ignored the ramp returned +0 +- 0, and `nan < 2.0` is
  False, so the bake gate accepted it. Every evaluator goes through `Policy`,
  `paired` flags a degenerate comparison, and the gate requires a finite t.
- **Global files are shared state.** A single `live_dials.json` let one probe
  redefine the search space of every later search. Dial lists live next to
  their checkpoint with its digest; searches freeze their checkpoint and
  write self-describing artefacts.
- **`KAG_*` environment variables change the agent.** None is set on Kaggle.
  The gate refuses to run with any set, and every ledger line lists them.
- **Nothing seeded torch.** `--seed` seeds the trainer and every worker;
  `--seed0` only seeds the boards.
- **A broken opponent is a weak opponent.** Exceptions are counted, never
  swallowed, and an opponent that throws on most turns is excluded.
- **Order matters on Kaggle.** Only two submissions count; the second
  replaces the first in ranking, so deploy the validated one last.

## 5. Refuted with measurement; do not reopen without new data

Checkpoint soups (-7,893, t -4.89). SWA. ASP planning. Narrowing the policy
(entropy falling is an alarm, not convergence). Sampled-versus-deterministic
mismatch (both fall the same). Elo as a criterion (988 -> 1730 while real
performance fell). PPO from any starting point. The rival's money as dense
shaping (it carries 99% of the reward variance and sinks the critic).
Cloning a public agent's trajectory (100% agreement on its trajectory =
13.9% of its money).

## 6. Where the gap is, measured

```
tiles under crop         ours 10.2 mean     v48 42.1
days 0-1                 HIRE 12 / BUY_ANIMAL 7 / BUY_SEED 1     v48: 5 / 1 / 7
money on day 10          ours $10           v48 $4,738
```

Both go broke on day 5; the error is not how much is spent but on what.
Seed compounds (wheat x1.78 per day), staff and livestock do not. It is not
the number of hands (8.6 vs 9.1), not routing (v48 moves more than we do), and
not `sustainable_tiles` (never binds). What we have in excess is PASS: 21.4%
of unit-turns against 4.9%.

## 7. What the ramp of partida_v5 is worth on the criterion

Paired on the band (74 opponents x 6 seeds x 2 seats = 900 boards), the ramp
stored in `partida_v5.pt` against the same weights without it:

```
money        +3,350 +- 559   t +6.0    better on 57% of boards
win rate     -0.007
```

Money that the criterion does not see. This is the measurement that closes
the money phase: an offset validated on dollars against v48 is not evidence
about the ladder.

## 8. The ladder of universes

A universe is the first `k` days of a 30-day game: the episode is cut at day
`k`, our policy plays as if 30 days remained (`--agent-horizon 30`), and the
position at the cut is valued with the full horizon (exact liquidation:
cash, shed, standing crops and animals that have time to pay). It is not a
shorter game: in an honest 8-day game nothing pays and the optimal policy is
inaction, which is how an earlier ladder promoted inert policies.

Each rung has a fixed, external yardstick: the position value of the public
agents at day `k` (they are schedules; they do the same thing every time).
Measured on 2026-09-24 at day 8, partida_v5 against v48: 11,000-12,500 $
against 25,000-28,000 $; against the 2945 expert: 10,500 $ against 33,000 $.

```
rung   days   gate to climb
  1      8    position value / v48's >= 0.6 on 30 reserved seeds, paired vs the previous rung
  2     14    same, >= 0.6
  3     20    same, >= 0.6
  4     30    the criterion: band win rate, paired vs the previous checkpoint
```

Rules that come from what already failed:

- Promotion is by the rung's external yardstick, never by self-play return
  or Elo (both rise while real strength falls).
- The cheap mechanism first at every rung (an opening offset searched by
  `tools/search.py --days k --until-day k --objective value`), the gradient
  only if the search plateaus, and every mechanism validated at full scale
  (`tools/validate_offset.py` plays the whole game) before the next rung
  starts from it.
- The public agents survive reduced episodes (0 failures at 8 days), so a
  rung is played against them, not only against ourselves; the mirror test
  (a policy against itself reads 0.5) stays in the test suite as the null
  case.
- The last rung is the only place where the band decides.

**Training never runs on a short calendar.** Measured by the pipeline's
progress gate: eight updates on an honest 24h x 8d game left the policy
14,066 $ worse (t -10, worse on every board) at the full game than after one
update. In a short game nothing pays and the gradient learns inaction. The
universes are for search, validated at full scale.

## The ladder protocol (`tools/ladder.py`)

The reduced worlds are climbed one day at a time, 1 to 30, with the code
frozen for the whole climb (the report carries the commit; if the search or
the executor changes, the ladder starts again at 1). For every horizon:

1. **Search** on seeds 7101-7105: the constant-plan grid, then day-by-day
   refinement with exact rollouts (`tools/plan_daysearch.py`). Every
   decision is recorded: state at the day boundary, the day's plan, money.
2. **Check** the searched schedule on five seeds the search never saw
   (7106-7110). This is the number reported.
3. **Audit** the plan's execution (`tools/plan_audit.py`): produced, sold,
   lost (tiles, hands, shed overflow), crew turns split into work, needed
   moves, excess, orphan and pass.
4. **Compare** with the previous rung's schedule padded by one day on the
   same unseen seeds: what the extra day buys, and which plan fields
   changed. That is the decision the new day adds.
5. **Write** a section in `runs/ladder/report.md` and a row in
   `runs/ladder/ladder.jsonl` before starting the next rung.

The records of the climb are the dataset of the plan head (EXP-008),
which trains only after the ladder is complete.
