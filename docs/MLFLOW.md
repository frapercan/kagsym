# MLflow: how runs are organised and how to read them

Tracking database: `data/mlflow.db` (SQLite). Artifacts: `data/mlartifacts/`.

```
.venv312/bin/mlflow ui --backend-store-uri sqlite:///data/mlflow.db --host 127.0.0.1 --port 5000
```

`kagsym/tracking.py` is the only module that imports MLflow. Everything that
reports a number goes through it: the trainer, `tools/search.py`, and every
evaluation recorded by `kagsym.evaluate.record` (band, evaluate, validate).
Without MLflow installed, or with `KAGSYM_MLFLOW=0`, tracking is a no-op and
the ledger (`runs/ledger.jsonl`) still records evaluations.

## Experiments

| experiment | one run per | steps are |
|---|---|---|
| `kagsym/training` | trainer invocation | PPO updates |
| `kagsym/search` | search (`tools/search.py`) | CEM generations |
| `kagsym/evaluation` | measurement (band, evaluate, validate) | none (one point) |
| `legacy (mixed schema, before 2026-09-24)` | earlier runs | mixed Spanish/English keys, no context tags; history only |

## Context tags on every run

`kind`, `git.commit`, `git.branch`, `git.dirty`, `game_fingerprint`,
`model_fingerprint`, `checkpoint`, `checkpoint_digest`, `parent_checkpoint`,
`opponent`, `seed_family`, `objective`, `experiment` (preregistration id such
as `EXP-001`, from `--experiment` or `KAGSYM_EXPERIMENT`), `preregistration`
(path), `host`. Runs with `git.dirty = yes` were produced by code that is not
in any commit. Two runs with different `game_fingerprint` played different
games and their numbers are not comparable.

Every run also has a description (the "note" field) saying what was run,
against whom, and which metric to read.

## Metric schema

The prefix orders the groups in the UI; the group says what the number is for.

| group | contents | how to read |
|---|---|---|
| `1_result/` | `eval_money`, `eval_se`, `eval_margin_pct`, `eval_best` (deterministic policy, reserved seeds, vs uncapped v48); `anchor_*` (sampled policy vs the anchor worker's opponent); `train_money`, `train_margin_pct`, `train_win_rate`, `x_inaction` (sampled policy on training boards) | `eval_*` is level. `train_*` and `anchor_*` are not comparable across runs: the opponent and the boards differ. Never select a checkpoint by any of them (docs/PROCEDURE.md). |
| `2_policy/` | `kl_macro`, `kl_micro`, `kl_per_dim`, `kl_target`, `sigma_macro`, `sigma_micro_value`, `sigma_micro_verb`, `sigma_floor`, `ratio_saturation`, `sd_logratio`, `active_dims`, `verb_explore_pct`, `verb_signal_noise`, `macro_w_norm`, `micro_w_norm` | Is the policy moving. `kl_macro` at 5e-4 means the strategy head is frozen. `ratio_saturation` near 1 means the PPO clip removed the gradient. `sigma_*` falling is an alarm, not convergence. |
| `3_critic/` | `r2`, `reward_mean`, `jepa_sd`, `memory_*` | `r2` below 0.6 means the advantage is noise. |
| `4_optim/` | `lr_trunk`, `lr_macro`, `lr_micro`, `lr_heads`, `grad_norm`, `grad_zero_pct`, `grad_sigma_*`, `epochs_run` | The KL controller's effect. `lr_macro == lr_micro` pinned at the floor is the strangling case. |
| `5_opponent/` | `money`, `level`, `league`, `damage_<d>d_pct`, `rung_win_XX`, `rung_opponent_XX`, `rung_quota_XX`, `league_*` | Who the sampled policy plays and how it goes. In self-play money is relative. |
| `6_economy/` | `hands_requested`, `hands_real`, `hands_saturation`, `income_<product>`, `income_total`, `income_top2_share`, `income_effective_variety`, `units_<product>`, `units_total` | Where the money comes from. |
| `search/` | `base`, `centre`, `centre_minus_base`, `best`, `population_mean`, `sigma`, `disqualified`, `seconds` | Only `centre_minus_base` is informative: paired on the same boards. `best` is the expected maximum of noisy draws. |
| `band/`, `evaluate/` | `win_mean`, `beaten`, `contested`, `opponents`, `money_mean`, `money_se`, `opp_failures`, `our_failures`, `broken_opponents`, `episodes` | `band/win_mean` and `band/beaten` are the criterion. |
| `paired/` | `money_diff`, `money_se`, `t`, `win_diff`, `boards_better`, `n`, `n_expected`, `degenerate` | Validation against the checkpoint's own offset. `n < n_expected` means boards were dropped; `degenerate` means the two sides were the same policy. |

Evaluation runs also carry artifacts: `per_opponent.csv` (win and money per
opponent), `episodes.json` (every board), `ledger_entry.json`. Search runs
carry `offset.npz`, `history.json` and `meta.json`.

## Reading rules

- Compare only within a seed family and only with the same
  `game_fingerprint`. The ledger (`python tools/ledger.py`) prints both.
- A money curve is read against the references logged as params:
  inaction 3,000 $, legal random 8,692 $, v48 uncapped against us ~135,000 $,
  the public expert against a passive opponent 180,186 $.
- Training curves are the sampled policy on training boards; the deployed
  policy is the mean. The calibration between the two is not stable.
- A run left `RUNNING` is a process that died; the tracker closes its run
  at exit, and the 175 legacy zombies were closed on 2026-09-24.
