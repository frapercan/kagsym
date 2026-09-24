"""Experiment tracking: the only module that imports MLflow.

Three experiments, one per kind of run:

    kagsym/training     the PPO trainer (one run per invocation)
    kagsym/search       tools/search.py (one run per search, one step per generation)
    kagsym/evaluation   band, evaluate, validate (one run per measurement)

Every run carries the same context tags, so a number can always be traced
to the code, the checkpoint and the seeds that produced it:

    kind, git.commit, git.branch, game_fingerprint, model_fingerprint,
    checkpoint, checkpoint_digest, parent_checkpoint, opponent, seed_family,
    objective, experiment (e.g. EXP-001), preregistration (path), host

Metric names are `<group>/<name>`, in English, and the group says what the
number is for. The numeric prefix orders the groups in the UI:

    1_result/    what we optimise for and what we deploy on
                 train_money, train_margin_pct, train_win_rate, x_inaction,
                 anchor_money|margin_pct|se|smoothed (sampled policy vs the
                 anchor opponent), eval_money|se|margin_pct|best
                 (deterministic, reserved seeds)
    2_policy/    is the policy moving, and how: kl_macro, kl_micro,
                 kl_per_dim, kl_target, sigma_*, ratio_saturation,
                 sd_logratio, active_dims, verb_explore_pct,
                 verb_signal_noise, macro_w_norm, micro_w_norm
    3_critic/    r2, reward_mean, jepa_sd, memory_*
    4_optim/     lr_trunk|macro|micro|heads, grad_norm, grad_zero_pct,
                 grad_sigma_*, epochs_run
    5_opponent/  who we train against and how it goes: money, level, league,
                 over_inaction, damage_<d>d_pct, rung_win|opponent|quota_XX,
                 league_*
    6_economy/   what the farm produces: hands_requested|real|saturation,
                 income_<product>, income_total, income_top2_share,
                 income_effective_variety, units_<product>, units_total
    search/      base, centre, centre_minus_base, best, population_mean,
                 sigma, seconds, opponents (per generation)
    band/ evaluate/ validate/ paired/   evaluation summaries

Tracking never breaks a run: without MLflow installed, or with
KAGSYM_MLFLOW=0, everything here is a no-op. A run is closed at process exit
even if the caller forgets, so no run is left RUNNING forever (175 were).
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import tempfile
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URI = "sqlite:///" + os.path.join(ROOT, "data", "mlflow.db")
ARTIFACTS = os.path.join(ROOT, "data", "mlartifacts")

EXPERIMENTS = {
    "training": "kagsym/training",
    "search": "kagsym/search",
    "evaluation": "kagsym/evaluation",
}
METRIC_GROUPS = {"1_result", "2_policy", "3_critic", "4_optim", "5_opponent",
                 "6_economy", "search", "band", "evaluate", "validate", "paired"}


def available() -> bool:
    if os.environ.get("KAGSYM_MLFLOW", os.environ.get("KAGWORLD_MLFLOW", "1")) == "0":
        return False
    try:
        import mlflow  # noqa: F401
        return True
    except Exception:
        return False


# -- context ------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _digest(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return ""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def context_tags(kind: str, checkpoint: str | None = None, parent: str | None = None,
                 opponent: str | None = None, seed_family: str | None = None,
                 objective: str | None = None, experiment: str | None = None,
                 preregistration: str | None = None, **extra: Any) -> dict:
    """The tags every run carries. Unknown values are left out, never guessed."""
    from .version import fingerprint, model_fingerprint
    tags = {
        "kind": kind,
        "git.commit": _git("rev-parse", "--short", "HEAD"),
        "git.branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git.dirty": "yes" if _git("status", "--porcelain") else "no",
        "game_fingerprint": fingerprint(),
        "model_fingerprint": model_fingerprint(),
        "host": socket.gethostname(),
    }
    if checkpoint:
        tags["checkpoint"] = os.path.relpath(os.path.abspath(checkpoint), ROOT)
        tags["checkpoint_digest"] = _digest(checkpoint)
    if parent:
        tags["parent_checkpoint"] = os.path.relpath(os.path.abspath(parent), ROOT)
    if opponent:
        tags["opponent"] = opponent
    if seed_family:
        tags["seed_family"] = seed_family
    if objective:
        tags["objective"] = objective
    exp = experiment or os.environ.get("KAGSYM_EXPERIMENT")
    if exp:
        tags["experiment"] = exp
    pre = preregistration or os.environ.get("KAGSYM_PREREGISTRATION")
    if pre:
        tags["preregistration"] = pre
    tags.update({k: str(v) for k, v in extra.items() if v is not None})
    return tags


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v[:12]))[:500]
        else:
            out[key] = v if v is not None else "None"
    return out


# -- the tracker ------------------------------------------------------------------

class Tracker:
    """One MLflow run. Use as a context manager or call start()/end()."""

    def __init__(self, kind: str, name: str, params: dict | None = None,
                 tags: dict | None = None, description: str | None = None,
                 enabled: bool | None = None, uri: str | None = None):
        if kind not in EXPERIMENTS:
            raise ValueError(f"unknown run kind {kind!r}; known: {sorted(EXPERIMENTS)}")
        self.kind, self.name = kind, name
        self.enabled = available() if enabled is None else (enabled and available())
        self._params, self._tags, self._description = params or {}, tags or {}, description
        self._uri = uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI)
        self._mlflow = None
        self._run = None
        self.run_id: str | None = None
        self._ended = False

    # lifecycle -----------------------------------------------------------------
    def start(self) -> "Tracker":
        if not self.enabled:
            return self
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            os.makedirs(ARTIFACTS, exist_ok=True)
            import mlflow
            mlflow.set_tracking_uri(self._uri)
            exp_name = EXPERIMENTS[self.kind]
            if mlflow.get_experiment_by_name(exp_name) is None:
                mlflow.create_experiment(exp_name, artifact_location=ARTIFACTS)
            mlflow.set_experiment(exp_name)
            self._mlflow = mlflow
            self._run = mlflow.start_run(run_name=self.name, description=self._description)
            self.run_id = self._run.info.run_id
            if self._tags:
                self.set_tags({"kind": self.kind, **self._tags})
            if self._params:
                self.log_params(self._params)
            atexit.register(self._end_at_exit)
            print(f"[mlflow] {exp_name} / {self.name}  run {self.run_id[:8]}", flush=True)
        except Exception as e:
            # Never silent: a NameError once switched tracking off for two runs
            # and nobody noticed until the database was queried by hand.
            print(f"[mlflow] DISABLED: {type(e).__name__}: {e}", flush=True)
            self._mlflow, self.enabled = None, False
        return self

    def end(self, status: str = "FINISHED") -> None:
        if self._mlflow is None or self._ended:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as e:
            print(f"[mlflow] could not end run: {e}", flush=True)
        self._ended = True

    def _end_at_exit(self) -> None:
        # An unhandled exception leaves sys.last_type set in interactive
        # sessions only; for scripts, exit status is what we have.
        self.end("FINISHED" if getattr(sys, "last_type", None) is None else "FAILED")

    def __enter__(self) -> "Tracker":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.end("FAILED" if exc_type is not None else "FINISHED")
        return False

    # logging ---------------------------------------------------------------
    def log(self, metrics: dict, step: int | None = None) -> None:
        if self._mlflow is None:
            return
        clean = {}
        for k, v in metrics.items():
            if not isinstance(v, (int, float)) or v != v:
                continue
            group = k.split("/", 1)[0]
            if group not in METRIC_GROUPS:
                print(f"[mlflow] metric {k!r} is outside the schema; see kagsym/tracking.py",
                      flush=True)
            clean[k] = float(v)
        if clean:
            try:
                self._mlflow.log_metrics(clean, step=step)
            except Exception as e:
                print(f"[mlflow] could not log metrics: {e}", flush=True)

    log_metrics = log

    def log_params(self, params: dict) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_params({k: str(v)[:500] for k, v in _flatten(params).items()})
        except Exception as e:
            print(f"[mlflow] could not log params: {e}", flush=True)

    def set_tags(self, tags: dict) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})
        except Exception as e:
            print(f"[mlflow] could not set tags: {e}", flush=True)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        if self._mlflow is None or not os.path.exists(path):
            return
        try:
            self._mlflow.log_artifact(path, artifact_path)
        except Exception as e:
            print(f"[mlflow] could not upload {path}: {e}", flush=True)

    def log_json(self, obj: Any, filename: str) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_dict(obj, filename)
        except Exception as e:
            print(f"[mlflow] could not upload {filename}: {e}", flush=True)

    def log_text(self, text: str, filename: str) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_text(text, filename)
        except Exception as e:
            print(f"[mlflow] could not upload {filename}: {e}", flush=True)

    def log_table(self, rows: list[dict], filename: str) -> None:
        """A list of dicts as a CSV artifact (per-opponent tables, histories)."""
        if self._mlflow is None or not rows:
            return
        import csv
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        try:
            self._mlflow.log_artifact(path, os.path.dirname(filename) or None)
        except Exception as e:
            print(f"[mlflow] could not upload {filename}: {e}", flush=True)
        finally:
            os.unlink(path)


def describe(lines: list[str]) -> str:
    """A run description: what was run, against whom, why; shown in the UI."""
    return "\n".join(lines)


def ui_command() -> str:
    return f"mlflow ui --backend-store-uri {DEFAULT_URI} --host 127.0.0.1 --port 5000"
