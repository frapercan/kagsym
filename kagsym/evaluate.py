"""The one evaluation loop: a policy against named opponents on named seeds.

Every tool that plays episodes for a number goes through `play` and `run`.
The loop drives `kagsym.policy.Policy`, i.e. the same code the submission
runs, so a number here describes the agent that is uploaded. The submission
wrapper is checked against this loop at the dollar by `tools/check_submission.py`.

What is recorded per episode: opponent, seed, seat, our money, their money,
win (1, 0.5 on a tie, 0), and how many turns the opponent threw an exception
and was played as PASS. A broken opponent is a weak opponent, so that count is
never hidden: an evaluation with opponent failures is flagged in its summary.

Failures on OUR side propagate: a wide `try` that relabels our crash as "the
opponent did not load" removes us from the denominator, and that shape of bug
already cost a night.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from . import seeds as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(ROOT, "agents_pub")
LEDGER = os.path.join(ROOT, "runs", "ledger.jsonl")

HOURS, DAYS, CASH = 24, 30, 3000
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


# -- opponents ---------------------------------------------------------------

@dataclass(frozen=True)
class Opponent:
    """A picklable description of who we play against.

    kind: "public" (a file in agents_pub/, optionally hand-capped),
          "passive" (always PASS), "checkpoint" (one of our own policies).
    """
    kind: str
    name: str = ""
    cap: int | None = None

    @property
    def label(self) -> str:
        if self.kind == "public":
            return self.name if self.cap is None else f"{self.name}@cap{self.cap}"
        if self.kind == "checkpoint":
            return os.path.basename(self.name)
        return self.kind


def public(name: str, cap: int | None = None) -> Opponent:
    return Opponent("public", name, cap)


def passive() -> Opponent:
    return Opponent("passive")


def checkpoint(path: str) -> Opponent:
    return Opponent("checkpoint", os.path.abspath(path))


def public_names() -> list[str]:
    return sorted(f[:-3] for f in os.listdir(PUBLIC_DIR) if f.endswith(".py"))


def band(limit: int | None = None) -> list[Opponent]:
    """Every public agent, uncapped: the pool the ladder scores us against."""
    names = public_names()
    return [public(n) for n in (names[:limit] if limit else names)]


V48 = public("v48-fast-routes")

_OPPONENT_CACHE: dict = {}


def _opponent_callable(opp: Opponent):
    """obs -> action. Cached per process; public modules are heavy to load."""
    key = (opp.kind, opp.name, opp.cap) if opp.kind != "checkpoint" else ("checkpoint", _file_key(opp.name), opp.cap)
    fn = _OPPONENT_CACHE.get(key)
    if fn is not None:
        return fn
    if opp.kind == "passive":
        fn = lambda obs: dict(PASS_ACTION)
    elif opp.kind == "public":
        from .environment import load_public, public_with_cap
        fn = load_public(opp.name) if opp.cap is None else public_with_cap(opp.name, opp.cap)
    elif opp.kind == "checkpoint":
        from .policy import Policy
        pol = Policy.from_checkpoint(opp.name)

        def fn(obs, _p=pol):
            if int(obs["step"]) == 0 or _p.agent is None:
                _p.reset()
            return _p.act(obs)
    else:
        raise ValueError(f"unknown opponent kind {opp.kind!r}")
    _OPPONENT_CACHE[key] = fn
    return fn


# -- policies ------------------------------------------------------------------

@dataclass(frozen=True)
class PolicySpec:
    """A picklable description of the policy under evaluation.

    `offset` is "checkpoint" (use what the file carries), None (disable), or
    a tuple (delta, live, ramp) for a candidate a search is trying.
    """
    path: str
    offset: Any = "checkpoint"

    @property
    def label(self) -> str:
        base = os.path.basename(self.path)
        if self.offset is None:
            return base + "@no-offset"
        if self.offset == "checkpoint":
            return base
        return base + "@offset"


_POLICY_CACHE: dict = {}


def _file_key(path: str) -> tuple:
    """Cache key that changes when the file is rewritten under the same name."""
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


def _policy(spec: PolicySpec):
    from .policy import Offset, Policy
    key = _file_key(spec.path)
    if key not in _POLICY_CACHE:
        _POLICY_CACHE.clear()      # one live checkpoint per worker; old versions go
        pol = Policy.from_checkpoint(spec.path, offset="checkpoint")
        _POLICY_CACHE[key] = (pol, pol.offset)
    pol, stored = _POLICY_CACHE[key]
    if spec.offset == "checkpoint":
        pol.offset = stored
    elif spec.offset is None:
        pol.offset = None
    else:
        delta, live = spec.offset[0], spec.offset[1]
        ramp = spec.offset[2] if len(spec.offset) > 2 else (len(delta) == 2 * len(live))
        pol.offset = Offset(np.asarray(delta, dtype=np.float32), list(live), bool(ramp))
    return pol


# -- one episode ---------------------------------------------------------------

class OpponentLoadError(RuntimeError):
    """The opponent could not be built; the episode was never played."""


# An opponent that throws on more than this share of turns is BROKEN, not
# weak: it played PASS and a 1.00 against it is not a win the ladder would
# give us. Broken opponents are reported and excluded from the criterion.
BROKEN_FAILURE_SHARE = 0.5


@dataclass
class Episode:
    opponent: str
    seed: int
    seat: int
    money: float
    opp_money: float
    win: float
    opp_failures: int
    error: str | None = None


def play(spec: PolicySpec, opp: Opponent, seed: int, seat: int = 0,
         hours: int = HOURS, days: int = DAYS, cash: int = CASH) -> Episode:
    """Play one full episode. Our failure raises; the opponent's is counted."""
    import torch
    torch.set_num_threads(1)
    from .fastenv import FastEnv

    steps = hours * days
    config = {"episodeSteps": steps, "turnsPerDay": hours, "startingMoney": cash}
    pol = _policy(spec)
    pol.configure(config)
    pol.reset()
    try:
        rival = _opponent_callable(opp)
    except Exception as e:
        raise OpponentLoadError(f"{opp.label}: {type(e).__name__}: {e}") from e
    env = FastEnv(configuration=config, seed=seed)
    obs = env.reset()
    me, other = seat, 1 - seat
    failures = 0
    while not env.done:
        actions = [None, None]
        actions[me] = pol.act(obs[me])
        try:
            actions[other] = rival(obs[other])
        except Exception:
            failures += 1
            actions[other] = dict(PASS_ACTION)
        obs, _ = env.step(actions)
    money = env.rewards()
    mine, theirs = float(money[me]), float(money[other])
    win = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return Episode(opp.label, int(seed), int(seat), mine, theirs, win, failures)


def _play_task(task):
    spec, opp, seed, seat = task[:4]          # extra elements are caller tags
    try:
        return play(spec, opp, seed, seat)
    except OpponentLoadError as e:
        return Episode(opp.label, int(seed), int(seat), float("nan"), float("nan"),
                       float("nan"), 0, error="opponent-load: " + str(e))
    except Exception as e:  # our side: keep the traceback, never a silent NaN
        import traceback
        return Episode(opp.label, int(seed), int(seat), float("nan"), float("nan"),
                       float("nan"), 0, error="ours: " + (traceback.format_exc()[-2000:] or repr(e)))


# -- many episodes -------------------------------------------------------------

def tasks(spec: PolicySpec, opponents: Sequence[Opponent], seeds: Sequence[int],
          seats: Sequence[int] = (0, 1)) -> list[tuple]:
    return [(spec, o, s, seat) for o in opponents for s in seeds for seat in seats]


def run_tasks(todo: Sequence[tuple], procs: int | None = None,
              progress: bool = False) -> list[tuple[tuple, Episode]]:
    """Play arbitrary (spec, opponent, seed, seat, *tags) tasks in parallel.

    Returns (task, episode) pairs; extra elements of a task come back
    untouched so callers with several policies (a search population) can
    group the records by their own tag.
    """
    import multiprocessing as mp
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    out: list[tuple[tuple, Episode]] = []
    t0 = time.time()
    with mp.get_context("fork").Pool(procs) as pool:
        for i, (task, ep) in enumerate(
                pool.imap_unordered(_play_task_keyed, todo, chunksize=1), 1):
            out.append((task, ep))
            if progress and (i % max(1, len(todo) // 20) == 0 or i == len(todo)):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} episodes  {el:.0f}s  "
                      f"ETA {el / i * (len(todo) - i):.0f}s", flush=True)
    ours = [e for _, e in out if e.error and e.error.startswith("ours")]
    not_loaded = sorted({e.opponent for _, e in out if e.error and e.error.startswith("opponent-load")})
    if ours:
        print(f"[evaluate] {len(ours)} episodes FAILED on our side; first:\n{ours[0].error}", flush=True)
    if not_loaded:
        print(f"[evaluate] opponents that did not load (excluded): {not_loaded}", flush=True)
    return out


def _play_task_keyed(task):
    return task, _play_task(task)


def run(spec: PolicySpec, opponents: Sequence[Opponent], seeds: Sequence[int],
        seats: Sequence[int] = (0, 1), procs: int | None = None,
        progress: bool = False) -> list[Episode]:
    """Play every (opponent, seed, seat) in parallel and return the records."""
    return [ep for _, ep in run_tasks(tasks(spec, opponents, seeds, seats), procs, progress)]


# -- summaries -----------------------------------------------------------------

def by_opponent(episodes: Iterable[Episode]) -> dict[str, dict]:
    rows: dict[str, list[Episode]] = {}
    for e in episodes:
        if e.error:
            continue
        rows.setdefault(e.opponent, []).append(e)
    out = {}
    for name, eps in rows.items():
        w = np.array([e.win for e in eps])
        failures = int(sum(e.opp_failures for e in eps))
        turns = len(eps) * HOURS * DAYS
        out[name] = {"n": len(eps), "win": float(w.mean()),
                     "money": float(np.mean([e.money for e in eps])),
                     "opp_money": float(np.mean([e.opp_money for e in eps])),
                     "opp_failures": failures,
                     "broken": failures > BROKEN_FAILURE_SHARE * turns}
    return out


def summary(episodes: Sequence[Episode]) -> dict:
    """The criterion: mean win rate over opponents and how many we beat."""
    per_all = by_opponent(episodes)
    broken = sorted(k for k, v in per_all.items() if v["broken"])
    per = {k: v for k, v in per_all.items() if not v["broken"]}
    wins = np.array([v["win"] for v in per.values()]) if per else np.array([])
    money = np.array([e.money for e in episodes if not e.error])
    not_loaded = sorted({e.opponent for e in episodes
                         if e.error and e.error.startswith("opponent-load")})
    return {
        "opponents": len(per),
        "broken_opponents": broken,
        "opponents_not_loaded": not_loaded,
        "episodes": int(sum(v["n"] for v in per.values())),
        "failed": int(sum(1 for e in episodes if e.error and e.error.startswith("ours"))),
        "win_mean": float(wins.mean()) if len(wins) else float("nan"),
        "beaten": int((wins > 0.5).sum()),
        "contested": int(((wins > 0.35) & (wins < 0.65)).sum()),
        "money_mean": float(money.mean()) if len(money) else float("nan"),
        "money_se": float(money.std(ddof=1) / np.sqrt(len(money))) if len(money) > 1 else float("nan"),
        "opp_failures": int(sum(v["opp_failures"] for v in per.values())),
    }


def paired(a: Sequence[Episode], b: Sequence[Episode]) -> dict:
    """Per-board difference a - b on identical (opponent, seed, seat) keys.

    `n_expected` is the number of boards either side played (including
    failures); a smaller `n` means boards were dropped and the reader must
    know. `degenerate` is True when every board gives exactly zero
    difference: that is not "no effect", it is the same policy measured
    twice, which is how a blind instrument reads.
    """
    ka = {(e.opponent, e.seed, e.seat): e for e in a if not e.error}
    kb = {(e.opponent, e.seed, e.seat): e for e in b if not e.error}
    n_expected = max(len(set((e.opponent, e.seed, e.seat) for e in a)),
                     len(set((e.opponent, e.seed, e.seat) for e in b)))
    keys = sorted(set(ka) & set(kb))
    dm = np.array([ka[k].money - kb[k].money for k in keys])
    dw = np.array([ka[k].win - kb[k].win for k in keys])
    se = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else float("nan")
    degenerate = bool(len(dm) > 0 and np.all(dm == 0) and np.all(dw == 0))
    return {"n": len(keys), "n_expected": int(n_expected),
            "money_diff": float(dm.mean()) if len(dm) else float("nan"),
            "money_se": float(se), "t": float(dm.mean() / se) if se and se > 0 else float("nan"),
            "win_diff": float(dw.mean()) if len(dw) else float("nan"),
            "boards_better": float((dm > 0).mean()) if len(dm) else float("nan"),
            "degenerate": degenerate}


# -- provenance and the ledger -----------------------------------------------

def file_digest(path: str, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def record(kind: str, spec: PolicySpec, opponents: Sequence[Opponent],
           seeds: Sequence[int], episodes: Sequence[Episode], extra: dict | None = None,
           path: str = LEDGER) -> dict:
    """Append one line to the experiment ledger and return it."""
    from .version import provenance
    fams = sorted({S.family_of(s) or "unregistered" for s in seeds})
    prov = provenance()
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "commit": prov["commit"],
        "dirty": prov["dirty"],
        "game_fingerprint": prov["game_fingerprint"],
        "model_fingerprint": prov["model_fingerprint"],
        "game_env": prov["game_env"],
        "argv": prov["argv"], "pid": prov["pid"],
        "checkpoint": os.path.relpath(spec.path, ROOT),
        "checkpoint_digest": file_digest(spec.path),
        "offset": spec.label.split("@")[1] if "@" in spec.label else "checkpoint",
        "opponents": [o.label for o in opponents] if len(opponents) <= 8 else f"{len(opponents)} opponents",
        "seed_families": fams, "seeds": [int(min(seeds)), int(max(seeds)), len(seeds)],
        "summary": summary(episodes),
    }
    if extra:
        entry.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    _track(kind, spec, opponents, seeds, episodes, entry, extra or {})
    return entry


def _track(kind, spec, opponents, seeds, episodes, entry, extra) -> None:
    """The same measurement as an MLflow run in kagsym/evaluation."""
    from .tracking import Tracker, context_tags, describe
    s = entry["summary"]
    fams = "+".join(entry["seed_families"])
    opp_label = opponents[0].label if len(opponents) == 1 else f"band({len(opponents)})"
    tr = Tracker(
        "evaluation", f"{kind}: {spec.label} vs {opp_label}",
        params={"checkpoint": entry["checkpoint"], "offset": entry["offset"],
                "opponents": len(opponents), "seeds": entry["seeds"][2],
                "seed_first": entry["seeds"][0], "seats": sorted({e.seat for e in episodes}),
                "hours": HOURS, "days": DAYS, "cash": CASH},
        tags=context_tags("evaluation", checkpoint=spec.path, opponent=opp_label,
                          seed_family=fams, measurement=kind),
        description=describe([
            f"`{kind}` of {spec.label} against {opp_label} on {entry['seeds'][2]} {fams} seeds.",
            "Deterministic policy, the same code the submission runs (kagsym.policy).",
            "band/win_mean and band/beaten are the criterion; money is a diagnostic.",
            "Paired numbers (paired/*) compare against the checkpoint's own stored offset.",
        ]))
    with tr:
        prefix = "band" if kind.startswith("band") or len(opponents) > 1 else "evaluate"
        tr.log({f"{prefix}/win_mean": s["win_mean"], f"{prefix}/beaten": s["beaten"],
                f"{prefix}/contested": s["contested"], f"{prefix}/opponents": s["opponents"],
                f"{prefix}/money_mean": s["money_mean"], f"{prefix}/money_se": s["money_se"],
                f"{prefix}/opp_failures": s["opp_failures"], f"{prefix}/our_failures": s["failed"],
                f"{prefix}/broken_opponents": len(s["broken_opponents"]),
                f"{prefix}/episodes": s["episodes"]})
        pv = extra.get("paired_vs_checkpoint")
        if pv:
            tr.log({f"paired/{k}": v for k, v in pv.items()})
        per = by_opponent(episodes)
        tr.log_table([{"opponent": k, **v} for k, v in sorted(per.items(), key=lambda kv: -kv[1]["win"])],
                     "per_opponent.csv")
        tr.log_json([asdict(e) for e in episodes], "episodes.json")
        tr.log_json(entry, "ledger_entry.json")


def save_episodes(episodes: Sequence[Episode], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(e) for e in episodes], f)


def load_episodes(path: str) -> list[Episode]:
    with open(path) as f:
        return [Episode(**d) for d in json.load(f)]
