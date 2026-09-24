"""The deployed policy, defined once.

`Policy` is the only object that turns a checkpoint into actions. The
submission wrapper (`submit_kagsym/main.py`), the evaluator
(`kagsym.evaluate`) and every search tool drive this class and nothing else.

Why one definition. Two nights were lost to measurement loops that were not
the agent being uploaded: one fed zeros where the agent feeds the opponent's
supply forecast (same board: $76,607 became $25,335), another did not pass the
executor's destinations, a third ignored the ramp stored in the checkpoint.
Each loop looked healthy and described a different agent.

What a policy does per episode:
  * loads the network ONCE (the first callback is where submissions time out),
  * decides ONCE a day: the micro map is a plan for the day (re-emitting it
    every turn cost -$13,346, t -21.5, and multiplies the cost by 24),
  * keeps the executor alive across turns (it holds destinations and a
    self-calibrated turns-per-tile estimate; recreating it cost -$342),
  * plays the MEAN of the policy: no exploration noise,
  * adds the macro offset from the checkpoint (`kagsym.offset`) before the
    sigmoid, with progress = day / days derived from the engine configuration.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _config_value(config: Any, key: str, default):
    if config is None:
        return default
    v = getattr(config, key, None)
    if v is None and isinstance(config, dict):
        v = config.get(key)
    return default if v is None else v


@dataclass
class Offset:
    """Macro offset in logit space: constant (`a`) or ramp (`a + b*progress`).

    `live` are the macro indices the offset touches, sorted by index; dead
    dials are never perturbed. `ramp` is an explicit field: it used to be
    inferred from the vector length, and a vector of the wrong length was read
    as a ramp shifted by one dial without any error. The length is now exact:
    `len(delta) == len(live)` for a constant, `2 * len(live)` for a ramp.

    A constant can be baked into the bias of `macro_mu`; a ramp depends on
    the day and travels in the checkpoint as `offset`.
    """
    delta: np.ndarray
    live: list[int]
    ramp: bool = False

    def __post_init__(self):
        self.delta = np.asarray(self.delta, dtype=np.float32).reshape(-1)
        self.live = [int(i) for i in self.live]
        n = len(self.live)
        expected = 2 * n if self.ramp else n
        if len(self.delta) != expected:
            raise ValueError(f"offset has {len(self.delta)} values for {n} live dials "
                             f"({'ramp' if self.ramp else 'constant'}: expected {expected})")
        if len(set(self.live)) != n:
            raise ValueError("live dials repeat")

    @property
    def is_ramp(self) -> bool:
        return self.ramp

    def vector(self, progress: float, n_macro: int) -> np.ndarray:
        out = np.zeros(int(n_macro), dtype=np.float32)
        n = len(self.live)
        d = self.delta
        v = d[:n] + d[n:2 * n] * float(progress) if self.ramp else d[:n]
        out[self.live] = v
        return out

    def to_checkpoint(self) -> dict:
        return {"delta": [float(x) for x in self.delta], "live": list(self.live),
                "ramp": bool(self.ramp)}

    @staticmethod
    def from_checkpoint(ck: dict) -> "Offset | None":
        r = ck.get("offset") or ck.get("delta_rampa")   # old field name
        if not r:
            return None
        live = [int(i) for i in r.get("live", r.get("vivos"))]
        delta = np.asarray(r["delta"], dtype=np.float32)
        ramp = bool(r["ramp"]) if "ramp" in r else len(delta) == 2 * len(live)
        return Offset(delta, live, ramp)

    # self-describing files: the vector never travels without its dial map
    def save(self, path: str, checkpoint_digest: str, **meta) -> None:
        tmp = path + ".tmp"
        np.savez(tmp, delta=self.delta, live=np.asarray(self.live, dtype=np.int64),
                 ramp=np.asarray(self.ramp), checkpoint_digest=np.asarray(checkpoint_digest),
                 meta=np.asarray(json.dumps(meta)))
        os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, path)

    @staticmethod
    def load(path: str) -> "tuple[Offset, str, dict]":
        z = np.load(path, allow_pickle=False)
        off = Offset(z["delta"], z["live"].tolist(), bool(z["ramp"]))
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        return off, str(z["checkpoint_digest"]), meta


def load_network(path: str, strict: bool = False, verbose: bool = False):
    """Build the network exactly as the checkpoint describes it."""
    import torch
    from .migrate_ckpt import load_strict, load_tolerant
    from .nets.world import E2EAgent, WorldConfig

    # One thread, always. Measured: the same seed gave $42,242 with the
    # default thread count and $48,387 with one, because the summation order
    # in the network moves the macro by ~1e-7 and the executor thresholds it.
    # Every evaluator and the submission must run the network identically.
    torch.set_num_threads(1)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if verbose or os.environ.get("KAGSYM_CHECK_FINGERPRINT") == "1":
        from .version import check
        check(ck.get("fingerprint"), what=os.path.basename(path))
    cfg = ck["cfg"]
    cfg = WorldConfig(**cfg) if isinstance(cfg, dict) else cfg
    cfg.device = "cpu"
    net = E2EAgent(cfg)
    if strict:
        load_strict(net, ck["sd"], path, macro_fields=ck.get("macro_fields"))
    else:
        load_tolerant(net, ck["sd"], path, verbose=verbose,
                      macro_fields=ck.get("macro_fields"))
    net.eval()
    return net, ck


@dataclass
class Policy:
    net: Any
    offset: Offset | None = None
    steps: int = 720
    hours: int = 24
    hand_cap: int | None = None
    _agent: Any = field(default=None, repr=False)
    _day: int | None = field(default=None, repr=False)
    _map: Any = field(default=None, repr=False)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str, offset: "Offset | str | None" = "checkpoint",
                        strict: bool = False, verbose: bool = False) -> "Policy":
        """`offset="checkpoint"` uses what the file carries; `None` disables it;
        an `Offset` overrides it (what a search does while exploring)."""
        net, ck = load_network(path, strict=strict, verbose=verbose)
        if offset == "checkpoint":
            offset = Offset.from_checkpoint(ck)
        return cls(net=net, offset=offset)

    # -- episode lifecycle ----------------------------------------------------
    def configure(self, config: Any = None) -> None:
        """Read episode length from the engine configuration, never from a 30."""
        self.steps = int(_config_value(config, "episodeSteps", 720))
        self.hours = int(_config_value(config, "turnsPerDay", 24))

    @property
    def days(self) -> int:
        return max(1, self.steps // max(1, self.hours))

    def reset(self, config: Any = None) -> None:
        from . import spec
        from . import macro as M
        from .symbolic.executor import Agent

        if config is not None:
            self.configure(config)
        spec.set_turns_per_day(self.hours)
        spec.set_episode_steps(self.steps)
        # A process-wide global inherited by forked workers. The trainer sets
        # it for curricula; a policy that plays for real never wants a cap.
        M.HAND_CAP = self.hand_cap
        self._agent = Agent(episode_steps=self.steps,
                            macro=M.Macro.from_vector([0.5] * M.N_MACRO))
        self._agent.micro = self._micro
        self._day = None
        self._map = None

    @property
    def agent(self):
        return self._agent

    def _micro(self, _obs):
        from .environment import _split_micro
        return _split_micro(self._map)

    # -- acting ---------------------------------------------------------------
    def plan_day(self, obs) -> None:
        """Run the network once for today's plan."""
        import torch
        from . import obs as O
        from . import macro as M

        g, b = O.encode_obs(obs, self._agent._destinations)
        hf = np.asarray(O.rival_flow(obs), dtype=np.float32)
        with torch.no_grad():
            out = self.net(torch.from_numpy(g).unsqueeze(0),
                           torch.from_numpy(b).unsqueeze(0),
                           torch.from_numpy(hf).unsqueeze(0))
            mu = out["macro_mu"][0]
            if self.offset is not None:
                day = int(obs["day"])
                mu = mu + torch.from_numpy(
                    self.offset.vector(day / self.days, M.N_MACRO))
            self._agent.macro = M.Macro.from_vector(torch.sigmoid(mu).numpy())
            self._map = out["micro"][0].numpy()

    def act(self, obs) -> dict:
        if self._agent is None:
            self.reset()
        day = int(obs["day"])
        if self._day != day:
            self.plan_day(obs)
            self._day = day
        return self._agent(obs)

    __call__ = act


def find_checkpoint(config: Any = None, filename: str = "model.pt") -> str:
    """Locate the model next to the package without relying on `__file__`.

    Kaggle does not import the agent module: it compiles the source and
    `exec`s it, so `__file__` is undefined there. `KAGSYM_CKPT` overrides the
    search (tests and evaluators use it).
    """
    env_ = os.environ.get("KAGSYM_CKPT")
    if env_ and os.path.exists(env_):
        return env_
    candidates = []
    try:
        import kagsym as _k
        candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(_k.__file__))))
    except Exception:
        pass
    raw = _config_value(config, "__raw_path__", None)
    if raw:
        candidates.append(os.path.dirname(os.path.abspath(raw)))
    candidates.append(os.getcwd())
    for d in candidates:
        for name in (filename, "modelo.pt"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"{filename} not found; looked in {candidates}")
