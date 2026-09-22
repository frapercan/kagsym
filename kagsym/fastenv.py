"""Exact clone of the Kaggriculture engine, without the kaggle-environments wrapper.

The key point: the rules are not re-implemented. It calls the `interpreter` of
the installed engine itself over a duck-typed state. That is exact by
construction and cannot desynchronise when Kaggle ships a new version.

What it saves against `kaggle_environments.make(...)`:
  - schema validation and action copying every turn
  - recording the full state history (720 deep states)
  - the agent machinery (subprocesses, timeouts, logs)

Typical use::

    env = FastEnv(seed=7)
    obs = env.reset()
    while not env.done:
        obs, done = env.step([a0, a1])
    print(env.rewards())

For imagined rollouts from a real in-game observation::

    env = FastEnv.from_observation(obs, seed_guess=0)
    env.step([my_action, guessed_opponent_action])
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from kaggle_environments.envs.kaggriculture import kaggriculture as _eng
from kaggle_environments.utils import Struct, structify

from . import spec

PASS_ACTION: dict = {"farmer": ["PASS"], "hands": [], "market": []}


# --- fast copy --------------------------------------------------------------
# BEWARE of Struct: kaggle_environments.utils.Struct keeps a parallel copy in
# __dict__, so `s["k"] = v` does NOT change `s.k`. The interpreter reads the
# action by attribute (`s.action`) and the step by item (`get(obs,"step")`), so
# writes must ALWAYS go through the attribute to touch both sides.
def _fast_copy(x: Any) -> Any:
    t = type(x)
    if t is list:
        return [_fast_copy(v) for v in x]
    if t is Struct:
        return Struct(**{k: _fast_copy(v) for k, v in x.items()})
    if isinstance(x, dict):
        return {k: _fast_copy(v) for k, v in x.items()}
    return x                          # int/float/str/bool/None: inmutables


def _agent_state(player: int, observation: dict) -> Struct:
    """Only the top two levels need attribute access.

    Below that (farms, tiles, market, town) the engine uses item access and
    .get(), so leaving them as plain dicts is behaviourally identical and
    considerably cheaper to copy.
    """
    return Struct(
        action=_fast_copy(PASS_ACTION),
        status="ACTIVE",
        reward=0.0,
        info={},
        observation=Struct(player=player, **observation),
    )


def _blank_observation() -> dict:
    return {
        "step": 0,
        "remainingOverageTime": 60,
        "farms": [],
        "private": {},
        "market": {},
        "town": {},
        "day": 0,
        "hour": 0,
    }


class FastEnv:
    """The Kaggriculture engine over plain dicts."""

    def __init__(
        self,
        configuration: dict | None = None,
        seed: int | None = None,
        n_agents: int = 2,
    ):
        cfg = dict(spec.DEFAULT_CONFIG)
        if configuration:
            cfg.update(configuration)
        cfg.pop("seed", None)
        self.configuration = structify(cfg)
        self.info: dict = {"seed": seed if seed is not None else 0}
        self.n_agents = n_agents
        self.state: list = []
        self._n_recorded = 0

    # -- properties the interpreter queries ---------------------------------
    @property
    def done(self) -> bool:
        return all(s.status != "ACTIVE" for s in self.state) if self.state else False

    @property
    def step_index(self) -> int:
        return self.state[0].observation["step"] if self.state else 0

    @property
    def day(self) -> int:
        return self.state[0].observation["day"]

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> list:
        self.state = [_agent_state(i, _blank_observation()) for i in range(self.n_agents)]
        _eng.interpreter(self.state, self)          # rama _initialize
        self.state[0].observation.step = 0
        self._n_recorded = 1
        return self.observations()

    def step(self, actions: list[dict]) -> tuple[list, bool]:
        """Apply one turn. `actions[i]` is player i's action dict."""
        for i, a in enumerate(actions):
            self.state[i].action = a if isinstance(a, dict) else _fast_copy(PASS_ACTION)
        _eng.interpreter(self.state, self)
        self.state[0].observation.step = self._n_recorded
        self._n_recorded += 1
        if self.state[0].observation["step"] >= self.configuration["episodeSteps"] - 1:
            for s in self.state:
                if s.status in ("ACTIVE", "INACTIVE"):
                    s.status = "DONE"
        return self.observations(), self.done

    # -- reading -------------------------------------------------------------
    def observations(self) -> list:
        """Per-player observation, with shared fields already propagated."""
        obs0 = self.state[0].observation
        for i in range(1, self.n_agents):
            oi = self.state[i].observation
            oi.farms = obs0["farms"]
            oi.market = obs0["market"]
            oi.town = obs0["town"]
            oi.day = obs0["day"]
            oi.hour = obs0["hour"]
            oi.step = obs0["step"]
        return [s.observation for s in self.state]

    def rewards(self) -> list[float]:
        farms = self.state[0].observation["farms"]
        return [float(f["money"]) for f in farms]

    # -- cloning for imagined rollouts ---------------------------------------
    def clone(self) -> "FastEnv":
        new = FastEnv.__new__(FastEnv)
        new.configuration = self.configuration
        new.info = dict(self.info)
        new.n_agents = self.n_agents
        new._n_recorded = self._n_recorded
        new.state = [_fast_copy(s) for s in self.state]
        # Re-link the shared objects: the interpreter writes via farms[0], and
        # _fast_copy would have duplicated them per player.
        obs0 = new.state[0].observation
        for i in range(1, new.n_agents):
            oi = new.state[i].observation
            oi.farms = obs0["farms"]
            oi.market = obs0["market"]
            oi.town = obs0["town"]
        return new

    # -- reconstruction from a real observation ------------------------------
    @classmethod
    def from_observation(
        cls,
        obs: Any,
        configuration: dict | None = None,
        seed_guess: int | None = None,
        assume_opponent_private: Callable[[Any, int], dict] | None = None,
    ) -> "FastEnv":
        """Build a simulable state from an agent's observation.

        `seed_guess=None` (the default) draws a random seed. This matters:
        weeds and shop unlocks come from `Random((seed*1000003) ^ day)`, so
        with a FIXED seed an imagined rollout crossing a day boundary always
        generates the same wrong shop -and the shop decides which products the
        town drains-. With a random seed the engine samples them at the correct
        rate and distribution, which is the best that can be done with
        something unobservable.

        The opponent's shed, seeds and unit inventories are private and have to
        be assumed. `assume_opponent_private(obs, opp_id)` is the hook for
        plugging in an opponent model; by default they are assumed empty, which
        is exactly the bias a neural residual would have to learn to correct.
        """
        import random as _random
        me = int(obs["player"])
        n = len(obs["farms"])
        if seed_guess is None:
            seed_guess = _random.randrange(1 << 30)
        env = cls(configuration=configuration, seed=seed_guess, n_agents=n)
        farms = _fast_copy(obs["farms"])
        market = _fast_copy(obs["market"])
        town = _fast_copy(obs["town"])
        state = []
        for i in range(n):
            if i == me:
                private = _fast_copy(obs["private"])
            elif assume_opponent_private is not None:
                private = _fast_copy(assume_opponent_private(obs, i))
            else:
                private = _eng._new_private()
                # the opponent has as many inventories as hands hired
                private["inventories"] = [{} for _ in range(1 + len(farms[i]["hands"]))]
            state.append(_agent_state(i, {
                "step": int(obs["step"]),
                "remainingOverageTime": 60,
                "farms": farms,
                "private": private,
                "market": market,
                "town": town,
                "day": int(obs["day"]),
                "hour": int(obs["hour"]),
            }))
        env.state = state
        env._n_recorded = int(obs["step"]) + 1
        return env


def run_episode(
    policy0: Callable[[Any], dict],
    policy1: Callable[[Any], dict],
    seed: int = 0,
    configuration: dict | None = None,
    on_transition: Callable[[int, list, list, list], None] | None = None,
) -> tuple[list[float], int]:
    """Play a full episode. Returns (rewards, turns played).

    `on_transition(step, obs_before, actions, obs_after)` is called every turn;
    it is the data-collection hook.
    """
    env = FastEnv(configuration=configuration, seed=seed)
    obs = env.reset()
    steps = 0
    while not env.done:
        before = [_fast_copy(o) for o in obs] if on_transition else None
        actions = [policy0(obs[0]), policy1(obs[1])]
        obs, done = env.step(actions)
        steps += 1
        if on_transition:
            on_transition(steps - 1, before, actions, [_fast_copy(o) for o in obs])
        if done:
            break
    return env.rewards(), steps
