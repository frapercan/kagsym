"""The deployable policy: the symbolic executor driven by the plan head.

Same lifecycle as `kagsym.policy.Policy` (from_checkpoint, configure,
reset, act, .agent), so the Kaggle wrapper does not change. At every day
boundary the head reads the state and emits the day's plan; the executor
plays it literally. No rollouts at play time: microseconds a day.
"""
from __future__ import annotations

from typing import Any

from ..plan import Plan, plan_macro, set_plan
from .data import features
from .model import PlanHead
from .state import day_state

FIELDS = ("crop", "tiles", "hands", "load", "water_last", "selling", "animals")


def _config_value(config, key, default):
    if config is None:
        return default
    try:
        return config[key]
    except Exception:
        return getattr(config, key, default)


class PlanHeadPolicy:
    def __init__(self, model: PlanHead):
        self.model = model
        self.steps, self.hours = 720, 24
        self._agent = None
        self._day = None
        self._fields = None
        self._land = 0

    @classmethod
    def from_checkpoint(cls, path: str) -> "PlanHeadPolicy":
        import torch
        torch.set_num_threads(1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = PlanHead(int(ck["n_in"]), int(ck.get("width", 256)))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return cls(model)

    def configure(self, config: Any = None) -> None:
        self.steps = int(_config_value(config, "episodeSteps", 720))
        self.hours = int(_config_value(config, "turnsPerDay", 24))

    @property
    def days(self) -> int:
        return max(1, self.steps // max(1, self.hours))

    def reset(self, config: Any = None) -> None:
        from .. import spec
        from ..symbolic.executor import Agent
        if config is not None:
            self.configure(config)
        spec.set_turns_per_day(self.hours)
        spec.set_episode_steps(self.steps)
        self._fields = {f: [] for f in FIELDS}
        self._land = 0
        self._plan = Plan()
        self._agent = Agent(episode_steps=self.steps, macro=plan_macro(self._plan))
        self._day = None

    @property
    def agent(self):
        return self._agent

    def _plan_day(self, obs) -> None:
        p = self.model.plan_for(features(day_state(obs), self.days))
        for f in FIELDS:
            self._fields[f].append(p[f])
        if int(obs["day"]) == 0:
            self._land = int(p["land"])
        fl = self._fields
        self._plan = Plan(crop=tuple(fl["crop"]), tiles=tuple(fl["tiles"]), hands=tuple(fl["hands"]), land=self._land,
                          animals=tuple(fl["animals"]), selling=tuple(fl["selling"]), load=tuple(fl["load"]),
                          water_last=tuple(fl["water_last"]))

    def act(self, obs) -> dict:
        if self._agent is None:
            self.reset()
        day = int(obs["day"])
        if self._day != day:
            self._plan_day(obs)
            self._day = day
        # The plan is a process global; set it for this call only, so another
        # policy in the same process (an evaluator's paired run) never sees it.
        set_plan(self._plan)
        try:
            return self._agent(obs)
        finally:
            set_plan(None)
