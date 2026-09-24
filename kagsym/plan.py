"""An explicit plan: the finer model of the problem.

The macro's 67 dials are the executor's internal constants exposed; none of
them says "50 tiles of carrot now, replant on harvest". A `Plan` does. It is
a handful of variables with game meaning, executed literally by the symbolic
layer (routes, assignment and legality stay exact), so that the plan space
can be searched exhaustively on the exact engine and, later, learned.

    plan = Plan(crop="CARROT", tiles=50, hands=6, land=1)
    with active(plan):
        ...  # every Agent built inside obeys the plan

Fields (per game, for the reduced solitaire; a schedule of plans per day is
the next step):
  crop     what to plant
  tiles    planted tiles to keep (replanted as they are harvested)
  hands    hands hired every day
  land     quadrants to buy at the start (0-3)
  animals  animals to keep (0 in the 8-day world: none pays)
  selling  the macro's selling dial in [0, 1]: 0 sells at once
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict


def _by_day(v, day: int):
    """A field is a number (every day) or a tuple indexed by day (last value
    repeats past the end)."""
    if isinstance(v, (tuple, list)):
        return v[min(int(day), len(v) - 1)] if v else 0
    return v


@dataclass(frozen=True)
class Plan:
    crop: str | tuple = "CARROT"
    tiles: int | tuple = 25
    hands: int | tuple = 3
    land: int | tuple = 0
    animals: int | tuple = 0
    selling: float | tuple = 0.05

    def crop_on(self, day: int) -> str:
        return str(_by_day(self.crop, day))

    def tiles_on(self, day: int) -> int:
        return int(_by_day(self.tiles, day))

    def hands_on(self, day: int) -> int:
        return int(_by_day(self.hands, day))

    def land_on(self, day: int) -> int:
        return int(_by_day(self.land, day))

    def animals_on(self, day: int) -> int:
        return int(_by_day(self.animals, day))

    def selling_on(self, day: int) -> float:
        return float(_by_day(self.selling, day))

    def to_dict(self) -> dict:
        return asdict(self)


PLAN: Plan | None = None      # process-global, like HAND_CAP; set per episode


def set_plan(plan: Plan | None) -> None:
    global PLAN
    PLAN = plan


def get_plan() -> Plan | None:
    return PLAN


@contextmanager
def active(plan: Plan | None):
    prev = PLAN
    set_plan(plan)
    try:
        yield
    finally:
        set_plan(prev)


def plan_macro(plan: Plan):
    """The macro vector that goes with a plan: defaults except the dials the
    plan replaces outright are irrelevant, and selling is set by the plan."""
    from .macro import Macro
    m = Macro()
    m.selling = plan.selling_on(0)
    return m


def play_plan(plan: Plan, seed: int, days: int = 8, hours: int = 24, cash: int = 3000) -> float:
    """One solitaire game under `plan`, deterministic; returns the final cash."""
    from . import spec
    from .fastenv import FastEnv
    from .symbolic.executor import Agent
    from .evaluate import PASS_ACTION
    steps = hours * days
    spec.set_turns_per_day(hours)
    spec.set_episode_steps(steps)
    with active(plan):
        env = FastEnv(configuration={"episodeSteps": steps, "turnsPerDay": hours, "startingMoney": cash}, seed=seed)
        obs = env.reset()
        ag = Agent(episode_steps=steps, macro=plan_macro(plan))
        while not env.done:
            obs, _ = env.step([ag(obs[0]), dict(PASS_ACTION)])
        return float(env.rewards()[0])
