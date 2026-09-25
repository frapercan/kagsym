"""Play a solitaire where the plan head decides each day's plan."""
from __future__ import annotations


from .. import evaluate as E, spec
from ..fastenv import FastEnv
from ..plan import Plan, plan_macro, set_plan
from ..symbolic.executor import Agent
from .data import features


def _state(ob) -> dict:
    from tools.plan_daysearch import _state as s
    return s(ob)


def play_head(model, seed: int, days: int, hours: int = 24, cash: int = 3000) -> float:
    steps = hours * days
    spec.set_turns_per_day(hours)
    spec.set_episode_steps(steps)
    env = FastEnv(configuration={"episodeSteps": steps, "turnsPerDay": hours, "startingMoney": cash}, seed=seed)
    obs = env.reset()
    plan = Plan()
    set_plan(plan)
    ag = Agent(episode_steps=steps, macro=plan_macro(plan))
    fields = {f: [] for f in ("crop", "tiles", "hands", "load", "water_last", "selling", "animals")}
    land = 0
    try:
        while not env.done:
            ob = obs[0]
            if ob["hour"] == 0:
                p = model.plan_for(features(_state(ob), days))
                for f in fields:
                    fields[f].append(p[f])
                if ob["day"] == 0:
                    land = int(p["land"])
                plan = Plan(crop=tuple(fields["crop"]), tiles=tuple(fields["tiles"]), hands=tuple(fields["hands"]),
                            land=land, animals=tuple(fields["animals"]), selling=tuple(fields["selling"]), load=tuple(fields["load"]),
                            water_last=tuple(fields["water_last"]))
                set_plan(plan)
            obs, _ = env.step([ag(ob), dict(E.PASS_ACTION)])
        return float(env.rewards()[0])
    finally:
        set_plan(None)
