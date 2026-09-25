"""Play a solitaire where the plan head decides each day's plan."""
from __future__ import annotations


from .. import spec
from ..fastenv import FastEnv
from ..plan import Plan, plan_macro, set_plan
from ..symbolic.executor import Agent
from .data import features


from .state import day_state as _state  # noqa: E402


def _rival(opponent, seed: int):
    from ..evaluate import PASS_ACTION, _opponent_callable, public
    if not opponent:
        return lambda ob: dict(PASS_ACTION)
    if opponent.startswith("replay:"):
        import json
        from ..plan import _TAPES
        tape = _TAPES.get(opponent)
        if tape is None:
            tape = _TAPES[opponent] = json.load(open(opponent[len("replay:"):]))
        acts = tape[str(seed)]
        return lambda ob, _a=acts: _a[int(ob["step"])] if int(ob["step"]) < len(_a) else dict(PASS_ACTION)
    return _opponent_callable(public(opponent))


def play_head(model, seed: int, days: int, hours: int = 24, cash: int = 3000, opponent: str | None = None) -> float:
    steps = hours * days
    spec.set_turns_per_day(hours)
    spec.set_episode_steps(steps)
    env = FastEnv(configuration={"episodeSteps": steps, "turnsPerDay": hours, "startingMoney": cash}, seed=seed)
    obs = env.reset()
    rival = _rival(opponent, seed)
    plan = Plan()
    set_plan(plan)
    ag = Agent(episode_steps=steps, macro=plan_macro(plan))
    fields = {f: [] for f in ("crop", "tiles", "hands", "load", "water_last", "selling", "animals", "land")}
    try:
        while not env.done:
            ob = obs[0]
            if ob["hour"] == 0:
                p = model.plan_for(features(_state(ob), days))
                for f in fields:
                    fields[f].append(p[f])
                plan = Plan(crop=tuple(fields["crop"]), tiles=tuple(fields["tiles"]), hands=tuple(fields["hands"]),
                            land=tuple(fields["land"]), animals=tuple(fields["animals"]), selling=tuple(fields["selling"]),
                            load=tuple(fields["load"]), water_last=tuple(fields["water_last"]))
                set_plan(plan)
            obs, _ = env.step([ag(ob), rival(obs[1])])
        return float(env.rewards()[0])
    finally:
        set_plan(None)
