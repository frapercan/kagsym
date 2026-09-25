"""The state at a day boundary, as the day search records it and the plan
head reads it. Lives in the package so the Kaggle wrapper carries it."""
from __future__ import annotations

from .. import spec


def day_state(ob) -> dict:
    f = ob["farms"][int(ob["player"])]
    by = {}
    for r in f["tiles"]:
        for t in r:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                k = f"{t['crop']}@{int(ob['day']) - t['planted_day']}"
                by[k] = by.get(k, 0) + 1
    animals = {}
    for r in f["tiles"]:
        for t in r:
            if isinstance(t, dict) and t.get("animal"):
                animals[t["animal"]] = animals.get(t["animal"], 0) + 1
    other = ob["farms"][1 - int(ob["player"])]
    rival_planted, rival_animals = {}, {}
    for r in other["tiles"]:
        for t in r:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                rival_planted[t["crop"]] = rival_planted.get(t["crop"], 0) + 1
            elif isinstance(t, dict) and t.get("animal"):
                rival_animals[t["animal"]] = rival_animals.get(t["animal"], 0) + 1
    return dict(day=int(ob["day"]), cash=round(float(f["money"])), hands=len(f["hands"]),
                quadrants=len(f["unlocked_quadrants"]), planted=by, animals=animals,
                seeds={k: int(v) for k, v in ob["private"]["seeds"].items() if v},
                shed={k: int(v) for k, v in ob["private"]["shed"].items() if v},
                prices={k: ob["market"]["prices"][k] for k in spec.PRODUCTS},
                inventory={k: int(ob["market"]["inventory"][k]) - 10000 for k in spec.PRODUCTS},
                shops=list(ob["town"].get("unlocked_shops", [])),
                rival=dict(money=round(float(other["money"])), hands=len(other["hands"]),
                           quadrants=len(other["unlocked_quadrants"]), planted=rival_planted, animals=rival_animals))
