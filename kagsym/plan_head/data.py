"""Records of the day search -> tensors.

A record (tools/plan_daysearch.py) holds the state at a day boundary and
the plan chosen for that day. The horizon is in the record (`days`) or in
the file name (daysearch_<d>d.jsonl).
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from .. import spec

CROPS = list(spec.CROP_LIST)
MIXES = [(a, b) for i, a in enumerate(CROPS) for b in CROPS[i + 1:]]
CROP_CLASSES = CROPS + [f"{a}+{b}" for a, b in MIXES]
FIELDS = {
    "hands": list(range(0, 9)),
    "load": [0, 3, 6, 9, 12, 15],
    "water_last": [0, 1],
    "tiles": [0, 10, 15, 20, 25, 30, 40, 50, 75],
    "crop": CROP_CLASSES,
    "selling": [0.05, 0.25, 0.5, 0.75, 0.95],
    "land": [0, 1, 2],
}
MAX_AGE = 16


def crop_class(c) -> str:
    if isinstance(c, dict):
        ks = sorted(c)
        return f"{ks[0]}+{ks[1]}" if len(ks) == 2 else ks[0]
    return str(c)


def class_to_crop(name: str):
    return {k: 0.5 for k in name.split("+")} if "+" in name else name


def features(state: dict, days: int) -> np.ndarray:
    """The state at the day boundary as a flat vector."""
    day = int(state["day"])
    x = [day / 30.0, (days - day) / 30.0, days / 30.0,
         state["cash"] / 10000.0, state["hands"] / 8.0, state["quadrants"] / 4.0,
         sum(state["seeds"].values()) / 50.0, sum(state["shed"].values()) / 100.0]
    x += [state["prices"].get(c, 0) / 250.0 for c in CROPS]
    grid = np.zeros((len(CROPS), MAX_AGE), dtype=np.float32)
    for k, n in state["planted"].items():
        c, age = k.split("@")
        grid[CROPS.index(c), min(int(age), MAX_AGE - 1)] += n / 25.0
    x += grid.ravel().tolist()
    x += [(days - day - spec.CROPS[c]["first_yield_day"]) / 30.0 for c in CROPS]   # can it still yield?
    return np.asarray(x, dtype=np.float32)


def targets(plan: dict) -> dict:
    out = {}
    for f, vals in FIELDS.items():
        v = plan[f]
        if f == "crop":
            v = crop_class(v)
        elif f == "selling":
            v = min(vals, key=lambda a: abs(a - float(v)))
        elif f == "tiles":
            v = min(vals, key=lambda a: abs(a - int(v)))
        out[f] = vals.index(v)
    return out


def load(pattern: str = "runs/ladder/daysearch_*d.jsonl") -> list:
    rows = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"daysearch_(\d+)d", os.path.basename(path))
        if not m or "test" in path or "pairs" in path:
            continue
        d_file = int(m.group(1))
        for line in open(path):
            r = json.loads(line)
            days = int(r.get("days", d_file))
            rows.append(dict(days=days, seed=int(r["seed"]), x=features(r["state"], days),
                             y=targets(r["plan"]), plan=r["plan"], money=float(r["best"])))
    return rows
