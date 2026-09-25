"""Amortisation by retrieval: at every day boundary, the plan of the most
similar recorded state (same day; then the standardised feature distance,
with the demand and the rival weighted up). No training, no compounding
from a misfit model: a state the search saw gets the search's plan.
"""
from __future__ import annotations

import numpy as np

from .data import FIELDS, KIND_OF, class_to_crop, crop_class, features


class PlanRetrieval:
    def __init__(self, xs: np.ndarray, days: np.ndarray, plans: list, k: int = 1, weights: np.ndarray | None = None):
        self.xs = np.asarray(xs, dtype=np.float32)
        self.days = np.asarray(days, dtype=np.int32)
        self.plans = list(plans)
        self.k = int(k)
        self.mu = self.xs.mean(0)
        self.sd = self.xs.std(0) + 1e-6
        self.w = np.ones(self.xs.shape[1], dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)

    @classmethod
    def from_records(cls, rows: list, k: int = 1) -> "PlanRetrieval":
        xs = np.stack([r["x"] for r in rows])
        days = np.array([int(round(r["x"][0] * 30)) for r in rows])
        plans = [r["plan"] for r in rows]
        return cls(xs, days, plans, k=k)

    def plan_for(self, x, day: int | None = None) -> dict:
        x = np.asarray(x, dtype=np.float32)
        d = int(round(float(x[0]) * 30)) if day is None else int(day)
        mask = self.days == d
        if not mask.any():
            mask = np.ones(len(self.days), dtype=bool)
        z = (self.xs[mask] - self.mu) / self.sd
        q = (x - self.mu) / self.sd
        dist = (((z - q) * self.w) ** 2).sum(1)
        idx = np.flatnonzero(mask)[np.argsort(dist)[: self.k]]
        cands = [self.plans[i] for i in idx]
        if self.k == 1:
            p = dict(cands[0])
        else:                                   # vote: mode for the discrete, median for the numeric
            p = {}
            for f in ("crop", "water_last", "land"):
                vals = [crop_class(c[f]) if f == "crop" else c[f] for c in cands]
                best = max(set(map(str, vals)), key=lambda s: sum(str(v) == s for v in vals))
                p[f] = class_to_crop(best) if f == "crop" else type(vals[0])(best) if not isinstance(vals[0], str) else best
            for f in ("tiles", "hands", "load", "selling"):
                p[f] = float(np.median([c[f] for c in cands])) if f == "selling" else int(np.median([c[f] for c in cands]))
            an = {}
            for kind in KIND_OF.values():
                an[kind] = int(np.median([(c.get("animals") or {}).get(kind, 0) if isinstance(c.get("animals"), dict) else 0 for c in cands]))
            p["animals"] = {k: v for k, v in an.items() if v > 0} or 0
        p.setdefault("animals", 0)
        return p

    def state_dict(self) -> dict:
        return {"kind": "plan_knn", "xs": self.xs, "days": self.days, "plans": self.plans, "k": self.k, "n_in": int(self.xs.shape[1])}


def build(pattern: str, k: int = 1) -> PlanRetrieval:
    from .data import load
    rows = load(pattern)
    return PlanRetrieval.from_records(rows, k=k)


__all__ = ["PlanRetrieval", "build", "features", "FIELDS"]
