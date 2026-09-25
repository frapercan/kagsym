from __future__ import annotations

import torch
import torch.nn as nn

from .data import FIELDS


class PlanHead(nn.Module):
    """One softmax head per plan field on a shared trunk."""

    def __init__(self, n_in: int, width: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.heads = nn.ModuleDict({f: nn.Linear(width, len(v)) for f, v in FIELDS.items()})

    def forward(self, x):
        h = self.trunk(x)
        return {f: head(h) for f, head in self.heads.items()}

    def plan_for(self, x) -> dict:
        """The argmax plan of one state vector, as plan values."""
        from .data import class_to_crop
        with torch.no_grad():
            logits = self(torch.as_tensor(x).unsqueeze(0))
        from .data import KIND_OF
        out = {}
        for f, vals in FIELDS.items():
            j = int(logits[f].argmax(-1))
            out[f] = class_to_crop(vals[j]) if f == "crop" else vals[j]
        out["animals"] = {KIND_OF[f]: out.pop(f) for f in list(KIND_OF) if f in out}
        out["animals"] = {k: v for k, v in out["animals"].items() if v > 0} or 0
        return out
