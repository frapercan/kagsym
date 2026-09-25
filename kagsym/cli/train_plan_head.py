"""EXP-008: train the plan head by imitation of the day search and measure
its regret against the searched plans on held-out horizons and seeds.

    python -m kagsym.cli.train_plan_head --holdout-days 6,9,12 --epochs 300
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from kagsym.plan_head.data import FIELDS, load  # noqa: E402
from kagsym.plan_head.model import PlanHead  # noqa: E402
from kagsym.plan_head.play import play_head  # noqa: E402
from kagsym.plan import play_plan, Plan  # noqa: E402
from kagsym.seeds import RESERVED  # noqa: E402


def _batch(rows):
    x = torch.as_tensor(np.stack([r["x"] for r in rows]))
    y = {f: torch.as_tensor([r["y"][f] for r in rows]) for f in FIELDS}
    return x, y


def searched_plan(rows_of_horizon) -> Plan:
    """The searched schedule of one seed at one horizon, as a Plan."""
    rows_of_horizon = sorted(rows_of_horizon, key=lambda r: r["x"][0])
    return Plan(crop=tuple(r["plan"]["crop"] for r in rows_of_horizon),
                tiles=tuple(r["plan"]["tiles"] for r in rows_of_horizon),
                hands=tuple(r["plan"]["hands"] for r in rows_of_horizon),
                land=tuple(r["plan"]["land"] for r in rows_of_horizon),
                animals=tuple(r["plan"].get("animals", 0) for r in rows_of_horizon),
                selling=tuple(r["plan"]["selling"] for r in rows_of_horizon),
                load=tuple(r["plan"]["load"] for r in rows_of_horizon),
                water_last=tuple(r["plan"]["water_last"] for r in rows_of_horizon))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-days", default="")
    ap.add_argument("--holdout-seeds", default="", help="seeds held out (a single-horizon dataset splits by seed)")
    ap.add_argument("--data", default="runs/ladder/daysearch_*d.jsonl", help="glob of day-search records")
    ap.add_argument("--opponent", default=None, help="regret played against this rival (replay:<tape> or a public name)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seeds", type=int, default=5, help="reserved seeds 7106.. for the regret")
    ap.add_argument("--out", default="runs/plan_head.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    rows = load(a.data)
    hold = {int(d) for d in a.holdout_days.split(",") if d}
    hold_seeds = {int(x) for x in a.holdout_seeds.split(",") if x}
    is_test = lambda r: (r["days"] in hold) or (r["seed"] in hold_seeds)
    train = [r for r in rows if not is_test(r)]
    test = [r for r in rows if is_test(r)]
    print(f"{len(rows)} records, horizons {sorted({r['days'] for r in rows})}, seeds {len({r['seed'] for r in rows})}; "
          f"train {len(train)}, held-out {len(test)} (days {sorted(hold)}, seeds {sorted(hold_seeds)})")
    model = PlanHead(len(rows[0]["x"]), a.width)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    xt, yt = _batch(train)
    xv, yv = _batch(test) if test else (None, None)
    ce = torch.nn.CrossEntropyLoss()
    for ep in range(1, a.epochs + 1):
        model.train()
        out = model(xt)
        loss = sum(ce(out[f], yt[f]) for f in FIELDS)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == a.epochs:
            model.eval()
            with torch.no_grad():
                agree = {f: float((model(xt)[f].argmax(-1) == yt[f]).float().mean()) for f in FIELDS}
                line = f"ep {ep:4d} loss {float(loss):.3f} train agree " + " ".join(f"{f[:5]} {v:.2f}" for f, v in agree.items())
                if xv is not None:
                    va = {f: float((model(xv)[f].argmax(-1) == yv[f]).float().mean()) for f in FIELDS}
                    line += " | held-out " + " ".join(f"{f[:5]} {v:.2f}" for f, v in va.items())
            print(line, flush=True)
    torch.save({"state_dict": model.state_dict(), "n_in": len(rows[0]["x"]), "width": a.width, "fields": FIELDS}, a.out)
    # Regret on held-out horizons, reserved seeds the search never saw.
    model.eval()
    print(f"regret (money of the head's plan vs the searched plan; opponent {a.opponent or 'passive'}):")
    for d in sorted({r["days"] for r in rows}):
        by_seed = {}
        for r in rows:
            if r["days"] == d:
                by_seed.setdefault(r["seed"], []).append(r)
        if hold_seeds:
            # per seed: the searched schedule of THAT seed against the head on the same seed
            for s in sorted(by_seed):
                ref = searched_plan(by_seed[s])
                m_ref = play_plan(ref, s, days=d, opponent=a.opponent)
                m_head = play_head(model, s, d, opponent=a.opponent)
                tag = "HELD-OUT" if s in hold_seeds else "train"
                print(f"  {d:2d} days seed {s} {tag:8s} searched {m_ref:9,.0f}  head {m_head:9,.0f}  regret {m_ref - m_head:+9,.0f} ({(m_ref - m_head) / max(1, m_ref):+.1%})", flush=True)
            continue
        seeds = RESERVED.seeds(a.eval_seeds, 5)
        ref = searched_plan(next(iter(by_seed.values())))
        m_ref = np.mean([play_plan(ref, s, days=d, opponent=a.opponent) for s in seeds])
        m_head = np.mean([play_head(model, s, d, opponent=a.opponent) for s in seeds])
        tag = "HELD-OUT" if d in hold else "train"
        print(f"  {d:2d} days {tag:8s} searched {m_ref:9,.0f}  head {m_head:9,.0f}  regret {m_ref - m_head:+9,.0f} ({(m_ref - m_head) / max(1, m_ref):+.1%})", flush=True)


if __name__ == "__main__":
    main()
