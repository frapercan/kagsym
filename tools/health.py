#!/usr/bin/env python
"""Health of a trained checkpoint: did learning happen, and where.

Reads what the optimiser state proves rather than what the log says:

  * a parameter with no Adam state, or with exp_avg_sq exactly zero, never
    received a gradient (the aux/JEPA heads were found this way);
  * the exploration widths (log_sigma, log_sigma_micro) must be alive, i.e.
    away from the floor and not collapsed;
  * the game fingerprint stored in the checkpoint must be the current code's,
    or every number measured on it describes another game.

Exit code 1 on any failed check. Used by tools/pipeline.py after training.

    python tools/health.py runs/model.pt [--require critic,macro_mu,micro]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def head_of(name: str) -> str:
    return name.split(".")[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--require", default="world,cuerpo,macro_mu,micro,critic",
                   help="heads that must have received gradient")
    p.add_argument("--sigma-macro-min", type=float, default=0.05)
    p.add_argument("--sigma-macro-max", type=float, default=2.0)
    p.add_argument("--allow-fingerprint-mismatch", action="store_true")
    a = p.parse_args()
    import torch
    from kagsym.version import fingerprint
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
        ok = ok and bool(cond)

    print(f"[health] {a.checkpoint}  upd {ck.get('upd')}  seed {ck.get('seed')}")
    fp = ck.get("fingerprint")
    check(fp == fingerprint() or a.allow_fingerprint_mismatch,
          f"game fingerprint {fp} vs current {fingerprint()}")

    opt, names = ck.get("opt"), ck.get("opt_nombres")
    check(opt is not None and names is not None, "optimizer state and parameter names stored")
    if opt and names:
        idx2name = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
        state = opt["state"]
        with_grad, without = {}, {}
        for g in opt["param_groups"]:
            for i in g["params"]:
                name = idx2name.get(i, idx2name.get(str(i), f"?{i}"))
                h = head_of(str(name))
                s = state.get(i)
                got = s is not None and float(s["exp_avg_sq"].abs().max()) > 0
                (with_grad if got else without).setdefault(h, 0)
                if got:
                    with_grad[h] += 1
                else:
                    without[h] += 1
        for h in a.require.split(","):
            check(with_grad.get(h, 0) > 0 and without.get(h, 0) == 0,
                  f"head {h}: {with_grad.get(h, 0)} tensors with gradient, {without.get(h, 0)} without")
        never = sorted(h for h in without if h not in with_grad)
        print(f"  info: heads that never received gradient: {never or 'none'}")

    sd = ck["sd"]
    if "log_sigma" in sd:
        s = sd["log_sigma"].exp()
        check(a.sigma_macro_min < float(s.mean()) < a.sigma_macro_max,
              f"macro sigma mean {float(s.mean()):.3f} (min {float(s.min()):.3f}, max {float(s.max()):.3f})")
    if "log_sigma_micro" in sd:
        s = sd["log_sigma_micro"].exp()
        check(float(s[0]) > 0.01, f"micro value sigma {float(s[0]):.3f}")
    for k, v in sd.items():
        if not torch.isfinite(v).all():
            check(False, f"non-finite values in {k}")
            break
    else:
        check(True, "all tensors finite")
    print(f"[health] {'OK' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
