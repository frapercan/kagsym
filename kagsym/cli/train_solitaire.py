"""A minimal, clean learner for the solitaire: policy gradient with an exact baseline.

Why another trainer. The PPO trainer learned the 8-day solitaire from nothing
to 5,788 $ in 300 updates and then lost 2,182 $ over the next 900: its
critic never fitted (R^2 <= 0 throughout), its exploration width never
moved from its initial value (the learning rate on `log_sigma` was 7.5e-5),
and it optimises the return of the NOISY policy while what is deployed is
the mean. In an 8-day solitaire none of that machinery is needed:

  * the return is the final money, observed exactly, no critic;
  * the baseline is the deterministic policy's money on the SAME seed
    (common random numbers), which removes the board's variance exactly;
  * the mean policy's value on reserved seeds is measured inside the loop,
    every few updates, so the convergence curve is the log itself;
  * the exploration width has its own learning rate.

One update: B seeds from the TRAINING family; for each seed the mean policy
plays once (baseline) and M sampled policies play once each (macro sampled
per day from N(mu, sigma) on the live dials; the micro map stays the mean).
The gradient is the REINFORCE estimate with advantage (R - R_mean(seed)).

    python -m kagsym.cli.train_solitaire --scratch --days 8 --updates 300 --out runs/sol8.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from kagsym import evaluate as E, seeds as S  # noqa: E402
from kagsym.fastenv import FastEnv  # noqa: E402
from kagsym.macro import N_MACRO  # noqa: E402
from kagsym.policy import Policy  # noqa: E402

_G: dict = {}   # per-process: the network weights of the current update, the config


def _build_policy():
    """A Policy over the weights broadcast for this update (fork copies _G)."""
    import torch
    from kagsym.nets.world import E2EAgent, WorldConfig
    if "pol" not in _G:
        torch.set_num_threads(1)
        net = E2EAgent(WorldConfig(**_G["cfg"]))
        net.eval()
        _G["pol"] = Policy(net=net, offset="stored", stored_offset=_G.get("schedule"))
    _G["pol"].net.load_state_dict(_G["sd"])
    return _G["pol"]


def _episode(task):
    """(seed, sample index, update) -> money and the per-day samples.

    Sample index 0 is the mean policy: the baseline of its seed."""
    import torch
    from kagsym import obs as O
    seed, k, upd = task
    pol = _build_policy()
    world = _G["world"]
    cfg = {"episodeSteps": world["hours"] * world["days"], "turnsPerDay": world["hours"], "startingMoney": 3000}
    pol.configure(cfg)
    pol.reset()
    env = FastEnv(configuration=cfg, seed=seed)
    obs = env.reset()
    live = _G["live"]
    gen = torch.Generator().manual_seed(int(1_000_003 * upd + 1_009 * seed + k))
    records = []
    day = None
    while not env.done:
        ob = obs[0]
        if ob["day"] != day:
            day = ob["day"]
            g, b = O.encode_obs(ob, pol.agent._destinations)
            hf = np.asarray(O.rival_flow(ob), dtype=np.float32)
            with torch.no_grad():
                out = pol.net(torch.from_numpy(g).unsqueeze(0), torch.from_numpy(b).unsqueeze(0),
                              torch.from_numpy(hf).unsqueeze(0))
                mu = out["macro_mu"][0]
                off = pol.offset_for(int(day))
                if off is not None:          # the checkpoint's schedule shifts the mean
                    mu = mu + torch.from_numpy(off.vector(day / pol.days, N_MACRO))
                sigma = pol.net.log_sigma.exp()
                eps = torch.zeros(N_MACRO)
                if k > 0:
                    eps[live] = torch.randn(len(live), generator=gen)
                z = mu + sigma * eps
                vec = torch.sigmoid(z).numpy()
            shift = (off.vector(day / pol.days, N_MACRO) if off is not None
                     else np.zeros(N_MACRO, dtype=np.float32))
            records.append((g, b, hf, z.numpy(), shift))
            pol.plan_day(ob, macro_override=vec)
            pol._day = day
        a = pol.act(ob)
        obs, _ = env.step([a, dict(E.PASS_ACTION)])
    return seed, k, float(env.rewards()[0]), records


def _eval_task(seed):
    pol = _build_policy()
    world = _G["world"]
    cfg = {"episodeSteps": world["hours"] * world["days"], "turnsPerDay": world["hours"], "startingMoney": 3000}
    pol.configure(cfg)
    pol.reset()
    env = FastEnv(configuration=cfg, seed=seed)
    obs = env.reset()
    while not env.done:
        obs, _ = env.step([pol.act(obs[0]), dict(E.PASS_ACTION)])
    return float(env.rewards()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="start from these weights")
    ap.add_argument("--scratch", action="store_true", help="start from random weights")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--seeds-per-update", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4, help="sampled episodes per seed (plus the mean)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma0", type=float, default=None, help="reset the exploration width at start")
    ap.add_argument("--heads-only", action="store_true",
                    help="train only the macro head and sigma; the trunk and the micro map stay as loaded")
    ap.add_argument("--mu-penalty", type=float, default=1e-3,
                    help="quadratic penalty on pre-activations beyond +-4: a saturated sigmoid explores nothing")
    ap.add_argument("--lr-sigma", type=float, default=1e-2)
    ap.add_argument("--adv-scale", type=float, default=100.0, help="floor of the advantage scale, in dollars")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--live", default=None, help="live-dial json; default: every dial")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="solitaire")
    ap.add_argument("--out", default=None)
    ap.add_argument("--experiment", default=None)
    a = ap.parse_args()
    import multiprocessing as mp
    import torch
    from kagsym.nets.world import E2EAgent, WorldConfig
    from kagsym.policy import load_network
    from kagsym.tracking import Tracker, context_tags, describe
    from kagsym.version import fingerprint, model_fingerprint

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.out = a.out or os.path.join("runs", a.run_name + ".pt")
    if a.scratch == (a.checkpoint is not None):
        raise SystemExit("give exactly one of --scratch or --checkpoint")
    from kagsym.policy import Offset
    schedule = []
    if a.checkpoint:
        net, ck = load_network(a.checkpoint)
        cfg = ck["cfg"] if isinstance(ck["cfg"], dict) else vars(ck["cfg"])
        macro_fields = ck.get("macro_fields")
        schedule = Offset.schedule_from_checkpoint(ck)     # kept as the mean's shift, not learned
    else:
        cfg = vars(WorldConfig(device="cpu"))
        net = E2EAgent(WorldConfig(**cfg))
        macro_fields = None
    cfg = dict(cfg, device="cpu")
    net.train()
    live = list(range(N_MACRO))
    if a.live:
        with open(a.live) as f:
            live = sorted(int(i) for i in json.load(f)["live"])
    _G.update(cfg=cfg, world={"hours": a.hours, "days": a.days}, live=live, schedule=schedule)
    if a.sigma0 is not None:
        with torch.no_grad():
            net.log_sigma.fill_(float(np.log(a.sigma0)))

    sigma_params = [net.log_sigma]
    trained = ("macro_mu.",) if a.heads_only else ("world.", "cuerpo.", "macro_mu.")
    other = [p for n, p in net.named_parameters() if n.startswith(trained)]
    for n, p in net.named_parameters():
        p.requires_grad_(n.startswith(trained) or n == "log_sigma")
    opt = torch.optim.Adam([{"params": other, "lr": a.lr}, {"params": sigma_params, "lr": a.lr_sigma}])
    train_seeds = S.TRAINING.seeds(a.updates * a.seeds_per_update, offset=a.seed * 100_000)
    eval_seeds = S.RESERVED.seeds(a.eval_n)

    tracker = Tracker("training", a.run_name, params={**vars(a), "live_dials": len(live), "learner": "solitaire-pg"},
                      tags=context_tags("training", checkpoint=a.checkpoint, opponent="passive",
                                        seed_family="training", experiment=a.experiment,
                                        universe=f"{a.hours}hx{a.days}d", learner="solitaire-pg"),
                      description=describe([
                          f"Policy gradient with an exact per-seed baseline in the {a.hours}h x {a.days}d solitaire.",
                          f"{a.seeds_per_update} seeds x ({a.samples} samples + the mean) per update; sigma learns at lr {a.lr_sigma}.",
                          "1_result/eval_money is the MEAN policy on reserved seeds, measured inside the loop."])).start()
    print(f"[solitaire] {a.hours}h x {a.days}d  {'scratch' if a.scratch else a.checkpoint}  "
          f"{a.seeds_per_update} seeds x {a.samples} samples/update  live dials {len(live)}  -> {a.out}", flush=True)

    def save(upd):
        torch.save({"sd": net.state_dict(), "cfg": cfg, "macro_fields": macro_fields, "upd": upd,
                    "offset_schedule": [o.to_checkpoint() for o in schedule if o.until_day is not None or o.from_day is not None],
                    "offset": next((o.to_checkpoint() for o in schedule if o.until_day is None and o.from_day is None), None),
                    "seed": a.seed, "fingerprint": fingerprint(), "model_fingerprint": model_fingerprint(),
                    "learner": "solitaire-pg", "world": _G["world"]}, a.out + ".tmp")
        os.replace(a.out + ".tmp", a.out)

    ctx = mp.get_context("fork")
    best_eval = -np.inf
    for upd in range(1, a.updates + 1):
        t0 = time.time()
        _G["sd"] = {k: v.detach().clone() for k, v in net.state_dict().items()}
        seeds = train_seeds[(upd - 1) * a.seeds_per_update: upd * a.seeds_per_update]
        tasks = [(s, k, upd) for s in seeds for k in range(a.samples + 1)]
        with ctx.Pool(a.procs) as pool:
            results = pool.map(_episode, tasks, chunksize=1)
            evals = pool.map(_eval_task, eval_seeds, chunksize=1) if upd % a.eval_every == 0 or upd == 1 else None
        base = {s: r for s, k, r, _ in results if k == 0}
        sampled = [(s, r, rec) for s, k, r, rec in results if k > 0]
        raw = torch.tensor([(r - base[s]) for s, r, _ in sampled], dtype=torch.float32)
        adv = raw / max(float(raw.std()), a.adv_scale)      # scale by the batch's own spread, floored
        # recompute the log-probability of every sampled day with gradient
        G = torch.from_numpy(np.stack([rec[0] for _, _, recs in sampled for rec in recs]))
        B = torch.from_numpy(np.stack([rec[1] for _, _, recs in sampled for rec in recs]))
        H = torch.from_numpy(np.stack([rec[2] for _, _, recs in sampled for rec in recs]))
        Z = torch.from_numpy(np.stack([rec[3] for _, _, recs in sampled for rec in recs]))
        SH = torch.from_numpy(np.stack([rec[4] for _, _, recs in sampled for rec in recs]))
        owner = torch.tensor([i for i, (_, _, recs) in enumerate(sampled) for _ in recs])
        out = net(G, B, H)
        mu = out["macro_mu"] + SH
        sigma = net.log_sigma.exp()
        lp = torch.distributions.Normal(mu[:, live], sigma[live]).log_prob(Z[:, live]).sum(-1)
        pen = a.mu_penalty * torch.relu(mu.abs() - 4.0).pow(2).mean()
        loss = -(lp * adv[owner]).sum() / max(1, len(sampled)) + pen
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(other) + sigma_params, a.grad_clip)
        opt.step()
        with torch.no_grad():
            net.log_sigma.clamp_(min=np.log(0.02), max=np.log(1.5))
        m_base = float(np.mean(list(base.values())))
        m_samp = float(np.mean([r for _, r, _ in sampled]))
        metrics = {"1_result/train_money_mean_policy": m_base, "1_result/train_money": m_samp,
                   "2_policy/sigma_macro": float(sigma.mean()), "2_policy/adv_std": float(raw.std()),
                   "2_policy/mu_abs_max": float(mu.abs().max()),
                   "4_optim/grad_norm": float(gn), "4_optim/seconds": time.time() - t0}
        line = (f"upd {upd:4d}/{a.updates}  mean-policy {m_base:7,.0f}  sampled {m_samp:7,.0f}  "
                f"adv sd {float(raw.std()):5.0f}$  sigma {float(sigma.mean()):.3f}  |mu|max {float(mu.abs().max()):.1f}  |g| {float(gn):.1f}")
        if evals is not None:
            ev = float(np.mean(evals))
            se = float(np.std(evals, ddof=1) / np.sqrt(len(evals)))
            metrics.update({"1_result/eval_money": ev, "1_result/eval_se": se})
            line += f"  EVAL reserved {ev:7,.0f} +- {se:,.0f}"
            best_eval = max(best_eval, ev)
            save(upd)                       # the LAST state, every eval; never "the best"
        tracker.log(metrics, step=upd)
        print(line + f"  {time.time() - t0:.1f}s", flush=True)
    save(a.updates)
    tracker.end()


if __name__ == "__main__":
    main()
