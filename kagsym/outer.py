"""The outer loop: the KL target is set by measured money, not by hand.

The learning rate is not chosen -- a controller moves it to hold a measured
KL. But the KL TARGET it holds was a hand-picked 0.01, and in a project whose
rule is that nothing outside the engine stays guessed, that was the most
expensive guess left: it governs the speed of everything below it.

It cannot be learned by the inner loop, and not because the policy would
cheat. There is simply no gradient: the target does not affect the return of
an episode, it affects how the policy is updated BETWEEN episodes. It lives
outside the episode, like the weights of the reward, and like them it belongs
to an outer level evaluated against the TRUE objective.

So: every N updates, play the current policy and the previous one on the SAME
seeds against a fixed opponent, and difference them. Paired, because with 8
seeds the standard error of an absolute measurement is about $4,700 and a
controller chasing that noise is worse than a constant; playing the same seeds
cancels the scenario variance. If the policy has genuinely improved, the
target goes up and learning is allowed to be faster. If it has degraded, the
target comes down.

This is what nobody was doing when four runs in a row improved against their
training opponents while getting worse against a fixed one. The only thing
that noticed was a person reading a table.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# Same anchor as the lr controller: symmetric in log space, so a good check
# and a bad one move the target by the same factor and there is no ratchet.
FACTOR = 0.7
FLOOR, CEILING = 0.002, 0.08


def _play(net, seeds, hours, days, cash, passive):
    """Final money of `net` on each seed, against a fixed opponent."""
    from . import obs as O, spec
    from .symbolic import tasks as T
    from .symbolic.executor import Agent
    from .environment import public_with_cap
    from .fastenv import FastEnv
    from .macro import Macro, N_MACRO
    from .nets.world import N_HIST
    out = []
    K = int(getattr(net, "n_keys", 0))
    nops = int(getattr(net, "n_ops", 0))
    hist = torch.zeros(1, N_HIST)
    for s in seeds:
        env = FastEnv(configuration={"episodeSteps": hours * days,
                                     "turnsPerDay": hours,
                                     "startingMoney": cash}, seed=s)
        o = env.reset()
        ag = Agent(episode_steps=hours * days,
                   macro=Macro.from_vector([0.5] * N_MACRO))
        if passive:
            from kaggle_environments.envs.kaggriculture import kaggriculture as _E
            rv = _E.pass_agent
        else:
            rv = public_with_cap("v48-fast-routes", 10 ** 6)
        day = None
        with torch.no_grad():
            while not env.done:
                ob = o[0]
                if day != ob["day"]:
                    gr, b = O.encode_obs(ob)
                    res = net(torch.from_numpy(gr).unsqueeze(0),
                              torch.from_numpy(b).unsqueeze(0), hist)
                    ag.macro = Macro.from_vector(
                        torch.sigmoid(res["macro_mu"])[0].cpu().numpy())
                    mp = res["micro"][0].cpu().numpy()
                    if K:
                        ag.micro = (lambda _o, m=(mp[0], mp[1:1 + nops],
                                                  mp[1 + nops:1 + nops + K],
                                                  mp[1 + nops + K:]): m)
                    else:
                        ag.micro = (lambda _o, m=(mp[0], mp[1:]): m)
                    day = ob["day"]
                try:
                    a1 = rv(o[1])
                except Exception:
                    a1 = {"farmer": ["PASS"], "hands": [], "market": []}
                o, _ = env.step([ag(ob), a1])
        out.append(float(env.rewards()[0]))
    return np.array(out)


def check(net, prev_sd, cfg, seeds, hours, days, cash, passive=False):
    """(paired difference, its standard error, money now) against `prev_sd`.

    `prev_sd` is the state dict of the policy at the previous check, so the
    comparison is against ourselves N updates ago on the same seeds.
    """
    from .nets.world import E2EAgent
    was = net.training
    net.eval()
    dev = next(net.parameters()).device
    now = _play(net.to("cpu"), seeds, hours, days, cash, passive)
    net.to(dev)
    if prev_sd is None:
        if was:
            net.train()
        return None, None, float(now.mean())
    old = E2EAgent(cfg)
    old.load_state_dict(prev_sd)
    old.eval()
    before = _play(old, seeds, hours, days, cash, passive)
    if was:
        net.train()
    d = now - before
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
    return float(d.mean()), se, float(now.mean())


def new_target(target, diff, se):
    """Raise the target when the money improved, lower it when it degraded.

    The dead zone is 2 standard errors of the PAIRED difference, so the
    controller never moves on noise -- which is the only way a loop closed on a
    noisy measurement is better than a constant.
    """
    if diff is None or se is None or not math.isfinite(se):
        return target, "sin medida"
    if diff > 2 * se:
        return min(CEILING, target / FACTOR), "mejora"
    if diff < -2 * se:
        return max(FLOOR, target * FACTOR), "empeora"
    return target, "empate"
