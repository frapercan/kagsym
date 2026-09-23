"""Rolling-horizon search on the exact engine.

At every day boundary: draw K macro vectors from the policy's OWN Gaussian,
apply each for that one day, roll the rest of the episode out with the policy,
and keep the one that ends with most money. Then advance one real day and
repeat.

WHY IT IS SOUND HERE AND NOT ELSEWHERE. `FastEnv` is an exact clone of the
engine, so a rollout is not a prediction: with a passive opponent and a fixed
seed it is exactly what would happen. The search returns the exact best of its
candidates, and nothing about it is limited by model error -- only by how many
candidates the budget buys. Against an unknown opponent that exactness breaks
and the rollout is only as good as the opponent model.

WHY THE CANDIDATES COME FROM `net.log_sigma`. It introduces no new constant:
the search explores exactly the uncertainty the policy already declares. That
is also what makes the result distillable -- it is a better sample from a
distribution the network already represents.

THE BUDGET -- READ FROM THE ENGINE, after estimating it wrong twice. The
configuration says `actTimeout` 1 s and the reset observation carries
`remainingOverageTime` 60. So the agent gets ONE SECOND PER CALL, and it is
called every turn -- 720 times -- plus a single 60 s pool for the whole episode.
Unused per-turn time does NOT accumulate; only the pool carries.

Timed on one thread, K=16 costs 11.67 s on day 0, where every rollout is a full
720 turns, and less as the horizon shortens: ~181 s over an episode.

    available at the 30 day boundaries    30 s   (1 s each)
    needed from the pool                 ~151 s
    the pool                               60 s
    -> AS WRITTEN IT DOES NOT FIT

That does not touch the measurements below, which are facts about the game. It
means the search is not deployable in this shape. The way in is to spread it
across the 24 calls of a day, ~1 s each, choosing the NEXT day's vector: that
buys 24 s per decision without touching the pool, at the cost of starting the
rollout from a state one day old -- so it stops being exact, and how much that
approximation costs is itself measurable.

MEASURED, paired seeds. The opponent is part of the result, so both regimes are
reported -- and the search is the same size in each, which is what says the gain
is not an artefact of a passive market:

    8h x 14d   K=16  passive  19,294 -> 19,697  +403 +- 27       t +14.8  24/24
    24h x 30d  K=16  passive  63,996 -> 87,944  +23,948 +- 2,606 t  +9.2  12/12
    24h x 30d  K=48  passive  70,214 -> 99,125  +28,911 +- 3,331 t  +8.7  12/12
    24h x 30d  K=16  v48      41,246 -> 65,180  +23,934 +- 1,857 t +12.9  12/12

Against v48 the rollouts carry v48 too, so they stay exact. That run moves the
margin against v48 from -64.9% to about -46%.

WHAT DOES NOT WORK, so it is not tried again. Distilling the search into
`macro_mu` does not: with honest held-out validation the head cannot improve on
its prediction of the search's choices at all (early stopping fired at epoch 90
with the validation mse unchanged), the mean displacement of those choices from
the policy's own mean is 0.017 sigmas -- noise -- and random one-sigma
perturbations of the head's bias lost 10 times out of 10. One unvalidated fit
did gain +4,745 against a passive opponent, but it lost -1,753 against v48
(t -2.02): it had specialised to a market nobody competes for. The search wins
by LOOKING, not by knowing where to look, so it is executed and not learned.

AND THE TRAP THAT ALMOST BURIED IT. The first measurement came out at -342
with t -2.5 and looked like a clean negative. The rollout was starting from a
FRESH executor, and `Agent` keeps state across turns -- `_destinations`,
`prev`, and above all `turns_per_tile`, a self-recalibrating estimate of what
it costs to keep a tile alive. Verified: rolling out the policy's own vector
reproduced the real continuation exactly at day 3 and was $389 off at day 7,
with every seed collapsing to the same number. `agent` must therefore be a deep
copy of the executor that is actually playing. Any counterfactual measurement
here should first check that its null case reproduces exactly.
"""
from __future__ import annotations

import copy

import numpy as np
import torch


def _rollout(env, net, agent, encode, first_macro=None, rival_fn=None):
    """Play `env` to the end. `first_macro` overrides only the day in progress."""
    from . import spec
    from .macro import Macro
    from .nets import world as M
    hist = torch.zeros(1, M.N_HIST)
    ag = copy.deepcopy(agent)
    day, forced = None, first_macro
    with torch.no_grad():
        while not env.done:
            ob = env.observations()[0]
            if day != ob["day"]:
                gr, b = encode(ob)
                out = net(torch.from_numpy(gr).unsqueeze(0),
                          torch.from_numpy(b).unsqueeze(0), hist)
                mv = (forced if forced is not None
                      else torch.sigmoid(out["macro_mu"])[0].numpy())
                forced = None
                ag.macro = Macro.from_vector(mv)
                mp = out["micro"][0].numpy()
                ag.micro = (lambda _o, m=(mp[0], mp[1:]): m)
                day = ob["day"]
            rv = rival_fn(env.observations()[1]) if rival_fn else {
                "farmer": ["PASS"], "hands": [], "market": []}
            env.step([ag(ob), rv])
    return float(env.rewards()[0])


def choose_macro(env, net, agent, ob, out, k: int, generator=None,
                 rival_fn=None):
    """The macro vector for today, chosen by exact rollout among k candidates.

    Returns (vector, index chosen, values). Index 0 is always the policy's own
    mean, so `index != 0` counts how often the search improved on it.
    """
    from . import obs as O
    mu = out["macro_mu"]
    sigma = net.log_sigma.exp()
    cands = [torch.sigmoid(mu)[0].numpy()]
    for _ in range(max(0, k - 1)):
        e = torch.randn(mu.shape, generator=generator)
        cands.append(torch.sigmoid(mu + sigma * e)[0].numpy())
    vals = [_rollout(env.clone(), net, agent, O.encode_obs, first_macro=c,
                     rival_fn=rival_fn) for c in cands]
    i = int(np.argmax(vals))
    return cands[i], i, vals
