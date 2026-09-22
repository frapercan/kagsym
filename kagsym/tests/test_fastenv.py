"""FastEnv tiene que ser indistinguible del motor real, turno a turno.

Si este test falla, todo lo que hay encima (datos, residual, DAgger) esta
aprendiendo sobre una dinamica equivocada. Es el test que no se puede saltar.
"""
from __future__ import annotations

import json
import random
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as _eng

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kagsym.fastenv import FastEnv, _fast_copy


def _plain(x):
    """Normaliza Struct/dict/list a algo comparable y hasheable por json."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def _snapshot(obs_list):
    o0 = obs_list[0]
    return _plain({
        "day": o0["day"], "hour": o0["hour"], "step": o0["step"],
        "farms": o0["farms"], "market": o0["market"], "town": o0["town"],
        "private": [o["private"] for o in obs_list],
    })


# random_agent del motor usa random.Random() sin semilla: no es reproducible.
# Usamos nuestras politicas deterministas para poder comparar los dos motores.
from kagworld.policies import make_random_policy, make_scripted_policy


def test_equivalence(n_steps=240, seed=1234, verbose=True):
    cfg = {"episodeSteps": n_steps + 2, "seed": seed}

    real = make("kaggriculture", configuration=cfg, debug=False)
    real.reset()
    real_obs = [s.observation for s in real.state]

    fast = FastEnv(configuration={"episodeSteps": n_steps + 2}, seed=seed)
    fast_obs = fast.reset()

    # El seed resuelto tiene que coincidir; si no, las hierbas divergen.
    assert real.info.get("seed") == fast.info["seed"], (
        f"seed real={real.info.get('seed')} fast={fast.info['seed']}")

    a = _snapshot(real_obs)
    b = _snapshot(fast_obs)
    assert a == b, "el estado inicial ya difiere"

    pol_real = [make_random_policy(seed=11), make_scripted_policy(seed=12, crop="WHEAT", ranch=True)]
    pol_fast = [make_random_policy(seed=11), make_scripted_policy(seed=12, crop="WHEAT", ranch=True)]

    for t in range(n_steps):
        acts_real = [pol_real[i](real_obs[i]) for i in range(2)]
        acts_fast = [pol_fast[i](fast_obs[i]) for i in range(2)]
        assert _plain(acts_real) == _plain(acts_fast), f"acciones divergen en t={t}"

        real.step([_fast_copy(x) for x in acts_real])
        real_obs = [s.observation for s in real.state]
        fast_obs, _ = fast.step([_fast_copy(x) for x in acts_fast])

        a, b = _snapshot(real_obs), _snapshot(fast_obs)
        if a != b:
            ka = json.dumps(a, sort_keys=True)
            kb = json.dumps(b, sort_keys=True)
            for key in a:
                if a[key] != b[key]:
                    print(f"  campo divergente: {key}")
                    print(f"    real: {json.dumps(a[key])[:400]}")
                    print(f"    fast: {json.dumps(b[key])[:400]}")
            raise AssertionError(f"estado divergente en t={t} (len {len(ka)} vs {len(kb)})")

    if verbose:
        print(f"OK: {n_steps} turnos identicos bit a bit (seed={seed})")
        print(f"    dinero final real={[f['money'] for f in real_obs[0]['farms']]}")
        print(f"    dinero final fast={[f['money'] for f in fast_obs[0]['farms']]}")
    return True


if __name__ == "__main__":
    for s in (1234, 99, 20260919):
        test_equivalence(n_steps=240, seed=s)
    print("\nTODOS LOS SEEDS OK")
