"""Clon exacto del motor de Kaggriculture, sin el envoltorio de kaggle-environments.

La clave: no re-implementamos las reglas, llamamos al `interpreter` del propio
motor instalado sobre un estado duck-typeado. Es exacto por construccion y no se
desincroniza cuando Kaggle publique una version nueva.

Lo que nos ahorramos frente a `kaggle_environments.make(...)`:
  - validacion de esquema y copia de acciones en cada turno
  - grabacion del historico completo de estados (720 estados profundos)
  - la maquinaria de agentes (subprocesos, timeouts, logs)

Uso tipico::

    env = FastEnv(seed=7)
    obs = env.reset()
    while not env.done:
        obs, done = env.step([a0, a1])
    print(env.rewards())

Para rollouts imaginados desde una observacion real (en partida)::

    env = FastEnv.from_observation(obs, seed_guess=0)
    env.step([my_action, guessed_opponent_action])
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from kaggle_environments.envs.kaggriculture import kaggriculture as _eng
from kaggle_environments.utils import Struct, structify

from . import spec

PASS_ACTION: dict = {"farmer": ["PASS"], "hands": [], "market": []}


# --- copia rapida -----------------------------------------------------------
# OJO con Struct: kaggle_environments.utils.Struct mantiene una copia paralela
# en __dict__, asi que `s["k"] = v` NO cambia `s.k`. El interpreter lee la
# accion por atributo (`s.action`) y el step por item (`get(obs,"step")`), asi
# que hay que escribir SIEMPRE por atributo para tocar los dos lados.
def _fast_copy(x: Any) -> Any:
    t = type(x)
    if t is list:
        return [_fast_copy(v) for v in x]
    if t is Struct:
        return Struct(**{k: _fast_copy(v) for k, v in x.items()})
    if isinstance(x, dict):
        return {k: _fast_copy(v) for k, v in x.items()}
    return x                          # int/float/str/bool/None: inmutables


def _agent_state(player: int, observation: dict) -> Struct:
    """Solo los dos niveles superiores necesitan acceso por atributo.

    Por debajo (farms, tiles, market, town) el motor usa item access y .get(),
    asi que dejarlos como dicts planos es identico en comportamiento y bastante
    mas barato de copiar.
    """
    return Struct(
        action=_fast_copy(PASS_ACTION),
        status="ACTIVE",
        reward=0.0,
        info={},
        observation=Struct(player=player, **observation),
    )


def _blank_observation() -> dict:
    return {
        "step": 0,
        "remainingOverageTime": 60,
        "farms": [],
        "private": {},
        "market": {},
        "town": {},
        "day": 0,
        "hour": 0,
    }


class FastEnv:
    """Motor de Kaggriculture sobre dicts planos."""

    def __init__(
        self,
        configuration: dict | None = None,
        seed: int | None = None,
        n_agents: int = 2,
    ):
        cfg = dict(spec.DEFAULT_CONFIG)
        if configuration:
            cfg.update(configuration)
        cfg.pop("seed", None)
        self.configuration = structify(cfg)
        self.info: dict = {"seed": seed if seed is not None else 0}
        self.n_agents = n_agents
        self.state: list = []
        self._n_recorded = 0

    # -- propiedades que el interpreter consulta ----------------------------
    @property
    def done(self) -> bool:
        return all(s.status != "ACTIVE" for s in self.state) if self.state else False

    @property
    def step_index(self) -> int:
        return self.state[0].observation["step"] if self.state else 0

    @property
    def day(self) -> int:
        return self.state[0].observation["day"]

    # -- ciclo de vida -------------------------------------------------------
    def reset(self) -> list:
        self.state = [_agent_state(i, _blank_observation()) for i in range(self.n_agents)]
        _eng.interpreter(self.state, self)          # rama _initialize
        self.state[0].observation.step = 0
        self._n_recorded = 1
        return self.observations()

    def step(self, actions: list[dict]) -> tuple[list, bool]:
        """Aplica un turno. `actions[i]` es el dict de accion del jugador i."""
        for i, a in enumerate(actions):
            self.state[i].action = a if isinstance(a, dict) else _fast_copy(PASS_ACTION)
        _eng.interpreter(self.state, self)
        self.state[0].observation.step = self._n_recorded
        self._n_recorded += 1
        if self.state[0].observation["step"] >= self.configuration["episodeSteps"] - 1:
            for s in self.state:
                if s.status in ("ACTIVE", "INACTIVE"):
                    s.status = "DONE"
        return self.observations(), self.done

    # -- lectura -------------------------------------------------------------
    def observations(self) -> list:
        """Observacion por jugador, con los campos compartidos ya propagados."""
        obs0 = self.state[0].observation
        for i in range(1, self.n_agents):
            oi = self.state[i].observation
            oi.farms = obs0["farms"]
            oi.market = obs0["market"]
            oi.town = obs0["town"]
            oi.day = obs0["day"]
            oi.hour = obs0["hour"]
            oi.step = obs0["step"]
        return [s.observation for s in self.state]

    def rewards(self) -> list[float]:
        farms = self.state[0].observation["farms"]
        return [float(f["money"]) for f in farms]

    # -- clonado para rollouts imaginados ------------------------------------
    def clone(self) -> "FastEnv":
        new = FastEnv.__new__(FastEnv)
        new.configuration = self.configuration
        new.info = dict(self.info)
        new.n_agents = self.n_agents
        new._n_recorded = self._n_recorded
        new.state = [_fast_copy(s) for s in self.state]
        # Re-enlaza los objetos compartidos: el interpreter escribe via farms[0],
        # y _fast_copy los habria duplicado por jugador.
        obs0 = new.state[0].observation
        for i in range(1, new.n_agents):
            oi = new.state[i].observation
            oi.farms = obs0["farms"]
            oi.market = obs0["market"]
            oi.town = obs0["town"]
        return new

    # -- reconstruccion desde una observacion real ---------------------------
    @classmethod
    def from_observation(
        cls,
        obs: Any,
        configuration: dict | None = None,
        seed_guess: int | None = None,
        assume_opponent_private: Callable[[Any, int], dict] | None = None,
    ) -> "FastEnv":
        """Construye un estado simulable desde la observacion de un agente.

        `seed_guess=None` (por defecto) sortea una semilla. Importa: hierbas y
        desbloqueo de tiendas salen de `Random((seed*1000003) ^ day)`, asi que
        con una semilla FIJA un rollout imaginado que cruce un cambio de dia
        genera siempre la misma tienda equivocada -y la tienda decide que
        productos drena el pueblo-. Con semilla aleatoria el motor las muestrea
        con la tasa y la distribucion correctas, que es lo mejor que se puede
        hacer con algo que no es observable.

        El cobertizo, las semillas y los inventarios del rival son privados: hay
        que suponerlos. `assume_opponent_private(obs, opp_id)` es el gancho donde
        enchufar un modelo de rival; por defecto se asume vacio, que es justo el
        sesgo que el residual neuronal tiene que aprender a corregir.
        """
        import random as _random
        me = int(obs["player"])
        n = len(obs["farms"])
        if seed_guess is None:
            seed_guess = _random.randrange(1 << 30)
        env = cls(configuration=configuration, seed=seed_guess, n_agents=n)
        farms = _fast_copy(obs["farms"])
        market = _fast_copy(obs["market"])
        town = _fast_copy(obs["town"])
        state = []
        for i in range(n):
            if i == me:
                private = _fast_copy(obs["private"])
            elif assume_opponent_private is not None:
                private = _fast_copy(assume_opponent_private(obs, i))
            else:
                private = _eng._new_private()
                # el rival tiene tantos inventarios como peones contratados
                private["inventories"] = [{} for _ in range(1 + len(farms[i]["hands"]))]
            state.append(_agent_state(i, {
                "step": int(obs["step"]),
                "remainingOverageTime": 60,
                "farms": farms,
                "private": private,
                "market": market,
                "town": town,
                "day": int(obs["day"]),
                "hour": int(obs["hour"]),
            }))
        env.state = state
        env._n_recorded = int(obs["step"]) + 1
        return env


def run_episode(
    policy0: Callable[[Any], dict],
    policy1: Callable[[Any], dict],
    seed: int = 0,
    configuration: dict | None = None,
    on_transition: Callable[[int, list, list, list], None] | None = None,
) -> tuple[list[float], int]:
    """Juega un episodio completo. Devuelve (recompensas, turnos jugados).

    `on_transition(step, obs_antes, acciones, obs_despues)` se llama en cada
    turno; es el gancho de recoleccion de datos.
    """
    env = FastEnv(configuration=configuration, seed=seed)
    obs = env.reset()
    steps = 0
    while not env.done:
        before = [_fast_copy(o) for o in obs] if on_transition else None
        actions = [policy0(obs[0]), policy1(obs[1])]
        obs, done = env.step(actions)
        steps += 1
        if on_transition:
            on_transition(steps - 1, before, actions, [_fast_copy(o) for o in obs])
        if done:
            break
    return env.rewards(), steps
