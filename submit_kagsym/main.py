"""Kaggle submission wrapper: `agent(obs, config)` returns the turn's action.

Everything that decides lives in `kagsym.policy.Policy`; this file only
survives the way Kaggle loads agents and never guesses at a plan.

How the Kaggle runner loads this file (kaggle_environments/agent.py):
  * it does NOT import it: it compiles the source and `exec`s it in a fresh
    namespace, so `__file__` is undefined and using it fails before the agent
    even loads ("Invalid raw Python: NameError");
  * it adds the agent's directory to `sys.path` only WHILE executing this
    source and pops it right after, so `kagsym` must be imported here, at
    exec time, not lazily inside `agent()`;
  * it returns the LAST callable defined in the file, so `agent` stays last;
  * it passes the agent's own path as `configuration["__raw_path__"]`.

Budget: 1 s per call, not cumulative, plus a 60 s overage pool per episode.
"""
import sys
import traceback

try:
    import kagsym                      # binds kagsym.__path__ while sys.path allows it
except Exception:
    kagsym = None

_STATE = {"policy": None, "episode_key": None}
_PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _load(config):
    from kagsym.policy import Policy, find_checkpoint
    policy = Policy.from_checkpoint(find_checkpoint(config), offset="checkpoint")
    policy.configure(config)
    return policy


def agent(obs, config=None):
    try:
        if _STATE["policy"] is None:
            _STATE["policy"] = _load(config)
        policy = _STATE["policy"]
        # A new episode in a reused process must not inherit executor state.
        if int(obs.get("step", 0)) == 0 or policy.agent is None:
            policy.reset(config)
        return policy.act(obs)
    except Exception:
        # The engine turns the impossible into a no-op, so a crash here would
        # be a silent $3,000 game. Leave the trace where the log shows it.
        traceback.print_exc(file=sys.stderr)
        return dict(_PASS)
