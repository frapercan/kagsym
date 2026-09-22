"""The symbolic layer: nothing here has learned parameters of its own.

Everything it decides is either a fact of the engine or a function of the
vector the network emits once per day.
"""
from . import assignment, executor, market_ops, tasks  # noqa: F401
