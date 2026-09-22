"""Kaggriculture: exact executor plus learned policies.

The split follows the action space, and it is a split by competence:

    symbolic/  the engine and everything with an exact algorithm that fits in
               one second: legality, integer arithmetic, Hungarian assignment,
               end-of-season liquidation.
    nets/      the only parts that depend on a value judgement or on the
               opponent: what each tile is worth and which verb to apply
               (micro), and the day's targets (macro).

Legality is a fact and belongs to the engine. Preference is not, so it is
learned. Nothing in between is guessed: every constant that cannot be derived
from the engine is exposed as a learnable parameter of the macro vector.
"""
