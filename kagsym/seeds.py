"""Seed families: every episode played by any tool comes from a named family.

Why this exists. Three evaluators used three different seed ranges (601+,
7101+, 9001+) and the same two checkpoints differed by $6,800 between two of
them. A search ran on 9000-9959 while its "independent" contrast defaulted to
9001-9200. Training seeds advanced into the "reserved" evaluation range after
~645 updates. None of that is visible when ranges live as literals in scripts.

Rules:
  * A number is only comparable with another number from the SAME family.
  * SEARCH is what optimisers see. RESERVED is for validating what a search
    produced. CLEAN is a second, independent instrument. TRAINING is what the
    trainer may consume and it never overlaps the other three.
  * A family is a contiguous range; `seeds(family, n)` returns its first `n`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    name: str
    start: int
    size: int
    purpose: str

    @property
    def stop(self) -> int:
        return self.start + self.size

    def seeds(self, n: int | None = None, offset: int = 0) -> list[int]:
        n = self.size - offset if n is None else n
        if offset < 0 or offset + n > self.size:
            raise ValueError(f"family {self.name} has {self.size} seeds; "
                             f"asked for {n} from offset {offset}")
        return list(range(self.start + offset, self.start + offset + n))

    def __contains__(self, seed: int) -> bool:
        return self.start <= int(seed) < self.stop


RESERVED = Family("reserved", 7101, 200, "validate what a search produced")
MATRIX = Family("matrix", 7401, 200, "duels between our own checkpoints")
SEARCH = Family("search", 9000, 1000, "what optimisers see; never a result")
CLEAN = Family("clean", 30000, 1000, "second, independent instrument")
DIALS = Family("dials", 7401, 20, "live-dial probes (shares the matrix range)")
TRAINING = Family("training", 100000, 10_000_000, "what the trainer consumes")

FAMILIES = {f.name: f for f in (RESERVED, MATRIX, SEARCH, CLEAN, DIALS, TRAINING)}


def family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown seed family {name!r}; known: {sorted(FAMILIES)}")


def family_of(seed: int) -> str | None:
    for f in FAMILIES.values():
        if seed in f:
            return f.name
    return None


def assert_disjoint(a: Family, b: Family) -> None:
    if a.start < b.stop and b.start < a.stop:
        raise ValueError(f"seed families {a.name} and {b.name} overlap")


for _a in (RESERVED, MATRIX, SEARCH, CLEAN):
    assert_disjoint(_a, TRAINING)
assert_disjoint(RESERVED, SEARCH)
assert_disjoint(RESERVED, CLEAN)
assert_disjoint(SEARCH, CLEAN)
assert_disjoint(RESERVED, MATRIX)
