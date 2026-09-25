from kagsym import seeds as S


def test_families_are_disjoint():
    fams = [S.RESERVED, S.SEARCH, S.CLEAN, S.TRAINING]
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            S.assert_disjoint(a, b)


def test_seeds_and_lookup():
    assert S.RESERVED.seeds(3) == [7101, 7102, 7103]
    assert S.SEARCH.seeds(2, offset=16) == [9016, 9017]
    assert S.family_of(7150) == "reserved"
    assert S.family_of(9500) == "search"
    assert S.family_of(30001) == "clean"
    assert S.family_of(5) is None


def test_overrun_is_an_error():
    import pytest
    with pytest.raises(ValueError):
        S.RESERVED.seeds(201)
