"""Tests
"""

import numpy as np

from fuzzy_expert.variable import FuzzyVariable


def test_variable_init_with_no_terms() -> None:
    """Test fuzzy variable creation"""
    universe_range: tuple[int, int] = (0, 1)
    terms: dict = {}

    fuzzyvar: FuzzyVariable = FuzzyVariable(universe_range=universe_range, terms=terms)

    assert fuzzyvar.universe_range == universe_range
    assert fuzzyvar.terms == terms
    assert (fuzzyvar.universe == np.linspace(start=0, stop=1, num=11)).all()


def test_add_terms_to_universe() -> None:
    """Verify universe expansion."""
    universe_range: tuple[int, int] = (0, 1)
    terms: dict = {}
    step: float = 0.5

    fuzzyvar: FuzzyVariable = FuzzyVariable(
        universe_range=universe_range, terms=terms, step=step
    )

    fuzzyvar.terms["A"] = np.linspace(start=0, stop=1, num=3)

    new_points: list = [0.0, 0.25, 0.75, 1.0]
    fuzzyvar.add_points_to_universe(points=new_points)

    assert (fuzzyvar.universe == np.linspace(start=0, stop=1, num=5)).all()
    assert (fuzzyvar.terms["A"] == np.linspace(start=0, stop=1, num=5)).all()


def test_set_term_from_list() -> None:
    """Check term addition speecified as a list"""

    universe_range: tuple[int, int] = (0, 1)
    terms: dict = {}
    step: float = 0.5

    fuzzyvar: FuzzyVariable = FuzzyVariable(
        universe_range=universe_range, terms=terms, step=step
    )

    term = "A"
    membership: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]

    fuzzyvar[term] = membership

    assert list(fuzzyvar.terms.keys()) == [term]
    assert (fuzzyvar.terms["A"] == np.linspace(start=0, stop=1, num=3)).all()


def test_set_term_from_tuple() -> None:
    """Check term addition speecified as a list"""

    universe_range: tuple[int, int] = (0, 1)
    terms: dict = {}
    step: float = 0.5

    fuzzyvar: FuzzyVariable = FuzzyVariable(
        universe_range=universe_range, terms=terms, step=step
    )

    term = "A"
    membership: tuple = ("trimf", 0, 0.5, 1)

    fuzzyvar[term] = membership

    assert list(fuzzyvar.terms.keys()) == [term]
    assert (fuzzyvar.terms["A"] == [0, 1, 0]).all()
