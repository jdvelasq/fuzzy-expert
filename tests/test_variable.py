"""Tests
"""

import numpy as np

from fuzzy_expert.variable import FuzzyVariable


def test_variable_init() -> None:
    """Test fuzzy variable creation"""
    name: str = "name"
    universe_range: tuple[int, int] = (0, 1)
    terms: dict = {}

    fuzzyvar: FuzzyVariable = FuzzyVariable(
        name=name, universe_range=universe_range, terms=terms
    )

    assert fuzzyvar.name == name
    assert fuzzyvar.universe_range == universe_range
    assert fuzzyvar.terms == terms
    assert (fuzzyvar.universe == np.linspace(start=0, stop=1, num=11)).all()


# score = FuzzyVariable(
#     name="score",
#     universe=(150, 200),
#     terms={
#         "High": [(175, 0), (180, 0.2), (185, 0.7), (190, 1)],
#         "Low": [(155, 1), (160, 0.8), (165, 0.5), (170, 0.2), (175, 0)],
#     },
# )


# class TestVariable:
#     def test_interpolate(self):

#         assert score.fuzzificate(value=170, term="High", modifiers=None) == 0
#         assert score.fuzzificate(value=200, term="High", modifiers=None) == 1
#         assert score.fuzzificate(value=187.5, term="High", modifiers=None) == 0.85

#     def test_interpolate_with_modifiers(self):

#         assert (
#             round(score.fuzzificate(value=187.5, term="High", modifiers=("NOT",)), 3)
#             == 0.15
#         )

#         assert (
#             score.fuzzificate(value=187.5, term="High", modifiers=("NOT", "NOT"))
#             == 0.85
#         )

#     def test_variable_creation_with_mf_functions(self):

#         temp = FuzzyVariable(
#             name="temp",
#             universe=(0, 100),
#             terms={
#                 "cold": ("zmf", 10, 26),
#                 "ok": ("pimf", 2, 16, 18, 36),
#                 "hot": ("smf", 37, 60),
#             },
#         )

#         assert len(temp["cold"]) == 9
#         assert temp["cold"][0] == (10.0, 1.0)
#         assert temp["cold"][-1] == (26.0, 0.0)
#         assert temp["cold"][4] == (18.0, 0.5)

#     def test_plot(self):

#         temp = FuzzyVariable(
#             name="temp",
#             universe=(0, 100),
#             terms={
#                 "cold": ("zmf", 10, 26),
#                 "ok": ("pimf", 2, 16, 18, 36),
#                 "hot": ("smf", 37, 60),
#             },
#         )

#         temp.plot()

#     def test_variable_creation_with_complex_functions(self):

#         temp = FuzzyVariable(
#             name="temp",
#             universe=(0, 100),
#             terms={
#                 "cold": ("zmf", 10, 26),
#                 "hot": ("smf", 37, 60),
#                 "warm": ("NOT", ("OR", ("hot", "cold"))),
#             },
#         )

#         assert len(temp["cold"]) == 9
#         assert temp["cold"][0] == (10.0, 1.0)
#         assert temp["cold"][-1] == (26.0, 0.0)
#         assert temp["cold"][4] == (18.0, 0.5)
