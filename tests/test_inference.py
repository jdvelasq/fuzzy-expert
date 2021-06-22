"""
Test inferecem method
"""
# from typing import Union

from fuzzy_expert.rule import FuzzyRule
from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.inference import DecompositionalInference


def test_loan_decision_problem() -> None:
    """
    Loan Bank Decision Problem

    """
    variables = [
        #
        FuzzyVariable(
            name="score",
            universe_range=(150, 200),
            terms={
                "High": [(175, 0), (180, 0.2), (185, 0.7), (190, 1)],
                "Low": [(155, 1), (160, 0.8), (165, 0.5), (170, 0.2), (175, 0)],
            },
        ),
        #
        FuzzyVariable(
            name="ratio",
            universe_range=(0.1, 1),
            terms={
                "Goodr": [(0.3, 1), (0.4, 0.7), (0.41, 0.3), (0.42, 0)],
                "Badr": [(0.44, 0), (0.45, 0.3), (0.5, 0.7), (0.7, 1)],
            },
        ),
        #
        FuzzyVariable(
            name="credit",
            universe_range=(0, 10),
            terms={
                "Goodc": [(2, 1), (3, 0.7), (4, 0.3), (5, 0)],
                "Badc": [(5, 0), (6, 0.3), (7, 0.7), (8, 1)],
            },
        ),
        #
        FuzzyVariable(
            name="decision",
            universe_range=(0, 10),
            terms={
                "Approve": [(5, 0), (6, 0.3), (7, 0.7), (8, 1)],
                "Reject": [(2, 1), (3, 0.7), (4, 0.3), (5, 0)],
            },
        ),
    ]

    rules = [
        #
        # Rule 1
        #
        FuzzyRule(
            cf=0.8,
            premises=[
                ("score", "High"),
                ("AND", "ratio", "Goodr"),
                ("AND", "credit", "Goodc"),
            ],
            consequences=[
                ("decision", "Approve"),
            ],
        ),
        #
        # Rule 2
        #
        FuzzyRule(
            cf=0.7,
            premises=[
                ("score", "Low"),
                ("AND", "ratio", "Badr"),
                ("OR", "credit", "Badc"),
            ],
            consequences=[
                ("decision", "Reject"),
            ],
        ),
    ]

    model = DecompositionalInference(
        and_operator="min",
        or_operator="max",
        implication_operator="Rc",
        composition_operator="max-min",  # min / prod
        production_link="max",
        defuzzification_operator="cog",
    )

    model(
        variables=variables,
        rules=rules,
        score=(190, 0.9),
        ratio=(0.39, 1),
        credit=(1.5, 1),
    )
