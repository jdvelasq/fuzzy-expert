"""Tests for fuzzy rules"""


from fuzzy_expert.rule import FuzzyRule


def test_repr():
    """
    Test function printing.

    """

    rule = FuzzyRule(
        premise=[("score", "HIGH"), ("AND", "ratio", "very", "LOW")],
        consequence=[("decision1", "BAD"), ("decision2", "very", "GOOD")],
    )

    text = rule.__repr__()
    assert (
        text
        == """IF  score IS HIGH
    AND ratio IS very LOW
THEN
    decision1 IS BAD
    decision2 IS very GOOD
CF = 1.00
Threshold-CF = 0.00
"""
    )
