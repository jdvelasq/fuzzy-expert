"""
Zadeh-Mamdani Rules
===============================================================================

"""

from __future__ import annotations


class FuzzyRule:
    """Creates a Zadeh-Mamdani fuzzy rule.

    :param premise: List of propositions in rule premise.
    :param consequence: List of propositions in rule consequence.
    :param cf: Certainty factor of the rule.
    :param threshold_cf: Minimum certainty factor for rule firing.

    >>> from fuzzy_expert.rule import FuzzyRule
    >>> rule = FuzzyRule(
    ...     premise=[
    ...         ("score", "High"),
    ...         ("AND", "ratio", "Goodr"),
    ...         ("AND", "credit", "Goodc"),
    ...     ],
    ...     consequence=[("decision", "Approve")],
    ... )
    >>> rule
    IF  score IS High
        AND ratio IS Goodr
        AND credit IS Goodc
    THEN
        decision IS Approve
    CF = 1.00
    Threshold-CF = 0.00
    <BLANKLINE>
    """

    def __init__(
        self,
        premise,
        consequence,
        cf: float = 1.0,
        threshold_cf: float = 0,
    ):
        self.premise = premise
        self.consequence = consequence
        self.rule_cf: float = cf
        self.threshold_cf: float = threshold_cf

    def __repr__(self):

        text = "IF  "
        space = " " * 4

        #
        # Premise
        #
        for i_proposition, proposition in enumerate(self.premise):

            if i_proposition == 0:
                text += proposition[0] + " IS"
                for t in proposition[1:]:
                    text += " " + t
                text += "\n"
            else:
                text += space + proposition[0] + " " + proposition[1] + " IS"
                for t in proposition[2:]:
                    text += " " + t
                text += "\n"

        text += "THEN\n"

        #
        # Consequences
        #
        for proposition in self.consequence:
            text += space + proposition[0] + " IS"
            for t in proposition[1:]:
                text += " " + t
            text += "\n"
        #
        # Certainty factors
        #
        text += "CF = {:.2f}\n".format(self.rule_cf)
        text += "Threshold-CF = {:.2f}\n".format(self.threshold_cf)

        return text
