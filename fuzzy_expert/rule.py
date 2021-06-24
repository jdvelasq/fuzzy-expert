"""
Zadeh-Mamdani Rules
===============================================================================

"""

from __future__ import annotations


class FuzzyRule:
    """Creates a Zadeh-Mamdani fuzzy rule.

    Args:
        antecedents (list of tuples): Fuzzy variables in the rule antecedent.
        consequent (tuple): Fuzzy variable in the consequence.
        is_and (bool): When True, membership values are combined using the specified AND operator; when False, the OR operator is used.

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
        text += "Threshold-CF = {:.2f}\n".format(self.rule_cf)

        return text
