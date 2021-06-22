"""
Zadeh-Mamdani Rules
===============================================================================

"""


class FuzzyRule:
    """Creates a Zadeh-Mamdani fuzzy rule.

    Args:
        antecedents (list of tuples): Fuzzy variables in the rule antecedent.
        consequent (tuple): Fuzzy variable in the consequence.
        is_and (bool): When True, membership values are combined using the specified AND operator; when False, the OR operator is used.

    """

    def __init__(
        self,
        premises,
        consequences,
        cf: float = 1.0,
        threshold_cf: float = 0,
    ):
        self.premises = premises
        self.consequences = consequences
        self.rule_cf: float = cf
        self.threshold_cf: float = threshold_cf

    def __repr__(self):

        text = "IF  "
        space = " " * 4

        #
        # Premises
        #
        for i_premise, premise in enumerate(self.premises):

            if i_premise == 0:
                text += premise[0] + " IS"
                for t in premise[1:]:
                    text += " " + t
                text += "\n"
            else:
                text += space + premise[0] + " " + premise[1] + " IS"
                for t in premise[2:]:
                    text += " " + t
                text += "\n"

        text += "THEN\n"

        #
        # Consequences
        #
        for consequence in self.consequences:
            text += space + consequence[0] + " IS"
            for t in consequence[1:]:
                text += " " + t
            text += "\n"
        #
        # Certainty factors
        #
        text += "CF = {:.2f}\n".format(self.rule_cf)
        text += "Threshold-CF = {:.2f}\n".format(self.rule_cf)

        return text
