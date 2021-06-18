import numpy as np

from .modifiers import apply_modifiers


class FuzzyVariable:
    """Creates a fuzzy variable.

    Args:
        name (string): variable name.
        universe (list, numpy.array): list of points defining the universe of the variable.
        sets (dict): dictionary where keys are the name of the sets, and the values correspond to the membership for each point of the universe.

    Returns:
        A fuzzy variable.

    """

    def __init__(self, name, universe, terms):
        self.name = name
        self.universe = universe
        self.terms = terms

    def __getitem__(self, term):
        """Returns the membership function for the specified fuzzy set.

        Args:
            term (string): primary term

        Returns:
            A numpy array.

        """
        return self.terms[term]

    def __setitem__(self, term, membership):
        """Sets the membership function values for the specified fuzzy set.

        Args:
            term (string): Fuzzy set name.
            memberships (list, numpy.array): membership values.

        """
        self.terms[term] = membership

    def fuzzificate(self, value, term, modifiers=None):
        """Computes the value of the membership function on a specifyied point of the universe for the fuzzy set.

        Args:
            value (float, numpy.array): point to evaluate the value of the membership function.
            term (string): name of the fuzzy set.
            modifiers (string): membership function modifier.


        Returns:
            A float number or numpy.array.
        """

        membership = self.terms[term]

        if modifiers is not None:
            membership = apply_modifiers(membership=membership, modifiers=modifiers)

        xp = [xp for xp, _ in membership]
        fp = [fp for _, fp in membership]

        if value <= xp[0]:
            return fp[0]

        if value >= xp[-1]:
            return fp[-1]

        ## membership = self.apply_modifiers(term, modifiers)

        return np.interp(
            x=value,
            xp=xp,
            fp=fp,
        )
