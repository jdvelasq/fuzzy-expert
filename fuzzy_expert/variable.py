import numpy as np

from .modifiers import apply_modifiers
from .mf import gaussmf, gbellmf, pimf, smf, sigmf, trimf, zmf, trapmf
from .core import plot_fuzzyvariable


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

        self.evaluate_membeships()

    def evaluate_tuple(self, expression):

        fn, *params = expression

        if fn == "gaussmf":
            return gaussmf(center=params[0], sigma=params[1])

        if fn == "gbellmf":
            return gbellmf(center=params[0], sigma=params[1], b=params[2])

        if fn == "pimf":
            return pimf(a=params[0], b=params[1], c=params[2], d=params[3])

        if fn == "sigmf":
            return sigmf(center=params[0], alpha=params[1])

        if fn == "smf":
            return smf(a=params[0], b=params[1])

        if fn == "trapmf":
            return trapmf(a=params[0], b=params[1], c=params[2], d=params[3])

        if fn == "trimf":
            return trimf(a=params[0], b=params[1], c=params[2])

        if fn == "zmf":
            return zmf(a=params[0], b=params[1])

    def evaluate_membeships(self):
        """Transforms fuzzysets specified using membership functions to tuples"""

        for key, value in self.terms.items():
            if isinstance(value, tuple):
                self[key] = self.evaluate_tuple(expression=value)

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

    def fuzzificate(self, value, term, modifiers):
        """Computes the value of the membership function on a specifyied point of the universe for the fuzzy set.

        Args:
            value (float, numpy.array): point to evaluate the value of the membership function.
            term (string): name of the fuzzy set.
            modifiers (string): membership function modifier.


        Returns:
            A float number or numpy.array.
        """

        membership = self.terms[term]
        membership = apply_modifiers(membership=membership, modifiers=modifiers)

        xp = [xp for xp, _ in membership]
        fp = [fp for _, fp in membership]

        return np.interp(
            x=value,
            xp=xp,
            fp=fp,
        )

    def plot(self, fmt="-", linewidth=2, num=100):
        """Plots the fuzzy sets defined for the variable.

        Args:
            figsize (tuple): figure size.

        """
        memberships = []
        universe = np.linspace(start=self.universe[0], stop=self.universe[1], num=num)

        for term in self.terms.keys():
            memberships.append(self.fuzzificate(universe, term, None))

        plot_fuzzyvariable(
            universe=universe,
            memberships=memberships,
            labels=list(self.terms.keys()),
            title=self.name,
            fmt=fmt,
            linewidth=linewidth,
            view_xaxis=True,
            view_yaxis=True,
        )
