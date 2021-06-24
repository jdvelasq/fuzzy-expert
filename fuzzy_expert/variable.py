"""
Fuzzy Variables
===============================================================================

"""
from __future__ import annotations

from typing import List, Union

import numpy as np

from fuzzy_expert.mf import MembershipFunction
from fuzzy_expert.operators import apply_modifiers
from fuzzy_expert.plots import plot_crisp_input, plot_fuzzy_input, plot_fuzzy_variable


class FuzzyVariable:
    """Creates a fuzzy variable.

    :param unverse_range: Limits of the universe of discourse.
    :param terms: Dictionary where each term is the key of the dictionary, and the values is the membership function.
    :param step: Value controling the resolution for the discrete representation of the universe.

    >>> from fuzzy_expert.variable import FuzzyVariable
    >>> v = FuzzyVariable(
    ...     universe_range=(150, 200),
    ...     terms={
    ...         "High": [(175, 0), (180, 0.2), (185, 0.7), (190, 1)],
    ...         "Low": [(155, 1), (160, 0.8), (165, 0.5), (170, 0.2), (175, 0)],
    ...     },
    ... )
    >>> v.plot()

    .. image:: ./images/fuzzyvar.png
        :width: 350px
        :align: center

    """

    def __init__(
        self,
        universe_range: tuple[float, float],
        terms: Union[dict, None] = None,
        step: float = 0.1,
    ) -> None:

        if terms is None:
            terms: dict = {}
        self.universe_range: tuple[int, int] = universe_range
        self.terms: dict = terms

        self.min_u, self.max_u = universe_range
        num = int((self.max_u - self.min_u) / step) + 1
        self.universe = np.linspace(start=self.min_u, stop=self.max_u, num=num)

        for term in terms.keys():
            self.__setitem__(term, terms[term])

    def __setitem__(self, term: str, membership: Union[tuple, list]) -> None:
        """Sets the membership function values for the specified fuzzy set."""

        if isinstance(membership, tuple):
            self._set_term_from_tuple(term=term, membership=membership)
        if isinstance(membership, list):
            self._set_term_from_list(term=term, membership=membership)

    def _set_term_from_tuple(self, term: str, membership: type) -> None:
        """Sets the membership of a term when it is specified as a function"""

        mf = MembershipFunction()
        self._set_term_from_list(term=term, membership=mf(membership))

    def _set_term_from_list(
        self, term: str, membership: list[tuple[float, float]]
    ) -> None:
        """Sets the membership of a term when it is specified as a function"""

        xp: list[float] = [xp for xp, _ in membership]
        fp: list[float] = [fp for _, fp in membership]
        self.add_points_to_universe(points=xp)
        self.terms[term] = np.interp(x=self.universe, xp=xp, fp=fp)

    def add_points_to_universe(self, points):

        #
        # Adds new points to the universe
        #
        universe = np.append(self.universe, points)
        universe = np.where(universe < self.min_u, self.min_u, universe)
        universe = np.where(universe > self.max_u, self.max_u, universe)
        universe = np.unique(universe)
        universe = np.sort(universe)

        #
        # Expand existent membership functions with the new points
        #
        for term in self.terms.keys():

            if isinstance(self.terms[term], np.ndarray):
                self.terms[term] = np.interp(
                    x=universe, xp=self.universe, fp=self.terms[term]
                )

        #
        # Update the universe with the new points
        #
        self.universe = universe

    def __getitem__(self, term: str) -> np.ndarray:
        """
        Returns the membership function for the specified fuzzy set.

        """
        return self.terms[term]

    def get_modified_membeship(
        self, term: str, modifiers: Union[None, List[str]] = None
    ) -> np.ndarray:
        """Returns the membership modified values for the term.

        :param term: Name of the fuzzy set.
        :param modifiers: List of modifiers.

        >>> import matplotlib.pyplot as plt
        >>> from fuzzy_expert.variable import FuzzyVariable
        >>> v = FuzzyVariable(
        ...     universe_range=(150, 200),
        ...     terms={
        ...         "High": [(175, 0), (180, 0.2), (185, 0.7), (190, 1)],
        ...         "Low": [(155, 1), (160, 0.8), (165, 0.5), (170, 0.2), (175, 0)],
        ...     },
        ... )
        >>> y = v.get_modified_membeship('High' ,['extremely'])
        >>> _ = plt.plot(v.universe, v['High'], label='High')
        >>> _ = plt.plot(v.universe, y, label='extremely High')
        >>> _ = plt.legend()
        >>> plt.show()

        .. image:: ./images/hedges.png
            :width: 350px
            :align: center

        """

        membership: np.ndarray = self.terms[term]

        if modifiers is not None:
            membership: np.ndarray = apply_modifiers(membership, modifiers)

        return membership

    def plot(self, fmt: str = "-", linewidth: float = 3) -> None:
        """
        Plots a fuzzy variable.

        :param fmt: Format string passed to Matplotlib.pyplot.
        :param linewidth: Width of lines.

        """
        memberships = []

        for term in self.terms.keys():
            memberships.append(self.terms[term])

        plot_fuzzy_variable(
            universe=self.universe,
            memberships=memberships,
            labels=list(self.terms.keys()),
            title=None,
            fmt=fmt,
            linewidth=linewidth,
            view_xaxis=True,
            view_yaxis="left",
        )

    def plot_input(self, value, fuzzyset, view_xaxis=True, view_yaxis="left"):
        """Plots the fuzzy set, and the input.

        :param value: Crisp or fuzzy input value.
        :param fuzzyset: Term to plot.
        :param view_xaxis: Draw the x-axis of plot
        :param view_yaxis: Draw the y-axis of plot at left or right side.

        >>> from fuzzy_expert.variable import FuzzyVariable
        >>> v = FuzzyVariable(
        ...     universe_range=(150, 200),
        ...     terms={
        ...         "High": [(175, 0), (180, 0.2), (185, 0.7), (190, 1)],
        ...         "Low": [(155, 1), (160, 0.8), (165, 0.5), (170, 0.2), (175, 0)],
        ...     },
        ... )
        >>> v.plot_input(value=185, fuzzyset='High', view_xaxis=True, view_yaxis='right')


        .. image:: ./images/plot_crisp_input.png
            :width: 350px
            :align: center

        """

        if isinstance(value, (np.ndarray, list)):

            plot_fuzzy_input(
                value=value,
                universe=self.universe,
                membership=self.terms[fuzzyset],
                name=None,
                view_xaxis=view_xaxis,
                view_yaxis=view_yaxis,
            )

        else:

            plot_crisp_input(
                value=value,
                universe=self.universe,
                membership=self.terms[fuzzyset],
                name=None,
                view_xaxis=view_xaxis,
                view_yaxis=view_yaxis,
            )
