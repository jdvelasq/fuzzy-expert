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
    def __init__(
        self,
        universe_range: tuple[float, float],
        terms: Union[dict, tuple, None] = None,
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
            self.set_term_from_tuple(term=term, membership=membership)
        if isinstance(membership, list):
            self.set_term_from_list(term=term, membership=membership)

    def set_term_from_tuple(self, term: str, membership: type) -> None:
        """Sets the membership of a term when it is specified as a function"""

        mf = MembershipFunction()
        self.set_term_from_list(term=term, membership=mf(membership))

    def set_term_from_list(
        self, term: str, membership: list[tuple[float, float]]
    ) -> None:
        """Sets the membership of a term when it is specified as a function"""

        xp: list[float] = [xp for xp, _ in membership]
        fp: list[float] = [fp for _, fp in membership]
        self._add_points_to_universe(points=xp)
        self.terms[term] = np.interp(x=self.universe, xp=xp, fp=fp)

    def _add_points_to_universe(self, points):

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
        """
        Returns the membership modified values for the term.

        """

        membership: np.ndarray = self.terms[term]

        if modifiers is not None:
            membership: np.ndarray = apply_modifiers(membership, modifiers)

        return membership

    def plot(self, fmt: str = "-", linewidth: float = 3) -> None:
        """
        Plots a fuzzy variable.

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
