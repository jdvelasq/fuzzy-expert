"""
Fuzzy Variables
===============================================================================

"""
import numpy as np
from typing import List, Union
from fuzzy_expert.mf import MembershipFunction

from fuzzy_expert.operators import apply_modifiers
from fuzzy_expert.plots import plot_fuzzy_variable, plot_fuzzy_input, plot_crisp_input


class FuzzyVariable:
    def __init__(
        self,
        name: str,
        universe_range: tuple[int, int],
        terms: Union[dict, tuple, None] = None,
        step: float = 0.1,
    ) -> None:

        if terms is None:
            terms: dict = {}
        self.name: str = name
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
            title=self.name,
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
                name=self.name,
                view_xaxis=view_xaxis,
                view_yaxis=view_yaxis,
            )

        else:

            plot_crisp_input(
                value=value,
                universe=self.universe,
                membership=self.terms[fuzzyset],
                name=self.name,
                view_xaxis=view_xaxis,
                view_yaxis=view_yaxis,
            )


#
#
#   C O D E   T O   R E F A C T O R I N G
#
#


# import numpy as np

# from fuzzy_expert.mf import gaussmf, gbellmf, pimf, smf, sigmf, trimf, zmf, trapmf
#
# from fuzzy_expert.operators import get_modified_membership


# class FuzzyVariable:
#     """Creates a fuzzy variable.

#     Args:
#         name (string): variable name.
#         universe (list, numpy.array): list of points defining the universe of the variable.
#         sets (dict): dictionary where keys are the name of the sets, and the values correspond to the membership for each point of the universe.

#     Returns:
#         A fuzzy variable.

#     """

#     def __init__(
#         self,
#         name: str,
#         universe,
#         terms=None,
#         n_points: int = 9,
#         step_universe: float = 0.1,
#     ):

#         #
#         # universe -> (u_min, u_max)
#         # terms -> tuples / list of tuples
#         #
#         self.name = name
#         self.n_points = n_points

#         if terms is None:
#             self.terms = {}
#         else:
#             self.terms = terms

#         #
#         # internal attributes
#         #
#         if isinstance(universe, tuple):
#             self.min_u, self.max_u = universe
#             self.universe = np.arange(
#                 start=self.min_u, stop=self.max_u + 1e-3, step=step_universe
#             )
#         else:
#             self.universe = universe

#         #
#         # universe expansion
#         #
#         for term, membership in self.terms.items():

#             if isinstance(membership, tuple):
#                 self.expand_fuzzyset_from_tuple(term, membership)

#             if isinstance(membership, list):
#                 self.expand_fuzzyset_from_list(term, membership)


# # score = FuzzyVariable(
# #     name="score",
# #     n_points=20,
# #     universe=(150, 200),
# #     terms={
# #         "High": [(175, 0), (180, 0.2), (185, 0.7), (190, 1)],
# #         "Low": [(155, 1), (160, 0.8), (165, 0.5), (170, 0.2), (175, 0)],
# #         "test": ("trimf", 170, 180, 190),
# #     },
# # )
# # score.plot()

# #
# #
# #
# #


# # def expand_(self):

# #     for key, value in self.terms.items():
# #         if isinstance(value, tuple):
# #             self[key] = self.evaluate_tuple(expression=value)


# # def evaluate_tuple(self, expression):

# #     fn, *params = expression

# #     if fn == "gaussmf":
# #         return gaussmf(center=params[0], sigma=params[1])

# #     if fn == "gbellmf":
# #         return gbellmf(center=params[0], sigma=params[1], b=params[2])

# #     if fn == "pimf":
# #         return pimf(a=params[0], b=params[1], c=params[2], d=params[3])

# #     if fn == "sigmf":
# #         return sigmf(center=params[0], alpha=params[1])

# #     if fn == "smf":
# #         return smf(a=params[0], b=params[1])

# #     if fn == "trapmf":
# #         return trapmf(a=params[0], b=params[1], c=params[2], d=params[3])

# #     if fn == "trimf":
# #         return trimf(a=params[0], b=params[1], c=params[2])

# #     if fn == "zmf":
# #         return zmf(a=params[0], b=params[1])

# # def evaluate_membeships(self):
# #     """Transforms fuzzysets specified using membership functions to tuples"""

# #     for key, value in self.terms.items():
# #         if isinstance(value, tuple):
# #             self[key] = self.evaluate_tuple(expression=value)


# # def __getitem__(self, term):
# #     """Returns the membership function for the specified fuzzy set.

# #     Args:
# #         term (string): primary term

# #     Returns:
# #         A numpy array.

# #     """
# #     return self.terms[term]

# # def __setitem__(self, term, membership):
# #     """Sets the membership function values for the specified fuzzy set.

# #     Args:
# #         term (string): Fuzzy set name.
# #         memberships (list, numpy.array): membership values.

# #     """
# #     self.terms[term] = membership

# # def fuzzificate(self, value, term, modifiers):
# #     """Computes the value of the membership function on a specifyied point of the universe for the fuzzy set.

# #     Args:
# #         value (float, numpy.array): point to evaluate the value of the membership function.
# #         term (string): name of the fuzzy set.
# #         modifiers (string): membership function modifier.


# #     Returns:
# #         A float number or numpy.array.
# #     """

# #     membership = self.terms[term]
# #     membership = apply_modifiers(membership=membership, modifiers=modifiers)

# #     xp = [xp for xp, _ in membership]
# #     fp = [fp for _, fp in membership]

# #     return np.interp(
# #         x=value,
# #         xp=xp,
# #         fp=fp,
# #     )

# # def plot(self, fmt="-", linewidth=2, num=100):
# #     """Plots the fuzzy sets defined for the variable.

# #     Args:
# #         figsize (tuple): figure size.

# #     """
# #     memberships = []
# #     universe = np.linspace(start=self.universe[0], stop=self.universe[1], num=num)

# #     for term in self.terms.keys():
# #         memberships.append(self.fuzzificate(universe, term, None))

# #     plot_fuzzyvariable(
# #         universe=universe,
# #         memberships=memberships,
# #         labels=list(self.terms.keys()),
# #         title=self.name,
# #         fmt=fmt,
# #         linewidth=linewidth,
# #         view_xaxis=True,
# #         view_yaxis=True,
# #     )
