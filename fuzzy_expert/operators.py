"""
Modifiers and operators
===============================================================================

"""

import numpy as np
import numpy.typing as npt
from typing import List

# #############################################################################
#
#
# Unary operators
#
#
# #############################################################################


def extremely(membership: npt.ArrayLike) -> npt.ArrayLike:
    return np.power(membership, 3)


def intensify(membership: npt.ArrayLike) -> npt.ArrayLike:
    return np.where(
        membership <= 0.5, np.power(membership, 2), 1 - 2 * np.power(1 - membership, 2)
    )


def more_or_less(membership: npt.ArrayLike) -> npt.ArrayLike:
    return np.power(membership, 0.5)


def norm(membership: npt.ArrayLike) -> npt.ArrayLike:
    return membership / np.max(membership)


def not_(membership: npt.ArrayLike) -> npt.ArrayLike:
    return 1 - membership


def plus(membership: npt.ArrayLike) -> npt.ArrayLike:
    return np.power(membership, 1.25)


def somewhat(membership: npt.ArrayLike) -> npt.ArrayLike:
    return np.power(membership, 1.0 / 3.0)


def very(membership: npt.ArrayLike) -> npt.ArrayLike:
    return np.power(membership, 2)


def slightly(membership: npt.ArrayLike) -> npt.ArrayLike:
    plus_membership: npt.ArrayLike = np.power(membership, 1.25)
    not_very_membership: npt.ArrayLike = 1 - np.power(membership, 2)
    membership: npt.ArrayLike = np.where(
        membership < not_very_membership, plus_membership, not_very_membership
    )
    membership: npt.ArrayLike = membership / np.max(membership)
    return np.where(membership <= 0.5, membership ** 2, 1 - 2 * (1 - membership) ** 2)


def apply_modifiers(membership: npt.ArrayLike, modifiers: List[str]) -> npt.ArrayLike:
    """
    Apply a list of modifiers or hedges to an array of memberships.

    """
    if modifiers is None:
        return membership

    fn = {
        "EXTREMELY": extremely,
        "INTENSIFY": intensify,
        "MORE_OR_LESS": more_or_less,
        "NORM": norm,
        "NOT": not_,
        "PLUS": plus,
        "SLIGHTLY": slightly,
        "SOMEWHAT": somewhat,
        "VERY": very,
    }

    membership = membership.copy()
    modifiers = list(modifiers)
    modifiers.reverse()

    for modifier in modifiers:
        membership = fn[modifier.upper()](membership)

    return membership


# #############################################################################
#
#
# Advanced operators
#
#
# #############################################################################


def prob_or(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Probabilistic OR

    """
    result: npt.ArrayLike = memberships[0]
    for membership in memberships[1:]:
        result: npt.ArrayLike = result + membership - result * membership
    return np.maximum(1, np.minimum(1, result))


def bounded_prod(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Bounded product: max(0, u + v - 1)

    """
    result: npt.ArrayLike = memberships[0]
    for membership in memberships[1:]:
        result: npt.ArrayLike = np.maximum(0, result + membership - 1)
    return result


def bounded_sum(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Bounded sum: min(1, u + v)

    """
    result: npt.ArrayLike = memberships[0]
    for membership in memberships[1:]:
        result: npt.ArrayLike = np.minimum(1, result + membership)
    return result


def drastic_prod(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Drastic product: u if v == 0
                     v if u == 1
                     0 if a,v < 1

    """
    result: npt.ArrayLike = memberships[0]
    for membership in memberships[1:]:
        result: npt.ArrayLike = np.where(
            membership == 0, result, np.where(result == 1, membership, 0)
        )
    return result


def drastic_sum(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Drastic product: u if v == 0
                     v if u == 0
                     0 if a,v < 1

    """
    result: npt.ArrayLike = memberships[0]
    for membership in memberships[1:]:
        result: npt.ArrayLike = np.where(
            membership == 0, result, np.where(result == 1, membership, 1)
        )
    return result


def product(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    result: npt.ArrayLike = memberships[0]
    for membership in memberships[1:]:
        result: npt.ArrayLike = result * membership
    return result


def maximum(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    result: npt.ArrayLike = memberships[0]
    for membership in memberships[1:]:
        result: npt.ArrayLike = np.maximum(result, membership)
    return result


def minimum(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    result: npt.ArrayLike = memberships[0]
    for membership in memberships[1:]:
        result: npt.ArrayLike = np.minimum(result, membership)
    return result


def defuzzificate(universe, membership, operator="cog"):
    """Computes a representative crisp value for the fuzzy set.

    Args:
        fuzzyset (string): Fuzzy set to defuzzify
        operator (string): {"cog"|"bisection"|"mom"|"lom"|"som"}

    Returns:
        A float value.

    """

    def cog():
        start = np.min(universe)
        stop = np.max(universe)
        x = np.linspace(start, stop, num=200)
        m = np.interp(x, xp=universe, fp=membership)
        return np.sum(x * m) / sum(m)

    def coa():
        start = np.min(universe)
        stop = np.max(universe)
        x = np.linspace(start, stop, num=200)
        m = np.interp(x, xp=universe, fp=membership)
        area = np.sum(m)
        cum_area = np.cumsum(m)
        return np.interp(area / 2, xp=cum_area, fp=x)

    def mom():
        maximum = np.max(membership)
        maximum = np.array([u for u, m in zip(universe, membership) if m == maximum])
        return np.mean(maximum)

    def lom():
        maximum = np.max(membership)
        maximum = np.array([u for u, m in zip(universe, membership) if m == maximum])
        return np.max(maximum)

    def som():
        maximum = np.max(membership)
        maximum = np.array([u for u, m in zip(universe, membership) if m == maximum])
        return np.min(maximum)

    if np.sum(membership) == 0.0:
        return np.mean(universe)

    return {
        "cog": cog,
        "coa": coa,
        "mom": mom,
        "lom": lom,
        "som": som,
    }[operator]()
