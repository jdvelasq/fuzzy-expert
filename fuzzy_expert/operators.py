"""
Modifiers and operators
===============================================================================

"""

import numpy as np


# #############################################################################
#
#
# Unary operators
#
#
# #############################################################################


def extremely(membership: np.ndarray) -> np.ndarray:
    return np.power(membership, 3)


def intensify(membership: np.ndarray) -> np.ndarray:
    return np.where(
        membership <= 0.5, np.power(membership, 2), 1 - 2 * np.power(1 - membership, 2)
    )


def more_or_less(membership: np.ndarray) -> np.ndarray:
    return np.power(membership, 0.5)


def norm(membership: np.ndarray) -> np.ndarray:
    return membership / np.max(membership)


def not_(membership: np.ndarray) -> np.ndarray:
    return 1 - membership


def plus(membership: np.ndarray) -> np.ndarray:
    return np.power(membership, 1.25)


def somewhat(membership: np.ndarray) -> np.ndarray:
    return np.power(membership, 1.0 / 3.0)


def very(membership: np.ndarray) -> np.ndarray:
    return np.power(membership, 2)


def slightly(membership: np.ndarray) -> np.ndarray:
    plus_membership: np.ndarray = np.power(membership, 1.25)
    not_very_membership: np.ndarray = 1 - np.power(membership, 2)
    membership: np.ndarray = np.where(
        membership < not_very_membership, plus_membership, not_very_membership
    )
    membership: np.ndarray = membership / np.max(membership)
    return np.where(membership <= 0.5, membership ** 2, 1 - 2 * (1 - membership) ** 2)


def apply_modifiers(membership: np.ndarray, modifiers: list[str]) -> np.ndarray:
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


def prob_or(memberships: list[np.ndarray]) -> np.ndarray:
    """
    Probabilistic OR

    """
    result: np.ndarray = memberships[0]
    for membership in memberships[1:]:
        result: np.ndarray = result + membership - result * membership
    return np.maximum(1, np.minimum(1, result))


def bounded_prod(memberships: list[np.ndarray]) -> np.ndarray:
    """
    Bounded product: max(0, u + v - 1)

    """
    result: np.ndarray = memberships[0]
    for membership in memberships[1:]:
        result: np.ndarray = np.maximum(0, result + membership - 1)
    return result


def bounded_sum(memberships: list[np.ndarray]) -> np.ndarray:
    """
    Bounded sum: min(1, u + v)

    """
    result: np.ndarray = memberships[0]
    for membership in memberships[1:]:
        result: np.ndarray = np.minimum(1, result + membership)
    return result


def drastic_prod(memberships: list[np.ndarray]) -> np.ndarray:
    """
    Drastic product: u if v == 0
                     v if u == 1
                     0 if a,v < 1

    """
    result: np.ndarray = memberships[0]
    for membership in memberships[1:]:
        result: np.ndarray = np.where(
            membership == 0, result, np.where(result == 1, membership, 0)
        )
    return result


def drastic_sum(memberships: list[np.ndarray]) -> np.ndarray:
    """
    Drastic product: u if v == 0
                     v if u == 0
                     0 if a,v < 1

    """
    result: np.ndarray = memberships[0]
    for membership in memberships[1:]:
        result: np.ndarray = np.where(
            membership == 0, result, np.where(result == 1, membership, 1)
        )
    return result


def product(memberships: list[np.ndarray]) -> np.ndarray:
    result: np.ndarray = memberships[0]
    for membership in memberships[1:]:
        result: np.ndarray = result * membership
    return result


def maximum(memberships: list[np.ndarray]) -> np.ndarray:
    result: np.ndarray = memberships[0]
    for membership in memberships[1:]:
        result: np.ndarray = np.maximum(result, membership)
    return result


def minimum(memberships: list[np.ndarray]) -> np.ndarray:
    result: np.ndarray = memberships[0]
    for membership in memberships[1:]:
        result: np.ndarray = np.minimum(result, membership)
    return result


def aggregate(operator: str, memberships: list[np.ndarray]) -> np.ndarray:
    """Replace the fuzzy sets by a unique fuzzy set computed by the aggregation operator.

    Args:
        operator (string): {"max"|"sim"|"probor"} aggregation operator.

    Returns:
        A FuzzyVariable

    """
    result: np.ndarray = memberships[0]

    if operator == "max":
        for membership in memberships[1:]:
            result: np.ndarray = np.maximum(result, membership)
        return result

    if operator == "sum":
        for membership in memberships[1:]:
            result: np.ndarray = result + membership
        return np.minimum(1, result)

    if operator == "probor":
        for membership in memberships[1:]:
            result: np.ndarray = result + membership - result * membership
        return np.maximum(1, np.minimum(1, result))


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
        return 0.0

    return {
        "cog": cog,
        "coa": coa,
        "mom": mom,
        "lom": lom,
        "som": som,
    }[operator]()
