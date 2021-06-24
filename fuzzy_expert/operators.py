"""
Modifiers and operators
===============================================================================

"""
from __future__ import annotations

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


def apply_modifiers(membership: npt.ArrayLike, modifiers: List[str]) -> npt.ArrayLike:
    """
    Apply a list of modifiers or hedges to a numpy array.

    :param membership: Membership values to be modified.
    :param modifiers: List of modifiers or hedges.

    >>> from fuzzy_expert.operators import apply_modifiers
    >>> x = [0.0, 0.25, 0.5, 0.75, 1]
    >>> apply_modifiers(x, ('not', 'very'))
    array([1.    , 0.9375, 0.75  , 0.4375, 0.    ])

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


def extremely(membership: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies the element-wise function fn(u) = u^3.

    :param membership: Membership function to be modified.

    >>> from fuzzy_expert.operators import extremely
    >>> extremely([0, 0.25, 0.5, 0.75, 1])
    array([0.      , 0.015625, 0.125   , 0.421875, 1.      ])

    """
    return np.power(membership, 3)


def intensify(membership: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies the element-wise function fn(u) = u^2 if u <= 0.5 else 1 - 2 * (1 - u)^2.

    :param membership: Membership function to be modified.

    >>> from fuzzy_expert.operators import intensify
    >>> intensify([0, 0.25, 0.5, 0.75, 1])
    array([0.    , 0.0625, 0.25  , 0.875 , 1.    ])

    """
    membership = np.array(membership)
    return np.where(
        membership <= 0.5,
        np.power(membership, 2),
        1 - 2 * np.power(1 - membership, 2),
    )


def more_or_less(membership: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies the element-wise function  fn(u) = u^(1/2).

    :param membership: Membership function to be modified.

    >>> from fuzzy_expert.operators import more_or_less
    >>> more_or_less([0, 0.25, 0.5, 0.75, 1])
    array([0.        , 0.5       , 0.70710678, 0.8660254 , 1.        ])

    """
    return np.power(membership, 0.5)


def norm(membership: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies the element-wise function  fn(u) = u / max(u).

    :param membership: Membership function to be modified.

    >>> from fuzzy_expert.operators import norm
    >>> norm([0, 0.25, 0.5])
    array([0. , 0.5, 1. ])

    """
    return membership / np.max(membership)


def not_(membership: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies the element-wise function fn(u) = 1 - u.

    :param membership: Membership function to be modified.

    >>> from fuzzy_expert.operators import not_
    >>> not_([0, 0.25, 0.5, 0.75, 1])
    array([1.  , 0.75, 0.5 , 0.25, 0.  ])

    """
    return 1 - np.array(membership)


def plus(membership: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies the element-wise function fn(u) = u^1.25.

    :param membership: Membership function to be modified.

    >>> from fuzzy_expert.operators import plus
    >>> plus([0, 0.25, 0.5, 0.75, 1])
    array([0.        , 0.1767767 , 0.42044821, 0.69795364, 1.        ])


    """
    return np.power(membership, 1.25)


def somewhat(membership: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies the element-wise function fn(u) = u^(1/3).

    :param membership: Membership function to be modified.

    >>> from fuzzy_expert.operators import somewhat
    >>> somewhat([0, 0.25, 0.5, 0.75, 1])
    array([0.        , 0.62996052, 0.79370053, 0.9085603 , 1.        ])

    """
    return np.power(membership, 1.0 / 3.0)


def very(membership: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies the element-wise function fn(u) = u^2.

    :param membership: Membership function to be modified.

    >>> from fuzzy_expert.operators import very
    >>> very([0, 0.25, 0.5, 0.75, 1])
    array([0.    , 0.0625, 0.25  , 0.5625, 1.    ])

    """
    return np.power(membership, 2)


def slightly(membership: npt.ArrayLike) -> npt.ArrayLike:
    """
    Applies the element-wise function fn(u) = u^(1/2).

    :param membership: Membership function to be modified.

    >>> from fuzzy_expert.operators import slightly
    >>> slightly([0, 0.25, 0.5, 0.75, 1])
    array([0.        , 0.16326531, 0.99696182, 1.        , 0.        ])

    """
    plus_membership: npt.ArrayLike = np.power(membership, 1.25)
    not_very_membership: npt.ArrayLike = 1 - np.power(membership, 2)
    membership: npt.ArrayLike = np.where(
        membership < not_very_membership, plus_membership, not_very_membership
    )
    membership: npt.ArrayLike = membership / np.max(membership)
    return np.where(membership <= 0.5, membership ** 2, 1 - 2 * (1 - membership) ** 2)


# #############################################################################
#
#
# Advanced operators
#
#
# #############################################################################


def prob_or(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Applies the element-wise function fn(u, v) = u + v - u * v. Also known as algebraic-sum.

    :param memberships: List of arrays of membership values.

    >>> from fuzzy_expert.operators import prob_or
    >>> x = [0.1, 0.25, 0.5, 0.75, 0.3]
    >>> y = [0, 0.75, 0.5, 0.25, 0]
    >>> prob_or([x, y])
    array([0.1   , 0.8125, 0.75  , 0.8125, 0.3   ])

    """
    result: npt.ArrayLike = np.array(memberships[0])
    for membership in memberships[1:]:
        membership = np.array(membership)
        result: npt.ArrayLike = result + membership - result * membership
    return np.maximum(0, np.minimum(1, result))


def bounded_prod(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Applies the element-wise function max(0, u + v - 1).

    :param memberships: List of arrays of membership values.

    >>> from fuzzy_expert.operators import bounded_prod
    >>> x = [0.1, 0.25, 0.5, 0.75, 1]
    >>> bounded_prod([x, x])
    array([0. , 0. , 0. , 0.5, 1. ])

    """
    result: npt.ArrayLike = np.array(memberships[0])
    for membership in memberships[1:]:
        membership: npt.ArrayLike = np.array(membership)
        result: npt.ArrayLike = np.maximum(0, result + membership - 1)
    return result


def bounded_sum(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Applies the element-wise function min(1, u + v).

    :param memberships: List of arrays of membership values.

    >>> from fuzzy_expert.operators import bounded_sum
    >>> x = [0, 0.25, 0.5, 0.75, 1]
    >>> bounded_sum([x, x, x])
    array([0.  , 0.75, 1.  , 1.  , 1.  ])

    """
    result: npt.ArrayLike = np.array(memberships[0])
    for membership in memberships[1:]:
        membership: npt.ArrayLike = np.array(membership)
        result: npt.ArrayLike = np.minimum(1, result + membership)
    return result


def bounded_diff(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Applies the element-wise function max(0, u - v).

    :param memberships: List of arrays of membership values.

    >>> from fuzzy_expert.operators import bounded_diff
    >>> x = [0, 0.25, 0.5, 0.75, 1]
    >>> y = [0, 0.25, 0.5, 0.6, 0.7]
    >>> bounded_diff([x, y])
    array([0.  , 0.  , 0.  , 0.15, 0.3 ])

    """
    result: npt.ArrayLike = np.array(memberships[0])
    for membership in memberships[1:]:
        membership = np.array(membership)
        result: npt.ArrayLike = np.maximum(0, result - membership)
    return result


def drastic_prod(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Applies the element-wise function f(u, v) = u if v == 0 else v if u == 1 else 0.

    :param memberships: List of arrays of membership values.


    >>> from fuzzy_expert.operators import drastic_prod
    >>> x = [0, 0.25, 0.5, 0.75, 1]
    >>> y = [1, 0.75, 0.5, 0.25, 0]
    >>> drastic_prod([x, y])
    array([0., 0., 0., 0., 1.])

    """
    result: npt.ArrayLike = np.array(memberships[0])
    for membership in memberships[1:]:
        membership = np.array(membership)
        result: npt.ArrayLike = np.where(
            membership == np.float64(0),
            result,
            np.where(result == np.float64(1), membership, 0),
        )
    return result


def drastic_sum(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Applies the element-wise function f(u, v) = u if v == 0 else v if u == 0 else 1.

    :param memberships: List of arrays of membership values.

    >>> from fuzzy_expert.operators import drastic_sum
    >>> x = [0.1, 0.25, 0.5, 0.75, 0.3]
    >>> y = [0, 0.75, 0.5, 0.25, 0]
    >>> drastic_sum([x, y])
    array([0.1, 1. , 1. , 1. , 0.3])

    """
    result: npt.ArrayLike = memberships[0]
    for membership in memberships[1:]:
        result: npt.ArrayLike = np.where(
            membership == np.float64(0),
            result,
            np.where(result == np.float64(0), membership, 1),
        )
    return result


def product(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Applies the element-wise function f(u, v) = u * v.

    :param memberships: List of arrays of membership values.

    >>> from fuzzy_expert.operators import product
    >>> x = [0, 0.25, 0.5, 0.75, 1]
    >>> product([x, x, x])
    array([0.      , 0.015625, 0.125   , 0.421875, 1.      ])

    """
    result: npt.ArrayLike = np.array(memberships[0])
    for membership in memberships[1:]:
        membership = np.array(membership)
        result: npt.ArrayLike = result * membership
    return result


def maximum(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Applies the element-wise function f(u, v) = max(u, v).

    :param memberships: List of arrays of membership values.

    >>> from fuzzy_expert.operators import maximum
    >>> x = [0.1, 0.25, 0.5, 0.75, 0.3]
    >>> y = [0, 0.75, 0.5, 0.25, 0]
    >>> maximum([x, y])
    array([0.1 , 0.75, 0.5 , 0.75, 0.3 ])

    """
    result: npt.ArrayLike = np.array(memberships[0])
    for membership in memberships[1:]:
        membership = np.array(membership)
        result: npt.ArrayLike = np.maximum(result, membership)
    return result


def minimum(memberships: List[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Applies the element-wise function f(u, v) = min(u, v).

    :param memberships: List of arrays of membership values.

    >>> from fuzzy_expert.operators import minimum
    >>> x = [0.1, 0.25, 0.5, 0.75, 0.3]
    >>> y = [0, 0.75, 0.5, 0.25, 0]
    >>> minimum([x, y])
    array([0.  , 0.25, 0.5 , 0.25, 0.  ])

    """
    result: npt.ArrayLike = np.array(memberships[0])
    for membership in memberships[1:]:
        membership = np.array(membership)
        result: npt.ArrayLike = np.minimum(result, membership)
    return result


def defuzzificate(universe, membership, operator="cog") -> dict:
    """Computes a representative crisp value for the fuzzy set.

    :param universe: Array of values representing the universe of discourse.
    :param membership: Array of values representing the membership function.
    :param operator: Method used for computing the crisp representative value of the fuzzy set.

        * `"cog"`: Center of gravity.
        * `"boa"`: Bisector of area.
        * `"mom"`: Mean of the values for which the membership function is maximum.
        * `"lom"`: Largest value for which the membership function is maximum.
        * `"som"`: Smallest value for which the membership function is minimum.

    >>> from fuzzy_expert.operators import defuzzificate
    >>> u = [0, 1,   2, 3, 4]
    >>> m = [0, 0, 0.5, 1, 1]
    >>> defuzzificate(u, m, "cog")
    2.9166666666666665

    >>> defuzzificate(u, m, "boa")
    3.0

    >>> defuzzificate(u, m, "mom")
    3.5

    >>> defuzzificate(u, m, "lom")
    4

    >>> defuzzificate(u, m, "som")
    3

    """

    def cog():

        n_areas = len(universe) - 1

        areas = []
        centroids = []

        for i in range(n_areas):
            base = universe[i + 1] - universe[i]
            area_rect = np.minimum(membership[i], membership[i + 1]) * base
            centr_rect = universe[i] + base / 2.0
            if membership[i + 1] == membership[i]:
                area_tria = 0
                centr_tri = 0
            elif membership[i + 1] > membership[i]:
                area_tria = base * np.abs(membership[i + 1] - membership[i]) / 2.0
                centr_tri = universe[i] + 2.0 / 3.0 * base
            else:
                area_tria = base * np.abs(membership[i + 1] - membership[i]) / 2.0
                centr_tri = universe[i] + 1.0 / 3.0 * base

            area = area_rect + area_tria
            if area == np.float64(0):
                centr = np.float64(0)
            else:
                centr = (area_rect * centr_rect + area_tria * centr_tri) / (
                    area_rect + area_tria
                )

            areas.append(area)
            centroids.append(centr)

        num = sum([area * centroid for area, centroid in zip(areas, centroids)])
        den = sum(areas)
        return num / den

    def boa():

        n_areas = len(universe) - 1

        areas = []

        for i_area in range(n_areas):
            base = universe[i_area + 1] - universe[i_area]
            area = (membership[i_area] + membership[i_area + 1]) * base / 2.0
            areas.append(area)

        total_area = sum(areas)

        target = total_area / 2.0

        cum_area = 0
        for i_area in range(n_areas):
            cum_area += areas[i_area]
            if cum_area >= target:
                break

        xp = [universe[i_area], universe[i_area + 1]]
        fp = [cum_area - areas[i_area], cum_area]

        return np.interp(target, xp=fp, fp=xp)

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

    universe = np.array(universe)
    membership = np.array(membership)

    if np.sum(membership) == 0.0:
        return np.mean(universe)

    return {
        "cog": cog,
        "boa": boa,
        "mom": mom,
        "lom": lom,
        "som": som,
    }[operator]()
