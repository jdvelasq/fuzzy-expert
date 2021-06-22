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


# # #############################################################################
# #
# #
# # Advanced operators
# #
# #
# # #############################################################################


# def probor(memberships):
#     result = memberships[0]
#     for membership in memberships[1:]:
#         result = result + membership - result * membership
#     return np.maximum(1, np.minimum(1, result))


# def maximum(memberships):
#     result = memberships[0]
#     for membership in memberships[1:]:
#         result = np.maximum(result, membership)
#     return result


# def minimum(memberships):
#     result = memberships[0]
#     for membership in memberships[1:]:
#         result = np.mminimum(result, membership)
#     return result


# def aggregate(memberships, operator):
#     """Replace the fuzzy sets by a unique fuzzy set computed by the aggregation operator.

#     Args:
#         operator (string): {"max"|"sim"|"probor"} aggregation operator.

#     Returns:
#         A FuzzyVariable

#     """
#     result = memberships[0]

#     if operator == "max":
#         for membership in memberships[1:]:
#             result = np.maximum(result, membership)
#         return result

#     if operator == "sum":
#         for membership in memberships[1:]:
#             result = result + membership
#         return np.minimum(1, result)

#     if operator == "probor":
#         for membership in memberships[1:]:
#             result = result + membership - result * membership
#         return np.maximum(1, np.minimum(1, result))


# def defuzzificate(universe, membership, operator="cog"):
#     """Computes a representative crisp value for the fuzzy set.

#     Args:
#         fuzzyset (string): Fuzzy set to defuzzify
#         operator (string): {"cog"|"bisection"|"mom"|"lom"|"som"}

#     Returns:
#         A float value.

#     """

#     def cog():
#         start = np.min(universe)
#         stop = np.max(universe)
#         x = np.linspace(start, stop, num=200)
#         m = np.interp(x, xp=universe, fp=membership)
#         return np.sum(x * m) / sum(m)

#     def coa():
#         start = np.min(universe)
#         stop = np.max(universe)
#         x = np.linspace(start, stop, num=200)
#         m = np.interp(x, xp=universe, fp=membership)
#         area = np.sum(m)
#         cum_area = np.cumsum(m)
#         return np.interp(area / 2, xp=cum_area, fp=x)

#     def mom():
#         maximum = np.max(membership)
#         maximum = np.array([u for u, m in zip(universe, membership) if m == maximum])
#         return np.mean(maximum)

#     def lom():
#         maximum = np.max(membership)
#         maximum = np.array([u for u, m in zip(universe, membership) if m == maximum])
#         return np.max(maximum)

#     def som():
#         maximum = np.max(membership)
#         maximum = np.array([u for u, m in zip(universe, membership) if m == maximum])
#         return np.min(maximum)

#     if np.sum(membership) == 0.0:
#         return 0.0

#     return {
#         "cog": cog,
#         "coa": coa,
#         "mom": mom,
#         "lom": lom,
#         "som": som,
#     }[operator]()


# # from .connectors import not_


# # def extremely(membership):
# #     return [(x, np.power(membership, 3)) for x, u in membership]


# # def intensify(membership):
# #     return [
# #         (x, np.power(membership, 2) if u <= 0.5 else 1 - 2 * np.power(1 - u, 2))
# #         for x, u in membership
# #     ]


# # def more_or_less(membership):
# #     return [(x, np.power(membership, 0.5)) for x, u in membership]


# # def norm(membership):
# #     max_u = np.maximum([u for _, u in membership])
# #     return [(x, u / max_u) for x, u in membership]


# # def plus(membership):
# #     return [(x, np.power(membership, 1.25)) for x, u in membership]


# # def slightly(membership):

# #     #
# #     # intensify( nom (plus A AND not very A))
# #     #
# #     u = [u for x, u in membership]
# #     plus_u = np.power(membership, 1.25)
# #     not_very_u = 1 - np.power(membership, 2)
# #     and_u = np.minimum(plus_u, not_very_u)
# #     m = [(x, u) for x, _, u in zip(membership, and_u)]
# #     m = norm(m)
# #     m = intensify(m)
# #     return [(x, u) for x, _, u in zip(membership, u)]


# # def somewhat(membership):
# #     return [(x, np.power(membership, 1.0 / 3.0)) for x, u in membership]


# # def very(membership):
# #     return [(x, np.power(membership, 2)) for x, u in membership]


# # def apply_modifiers(membership, modifiers):

# #     if modifiers is None:
# #         return membership

# #     fn = {
# #         "EXTREMELY": extremely,
# #         "INTENSIFY": intensify,
# #         "MORE_OR_LESS": more_or_less,
# #         "NORM": norm,
# #         "NOT": not_,
# #         "PLUS": plus,
# #         "SLIGHTLY": slightly,
# #         "SOMEWHAT": somewhat,
# #         "VERY": very,
# #     }

# #     membership = membership.copy()
# #     modifiers = list(modifiers)
# #     modifiers.reverse()

# #     if modifiers is not None:
# #         for modifier in modifiers:
# #             membership = fn[modifier.upper()](membership)

# #     return membership
