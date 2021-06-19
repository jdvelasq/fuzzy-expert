"""
Modifiers and operators
===============================================================================

Definition of inguistic modifers:

"""
import numpy as np


# #############################################################################
#
#
# Basic membership functions
#
#
# #############################################################################


def extremely(membership):
    return np.power(membership, 3)


def intensify(membership):
    return np.where(
        membership <= 0.5, np.power(membership, 2), 1 - 2 * np.power(1 - membership, 2)
    )


def maximum(membership1, membership2):
    return np.maximum(membership1, membership2)


def minimum(membership1, membership2):
    return np.minimum(membership1, membership2)


def more_or_less(membership):
    return np.power(membership, 0.5)


def norm(membership):
    return membership / np.max(membership)


def not_(membership):
    return 1 - membership


def plus(membership):
    return np.power(membership, 1.25)


def somewhat(membership):
    return np.power(membership, 1.0 / 3.0)


def very(membership):
    return np.power(membership, 2)


def slightly(membership):
    plus_membership = np.power(membership, 1.25)
    not_very_membership = 1 - np.power(membership, 2)
    membership = np.where(
        membership < not_very_membership, plus_membership, not_very_membership
    )
    membership = membership / np.max(membership)
    return np.where(membership <= 0.5, membership ** 2, 1 - 2 * (1 - membership) ** 2)


# #############################################################################
#
#
# Advanced membership functions
#
#
# #############################################################################


def get_modified_membership(membership, modifiers):

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


# from .connectors import not_


# def extremely(membership):
#     return [(x, np.power(membership, 3)) for x, u in membership]


# def intensify(membership):
#     return [
#         (x, np.power(membership, 2) if u <= 0.5 else 1 - 2 * np.power(1 - u, 2))
#         for x, u in membership
#     ]


# def more_or_less(membership):
#     return [(x, np.power(membership, 0.5)) for x, u in membership]


# def norm(membership):
#     max_u = np.maximum([u for _, u in membership])
#     return [(x, u / max_u) for x, u in membership]


# def plus(membership):
#     return [(x, np.power(membership, 1.25)) for x, u in membership]


# def slightly(membership):

#     #
#     # intensify( nom (plus A AND not very A))
#     #
#     u = [u for x, u in membership]
#     plus_u = np.power(membership, 1.25)
#     not_very_u = 1 - np.power(membership, 2)
#     and_u = np.minimum(plus_u, not_very_u)
#     m = [(x, u) for x, _, u in zip(membership, and_u)]
#     m = norm(m)
#     m = intensify(m)
#     return [(x, u) for x, _, u in zip(membership, u)]


# def somewhat(membership):
#     return [(x, np.power(membership, 1.0 / 3.0)) for x, u in membership]


# def very(membership):
#     return [(x, np.power(membership, 2)) for x, u in membership]


# def apply_modifiers(membership, modifiers):

#     if modifiers is None:
#         return membership

#     fn = {
#         "EXTREMELY": extremely,
#         "INTENSIFY": intensify,
#         "MORE_OR_LESS": more_or_less,
#         "NORM": norm,
#         "NOT": not_,
#         "PLUS": plus,
#         "SLIGHTLY": slightly,
#         "SOMEWHAT": somewhat,
#         "VERY": very,
#     }

#     membership = membership.copy()
#     modifiers = list(modifiers)
#     modifiers.reverse()

#     if modifiers is not None:
#         for modifier in modifiers:
#             membership = fn[modifier.upper()](membership)

#     return membership
