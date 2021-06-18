"""
Modifiers
===============================================================================

Definition of inguistic modifers:

"""
import numpy as np


def extremely(membership):
    return [(x, np.power(u, 3)) for x, u in membership]


def intensify(membership):
    return [
        (x, np.power(u, 2) if u <= 0.5 else 1 - 2 * np.power(1 - u, 2))
        for x, u in membership
    ]


def more_or_less(membership):
    return [(x, np.power(u, 0.5)) for x, u in membership]


def norm(membership):
    max_u = np.maximum([u for _, u in membership])
    return [(x, u / max_u) for x, u in membership]


def not_(membership):
    return [(x, 1 - u) for x, u in membership]


def plus(membership):
    return [(x, np.power(u, 1.25)) for x, u in membership]


def slightly(membership):

    #
    # intensify( nom (plus A AND not very A))
    #
    u = [u for x, u in membership]
    plus_u = np.power(u, 1.25)
    not_very_u = 1 - np.power(u, 2)
    and_u = np.minimum(plus_u, not_very_u)
    m = [(x, u) for x, _, u in zip(membership, and_u)]
    m = norm(m)
    m = intensify(m)
    return [(x, u) for x, _, u in zip(membership, u)]


def somewhat(membership):
    return [(x, np.power(u, 1.0 / 3.0)) for x, u in membership]


def very(membership):
    return [(x, np.power(u, 2)) for x, u in membership]


def apply_modifiers(membership, modifiers):

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
    modifiers = modifiers.copy()

    for modifier in modifiers.reverse():
        membership = fn[modifier.upper()](membership)

    return membership
