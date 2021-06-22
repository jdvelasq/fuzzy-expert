"""Test for standard membership functions"""

from fuzzy_expert.mf import MembershipFunction


def test_gaussmf():
    """
    Gaussian membersip function.

    """


def test_gbellmf():
    """
    Generalized-bell membersip function.

    """


def test_pimf():
    """
    Pi-shaped membersip function.

    """


def test_sigmf():
    """
    Sigmoid membersip function.

    """


def test_smf():
    """
    S-shaped membersip function.

    """


def test_trapmf():
    """
    Trapezoidal membersip function.

    """
    obj: MembershipFunction = MembershipFunction()
    left_feet = 0
    left_peak = 1
    right_peak = 2
    right_feet = 3
    result = obj(mfspec=("trapmf", left_feet, left_peak, right_peak, right_feet))
    assert result == [(left_feet, 0), (left_peak, 1), (right_peak, 1), (right_feet, 0)]


def test_trimf():
    """
    Triangular membersip function.

    """
    obj: MembershipFunction = MembershipFunction()
    left = 0
    center = 1
    right = 2
    result = obj(mfspec=("trimf", left, center, right))
    assert result == [(left, 0), (center, 1), (right, 0)]


def test_zmf():
    """
    Z-shaped membersip function.

    """
