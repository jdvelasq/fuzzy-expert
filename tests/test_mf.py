"""Test for standard membership functions"""

from fuzzy_expert.mf import MembershipFunction


def test_gaussmf():
    """Gaussian membersip function"""


def test_gbellmf():
    """Generalized-bell membersip function"""


def test_pimf():
    """Pi-shaped membersip function"""


def test_sigmf():
    """Sigmoid membersip function"""


def test_smf():
    """S-shaped membersip function"""


def test_trapmf():
    """Trapezoidal membersip function"""


def test_trimf():
    """Triangular membersip function"""
    obj: MembershipFunction = MembershipFunction()
    left = 0
    center = 1
    right = 2
    result = obj(mfspec=("trimf", left, center, right))
    assert result == [(left, 0), (center, 1), (right, 0)]


def test_zmf():
    """Z-shaped membersip function"""
