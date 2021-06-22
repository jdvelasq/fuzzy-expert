"""Test for standard membership functions"""

import pytest
from fuzzy_expert.mf import MembershipFunction


def test_gaussmf():
    """
    Gaussian membersip function.

    """
    obj: MembershipFunction = MembershipFunction(n_points=3)
    cen = 0
    wth = 1
    result = obj(mfspec=("gaussmf", cen, wth))
    comp_xp = [x for x, _ in result]
    comp_fp = [y for _, y in result]

    expected_xp = [-3.0, -2.0, -1.2, -0.4, 0.4, 1.2, 2.0, 3.0]
    expected_xp = [pytest.approx(u) for u in expected_xp]
    expected_fp = [
        0.0,
        0.13533528,
        0.48675226,
        0.92311635,
        0.92311635,
        0.48675226,
        0.13533528,
        0.0,
    ]
    expected_fp = [pytest.approx(u) for u in expected_fp]
    assert expected_xp == comp_xp
    assert expected_fp == comp_fp


def test_gbellmf():
    """
    Generalized-bell membersip function.

    """
    obj: MembershipFunction = MembershipFunction(n_points=3)
    cen = 0
    wth = 1
    sha = 0.5
    result = obj(mfspec=("gbellmf", cen, wth, sha))
    comp_xp = [x for x, _ in result]
    comp_fp = [y for _, y in result]

    expected_xp = [
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.2,
        -1.0,
        -0.4,
        0.0,
        0.4,
        1.0,
        1.2,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]

    expected_xp = [pytest.approx(u) for u in expected_xp]
    expected_fp = [
        0.0,
        0.166666667,
        0.2,
        0.25,
        0.33333333,
        0.45454545,
        0.5,
        0.71428571,
        1.0,
        0.71428571,
        0.5,
        0.45454545,
        0.33333333,
        0.25,
        0.2,
        0.16666667,
        0.0,
    ]
    expected_fp = [pytest.approx(u) for u in expected_fp]

    assert expected_xp == comp_xp
    assert expected_fp == comp_fp


def test_pimf():
    """
    Pi-shaped membersip function.

    """


def test_sigmf():
    """
    Sigmoid membersip function.

    """
    obj: MembershipFunction = MembershipFunction(n_points=3)
    cen = 0
    wth = 1
    result = obj(mfspec=("sigmf", cen, wth))
    comp_xp = [x for x, _ in result]
    comp_fp = [y for _, y in result]

    expected_xp = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
    expected_xp = [pytest.approx(u) for u in expected_xp]
    expected_fp = [
        0.00669285,
        0.04742587,
        0.26894142,
        0.73105858,
        0.95257413,
        0.99330715,
    ]
    expected_fp = [pytest.approx(u) for u in expected_fp]

    assert expected_xp == comp_xp
    assert expected_fp == comp_fp


def test_smf():
    """
    S-shaped membersip function.

    """
    obj: MembershipFunction = MembershipFunction(n_points=3)
    foot = 0
    shld = 1
    result = obj(mfspec=("smf", foot, shld))
    comp_xp = [x for x, _ in result]
    comp_fp = [y for _, y in result]

    expected_xp = [-0, 0.5, 1]
    expected_xp = [pytest.approx(u) for u in expected_xp]
    expected_fp = [0, 0.5, 1]
    expected_fp = [pytest.approx(u) for u in expected_fp]

    assert expected_xp == comp_xp
    assert expected_fp == comp_fp


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
