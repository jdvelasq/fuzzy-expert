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

    expected_xp = [
        -3,
        -2.0,
        -1.2,
        -0.3999999999999999,
        0.0,
        0.40000000000000036,
        1.2000000000000002,
        2.0,
        3,
    ]
    expected_xp = [pytest.approx(u) for u in expected_xp]
    expected_fp = [
        0,
        0.1353352832366127,
        0.4867522559599717,
        0.9231163463866359,
        1.0,
        0.9231163463866356,
        0.48675225595997157,
        0.1353352832366127,
        0,
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
    obj: MembershipFunction = MembershipFunction(n_points=3)
    lfeet = 0
    lpeak = 1
    rpeak = 2
    rfeet = 3
    result = obj(mfspec=("pimf", lfeet, lpeak, rpeak, rfeet))
    comp_xp = [x for x, _ in result]
    comp_fp = [y for _, y in result]

    expected = MembershipFunction(n_points=3)(
        mfspec=("smf", lfeet, lpeak)
    ) + MembershipFunction(n_points=3)(mfspec=("zmf", rpeak, rfeet))
    expected_xp = [pytest.approx(xp) for xp, _ in expected]
    expected_fp = [pytest.approx(fp) for _, fp in expected]

    assert expected_xp == comp_xp
    assert expected_fp == comp_fp


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

    expected_xp = [-6, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 6]
    expected_xp = [pytest.approx(u) for u in expected_xp]
    expected_fp = [
        0,
        0.0066928509242848554,
        0.04742587317756678,
        0.2689414213699951,
        0.5,
        0.7310585786300049,
        0.9525741268224334,
        0.9933071490757153,
        1,
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
    obj: MembershipFunction = MembershipFunction(n_points=3)
    foot = 0
    shld = 1
    result = obj(mfspec=("zmf", foot, shld))
    comp_xp = [x for x, _ in result]
    comp_fp = [y for _, y in result]

    expected_xp = [0, 0.5, 1]
    expected_xp = [pytest.approx(u) for u in expected_xp]
    expected_fp = [1.0, 0.5, 0]
    expected_fp = [pytest.approx(u) for u in expected_fp]

    assert expected_xp == comp_xp
    assert expected_fp == comp_fp
