"""
Membership Functions
==============================================================================

Functions in this module returns a standard membership function specificaion as a list of points (x_i, u_i).

"""
from __future__ import annotations
from typing import Tuple, List
import numpy as np


## pag. 27, FuzzyCLIPS


class MembershipFunction:
    """Membership function constructor.

    :param n_points: Number base point for building the approximations.

    >>> from fuzzy_expert.mf import MembershipFunction
    >>> mf = MembershipFunction(n_points=3)
    >>> mf(('gaussmf', 5, 1))
    [(2, 0), (3.0, 0.1353352832366127), (3.8, 0.48675225595997157), (4.6, 0.9231163463866356), (5.0, 1.0), (5.4, 0.9231163463866356), (6.2, 0.48675225595997157), (7.0, 0.1353352832366127), (8, 0)]


    """

    def __init__(self, n_points: int = 9):
        self.n_points: int = n_points

    def __call__(self, mfspec: tuple):
        """Generates a list of poinnts representing the membership function.


        >>> from fuzzy_expert.mf import MembershipFunction
        >>> mf = MembershipFunction(n_points=3)
        >>> mf(('gaussmf', 5, 1))
        [(2, 0), (3.0, 0.1353352832366127), (3.8, 0.48675225595997157), (4.6, 0.9231163463866356), (5.0, 1.0), (5.4, 0.9231163463866356), (6.2, 0.48675225595997157), (7.0, 0.1353352832366127), (8, 0)]

        """
        fn, *params = mfspec

        fn = {
            "gaussmf": self.gaussmf,
            "gbellmf": self.gbellmf,
            "pimf": self.pimf,
            "sigmf": self.sigmf,
            "smf": self.smf,
            "trapmf": self.trapmf,
            "trimf": self.trimf,
            "zmf": self.zmf,
        }[fn]

        return fn(*params)

    def gaussmf(self, center: float, sigma: float) -> List[Tuple[float, float]]:
        """Gaussian membership function.

        :param center: Defines the center of the membership function.
        :param sigma: Defines the width of the membership function, where a larger value creates a wider membership function.

        >>> from fuzzy_expert.mf import MembershipFunction
        >>> mf = MembershipFunction(n_points=3)
        >>> mf.gaussmf(center=5, sigma=1)
        [(2, 0), (3.0, 0.1353352832366127), (3.8, 0.48675225595997157), (4.6, 0.9231163463866356), (5.0, 1.0), (5.4, 0.9231163463866356), (6.2, 0.48675225595997157), (7.0, 0.1353352832366127), (8, 0)]

        """
        xp: np.ndarray = np.linspace(
            start=center - 2 * sigma,
            stop=center + 2 * sigma,
            num=2 * self.n_points,
        )
        xp = np.append(xp, center)
        xp = np.unique(xp)
        xp.sort()
        fp: np.ndarray = np.exp(-((xp - center) ** 2) / (2 * sigma))
        return (
            [(center - 3 * sigma, 0)]
            + [(x, f) for x, f in zip(xp, fp)]
            + [(center + 3 * sigma, 0)]
        )

    def gbellmf(
        self,
        center: float,
        width: float,
        shape: float,
    ) -> List[Tuple[float, float]]:
        """Generalized bell-shaped membership function.

        :param center: Defines the center of the membership function.
        :param width: Defines the width of the membership function, where a larger value creates a wider membership function.
        :param shape: Defines the shape of the curve on either side of the central plateau, where a larger value creates a more steep transition.

        >>> from fuzzy_expert.mf import MembershipFunction
        >>> mf = MembershipFunction(n_points=3)
        >>> mf.gbellmf(center=5, width=1, shape=0.5)
        [(-1, 0), (0.0, 0.16666666666666666), (1.0, 0.2), (2.0, 0.25), (3.0, 0.3333333333333333), (3.8, 0.45454545454545453), (4.0, 0.5), (4.6, 0.7142857142857141), (5.0, 1.0), (5.4, 0.7142857142857141), (6.0, 0.5), (6.2, 0.45454545454545453), (7.0, 0.3333333333333333), (8.0, 0.25), (9.0, 0.2), (10.0, 0.16666666666666666), (11, 0)]

        """
        xp: np.ndarray = np.linspace(
            start=center - 2 * width, stop=center + 2 * width, num=2 * self.n_points
        )
        delta = center + width * np.linspace(start=-5, stop=5, num=11)
        xp: np.ndarray = np.append(xp, delta)
        xp = np.unique(xp)
        xp.sort()
        fp: np.ndarray = 1 / (1 + np.abs((xp - center) / width) ** (2 * shape))
        return (
            [(center - 6 * width, 0)]
            + [(x, f) for x, f in zip(xp, fp)]
            + [(center + 6 * width, 0)]
        )

    def pimf(
        self,
        left_feet: float,
        left_peak: float,
        right_peak: float,
        right_feet: float,
    ) -> List[Tuple[float, float]]:
        """Pi-shaped membership function.

        :param left_feet: Defines the left feet of the membership function.
        :param left_peak: Defines the left peak of the membership function.
        :param right_peak: Defines the right peak of the membership function.
        :param right_feet: Defines the right feet of the membership function.

        >>> from fuzzy_expert.mf import MembershipFunction
        >>> mf = MembershipFunction(n_points=4)
        >>> mf.pimf(left_feet=1, left_peak=2, right_peak=3, right_feet=4)
        [(1.0, 0.0), (1.3333333333333333, 0.22222222222222213), (1.6666666666666665, 0.7777777777777776), (2.0, 1.0), (3.0, 1.0), (3.3333333333333335, 0.7777777777777776), (3.6666666666666665, 0.22222222222222243), (4.0, 0.0)]

        """
        return self.smf(foot=left_feet, shoulder=left_peak) + self.zmf(
            shoulder=right_peak, feet=right_feet
        )

    def sigmf(
        self,
        center: float,
        width: float,
    ) -> List[Tuple[float, float]]:
        """Sigmoidal membership function.

        :param center: Defines the center of the membership function.
        :param width: Defines the width of the membership function.

        >>> from fuzzy_expert.mf import MembershipFunction
        >>> mf = MembershipFunction(n_points=3)
        >>> mf.sigmf(center=5, width=1)
        [(-1, 0), (0.0, 0.0066928509242848554), (2.0, 0.04742587317756678), (4.0, 0.2689414213699951), (5.0, 0.5), (6.0, 0.7310585786300049), (8.0, 0.9525741268224334), (10.0, 0.9933071490757153), (11, 1)]

        """
        xp: np.ndarray = np.linspace(
            start=center - 5 * width, stop=center + 5 * width, num=2 * self.n_points
        )
        xp: np.ndarray = np.append(xp, center)
        xp = np.unique(xp)
        xp.sort()
        fp: np.ndarray = 1 / (1 + np.exp(-np.abs(width) * (xp - center)))
        return (
            [(center - 6 * width, 0)]
            + [(x, f) for x, f in zip(xp, fp)]
            + [(center + 6 * width, 1)]
        )

    def smf(
        self,
        foot: float,
        shoulder: float,
    ) -> List[Tuple[float, float]]:
        """S-shaped membership function.

        :param foot: Defines the foot of the membership function.
        :param shoulder: Defines the shoulder of the membership function.

        >>> from fuzzy_expert.mf import MembershipFunction
        >>> mf = MembershipFunction(n_points=4)
        >>> mf.smf(foot=1, shoulder=2)
        [(1.0, 0.0), (1.3333333333333333, 0.22222222222222213), (1.6666666666666665, 0.7777777777777776), (2.0, 1.0)]

        """
        xp: np.ndarray = np.linspace(start=foot, stop=shoulder, num=self.n_points)
        fp: np.ndarray = np.where(
            xp <= foot,
            0,
            np.where(
                xp <= (foot + shoulder) / 2,
                2 * ((xp - foot) / (shoulder - foot)) ** 2,
                np.where(
                    xp <= shoulder,
                    1 - 2 * ((xp - shoulder) / (shoulder - foot)) ** 2,
                    1,
                ),
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]

    def trapmf(
        self,
        left_feet: float,
        left_peak: float,
        right_peak: float,
        right_feet: float,
    ) -> List[Tuple[float, float]]:
        """Trapezoidal membership function.

        :param left_feet: Defines the left feet of the membership function.
        :param left_peak: Defines the left peak of the membership function.
        :param right_peak: Defines the right peak of the membership function.
        :param right_feet: Defines the right feet of the membership function.

        >>> from fuzzy_expert.mf import MembershipFunction
        >>> mf = MembershipFunction(n_points=4)
        >>> mf.trapmf(left_feet=1, left_peak=2, right_peak=3, right_feet=4)
        [(1.0, 0.0), (2.0, 1.0), (3.0, 1.0), (4.0, 0.0)]


        """
        left_feet: np.ndarray = np.where(
            left_feet == left_peak, left_feet - 1e-4, left_feet
        )
        right_feet: np.ndarray = np.where(
            right_feet == right_peak, right_feet + 1e-4, right_feet
        )
        xp: np.ndarray = np.array([left_feet, left_peak, right_peak, right_feet])
        fp: np.ndarray = np.where(
            xp <= left_feet,
            0,
            np.where(
                xp <= left_peak,
                (xp - left_feet) / (left_peak - left_feet),
                np.where(
                    xp <= right_peak,
                    1,
                    np.where(
                        xp <= right_feet,
                        (right_feet - xp) / (right_feet - right_peak),
                        0,
                    ),
                ),
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]

    def trimf(
        self,
        left_feet: float,
        peak: float,
        right_feet: float,
    ) -> List[Tuple[float, float]]:
        """Triangular membership function.

        :param left_feet: Defines the left feet of the membership function.
        :param peak: Defines the peak of the membership function.
        :param right_feet: Defines the right feet of the membership function.

        >>> from fuzzy_expert.mf import MembershipFunction
        >>> mf = MembershipFunction(n_points=4)
        >>> mf.trimf(left_feet=1, peak=2, right_feet=4)
        [(1.0, 0.0), (2.0, 1.0), (4.0, 0.0)]

        """
        left_feet: np.ndarray = np.where(left_feet == peak, left_feet - 1e-4, left_feet)
        right_feet: np.ndarray = np.where(
            peak == right_feet, right_feet + 1e-4, right_feet
        )
        xp: np.ndarray = np.array([left_feet, peak, right_feet])
        fp: np.ndarray = np.where(
            xp <= left_feet,
            0,
            np.where(
                xp <= peak,
                (xp - left_feet) / (peak - left_feet),
                np.where(xp <= right_feet, (right_feet - xp) / (right_feet - peak), 0),
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]

    def zmf(
        self,
        shoulder: float,
        feet: float,
    ) -> List[Tuple[float, float]]:
        """Z-shaped membership function.

        :param shoulder: Defines the shoulder of the membership function.
        :param feet: Defines the feet of the membership function.

        >>> from fuzzy_expert.mf import MembershipFunction
        >>> mf = MembershipFunction(n_points=4)
        >>> mf.zmf(shoulder=1, feet=2)
        [(1.0, 1.0), (1.3333333333333333, 0.7777777777777779), (1.6666666666666665, 0.22222222222222243), (2.0, 0.0)]

        """
        xp: np.ndarray = np.linspace(start=shoulder, stop=feet, num=self.n_points)
        fp: np.ndarray = np.where(
            xp <= shoulder,
            1,
            np.where(
                xp <= (shoulder + feet) / 2,
                1 - 2 * ((xp - shoulder) / (feet - shoulder)) ** 2,
                np.where(xp <= feet, 2 * ((xp - feet) / (feet - shoulder)) ** 2, 0),
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]
