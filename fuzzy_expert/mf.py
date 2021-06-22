"""
Membership Functions
==============================================================================

Functions to compute fuzzy membership values for numpy.arrays.

"""
import numpy as np

## pag. 27, FuzzyCLIPS


class MembershipFunction:
    def __init__(self, n_points: int = 9):
        self.n_points: int = n_points

    def __call__(self, mfspec: tuple):

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

    def gaussmf(self, center: float, sigma) -> list[tuple[float, float]]:
        """
        Gaussian membership function.
        """
        xp: np.ndarray = np.linspace(
            start=center - 2 * sigma,
            stop=center + 2 * sigma,
            num=2 * self.n_points,
        )
        fp: np.ndarray = np.exp(-((xp - center) ** 2) / (2 * sigma))
        return (
            [(center - 3 * sigma, 0)]
            + [(x, f) for x, f in zip(xp, fp)]
            + [(center + 3 * sigma, 0)]
        )

    def gbellmf(
        self, center: float, width: float, shape: float
    ) -> list[tuple[float, float]]:
        """
        Generalized bell-shaped membership function.

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
    ) -> list[tuple[float, float]]:
        """
        Pi-shaped membership function.

        """
        return self.smf(a=left_feet, b=left_peak) + self.zmf(a=right_peak, b=right_feet)

    def sigmf(
        self,
        center: float,
        width: float,
    ) -> list[tuple[float, float]]:
        """
        Sigmoidal membership function.

        """
        xp: np.ndarray = np.linspace(
            start=center - 5 * width, stop=center + 5 * width, num=2 * self.n_points
        )
        fp: np.ndarray = 1 / (1 + np.exp(-np.abs(width) * (xp - center)))
        return [(x, f) for x, f in zip(xp, fp)]

    def smf(
        self,
        a: float,
        b: float,
    ) -> list[tuple[float, float]]:
        """
        S-shaped membership function.

        """
        xp: np.ndarray = np.linspace(start=a, stop=b, num=self.n_points)
        fp: np.ndarray = np.where(
            xp <= a,
            0,
            np.where(
                xp <= (a + b) / 2,
                2 * ((xp - a) / (b - a)) ** 2,
                np.where(xp <= b, 1 - 2 * ((xp - b) / (b - a)) ** 2, 1),
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]

    def trapmf(
        self,
        left_feet: float,
        left_peak: float,
        right_peak: float,
        right_feet: float,
    ) -> list[tuple[float, float]]:
        """
        Trapezoidal membership function.

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
    ) -> list[tuple[float, float]]:
        """
        Triangular membership function.

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
        a: float,
        b: float,
    ) -> list[tuple[float, float]]:
        """
        Z-shaped membership function.

        """
        xp: np.ndarray = np.linspace(start=a, stop=b, num=self.n_points)
        fp: np.ndarray = np.where(
            xp <= a,
            1,
            np.where(
                xp <= (a + b) / 2,
                1 - 2 * ((xp - a) / (b - a)) ** 2,
                np.where(xp <= b, 2 * ((xp - b) / (b - a)) ** 2, 0),
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]
