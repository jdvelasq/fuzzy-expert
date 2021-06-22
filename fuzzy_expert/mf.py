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

        print(params)

        return fn(*params)

    def gaussmf(self, center, sigma) -> list[tuple[float, float]]:
        """Gaussian membership function.

        This function computes fuzzy membership values using a Gaussian membership function using NumPy.

        Args:
            x (float, np.ndarray): input value.
            center (float): Center of the distribution.
            sigma (float): standard deviation.

        Returns:
            A float or numpy.ndarray.
        """
        # center, sigma = params
        xp = np.linspace(
            start=center - 2 * sigma,
            stop=center + 2 * sigma,
            num=2 * self.n_points,
        )
        xp = np.append(xp, [center - 3 * sigma, center * 3 + sigma])
        fp = np.exp(-((xp - center) ** 2) / (2 * sigma))
        return [(x, f) for x, f in zip(xp, fp)]

    def gbellmf(self, **params) -> list[tuple[float, float]]:
        """Generalized bell-shaped membership function.

        This function computes membership values using a generalized bell membership function using NumPy.

        Args:
            a (float): standard deviation.
            b (float): exponent.
            c (float): center.

        Returns:
            A numpy.array.
        """
        center, sigma, b = params
        xp = np.linspace(
            start=center - 2 * sigma, stop=center + 2 * sigma, num=2 * self.n_points
        )
        xp = np.append(xp, [center - 3 * sigma, center * 3 + sigma])
        fp = 1 / (1 + np.abs((xp - center) / sigma) ** (2 * b))
        return [(x, f) for x, f in zip(xp, fp)]

    def pimf(self, **params) -> list[tuple[float, float]]:
        """Pi-shaped membership function.

        This function computes membership values using a pi-shaped membership function using NumPy.

        Args:
            a (float): Left feet.
            b (float): Left peak.
            c (float): Right peak.
            d (float): Right feet.

        Returns:
            A numpy.array.
        """
        a, b, c, d = params
        return self.smf(a=a, b=b) + self.zmf(a=c, b=d)

    def sigmf(self, **params) -> list[tuple[float, float]]:
        """Sigmoidal membership function.

        This function computes fuzzy membership values using a sigmoidal membership function using NumPy.

        Args:
            x (float, np.array): input value.
            a (float): slope.
            c (float): center.

        Returns:
            A numpy.array.
        """
        center, alpha = params
        xp = np.linspace(
            start=center - 5 * alpha, stop=center + 5 * alpha, num=2 * self.n_points
        )
        fp = 1 / (1 + np.exp(-np.abs(alpha) * (xp - center)))
        return [(x, f) for x, f in zip(xp, fp)]

    def smf(self, **params) -> list[tuple[float, float]]:
        """S-shaped membership function

        This function computes fuzzy membership values using a S-shaped membership function using NumPy.

        Args:
            a (float): Left feet.
            b (float): Right peak.

        Returns:
            A numpy.array.
        """
        a, b = params
        xp = np.linspace(start=a, stop=b, num=self.n_points)
        fp = np.where(
            xp <= a,
            0,
            np.where(
                xp <= (a + b) / 2,
                2 * ((xp - a) / (b - a)) ** 2,
                np.where(xp <= b, 1 - 2 * ((xp - b) / (b - a)) ** 2, 1),
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]

    def trapmf(self, **params) -> list[tuple[float, float]]:
        """Trapezoida membership function

        This function computes fuzzy membership values using a trapezoidal membership function using NumPy.

        Args:
            a (float): Left feet.
            b (float): Left peak.
            c (float): Right peak.
            d (float): Right feet.

        Returns:
            A numpy.array.
        """
        a, b, c, d = params
        a = np.where(a == b, a - 1e-4, a)
        d = np.where(d == c, d + 1e-4, d)
        xp = np.array([a, b, c, d])
        fp = np.where(
            xp <= a,
            0,
            np.where(
                xp <= b,
                (xp - a) / (b - a),
                np.where(xp <= c, 1, np.where(xp <= d, (d - xp) / (d - c), 0)),
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]

    def trimf(self, a: float, b: float, c: float) -> list[tuple[float, float]]:
        """Triangular membership function.

        This function computes fuzzy membership values using a triangular membership function using NumPy.

        Args:
            a (float): Left feet.
            b (float): center or peak.
            c (float): right feet.

        Returns:
            A numpy.array.
        """
        a = np.where(a == b, a - 1e-4, a)
        c = np.where(b == c, c + 1e-4, c)
        xp = np.array([a, b, c])
        fp = np.where(
            xp <= a,
            0,
            np.where(
                xp <= b, (xp - a) / (b - a), np.where(xp <= c, (c - xp) / (c - b), 0)
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]

    def zmf(self, **params) -> list[tuple[float, float]]:
        """Z-shaped membership function

        This function computes fuzzy membership values using a Z-shaped membership function using NumPy.

        Args:
            x (float, np.array): input value.
            a (float): Left peak.
            b (float): Right feet.

        Returns:
            A numpy.array.
        """
        a, b = params
        xp = np.linspace(start=a, stop=b, num=self.n_points)
        fp = np.where(
            xp <= a,
            1,
            np.where(
                xp <= (a + b) / 2,
                1 - 2 * ((xp - a) / (b - a)) ** 2,
                np.where(xp <= b, 2 * ((xp - b) / (b - a)) ** 2, 0),
            ),
        )
        return [(x, f) for x, f in zip(xp, fp)]
