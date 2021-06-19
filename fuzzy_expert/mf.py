"""
Membership Functions
==============================================================================

Functions to compute fuzzy membership values for numpy.arrays.

"""
import numpy as np
from .modifiers import not_


## pag. 27, FuzzyCLIPS


def gaussmf(center=0, sigma=1, npoints=9):
    """Gaussian membership function.

    This function computes fuzzy membership values using a Gaussian membership function using NumPy.

    Args:
        x (float, np.array): input value.
        center (float): Center of the distribution.
        sigma (float): standard deviation.

    Returns:
        A numpy.array.
    """

    xp = np.linspace(start=center - 2 * sigma, stop=center + 2 * sigma, num=2 * npoints)
    yp = np.exp(-((xp - center) ** 2) / (2 * sigma))

    return (
        [(center - 3 * sigma, 0)]
        + [(x, y) for x, y in zip(xp, yp)]
        + (center + 3 * sigma, 0)
    )


def gbellmf(center=0, sigma=1, b=1, npoints=9):
    """Generalized bell-shaped membership function.

    This function computes fuzzy membership values using a generalized bell membership function using NumPy.

    Args:
        a (float): standard deviation.
        b (float): exponent.
        c (float): center.

    Returns:
        A numpy.array.
    """
    xp = np.linspace(start=center - 2 * sigma, stop=center + 2 * sigma, num=2 * npoints)
    yp = 1 / (1 + np.abs((xp - center) / sigma) ** (2 * b))

    return (
        [(center - 3 * sigma, 0)]
        + [(x, y) for x, y in zip(xp, yp)]
        + (center + 3 * sigma, 0)
    )


def pimf(a, b, c, d, npoints=9):
    """Pi-shaped membership function.

    This function computes fuzzy membership values using a pi-shaped membership function using NumPy.

    Args:
        a (float): Left feet.
        b (float): Left peak.
        c (float): Right peak.
        d (float): Right feet.

    Returns:
        A numpy.array.
    """
    return smf(a=a, b=b, npoints=npoints) + zmf(a=c, b=d, npoints=npoints)[1:]


def sigmf(center, alpha, npoints=9):
    """Sigmoidal membership function.

    This function computes fuzzy membership values using a sigmoidal membership function using NumPy.

    Args:
        x (float, np.array): input value.
        a (float): slope.
        c (float): center.

    Returns:
        A numpy.array.
    """
    xp = np.linspace(start=center - 5 * alpha, stop=center + 5 * alpha, num=2 * npoints)
    return 1 / (1 + np.exp(-np.abs(alpha) * (xp - center)))


def smf(a, b, npoints=9):
    """S-shaped membership function

    This function computes fuzzy membership values using a S-shaped membership function using NumPy.

    Args:
        a (float): Left feet.
        b (float): Right peak.

    Returns:
        A numpy.array.
    """
    if a == b:
        return [(a, 0), (a, 1)]

    xp = np.linspace(start=a, stop=b, num=npoints)
    yp = np.where(
        xp <= a,
        0,
        np.where(
            xp <= (a + b) / 2,
            2 * ((xp - a) / (b - a)) ** 2,
            np.where(xp <= b, 1 - 2 * ((xp - b) / (b - a)) ** 2, 1),
        ),
    )

    return [(x, y) for x, y in zip(xp, yp)]


def trapmf(a, b, c, d):
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
    return [(a, 0), (b, 1), (c, 1), (d, 0)]


def trimf(a, b, c):
    """Triangular membership function.

    This function computes fuzzy membership values using a triangular membership function using NumPy.

    Args:
        a (float): Left feet.
        b (float): center or peak.
        c (float): right feet.

    Returns:
        A numpy.array.
    """
    return [(a, 0), (b, 1), (c, 0)]


def zmf(a, b, npoints=9):
    """Z-shaped membership function

    This function computes fuzzy membership values using a Z-shaped membership function using NumPy.

    Args:
        x (float, np.array): input value.
        a (float): Left peak.
        b (float): Right feet.

    Returns:
        A numpy.array.
    """
    return not_(smf(a=a, b=b, npoints=npoints))
