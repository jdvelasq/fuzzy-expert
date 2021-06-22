"""
Membership Functions
==============================================================================

Functions to compute fuzzy membership values for numpy.arrays.

"""
# import numpy as np


## pag. 27, FuzzyCLIPS


# def gaussmf(x: float or np.ndarray, center: float, sigma: float) -> float or np.ndarray:
#     """Gaussian membership function.

#     This function computes fuzzy membership values using a Gaussian membership function using NumPy.

#     Args:
#         x (float, np.ndarray): input value.
#         center (float): Center of the distribution.
#         sigma (float): standard deviation.

#     Returns:
#         A float or numpy.ndarray.
#     """

#     return np.exp(-((x - center) ** 2) / (2 * sigma))


# def gbellmf(
#     x: float or np.ndarray, center: float, sigma: float, b: float
# ) -> float or np.ndarray:
#     """Generalized bell-shaped membership function.

#     This function computes membership values using a generalized bell membership function using NumPy.

#     Args:
#         a (float): standard deviation.
#         b (float): exponent.
#         c (float): center.

#     Returns:
#         A numpy.array.
#     """
#     return 1 / (1 + np.abs((x - center) / sigma) ** (2 * b))


# def pimf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
#     """Pi-shaped membership function.

#     This function computes membership values using a pi-shaped membership function using NumPy.

#     Args:
#         a (float): Left feet.
#         b (float): Left peak.
#         c (float): Right peak.
#         d (float): Right feet.

#     Returns:
#         A numpy.array.
#     """
#     return np.where(
#         x <= a,
#         0,
#         np.where(
#             x <= (a + b) / 2.0,
#             2 * ((x - a) / (b - a)) ** 2,
#             np.where(
#                 x <= c,
#                 1,
#                 np.where(
#                     x <= (c + d) / 2.0,
#                     1 - 2 * ((x - c) / (d - c)) ** 2,
#                     np.where(x <= d, 2 * ((x - d) / (d - c)) ** 2, 0),
#                 ),
#             ),
#         ),
#     )


# def sigmf(x, center, alpha) -> np.ndarray:
#     """Sigmoidal membership function.

#     This function computes fuzzy membership values using a sigmoidal membership function using NumPy.

#     Args:
#         x (float, np.array): input value.
#         a (float): slope.
#         c (float): center.

#     Returns:
#         A numpy.array.
#     """
#     return 1 / (1 + np.exp(-np.abs(alpha) * (x - center)))


# def smf(x, a, b) -> np.ndarray:
#     """S-shaped membership function

#     This function computes fuzzy membership values using a S-shaped membership function using NumPy.

#     Args:
#         a (float): Left feet.
#         b (float): Right peak.

#     Returns:
#         A numpy.array.
#     """

#     return np.where(
#         x <= a,
#         0,
#         np.where(
#             x <= (a + b) / 2,
#             2 * ((x - a) / (b - a)) ** 2,
#             np.where(x <= b, 1 - 2 * ((x - b) / (b - a)) ** 2, 1),
#         ),
#     )


# def trapmf(x, a, b, c, d) -> np.ndarray:
#     """Trapezoida membership function

#     This function computes fuzzy membership values using a trapezoidal membership function using NumPy.

#     Args:
#         a (float): Left feet.
#         b (float): Left peak.
#         c (float): Right peak.
#         d (float): Right feet.

#     Returns:
#         A numpy.array.
#     """
#     a = np.where(a == b, a - 1e-4, a)
#     d = np.where(d == c, d + 1e-4, d)

#     return np.where(
#         x <= a,
#         0,
#         np.where(
#             x <= b,
#             (x - a) / (b - a),
#             np.where(x <= c, 1, np.where(x <= d, (d - x) / (d - c), 0)),
#         ),
#     )


# def trimf(x, a, b, c) -> np.ndarray:
#     """Triangular membership function.

#     This function computes fuzzy membership values using a triangular membership function using NumPy.

#     Args:
#         a (float): Left feet.
#         b (float): center or peak.
#         c (float): right feet.

#     Returns:
#         A numpy.array.
#     """
#     a = np.where(a == b, a - 1e-4, a)
#     c = np.where(b == c, c + 1e-4, c)
#     return np.where(
#         x <= a,
#         0,
#         np.where(x <= b, (x - a) / (b - a), np.where(x <= c, (c - x) / (c - b), 0)),
#     )


# def zmf(x, a, b) -> np.ndarray:
#     """Z-shaped membership function

#     This function computes fuzzy membership values using a Z-shaped membership function using NumPy.

#     Args:
#         x (float, np.array): input value.
#         a (float): Left peak.
#         b (float): Right feet.

#     Returns:
#         A numpy.array.
#     """
#     return np.where(
#         x <= a,
#         1,
#         np.where(
#             x <= (a + b) / 2,
#             1 - 2 * ((x - a) / (b - a)) ** 2,
#             np.where(x <= b, 2 * ((x - b) / (b - a)) ** 2, 0),
#         ),
#     )
