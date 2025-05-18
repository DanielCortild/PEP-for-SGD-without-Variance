import numpy as np
import itertools
from .build_vectors import build_e, build_u


def build_Delta(sgd, vars_next, vars_curr):
    """
    Builds matrix Delta. Is such that tr(Delta @ G) = E_(k+1) - E_k.
    :param sgd: SGD class object.
    :param vars_next: Tuple of variables for the coefficients of the Lyapunov of the next iterate.
    :param vars_curr: Tuple of variables for the coefficients of the Lyapunov of the current iterate.
    :return: Matrix Delta.
    """

    # Setup functions to extract vectors
    e = lambda _: build_e(sgd, _)
    u = lambda _: build_u(sgd, _)

    # Initialize Delta
    Delta = np.zeros((sgd.NB_VARS, sgd.NB_VARS))

    # From Next Iterate
    ak1, ek1 = vars_next
    Delta = Delta + ak1 * sum([u(i).T @ u(i) for i in sgd.FULL_NEXT]) * sgd.PROBABILITY

    # From Previous Iterate
    ak0, ek0 = vars_curr
    Delta = Delta - ak0 * u(sgd.CURR).T @ u(sgd.CURR)
    Delta = Delta - ek0 * sum([e(i).T @ e(i) for i in sgd.PARTIAL_STAR]) * sgd.PROBABILITY

    return Delta


def build_A_star(sgd):
    """
    Builds matrix AStar. Is such that tr(AStar @ G) = |g_*|^2.
    :param sgd: SGD class object.
    :return: Matrix AStar.
    """

    # Setup functions to extract vectors
    e = lambda _: build_e(sgd, _)

    # Build AStar
    AStar = e(sgd.STAR).T @ e(sgd.STAR)

    return AStar


def build_A_interp(sgd, i, j):
    """
    Builds interpolation matrix A. Is such that tr(A_{i,j} @ G) represents the interpolation inequality,
    without function values.
    :param sgd: SGD class object.
    :param i: Index of the first point in the interpolation inequality.
    :param j: Index of the second point in the interpolation inequality.
    :return: Matrix A.
    """

    # Setup functions to extract vectors
    e = lambda _: build_e(sgd, _)
    u = lambda _: build_u(sgd, _)

    # Build interpolation A
    A = np.zeros((sgd.NB_VARS, sgd.NB_VARS))
    A += 1 / (2 * (sgd.L - sgd.mu)) * (e(i) - e(j)).T @ (e(i) - e(j))
    A += sgd.L * sgd.mu / (2 * (sgd.L - sgd.mu)) * (u(i) - u(j)).T @ (u(i) - u(j))
    A += sgd.L / (2 * (sgd.L - sgd.mu)) * (e(j).T @ (u(i) - u(j)) + (u(i) - u(j)).T @ e(j))
    A += sgd.mu / (2 * (sgd.L - sgd.mu)) * (e(i).T @ (u(j) - u(i)) + (u(j) - u(i)).T @ e(i))

    return A


def build_S(sgd, tau, lamb1s, lamb2s, lambs, vars_next, vars_curr):
    """
    Builds matrix S. Is such that S >> 0 describes the LMI.
    :param sgd: SGD class object.
    :param tau: Dual multiplier.
    :param lamb1s: Dual multiplier.
    :param lamb2s: Dual multiplier.
    :param lambs: Dual multiplier.
    :param vars_next: Tuple of variables for the coefficients of the Lyapunov of the next iterate.
    :param vars_curr: Tuple of variables for the coefficients of the Lyapunov of the current iterate.
    :return: Matrix S.
    """

    # Functions that should interpolate
    interp_group_f = [sgd.CURR, *sgd.FULL_NEXT, sgd.STAR]

    AStar = build_A_star(sgd)
    Delta = build_Delta(sgd, vars_next, vars_curr)
    A = lambda i, j: build_A_interp(sgd, i, j)

    S = tau * AStar - Delta \
        + sum([lamb1s[i] * A(i, j) for i, j in zip(sgd.PARTIAL_STAR, sgd.PARTIAL_CURR)]) \
        + sum([lamb2s[i] * A(i, j) for i, j in zip(sgd.PARTIAL_CURR, sgd.PARTIAL_STAR)]) \
        + sum([lambs[i][j] * A(i, j) for i, j in itertools.product(interp_group_f, interp_group_f)])

    return S
