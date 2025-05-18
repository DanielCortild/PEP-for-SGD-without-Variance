import numpy as np
import itertools


def build_e(sgd, i):
    """
    Extracts gradient vectors. Vector e(i) is such that e(i) * P = g_i.
    :param sgd: SGD class object.
    :param i: Index of which we wish to extract the vector.
    :return: Vector e(i).
    """

    e = np.zeros((1, sgd.NB_VARS))

    if i in [*sgd.PARTIAL_CURR, *sgd.FULL_NEXT, *sgd.PARTIAL_STAR]:
        e[0, i] = 1
        return e

    elif i == sgd.STAR:
        for j in sgd.PARTIAL_STAR:
            e[0, j] = sgd.PROBABILITY
        return e

    elif i == sgd.CURR:
        for j in sgd.PARTIAL_CURR:
            e[0, j] = sgd.PROBABILITY
        return e

    else:
        raise ValueError(f"Invalid index for e: i={i}")


def build_u(sgd, i):
    """
    Extracts point vectors. Vector u(i) is such that u(i) * P = x_i.
    :param sgd: SGD class object.
    :param i: Index of which we wish to extract the vector.
    :return: Vector u(i).
    """

    u = np.zeros((1, sgd.NB_VARS))

    if i in sgd.FULL_NEXT:
        u[0, sgd.CURR] = 1
        offset = sgd.FULL_NEXT[0] - sgd.PARTIAL_CURR[0]
        u[0, i - offset] = -sgd.gamma
        return u

    elif i in sgd.PARTIAL_CURR or i == sgd.CURR:
        u[0, sgd.CURR] = 1
        return u

    elif i in sgd.PARTIAL_STAR or i == sgd.STAR:
        return u

    else:
        raise ValueError(f"Invalid index for u: i={i}")


def build_f(sgd, i):
    """
    Extracts function values vectors. Vector f(i) is such that f(i) * F = f_i.
    :param sgd: SGD class object.
    :param i: Index of which we wish to extract the vector.
    :return: Vector f(i).
    """

    f = np.zeros((1, sgd.NB_VARS))

    if i in sgd.PARTIAL_CURR or i in sgd.PARTIAL_STAR or i in sgd.FULL_NEXT:
        f[0, i] = 1
        return f

    elif i == sgd.CURR:
        for j in sgd.PARTIAL_CURR:
            f[0, j] = sgd.PROBABILITY
        return f

    elif i == sgd.STAR:
        for j in sgd.PARTIAL_STAR:
            f[0, j] = sgd.PROBABILITY
        return f

    else:
        raise ValueError(f"Invalid index for f: i={i}")


def build_L(sgd, lamb1, lamb2, lamb, rho):
    f = lambda _: build_f(sgd, _)

    interp_group_f = [*sgd.FULL_NEXT, sgd.CURR, sgd.STAR]

    L = - rho * (f(sgd.CURR) - f(sgd.STAR)) \
        + sum([lamb1[i] * (f(j) - f(i)) for i, j in zip(sgd.PARTIAL_STAR, sgd.PARTIAL_CURR)]) \
        + sum([lamb2[i] * (f(j) - f(i)) for i, j in zip(sgd.PARTIAL_CURR, sgd.PARTIAL_STAR)]) \
        + sum([lamb[i][j] * (f(j) - f(i)) for i, j in itertools.product(interp_group_f, interp_group_f)])

    return L
