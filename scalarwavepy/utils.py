#!/usr/bin/env python3

import numpy as np


def discretize(ui, uf, nu):
    du = spacing(ui, uf, nu)
    ns = np.arange(0, nu + 1, 1)
    u = ui + ns * du
    return u


def spatial_derivative(f, dx):
    tmp = np.empty_like(f)
    tmp[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    tmp[0] = (f[1] - f[0]) / dx
    tmp[-1] = (f[-1] - f[-2]) / dx
    return tmp


def rk4(func, s, h):
    k0 = func(s)
    s1 = s + (h / 2) * k0
    k1 = func(s1)
    s2 = s + (h / 2) * k1
    k2 = func(s2)
    s3 = s + h * k2
    k3 = func(s3)
    s4 = s + (h / 6) * (k0 + 2 * k1 + 2 * k2 + k3)
    return s4


def L2_norm(vec, dx):
    return np.sqrt(integrate(vec * vec, dx))


def integrate(vec, dx, over="rows"):
    # trapezoidal rule
    if vec.ndim == 1:
        tmp = 0.5 * (vec[0] + vec[-1]) + np.sum(vec[1:-1], axis=0)
    elif vec.ndim == 2:
        if over == "rows":
            tmp = 0.5 * (vec[0, :] + vec[-1, :]) + np.sum(
                vec[1:-1, :], axis=0
            )
        else:
            tmp = 0.5 * (vec[:, 0] + vec[:, -1]) + np.sum(
                vec[:, 1:-1], axis=1
            )

    return dx * tmp


def npoints_from_dx(xi, xn, dx):
    n = int(round((xn - xi) / dx) + 1)
    return n

def ncells_from_dx(xi, xn, dx):
    n = int(round((xn-xi) / dx))
    return n

def spacing_of_array(x):
    dx = (x[-1]-x[0])/(len(x)-1)
    return dx

def spacing(xi, xn, n):
    """Calculates the dx for an interval, given the start/end
    of the interval and the number of points. The number of points is
    automatically added +1. If n=100, the result will be for n=101.

    """
    dx = (xn - xi) / n  # (n+1 -1)
    return dx


def check_monotonicity(vector):
    dv = np.diff(vector)
    return np.all(dv > 0) or np.all(dv < 0)
