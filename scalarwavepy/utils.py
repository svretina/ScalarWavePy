#!/usr/bin/env python3

import numpy as np


def discretize(ui, uf, du):
    n = uf / du + 1
    ns = np.arange(0, n, 1)
    uf = ui + ns * du
    return uf


def spatial_derivative(u, dx):
    u = np.gradient(u, dx, edge_order=1)
    return u


def rk4(func, s, h):
    k0 = func(s)
    s1 = s + h / 2 * k0
    k1 = func(s1)
    s2 = s + h / 2 * k1
    k2 = func(s2)
    s3 = s + h * k2
    k3 = func(s3)
    s4 = s + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)
    return s4


def L2_norm(dx, vec):
    return np.sqrt(integrate(dx, vec * vec))


def integrate(dx, vec):
    # trapezoidal rule
    tmp = 0.5 * (vec[0] + vec[-1]) + np.sum(vec[1:-1], axis=0)
    return dx * tmp
