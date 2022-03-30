#!/usr/bin/env python3

import numpy as np
from scalarwavepy import domains
from scalarwavepy import grids
from scalarwavepy import grid_functions as gf
from scalarwavepy import global_vars
from scalarwavepy import ode


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
                vec[1:-1, :],
                axis=0,
            )
        else:
            tmp = 0.5 * (vec[:, 0] + vec[:, -1]) + np.sum(
                vec[:, 1:-1],
                axis=1,
            )

    return dx * tmp


def npoints_from_dx(xi, xn, dx):
    n = int(round((xn - xi) / dx) + 1)
    return n


def ncells_from_dx(xi, xn, dx):
    n = int(round((xn - xi) / dx))
    return n


def spacing_of_array(x):
    dx = (x[-1] - x[0]) / (len(x) - 1)
    return dx


def spacing(xi, xn, n):
    """Calculates the dx for an interval, given the start/end
    of the interval and the number of points. The number of points is
    automatically added +1. If n=100, the result will be for n=101.

    """
    dx = (xn - xi) / n  # (n+1 -1)
    return dx


def check_monotonicity(vector):
    vector = np.atleast_1d(vector)
    dv = np.diff(vector)
    return np.all(dv > 0) or np.all(dv < 0)


def get_random(a, b, shape):
    return (b - a) * np.random.random_sample(shape) + a


def run(final_time, ncells, domain, noise, *args, **kwargs):
    if np.asarray(domain).ndim > 1:
        spatial_domain = domains.MultipleDomain(domain)
    else:
        spatial_domain = domains.SingleDomain(domain)

    if isinstance(spatial_domain, domains.MultipleDomain):
        spatial_grid = grids.MultipleGrid(spatial_domain, ncells)
    else:
        spatial_grid = grids.SingleGrid(spatial_domain, ncells)

    time_domain = domains.SingleDomain([0, final_time])
    time_grid = grids.TimeGrid_from_cfl(spatial_grid, time_domain)

    if not noise:
        f = global_vars.PULSE
        if np.asarray(domain).ndim > 1:
            state = gf.StateTensor(grid=spatial_grid, func=f)
        else:
            state = gf.StateVector(grid=spatial_grid, func=f)
    else:

        if np.asanyarray(domain).ndim > 1:
            u = gf.GridFunction(
                spatial_grid.ugrids[0], get_random(-1, 1, spatial_grid.ugrids[0].shape)
            )
            pi = gf.GridFunction(
                spatial_grid.ugrids[0], get_random(-1, 1, spatial_grid.ugrids[0].shape)
            )
            xi = gf.GridFunction(
                spatial_grid.ugrids[0], get_random(-1, 1, spatial_grid.ugrids[0].shape)
            )
            u2 = gf.GridFunction(
                spatial_grid.ugrids[1], get_random(-1, 1, spatial_grid.ugrids[1].shape)
            )
            pi2 = gf.GridFunction(
                spatial_grid.ugrids[1], get_random(-1, 1, spatial_grid.ugrids[1].shape)
            )
            xi2 = gf.GridFunction(
                spatial_grid.ugrids[1], get_random(-1, 1, spatial_grid.ugrids[1].shape)
            )
            st = np.array(
                [
                    gf.StateVector(
                        grid=spatial_grid.ugrids[0], vector=np.array([u, pi, xi])
                    ),
                    gf.StateVector(
                        grid=spatial_grid.ugrids[1], vector=np.array([u2, pi2, xi2])
                    ),
                ],
                dtype=object,
            )
            state = gf.StateTensor(grid=spatial_grid, tensor=st)
        else:
            u = gf.GridFunction(spatial_grid, get_random(-1, 1, spatial_grid.shape))
            pi = gf.GridFunction(spatial_grid, get_random(-1, 1, spatial_grid.shape))
            xi = gf.GridFunction(spatial_grid, get_random(-1, 1, spatial_grid.shape))
            sv = np.array([u, pi, xi])
            state = gf.StateVector(grid=spatial_grid, vector=sv)

    res = ode.evolve(state, spatial_grid, time_grid, *args, **kwargs)
    return res
