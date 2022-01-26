#!/usr/bin/env python3

import os
import numpy as np
from scalarwavepy import grid
from scalarwavepy import utils


def RHS(s, dx, alpha, ustar, vstar):
    # print("-RHS", s)
    u, pi, xi = s.state_vector
    alpha = 1.0
    pix = pi.differentiated()
    xix = xi.differentiated()
    # pi = xi
    # u = pi - xi
    # v = pi + xi

    # pi_t = xi_x
    # xi_t = pi_x

    # u_t = pi_t - xi_t = xi_x - pi_x = -u_x
    # v_t = pi_t + xi_t = xi_x + pi_x = v_x

    # Left B:
    # u_t = -u_x - (a/dx) * (u-u*) [u* is value that I want to impose]
    # v_t = v_x

    # Right B:
    # u_t = -u_x
    # v_t = v_x - (a/dx) * (v-v*) [v* is value that I want to impose]
    # then back to pi, xi

    # interior
    # dtu = pi[:]
    # dtpi = xix[:]
    # dtxi = pix[:]
    dtu = pi
    dtpi = xix
    dtxi = pix
    dx = pi.grid.dx
    # Weak boundary imposition
    # Left boundary
    ## why 2dx ??
    dtpi.values[0] = xix.values[0] - (alpha / (2 * dx)) * (
        pi.values[0] - xi.values[0] - ustar
    )
    dtxi.values[0] = pix.values[0] + (alpha / (2 * dx)) * (
        pi.values[0] - xi.values[0] - ustar
    )

    # Right boundary
    dtpi.values[-1] = xix.values[-1] - (alpha / (2 * dx)) * (
        pi.values[-1] + xi.values[-1] - vstar
    )
    dtxi.values[-1] = pix.values[-1] - (alpha / (2 * dx)) * (
        pi.values[-1] + xi.values[-1] - vstar
    )
    # it should return a StateVector object!!!
    sv = type(s)(np.array([dtu, dtpi, dtxi]), s.func)
    return sv


def rhs(dx, ustar, vstar, alpha=1.0):
    return lambda s: RHS(s, dx, alpha, ustar, vstar)


def calculate_diagnostics(state_vector, dx):
    _, pid, xid = state_vector[:, :, :]
    energy = utils.integrate(pid ** 2 + xid ** 2, dx, over="rows")
    energy_density = (1 / ud.shape[0]) * energy
    energy_density = energy_density / energy_density[0]
    return energy_density
    # if utils.check_monotonicity(energy_density):
    #     return energy_density
    # else:
    #     raise VallueError("Energy is not monotonically descreasing")
