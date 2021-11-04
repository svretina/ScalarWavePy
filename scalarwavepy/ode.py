#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scalarwavepy import utils
import os
import copy


def RHS(s, dx, ustar, vstar):
    u, pi, xi = s
    alpha = 1.

    pix = utils.spatial_derivative(pi, dx)
    xix = utils.spatial_derivative(xi, dx)

    # pi = xi
    # u = pi - xi
    # v = pi + xi

    # pi_t = xi_x
    # xi_t = pi_x

    # u_t = pi_t - xi_t = xi_x - pi_x = -u_x
    # v_t = pi_t + xi_T = xi_x + pi_x = v_x

    # Left B:
    # u_t = -u_x - (a/dx) * (u-u*) [u* is value that I want to impose]
    # v_t = v_x

    # Right B:
    # u_t = -u_x
    # v_t = v_x - (a/dx) * (v-v*) [v* is value that I want to impose]
    # then back to pi, xi

    # interior
    dtu = pi[:]
    dtpi = xix[:]
    dtxi = pix[:]

    # Weak boundary imposition
    # Left boundary
    dtpi[0] = xix[0] - (alpha/(2*dx)) * (pi[0]-xi[0]-ustar)
    dtxi[0] = pix[0] + (alpha/(2*dx)) * (pi[0]-xi[0]-ustar)

    # Right boundary
    dtpi[-1] = xix[-1] - (alpha/(2*dx)) * (pi[-1]+xi[-1] - vstar)
    dtxi[-1] = pix[-1] - (alpha/(2*dx)) * (pi[-1]+xi[-1] - vstar)

    # # pi_t - xi_t = 0 # here put the analytic solution for "generati on"
    # # pi_t + xi_t = xi_x + pi_x
    # # u_t = 0
    # # v_t = v_x
    # #
    # dtpi[0] = 0.5 * (
    #     xix[0] + pix[0]
    # )  ## left boundary pi_t = 0.5 * ( xi_x + pi_x )
    # dtpi[-1] = 0.5 * (
    #     xix[-1] - pix[-1]
    # )  ## right boundary  pi_t = 0.5 * ( xi_x - pi_x )

    # dtxi[0] = 0.5 * (
    #     xix[0] + pix[0]
    # )  ## left boundary   xi_t = 0.5 * ( xi_x + pi_x )
    # dtxi[-1] = 0.5 * (
    #     -xix[-1] + pix[-1]
    # )  ## right boundary xi_t = 0.5 * ( pi_x - xi_x )

    return np.array([dtu, dtpi, dtxi])


# def rhs(dx):
#     return lambda s: RHS(s, dx)

def rhs(dx, ustar, vstar):
    return lambda s: RHS(s, dx, ustar, vstar)


def calculate_diagnostics(dx, state_vector):
    ud, pid, xid = state_vector[:, :, :]
    energy = utils.integrate(dx, pid ** 2 + xid ** 2)
    energy_density = (1 / ud.shape[0]) * energy
    energy_density = energy_density / energy_density[0]
    if check_monotonicity(energy_density):
        return energy_density
    else:
        raise VallueError("Energy is not monotonically descreasing")

def check_monotonicity(vector):
    dv = np.diff(vector)
    return np.all(dv<=0)
