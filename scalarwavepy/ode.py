#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scalarwavepy import utils
import os
import copy


def RHS(s, dx):
    u, pi, xi = s

    pix = utils.spatial_derivative(pi, dx)
    xix = utils.spatial_derivative(xi, dx)

    # interior
    dtu = pi[:]
    dtpi = xix[:]
    dtxi = pix[:]

    # pi_t - xi_t = 0 # here put the analytic solution for "generation"
    # pi_t + xi_t = xi_x + pi_x
    dtpi[0] = 0.5 * (
        xix[0] + pix[0]
    )  ## left boundary pi_t = 0.5 * ( xi_x + pi_x )
    dtpi[-1] = 0.5 * (
        xix[-1] - pix[-1]
    )  ## right boundary  pi_t = 0.5 * ( xi_x - pi_x )

    dtxi[0] = 0.5 * (
        xix[0] + pix[0]
    )  ## left boundary   xi_t = 0.5 * ( xi_x + pi_x )
    dtxi[-1] = 0.5 * (
        -xix[-1] + pix[-1]
    )  ## right boundary xi_t = 0.5 * ( pi_x - xi_x )

    return np.array([dtu, dtpi, dtxi])


def rhs(dx):
    return lambda s: RHS(s, dx)


def calculate_diagnostics(dx, state_vector):
    ud, pid, xid = state_vector[:, :, :]
    energy = utils.integrate(dx, pid ** 2 + xid ** 2)
    energy_density = (1 / ud.shape[0]) * energy
    return energy_density
