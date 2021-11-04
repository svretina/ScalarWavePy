#!/usr/bin/env python3

import numpy as np
from scalarwavepy import ode
from scalarwavepy import wave
from scalarwavepy import utils
from scalarwavepy import analytic
from scalarwavepy import plotmod as pltm

import matplotlib.pyplot as plt
import os
import math


ampl = 8
sigma = 1 / 100
cntr = 0.4
pulse = analytic.Gaussian(cntr, ampl, sigma)


def convergence(
    dx_0,
    tf,
    t_eval,
    n=5,
    plot_convergence=False,
    plot_resolutions=False,
    savefigs=False,
):
    pold = np.nan
    dxs = []
    pis = []
    xis = []
    courant_factor = 0.4
    factor = 2 ** np.linspace(1, n, n, dtype=int)
    if plot_resolutions:
        Result = {}
        Result[0] = t_eval
        Result[1] = wave.ScalarWave(
            pulse,
            nx=int(round(1 / dx_0)),
            t_final=tf,
            courant_factor=courant_factor,
        )
    for i in factor:
        subresult = {}
        dxprime = dx_0 / i
        nxprime = utils.n_from_dx(dxprime)

        w = wave.ScalarWave(
            pulse,
            nx=nxprime,
            t_final=tf,
            courant_factor=courant_factor,
        )
        idx_eval = int(np.round(t_eval / w.dt))
        state_vector = w.evolve()

        numpi = state_vector[1, :, idx_eval]
        numxi = state_vector[2, :, idx_eval]

        analyticpi = pulse.dt(w.x, w.t[idx_eval])
        analyticxi = pulse.dx(w.x, w.t[idx_eval])

        diffpi = utils.L2_norm(dxprime, numpi - analyticpi)
        diffxi = utils.L2_norm(dxprime, numxi - analyticxi)

        dxs.append(dxprime)
        pis.append(diffpi)
        xis.append(diffxi)
        if plot_resolutions:
            subresult["x"] = w.x
            subresult["dx"] = dxprime
            subresult["factor"] = i
            subresult["errorpi2"] = diffpi
            subresult["errorxi2"] = diffxi
            subresult["api"] = analyticpi
            subresult["axi"] = analyticxi
            subresult["pi"] = numpi
            subresult["xi"] = numxi
            Result[i] = subresult

    pi_line = np.polyfit(np.log(dxs), np.log(pis), 1)
    xi_line = np.polyfit(np.log(dxs), np.log(xis), 1)

    if plot_resolutions:
        pltm.plot_resolutions(dx_0, Result, pulse, savefigs)
    if plot_convergence:
        pltm.plot_convergence(
            dxs, pis, xis, pi_line, xi_line, w.t[idx_eval], savefigs
        )
    return pi_line, xi_line


def convergence_over_time(dx_0, tf, plot=False, savefig=False):
    pi_convergence = []
    xi_convergence = []
    dt = 0.4 * dx_0
    nt = utils.n_from_dx(dt, tf)
    time = utils.discretize(0, tf, nt)
    for i in range(1, nt + 1):
        print(f"time[{i}] = {time[i]}")
        piline, xiline = convergence(dx_0, tf, time[i])
        pi_convergence.append(piline[0])
        xi_convergence.append(xiline[0])
    if plot:
        pltm.plot_convergence_over_time(
            time[1:], pi_convergence, xi_convergence, savefig
        )
    return pi_convergence, xi_convergence
