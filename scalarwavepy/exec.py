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
cntr = 0.8
pulse = analytic.Gaussian(cntr, ampl, sigma)


def convergence(dx_0, tf, t_eval, n=5, plot_convergence=False):
    pold = np.nan
    dxs = []
    pis = []
    xis = []
    courant_factor = 0.4
    factor = 2 * np.linspace(1, n, n, dtype=np.int)
    for i in factor:
        dxprime = dx / i

        w = wave.ScalarWave(
            pulse,
            dx=dxprime,
            t_final=tf,
            courant_factor=courant_factor,
        )
        idx_eval = int(np.round(t_eval / w.dt, 1))

        state_vector = w.evolve()
        numpi = state_vector[1, 0::i, idx_eval]
        numxi = state_vector[2, 0::i, idx_eval]

        analyticpi = pulse.expr_dt(w.x[0::i], w.t[idx_eval])
        analyticxi = pulse.expr_dx(w.x[0::i], w.t[idx_eval])

        diffpi = utils.L2_norm(dxprime, numpi - analyticpi)
        diffxi = utils.L2_norm(dxprime, numxi - analyticxi)

        pis.append(diffpi)
        xis.append(diffxi)
        dxs.append(dxprime)

        # try:
        #     p2 = math.log(diffpiold / diffpi, i / (i - step))
        #     p3 = math.log(diffxiold / diffxi, i / (i - step))
        #     # print(f"pi convergence: {p2}")
        #     # print(f"xi convergence: {p3}")
        # except:
        #     pass
        # diffpiold = diffpi
        # diffxiold = diffxi

    pi_line = np.polyfit(np.log(dxs), np.log(pis), 1)
    xi_line = np.polyfit(np.log(dxs), np.log(xis), 1)

    if plot_convergence:
        pltm.plot_convergence(
            dxs, pis, xis, pi_line, xi_line, w.t[idx_eval]
        )
    return pi_line, xi_line


def convergence_over_time(dx_0, tf, plot=False):
    pi_convergence = []
    xi_convergence = []
    time = utils.discretize(0, tf, 0.4 * dx_0)
    for i in range(1, len(time)):
        print(f"time[{i}] = {time[i]}")
        piline, xiline = convergence(dx_0, tf, time[i], False)
        pi_convergence.append(piline[0])
        xi_convergence.append(xiline[0])
    if plot:
        pltm.plot_convergence_over_time(
            time[1:], pi_convergence, xi_convergence
        )
    return pi_convergence, xi_convergence


dx = 1 / 100
tf = 2
piline, xiline = convergence(dx, tf, 1, 5, True)

# pi_convs, xi_convs = convergence_over_time(dx, tf, True)
# print("fit pi:", piline)
# print("fit xi:", xiline)

# energy_density = ode.calculate_diagnostics(dxprime, state_vector)

# plt.plot(t, energy_density)
# plt.savefig("energy_density.png")
# plt.clf()

# plt.plot(t, utils.L2_norm(dx, state_vector[0]))
# plt.savefig("l2norm.png")
# plt.clf()
w = wave.ScalarWave(
    pulse,
    dx=dx,
    t_final=4 * tf,
    courant_factor=0.4,
)
w.evolve()
pltm.plot_time_evolution(w, pulse, gif=True)

gif = False
if gif:
    os.chdir("./results")
    print("Converting to gif...")
    os.system(
        f"convert -delay 0.5 -loop 0 {{0..{nt-1}}}.png wave.gif"
    )
    os.system("mv wave.gif ../wave_absorbing.gif")
