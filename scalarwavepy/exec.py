#!/usr/bin/env python3

import numpy as np
from scalarwavepy import ode
from scalarwavepy import wave
from scalarwavepy import utils
from scalarwavepy import analytic
from scalarwavepy import plotmod as pltm
from scalarwavepy import convergence as c

cfl = 0.4
dx = 1 / 100
nx = utils.n_from_dx(0, 1, dx)
dt = wave.set_dt_from_courant(cfl, dx)
tf = 2  # 500 * dt
# teval = 23 * dt
# piline, xiline = c.convergence(dx, tf, teval, 5, True, True, True)
# pi_convs, xi_convs = c.convergence_over_time(dx, tf, True, True)

ampl = 8
sigma = 1 / 400
cntr = -0.2
pulse = analytic.Gaussian(cntr, ampl, sigma)


w = wave.ScalarWave(
    pulse,
    nx=nx,
    t_final=tf,
    courant_factor=cfl,
)
w.evolve()
# pltm.plot_time_evolution(w, pulse, gif=False)

energy_density = ode.calculate_diagnostics(w.state_vector, dx)
pltm.plot_energy_density(w.t, energy_density, savefig=True)

# plt.plot(t, utils.L2_norm(dx, state_vector[0]))
# plt.savefig("l2norm.png")
# plt.clf()
