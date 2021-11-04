#!/usr/bin/env python3

import numpy as np
from scalarwavepy import ode
from scalarwavepy import wave
from scalarwavepy import utils
from scalarwavepy import analytic
from scalarwavepy import convergence as c
from scalarwavepy import plotmod as pltm



dx = 1 / 10
nx = utils.n_from_dx(dx)
tf = 50 * 0.4 * dx
teval = 23 * 0.4 * dx
piline, xiline = c.convergence(dx, tf, teval, 5, True, True, True)
# pi_convs, xi_convs = c.convergence_over_time(dx, tf, True, True)


w = wave.ScalarWave(
    c.pulse,
    nx=nx,
    t_final= tf,
    courant_factor=0.4,
)
w.evolve()
# pltm.plot_time_evolution(w, pulse, gif=False)

energy_density = ode.calculate_diagnostics(dx, w.state_vector)
pltm.plot_energy_density(w.t, energy_density, savefig=True)

# plt.plot(t, utils.L2_norm(dx, state_vector[0]))
# plt.savefig("l2norm.png")
# plt.clf()
