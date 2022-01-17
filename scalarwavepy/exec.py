#!/usr/bin/env python3

import numpy as np
from scalarwavepy import ode
from scalarwavepy import grid
from scalarwavepy import wave
from scalarwavepy import utils
from scalarwavepy import globvars
from scalarwavepy import analytic
from scalarwavepy import plotmod as pltm
from scalarwavepy import convergence as c

ncells = 10
spatial_domain = grid.SingleDomain([0, 1])
time_domain = grid.SingleDomain([0, 1])
spatial_grid = grid.SingleGrid(spatial_domain, ncells)

time_grid = grid.TimeGrid_from_cfl(spatial_grid, time_domain)

f = globvars.PULSE
sv = grid.StateVector(func=f)
sv.initialize(spatial_grid)

res = wave.evolve(sv, spatial_grid, time_grid)
print(res[:, :, :])

# for i in 100 * np.array([1, 2, 3, 4, 5, 6]):
#     dx = 1 / i
#     nx = utils.n_from_dx(0, 1, dx)
#     dt = wave.set_dt_from_courant(globvars.CFL, dx)

#     tf = 2  # 500 * dt

#     w = wave.ScalarWave(
#         globvars.PULSE,
#         nx=nx,
#         t_final=tf,
#         courant_factor=globvars.CFL,
#     )
#     w.evolve()

#     # pltm.plot_time_evolution(w, pulse, gif=False)

#     energy_density = ode.calculate_diagnostics(w.state_vector, dx)
#     pltm.plot_energy_density(w.t, energy_density, savefig=True)


# plt.plot(t, utils.L2_norm(dx, state_vector[0]))
# plt.savefig("l2norm.png")
# plt.clf()
