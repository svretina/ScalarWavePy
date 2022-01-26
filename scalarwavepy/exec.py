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

ncells = 120
spatial_domain = grid.SingleDomain([0, 1])
time_domain = grid.SingleDomain([0, 2])
spatial_grid = grid.SingleGrid(spatial_domain, ncells)

time_grid = grid.TimeGrid_from_cfl(spatial_grid, time_domain)

f = globvars.PULSE
sv = grid.StateVector(func=f)
sv.initialize(spatial_grid)

res = wave.evolve(sv, spatial_grid, time_grid)
pltm.plot_time_evolution(time_grid, res, gif=True)
