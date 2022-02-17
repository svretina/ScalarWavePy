#!/usr/bin/env python3

import numpy as np
from scalarwavepy import ode
from scalarwavepy import grid
from scalarwavepy import utils
from scalarwavepy import grids
from scalarwavepy import domains
from scalarwavepy import analytic
from scalarwavepy import global_vars
from scalarwavepy import plotmod as pltm

ncells = [10, 20]
domain = [[0, 1], [2, 3]]

spatial_domain = domains.MultipleDomain(domain)
time_domain = domains.SingleDomain([0, 2])
spatial_grid = grids.MultipleGrid(spatial_domain, ncells)
time_grid = grids.TimeGrid_from_cfl(spatial_grid, time_domain)

# spatial_domain = grids.SingleDomain([0, 1])
# time_domain = grids.SingleDomain([0, 2])
# spatial_grid = grids.SingleGrid(spatial_domain, ncells)

f = global_vars.PULSE
sv = grid.StateVector(func=f)
sv.initialize(spatial_grid)
res = ode.evolve(sv, spatial_grid, time_grid)
exit()
pltm.plot_time_evolution(time_grid[0], res, gif=True)
