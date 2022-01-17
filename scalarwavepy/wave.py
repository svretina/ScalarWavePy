#!/usr/bin/env python3

import numpy as np
from scalarwavepy import ode
from scalarwavepy import grid
from scalarwavepy import utils
from scalarwavepy import globvars

## Fix this
## how will I store IO of state_vector
def evolve(state_vector, spatial_grid, time_grid, alpha=1.0 / 2.0):
    result = np.empty((3, 1, time_grid.npoints), dtype=object)
    result[:, :, 0] = state_vector
    for i in range(time_grid.ncells):
        ti = time_grid.coords[i]

        pistar0 = state_vector.func.dt(spatial_grid.domain[0], ti)
        pistarN = state_vector.func.dt(spatial_grid.domain[1], ti)

        xistar0 = state_vector.func.dx(spatial_grid.domain[0], ti)
        xistarN = state_vector.func.dx(spatial_grid.domain[1], ti)
        ustar = pistar0 - xistar0
        # vstar = pistarN + xistarN
        # ustar = 0
        vstar = 0

        rhs_func = ode.rhs(spatial_grid.dx, ustar, vstar, alpha)
        ########################
        state_vector = utils.rk4(
            rhs_func, state_vector, time_grid.spacing
        )
        result[:, :, i + 1] = state_vector
    return result
