#!/usr/bin/env python3

import copy
import numpy as np
from scalarwavepy import grids
from scalarwavepy import utils
from scalarwavepy import grid_functions as gf


def get_boundary_values(
    func,
    spatial_grid,
    time_instance,
):
    # pistar0 = func.dt(spatial_grid.domain[0], time_instance)
    # xistar0 = func.dx(spatial_grid.domain[0], time_instance)
    ustar = 0  # pistar0 - xistar0
    vstar = 0
    return ustar, vstar


def evolve(state, spatial_grid, time_grid, alpha):
    if isinstance(spatial_grid, grids.MultipleGrid):
        result = gf.Result(spatial_grid, time_grid)
        result.initialize(state)
        result.set_penalty(alpha)
        vold2to1 = 0
        for j in range(time_grid.ncells):
            ti = time_grid.coords[j]
            for i in range(spatial_grid.ndomains):
                ustar, vstar = get_boundary_values(
                    state.func, spatial_grid.ugrids[i], ti
                )
                if i > 0:
                    ustar = uold1to2
                if i == 0:
                    vstar = vold2to1

                rhs_func = rhs(spatial_grid.dx[i], ustar, vstar, alpha)
                state.state_tensor[i] = utils.rk4(
                    rhs_func, state.state_tensor[i], time_grid.spacing
                )

                if i == 0:
                    uold1to2 = (
                        state.state_tensor[i].pi.values[-1]
                        - state.state_tensor[i].xi.values[-1]
                    )
                if i > 0:
                    vold2to1 = (
                        state.state_tensor[i].pi.values[0]
                        + state.state_tensor[i].xi.values[0]
                    )
                result.tensor[i, j + 1] = state.state_tensor[i]

    elif isinstance(spatial_grid, grids.SingleGrid):
        result = gf.Result(spatial_grid, time_grid)
        result.initialize(state)
        result.set_penalty(alpha)
        for i in range(time_grid.ncells):
            # ti = time_grid.coords[i]
            # ustar, vstar = get_boundary_values(state.func, spatial_grid, ti)
            rhs_func = rhs(spatial_grid.dx, 0, 0, alpha)
            state = utils.rk4(rhs_func, state, time_grid.spacing)
            result.vector[i + 1] = state
    else:
        raise TypeError(
            f"""
            Grid can only be SingleGrid or MultipleGrid object. You passed a {type(spatial_grid)} object."""
        )
    return result


def RHS(s, dx, ustar, vstar, alpha):
    u, pi, xi = s.state_vector
    pix = pi.differentiated
    xix = xi.differentiated
    # u = pi - xi
    # v = pi + xi

    # pi_t = xi_x
    # xi_t = pi_x

    # u_t = pi_t - xi_t = xi_x - pi_x = -u_x
    # v_t = pi_t + xi_t = xi_x + pi_x = v_x

    # Left B:
    # u_t = -u_x - (2a/dx) * (u-u*) [u* is value that I want to impose]
    # v_t = v_x

    # Right B:
    # u_t = -u_x
    # v_t = v_x - (2a/dx) * (v-v*) [v* is value that I want to impose]
    # then back to pi, xi
    # interior
    dtu = copy.deepcopy(pi)
    dtpi = copy.deepcopy(xix)
    dtxi = copy.deepcopy(pix)
    # Weak boundary imposition
    # ## IMPORTANT => the weight of the stencil is canceling out with a factor of 2
    # when transforming back to pi, xi variables from u,v but that holds only for weight = 0.5
    # at the boundaries. if we upgrade to more accurate stencils this has to change.
    #  Left boundary
    dtpi.values[0] = xix.values[0] - (alpha / (1 * dx)) * (
        pi.values[0] - xi.values[0] - ustar
    )
    dtxi.values[0] = pix.values[0] + (alpha / (1 * dx)) * (
        pi.values[0] - xi.values[0] - ustar
    )

    # Right boundary
    dtpi.values[-1] = xix.values[-1] - (alpha / (1 * dx)) * (
        pi.values[-1] + xi.values[-1] - vstar
    )
    dtxi.values[-1] = pix.values[-1] - (alpha / (1 * dx)) * (
        pi.values[-1] + xi.values[-1] - vstar
    )

    return type(s)(
        grid=s.grid,
        vector=np.array([dtu, dtpi, dtxi]),
        func=s.func,
    )


def rhs(dx, ustar, vstar, alpha):
    return lambda s: RHS(s, dx, ustar, vstar, alpha)
