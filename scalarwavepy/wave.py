#!/usr/bin/env python3

import numpy as np
import copy
from scalarwavepy import ode
from scalarwavepy import utils


def set_dt_from_courant(c, dx):
    return c * dx


class ScalarWave:
    def __init__(
        self,
        initfunc,
        domain_x=[0, 1],
        dx=0.01,
        t_final=2,
        courant_factor=0.4,
    ):
        """Constructor of ScalarWave class."""
        self.c = courant_factor
        self.tf = t_final
        self.dx = dx
        self.dt = set_dt_from_courant(self.c, self.dx)
        self.x0 = domain_x[0]
        self.xn = domain_x[-1]
        self.check_courant()
        self.initfunc = initfunc
        self.x = utils.discretize(self.x0, self.xn, self.dx)
        self.t = utils.discretize(0, self.tf, self.dt)
        self.nx = len(self.x)
        self.nt = len(self.t)
        self.state_vector = np.zeros((3, self.nx, self.nt))
        self.state_vector[:, :, 0] = self.initialize_solution()

    def check_courant(self):
        if self.c >= 1:
            raise ValueError(
                f"Courant limit violation: c={self.c}"
            )

    def initialize_solution(self):
        u = self.initfunc.expr(self.x, 0)
        pi = self.initfunc.expr_dt(self.x, 0)
        xi = self.initfunc.expr_dx(self.x, 0)
        return np.array([u, pi, xi])

    # def prepare_gaussian_pulse(x, center=0.5, amplitude=1, sigma=1):
    #     dx = x[2]-x[1]
    #     u = analytic_sol(x, center, amplitude, sigma)
    #     pi = - analytic_derivative_solx(x, center, amplitude, sigma)
    #     xi = utils.spatial_derivative(u, dx)
    #     return np.array([u, pi, xi])

    def evolve(self):
        rhs_func = ode.rhs(self.dx)
        for i in range(0, self.nt - 1):
            self.state_vector[:, :, i + 1] = utils.rk4(
                rhs_func, self.state_vector[:, :, i], self.dt
            )
        return self.state_vector
