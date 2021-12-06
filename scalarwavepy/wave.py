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
        nx=100,
        t_final=2,
        courant_factor=0.4,
    ):
        """Constructor of ScalarWave class."""
        self.c = courant_factor
        self.tf = t_final
        assert type(nx) is int
        self.nx = nx
        self.x0 = domain_x[0]
        self.xn = domain_x[-1]
        self.dx = self.xn / self.nx
        self.dt = set_dt_from_courant(self.c, self.dx)
        self.nt = utils.n_from_dx(0, self.tf, self.dt)

        self.check_courant()
        self.initfunc = initfunc
        self.x = utils.discretize(self.x0, self.xn, self.nx)
        self.t = utils.discretize(0, self.tf, self.nt)

        self.state_vector = np.zeros((3, self.nx + 1, self.nt + 1))
        self.state_vector[:, :, 0] = self.initialize_solution()

    def check_courant(self):
        if self.c >= 1:
            raise ValueError(f"Courant limit violation: c={self.c}")

    def initialize_solution(self):
        u = self.initfunc(self.x, 0)
        pi = self.initfunc.dt(self.x, 0)
        xi = self.initfunc.dx(self.x, 0)
        return np.array([u, pi, xi])

    def evolve(self):
        alpha = 1.0 / 2
        for i in range(0, self.nt):
            ti = i * self.dt
            pistar0 = self.initfunc.dt(self.x0, ti)
            pistarN = self.initfunc.dt(self.xn, ti)
            xistar0 = self.initfunc.dx(self.x0, ti)
            xistarN = self.initfunc.dx(self.xn, ti)
            ustar = pistar0 - xistar0
            # vstar = pistarN + xistarN
            # ustar = 0
            vstar = 0
            rhs_func = ode.rhs(self.dx, ustar, vstar, alpha)
            self.state_vector[:, :, i + 1] = utils.rk4(
                rhs_func, self.state_vector[:, :, i], self.dt
            )
        return self.state_vector
