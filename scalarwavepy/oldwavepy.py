#!/usr/bin/env python3

import numpy as np
import scalarwavepy.pythran_lax as lax
import matplotlib.pyplot as plt
import copy


def set_dt_from_courant(c, dx):
    return c * dx


class ScalarWave:
    def __init__(
        self,
        mesh,
        alpha,
        dt,
        nt,
        initfunc,
        exfunc,
        t0=0,
    ):
        """Constructor of ScalarWave class."""
        self.c = alpha * dt / mesh.dx
        self.alpha = alpha
        self.check_courant()
        self.nt = nt
        self.dt = dt
        self.mesh = mesh
        self.exfunc = exfunc
        self.initfunc = initfunc
        self.t = self.discretize(t0, 0, nt, dt)
        self.u = self.create_solution_array()
        self.lastkey = tuple(self.u.keys())[-1]
        self.ghost = np.zeros((2, nt))

    def create_solution_array(self):
        u = dict()
        for key, value in self.mesh.x.items():
            u[key] = np.zeros((len(value), self.nt))
        return u

    def initialize_ghost_points(self):
        self.ghost[0, 0] = self.u[0][1, 0]
        self.ghost[1, 0] = self.u[self.lastkey][-2, 0]

    @staticmethod
    def discretize(u0, ni, nu, du):
        ui = u0 + ni * du
        uf = u0 + nu * du
        u = np.linspace(ui, uf, nu - ni)
        return u

    def check_courant(self):
        if self.c >= 1:
            raise ValueError(
                f"Courant limit violation: c={self.c}"
            )

    def calculate_ghost_points(self, time_slice):
        self.ghost[0, time_slice + 1] = self.u[0][
            1, time_slice + 1
        ]
        self.ghost[1, time_slice + 1] = self.u[self.lastkey][
            -2, time_slice + 1
        ]

    def calculate_outer_boundaries(
        self, time_slice, boundary_type="Neumann"
    ):
        if boundary_type == "Neumann":
            self.u[0][0, time_slice + 1] = self.lax_wendroff(
                self.ghost[0, time_slice],
                self.u[0][0, time_slice],
                self.u[0][1, time_slice],
                self.c,
            )
            self.u[self.lastkey][
                -1, time_slice + 1
            ] = self.lax_wendroff(
                self.ghost[1, time_slice],
                self.u[self.lastkey][-1, time_slice],
                self.u[self.lastkey][-2, time_slice],
                self.c,
            )

    def initialize_solution(self):
        for key, value in self.mesh.x.items():
            self.u[key][:, 0] = self.initfunc(self.alpha, value)

    def initialize_exc_boundary(self):
        for key, value in self.mesh.x.items():
            if value[0] != self.mesh.x0:
                self.u[key][0, 0] = self.exfunc(
                    self.alpha, value[0], t=0
                )
            if value[-1] != self.mesh.xn:
                self.u[key][-1, 0] = self.exfunc(
                    self.alpha, value[-1], t=0
                )

    def calculate_exc_boundary(self, key, time_slice):
        if self.mesh.x[key][0] != self.mesh.x0:
            self.u[key][0, time_slice + 1] = self.exfunc(
                self.alpha,
                self.mesh.x[key][0],
                t=self.t[time_slice + 1],
            )
        if self.mesh.x[key][-1] != self.mesh.xn:
            self.u[key][-1, time_slice + 1] = self.exfunc(
                self.alpha,
                self.mesh.x[key][-1],
                t=self.t[time_slice + 1],
            )

    @staticmethod
    def lax_wendroff(um1, u, up1, c):
        return (
            u
            - (c / 2) * (up1 - um1)
            + ((c * c) / 2.0) * (up1 - 2 * u + um1)
        )

    def solve_interior(self, key, time_slice):
        n = len(self.mesh.x[key])
        for i in range(1, n - 1):
            self.u[key][i, time_slice + 1] = self.lax_wendroff(
                self.u[key][i - 1, time_slice],
                self.u[key][i, time_slice],
                self.u[key][i + 1, time_slice],
                self.c,
            )

    def solve(self):
        for j in range(0, self.nt - 1):
            for key in self.u:
                self.solve_interior(key, j)
                self.calculate_exc_boundary(key, j)
            self.calculate_outer_boundaries(j)
            self.calculate_ghost_points(j)

        print("Done")
