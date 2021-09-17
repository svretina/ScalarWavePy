#!/usr/bin/env python3

import numpy as np
import scalarwavepy.pythran_lax as lax
import matplotlib.pyplot as plt

def set_dt_from_courant(c,dx):
    return c*dx


class ScalarWave:
    def __init__(self, alpha, dx, dt, nx, nt, t0 = 0,x0 = 0):
        """ Constructor of ScalarWave class.
        """
        self.c = alpha*dt/dx
        self.check_courant()
        self.nx = nx
        self.nt = nt
        self.x = self.discretize(x0, nx, dx)
        self.t = self.discretize(t0, nt, dt)
        self.u = np.zeros((nx,nt)) # u(x,t)
        self.ghost = np.zeros((2,nt))


    def initialize_ghost_points(self):
        self.ghost[0,0] = self.u[1,0]
        self.ghost[1,0] = self.u[-2,0]

    def discretize(self, u0, nu, du):
        uf = u0 + nu * du
        u = np.linspace(u0, uf, nu)
        return u

    def check_courant(self):
        if self.c >= 1:
            raise ValueError(f"Courant limit violation: c={self.c}")

    def calculate_ghost_points(self, time_slice):
        self.ghost[0, time_slice + 1] = self.u[1, time_slice + 1]
        self.ghost[1, time_slice + 1] = self.u[-2, time_slice + 1]

    def calculate_boundaries(self, time_slice, boundary_type = "Neumann"):
        if boundary_type == "Neumann":
            self.u[0, time_slice + 1] = self.lax_wendroff(self.ghost[0, time_slice], self.u[0, time_slice], self.u[1, time_slice], self.c)
            self.u[-1, time_slice + 1] = self.lax_wendroff(self.ghost[1, time_slice], self.u[-1, time_slice], self.u[-2, time_slice], self.c)

    def initialize_solution(self, func):
        self.u[:,0] = func(self.x)

    @staticmethod
    def lax_wendroff(um1, u, up1, c):
        return u - ( c / 2 ) * ( up1 - um1 ) + (( c*c ) / 2. ) * ( up1 - 2 * u + um1 )

    def solve_interior(self, time_slice):
        for i in range(1, self.nx-1):
            self.u[i, time_slice + 1] = self.lax_wendroff( self.u[i-1, time_slice],
                                                           self.u[i, time_slice],
                                                           self.u[i+1, time_slice],
                                                           self.c)

    def solve(self):
        for j in range(0, self.nt-1):
            self.solve_interior(j)
            self.calculate_boundaries(j)
            self.calculate_ghost_points(j)

        print("Done")
