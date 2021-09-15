#!/usr/bin/env python3

import numpy as np
import scalarwavepy.pythran_lax as lax
import matplotlib.pyplot as plt

def set_dt_from_courant(c,dx):
    return c*dx


class ScalarWave:
    def __init__(self, alpha, dx, dt, nx, nt, t0=0,x0=0):
        """ Constructor of ScalarWave class.
        """
        self.c = alpha*dt/dx
        self.check_courant()
        self.nx = nx
        self.nt = nt
        self.x = self.discreatize(x0, nx, dx)
        self.t = self.discreatize(t0, nt, dt)
        self.u = np.zeros((nx,nt))

    def discretize(self,u0, nu, du):
        uf = u0 + nu * du
        u = np.arange(u0, uf, du)
        return u

    def check_courant(self):
        if self.c >= 1:
            raise ValueError(f"Courant limit violation: c={self.c}")

    def set_boundaries(self,boundaries):
        """ Sets the boundaries. If self.boundaries contains numbers it sets them at u(0,t) and u(L,t). If functions are provided instead of scalars, then the functions are applied. Modifies the solution u(x,t) in place!
        """
        if isinstance(boundaries[0], int) or isinstance(boundaries[0], float):
            self.u[0,:] = boundaries[0]
        else:
            self.u[0,:] = boundaries[0](self.t)

        if isinstance(boundaries[1], int) or isinstance(boundaries[1], float):
            self.u[self.nx-1,:] = boundaries[1]
        else:
            self.u[nx-1,:] = boundaries[1](self.t)


    def initialize_solution(self,func):
        self.u[:,0] = func(self.x)

    def solve(self,method="Lax-Wendroff",pythran=True):
        if method=="Lax-Wendroff":
            if pythran:
                self.u = lax.solve(self.u,self.nx, self.nt, self.c, 8)
            else:
                for j in range(0,self.nt-1):
                    # print(f"{j},{self.t[j]}/{self.t[self.nt-1]}")
                    for i in range(1,self.nx-1):
                        self.u[i,j+1] = self.u[i,j] - ( self.c / 2) * ( self.u[i+1,j] - self.u[i-1,j] ) + ( (self.c*self.c) / 2. ) * ( self.u[i+1,j] - 2 * self.u[i,j] + self.u[i-1,j] )
        print("Done")
