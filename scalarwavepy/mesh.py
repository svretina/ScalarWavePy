#!/usr/bin/env python3

import numpy as np


class Mesh:
    def __init__(self, dx, nx, exc, x0=0):
        self.dx = dx
        self.nx = nx
        self.exc = exc
        self.x0 = x0
        self.exlen = len(exc)
        self.ndom = self.calculate_ndom(self.exlen)
        self.xn = x0 + nx * dx
        self.boundaries = (0,) + exc + (nx,)
        self.x = self.create_mesh()

    @staticmethod
    def calculate_ndom(nex):
        return int(nex / 2) + 1

    def create_mesh(self):
        x = dict()
        for i in range(self.ndom):
            x[i] = self.discretize(
                self.boundaries[2 * i],
                self.boundaries[2 * i + 1],
            )
        return x

    def discretize(self, ni, nf):
        xi = self.x0 + ni * self.dx
        xf = self.x0 + nf * self.dx
        x = np.linspace(xi, xf, nf - ni + 1)
        return x

    @staticmethod
    def check_even(num):
        if num % 2 == 0:
            pass  # Even
        else:
            raise ValueError("Number of boundaries must be even")
