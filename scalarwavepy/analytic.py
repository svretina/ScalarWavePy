#!/usr/bin/env python3

import numpy as np
from sympy.abc import x, t, A, s, c
import sympy.functions as sf
from sympy import lambdify
from functools import lru_cache


class Gaussian:
    def __init__(self, center, amplitude, sigma):
        self.A = amplitude
        self.sigma = sigma
        self.cntr = center
        self.base_expr = A * sf.exp(-((x - (c + t)) ** 2) / s)
        self.fx = self._expr()
        self.dt = self._expr_dt()
        self.dx = self._expr_dx()

    def __call__(self, xs, ts):
        return self.fx(xs, ts)

    def _expr(self):
<<<<<<< HEAD
        f = self.base_expr.subs(
            {A: self.A, s: self.sigma, c: self.cntr}
        )
=======
        f = self.base_expr.subs({A: self.A, s: self.sigma, c: self.cntr})
>>>>>>> 6bf743d (major changes)
        ff = lambdify((x, t), f, ["scipy", "numpy"])
        return ff

    def _expr_dt(self):
        f = self.base_expr.diff(t).subs(
            {A: self.A, s: self.sigma, c: self.cntr}
        )
        ff = lambdify((x, t), f, ["scipy", "numpy"])
        return ff

    def _expr_dx(self):
        f = self.base_expr.diff(x).subs(
            {A: self.A, s: self.sigma, c: self.cntr}
        )
        ff = lambdify((x, t), f, ["scipy", "numpy"])
        return ff
