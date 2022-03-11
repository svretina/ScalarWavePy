#!/usr/bin/env python3

import scipy
import numpy as np
from sympy import lambdify
from sympy import functions as sf
from sympy.abc import x, t, A, s, c


class Gaussian:
    def __init__(self, center, amplitude, sigma):
        self.A = amplitude
        self.s = sigma
        self.cntr = center
        self.base_expr = A * sf.exp(-((x - (c + t)) ** 2) / s)
        self.fx = self._expr()
        self.dt = self._expr_dt()
        self.dx = self._expr_dx()

    def __call__(self, xs, ts):
        return self.fx(xs, ts)

    def _expr(self):
        f = self.base_expr.subs({A: self.A, s: self.s, c: self.cntr})
        ff = lambdify((x, t), f, ["scipy", "numpy"])
        return ff

    def _expr_dt(self):
        f = self.base_expr.diff(t).subs({A: self.A, s: self.s, c: self.cntr})
        ff = lambdify((x, t), f, ["scipy", "numpy"])
        return ff

    def _expr_dx(self):
        f = self.base_expr.diff(x).subs({A: self.A, s: self.s, c: self.cntr})
        ff = lambdify((x, t), f, ["scipy", "numpy"])
        return ff
