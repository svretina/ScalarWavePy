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
        self.exprt = self.base_expr.diff(t)
        self.exprx = self.base_expr.diff(x)

    def expr(self, xs, ts):
        f = self.base_expr.subs(
            {A: self.A, s: self.sigma, c: self.cntr}
        )
        ff = lambdify((x, t), f, ["scipy", "numpy"])
        fff = ff(xs, ts)
        return fff

    def expr_dt(self, xs, ts):
        f = self.exprt.subs(
            {A: self.A, s: self.sigma, c: self.cntr}
        )
        ff = lambdify((x, t), f, ["scipy", "numpy"])
        fff = ff(xs, ts)
        return fff

    def expr_dx(self, xs, ts):
        f = self.exprx.subs(
            {A: self.A, s: self.sigma, c: self.cntr}
        )
        ff = lambdify((x, t), f, ["scipy", "numpy"])
        fff = ff(xs, ts)
        return fff
