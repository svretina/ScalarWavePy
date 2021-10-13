#!/usr/bin/env python3

import numpy as np

def discretize(u0, ni, nu, du):
    ui = u0 + ni * du
    uf = u0 + nu * du
    u = np.linspace(ui, uf, nu - ni)
    return u

def spatial_derivative(u, dx):
    u = np.gradient(u, dx, edge_order=1)
    return u


def rk2(func, s, h):
    k0 = func(s)
    s1 = s + (3/4)*h*k0
    k1 = func(s1)
    s2 = s + h*((1/3)*k0+(2/3)*k1)
    return s2

def rk4(func, s, h):
    k0 = func(s)
    s1 = s + h/2 * k0
    k1 = func(s1)
    s2 = s + h/2 * k1
    k2 = func(s2)
    s3 = s + h * k2
    k3 = func(s3)
    s4 = s + h/6 * (k0 + 2*k1 + 2*k2 + k3)
    return s4
