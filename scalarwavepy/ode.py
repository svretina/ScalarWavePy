#!/usr/bin/env python3

import numpy as np
from scipy.integrate import RK45
from scipy.stats import norm
import matplotlib.pyplot as plt
from scalarwavepy import mesh as m
from scalarwavepy import wave as w
import os

def discretize(u0, ni, nu, du):
    ui = u0 + ni * du
    uf = u0 + nu * du
    u = np.linspace(ui, uf, nu - ni)
    return u

def gaussian_pulse_t(x, t = 0, x0 = 0.2, c = 1):
    rv = norm(loc=x0 + c * t, scale=0.05)
    pulse = rv.pdf(x)
    return pulse

def spatial_derivative(u, dx):
    return np.gradient(u, dx, edge_order=2)

def rk2(func, s, h, dx):
    k0 = func(s, dx)
    s1 = s + h/2 * k0
    k1 = func(s1, dx)
    s2 = s + h * k1
    return s2

def rk2_ralston(func, s, h, dx):
    k0 = func(s, dx)
    s1 = s + (3/4)*h*k0
    k1 = func(s1, dx)
    s2 = s + h*((1/3)*k0+(2/3)*k1)
    return s2

def rk4(func, s, h, dx):
    k0 = func(s, dx)
    s1 = s + h/2 * k0
    k1 = func(s1, dx)
    s2 = s + h/2 * k1
    k2 = func(s2, dx)
    s3 = s + h * k2
    k3 = func(s3, dx)
    s3 = s + h/6 * (k0 + 2*k1 + 2*k2 + k3)
    return s3

def RHS(s, dx):
    u, pi, xi = s
    # enforce BCs
    u[0] = 0
    u[-1] = 0
    pi[0] = 0
    pi[-1] = 0

    dtu = pi
    dtpi = spatial_derivative(xi, dx)
    dtxi = spatial_derivative(pi, dx)
    return np.array([dtu, dtpi, dtxi])

dx = 1/100
dt = 0.4 * dx**(4/3)
nx = 100
nt = 2000

u = np.zeros((nx, nt))
pi = np.zeros((nx, nt))
xi = np.zeros((nx, nt))

x = discretize(0,0,nx,dx)
t = discretize(0,0,nt,dt)

u[:,0] = gaussian_pulse_t(x)
u[0,0] = 0
u[-1,0] = 0
pi[:,0] = 0
xi[:,0] = spatial_derivative(u[:,0], dx)

s = np.array([u[:,0], pi[:,0], xi[:,0]])

for i in range(1, nt):
    s = rk4(RHS, s, dt, dx)
    u[:,i], _, _ = s

for i in range(0, nt):
    plt.plot(x, u[:, i], "blue")
    plt.title(f"t={t[i]:.2f}")
    plt.ylim(-8,8)
    plt.tight_layout()
    plt.savefig(f"./results/{i}.png")
    plt.clf()


os.chdir("./results")
print("Converting to gif...")
os.system(f"convert -delay 0.5 -loop 0 {{0..{nt-1}}}.png wave.gif")
os.system("mv wave.gif ../wave_rk4.gif")
