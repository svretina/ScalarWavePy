#!/usr/bin/env python3

import numpy as np
from scalarwavepy import wave as w
import matplotlib.pyplot as plt
import os
from scipy.stats import norm


def sine_pulse(x):
    pulse = np.sin(16 * np.pi * x)
    pulse[x > 0.125] = 0
    return pulse


def gaussian_pulse(x, x0=0.2):
    rv = norm(loc=x0, scale=0.05)
    pulse = rv.pdf(x)
    return pulse


dx = 1 / 299
nx = 300
dt = w.set_dt_from_courant(0.4, dx)
nt = 600
c = 1
boundaries = [0, 0]
pythran = False

wave = w.ScalarWave(c, dx, dt, nx, nt)
wave.initialize_solution(gaussian_pulse)
wave.initialize_ghost_points()
wave.solve()

sol = wave.u

x = wave.x
t = wave.t

# for i in range(0,nx):
#     plt.plot(t,sol[i,:])
#     plt.ylim(0,8)
#     # plt.xlim(t[0],t[-1])
#     plt.title(f"x={x[i]:.2f}")
#     plt.savefig(f"./results/{i}.png")
#     plt.clf()

for i in range(0, nt):
    # print(i,x[-1], len(x),len(sol[:,i]))
    plt.plot(x, sol[:, i])
    # plt.ylim(-0.1,)
    plt.xlim(x[0], x[-1])
    plt.title(f"t={t[i]:.2f}")
    plt.savefig(f"./results/{i}.png")
    plt.clf()


os.chdir("./results")
print("Converting to gif...")
os.system(f"convert -delay 0.5 -loop 0 {{0..{nt-1}}}.png wave.gif")
os.system("mv wave.gif ..")
