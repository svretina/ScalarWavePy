#!/usr/bin/env python3

import numpy as np
from scalarwavepy import wave as w
from scalarwavepy import mesh as m

import matplotlib.pyplot as plt
import os
from scipy.stats import norm


def gaussian_pulse(x, x0=0.2):
    rv = norm(loc=x0, scale=0.05)
    pulse = rv.pdf(x)
    return pulse


def gaussian_pulse_t(c, x, t=0, x0=0.2):
    rv = norm(loc=x0 + c * t, scale=0.05)
    pulse = rv.pdf(x)
    return pulse


exc = (150, 200)
dx = 1 / 299
nx = 500
dt = w.set_dt_from_courant(0.4, dx)
nt = int((nx * dx) / dt)
alpha = 1

msh = m.Mesh(dx, nx, exc)

wave = w.ScalarWave(msh, alpha, dt, nt, gaussian_pulse_t, gaussian_pulse_t)
# print(wave.u)
wave.initialize_solution()
wave.initialize_exc_boundary()
wave.initialize_ghost_points()
wave.solve()

sol = wave.u
x = msh.x
t = wave.t

# compare at x-slice
# there's a small difference between numerical and theoretical
# solution, like a timeshift

# sol2 = gaussian_pulse_t(alpha, x[0][150], t=t)
# plt.plot(t, sol[0][150,:],'rx')
# plt.plot(t, sol2,"o")
# plt.show()
# exit()

maxsol = sol[0][:, 0].max()
minsol = sol[0][:, 0].min()
lastkey = tuple(x.keys())[-1]
for i in range(0, nt):
    for key, value in x.items():
        plt.plot(value, sol[key][:, i], "blue")
        # plt.xlim(value[0], value[-1])
    plt.ylim(minsol, maxsol)
    plt.axvline(x=x[0][-1], color="k", linestyle="--")
    plt.axvline(x=x[lastkey][0], color="k", linestyle="--")
    # for j in exc:
    #     plt.axvline(x=j*dx)
    plt.title(f"t={t[i]:.2f}")
    plt.savefig(f"./results/{i}.png")
    plt.clf()


os.chdir("./results")
print("Converting to gif...")
os.system(f"convert -delay 0.5 -loop 0 {{0..{nt-1}}}.png wave.gif")
os.system("mv wave.gif ..")
