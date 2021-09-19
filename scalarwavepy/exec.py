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

def gaussian_pulse_t(c , x, t=0, x0=0.2):
    rv = norm(loc=x0+c*t, scale=0.05)
    pulse = rv.pdf(x)
    return pulse

exc = (150,200)
dx = 1 / 299
nx = 500
dt = w.set_dt_from_courant(0.4, dx)
nt = int((nx*dx)/dt)
alpha = 1


# c = alpha * dt / dx
# t = w.ScalarWave.discretize(0,0,nt,dt)
# x = w.ScalarWave.discretize(0,0,nx,dx)


# for i in range(len(t)):
#     sol = gaussian_pulse_t(c, x, t=t[i])
#     plt.plot(x, sol)
#     plt.plot(x[150], sol[150],'rx')
#     plt.title(f"t={t[i]:.2f}")
#     plt.savefig(f"./results/{i}.png")
#     plt.clf()


# exit()
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

plt.plot(t[:-1], sol[0][-1,:])
plt.plot(t[:-1], sol[1][0,:])

plt.show()
exit()
# for i in range(0,nx):
#     plt.plot(t,sol[i,:])
#     plt.ylim(0,8)
#     # plt.xlim(t[0],t[-1])
#     plt.title(f"x={x[i]:.2f}")
#     plt.savefig(f"./results/{i}.png")
#     plt.clf()
# plt.plot(t[:-1], sol[0][-1,:])
# plt.show()
# exit()
for i in range(0, nt):
    for key, value in x.items():
        plt.plot(value, sol[key][:, i])
        # plt.xlim(value[0], value[-1])

    # for j in exc:
    #     plt.axvline(x=j*dx)
    plt.title(f"t={t[i]:.2f}")
    plt.savefig(f"./results/{i}.png")
    plt.clf()


os.chdir("./results")
print("Converting to gif...")
os.system(f"convert -delay 0.5 -loop 0 {{0..{nt-1}}}.png wave.gif")
os.system("mv wave.gif ..")
