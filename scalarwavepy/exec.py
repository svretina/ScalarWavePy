#!/usr/bin/env python3

import numpy as np
from scalarwavepy import wave as w
import matplotlib.pyplot as plt
import os

def sine_pulse(x):
    pulse = np.sin(16*np.pi*x)
    pulse[x>0.125] = 0
    return pulse


dx = 1/299
nx = 300
dt = w.set_dt_from_courant(0.4,dx)
nt = 400
alpha = 1
boundaries = [0,0]
pythran = False

wave = w.ScalarWave(alpha, dx, dt, nx, nt)
wave.set_boundaries(boundaries)
wave.initialize_solution(sine_pulse)
wave.solve(pythran=pythran)

sol = wave.u

x = wave.discreatize(0,nx,dx)
# t = wave.discreatize(0,nt,dt)

for i in range(0, nt):
    print(i,x[-1], len(x),len(sol[:,i]))
    plt.plot(x,sol[:,i],'-o')
    plt.ylim(-1,1)
    plt.xlim(0,1)
    plt.title(f"i={i}, t={t[i]:.2f}, x={x[-1]:.2f}")
    plt.savefig(f"./results/{i}.png")
    plt.clf()


os.chdir("./results")
print("Converting to gif...")
if pythran:
    os.system(f"convert -delay 0.5 -loop 0 {{0..{nt-1}}}.png wavep.gif")
    os.system("mv wavep.gif ..")
else:
    os.system(f"convert -delay 0.5 -loop 0 {{0..{nt-1}}}.png wave.gif")
    os.system("mv wave.gif ..")
