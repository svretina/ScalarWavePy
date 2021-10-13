#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scalarwavepy import utils
import os

def RHS(s, dx):
    u, pi, xi = s

    # pi[0] = xi[0]
    # pi[-1] = -xi[-1]
    # u_t - u_x = 0 # pi - xi = 0
    pix = utils.spatial_derivative(pi, dx)
    xix = utils.spatial_derivative(xi, dx)

    # interior
    dtu = pi[:]
    dtpi = xix[:]
    dtxi = pix[:]


    dtpi[0] = 0.5*(xix[0]+pix[0]) ## left boundary pi_t = 0.5 * ( xi_x + pi_x )
    dtpi[-1] = 0.5*(xix[-1]-pix[-1]) ## right boundary  pi_t = 0.5 * ( xi_x - pi_x )

    dtxi[0] = 0.5*(xix[0]+pix[0]) ## left boundary   xi_t = 0.5 * ( xi_x + pi_x )
    dtxi[-1] = 0.5*(-xix[-1]+pix[-1]) ## right boundary xi_t = 0.5 * ( pi_x - xi_x )

    # dtpi = np.hstack((
    #     0, ## left boundary pi_t = 0.5 * ( xi_x + pi_x )
    #     dtpi,
    #     0) ## right boundary  pi_t = 0.5 * ( xi_x - pi_x )
    # )

    # dtxi = np.hstack((
    #     0, ## left boundary   xi_t = 0.5 * ( xi_x + pi_x )
    #     dtxi,
    #     0) ## right boundary xi_t = 0.5 * ( pi_x - xi_x )
    # )


    # dtpi = xix
    # dtxi = pix
    # dtpi[0] = pix[0]
    # dtpi[-1] = -pix[-1]
    # dtxi[0] = xix[0]
    # dtxi[-1] = -xix[-1]

    return np.array([dtu, dtpi, dtxi])

# pi_t - xi_t = 0 # here put the analytic solution for "generation"
# pi_t + xi_t = xi_x + pi_x

def rhs(dx):
    return lambda s: RHS(s, dx)


def analytic_boundaries(x, center=0.5, amplitude=1, sigma=1):
    dtpi = 4*(((x-t)**2)/sigma)*analytic_sol(x,
                                      center,
                                      amplitude,
                                      sigma)
    dtxi = - dtpi
    return dtpi, dtxi

# pi_t - xi_t = 0 # here put the analytic solution for "generation"
# pi_t + xi_t = xi_x + pi_x

def analytic_sol(x, center=0.5, amplitude=1, sigma=1):
    return amplitude * np.exp(-((x-center)**2 )/sigma)

def analytic_derivative_solx(x, center=0.5, amplitude=1, sigma=1):
    return - (2 / sigma) * (x - center) * analytic_sol(x,
                                                       center,
                                                       amplitude,
                                                       sigma)

def analytic_derivative_solt(x, center=0.5, amplitude=1, sigma=1):
    return (2 / sigma) * (x - center) * analytic_sol(x,
                                                     center,
                                                     amplitude,
                                                     sigma)

def prepare_gaussian_pulse(x, center=0.5, amplitude=1, sigma=1):
    dx = x[2]-x[1]
    u = analytic_sol(x, center, amplitude, sigma)
    pi = - analytic_derivative_solx(x, center, amplitude, sigma)
    xi = utils.spatial_derivative(u, dx)
    return np.array([u, pi, xi])

dx = 1/100
dt = 0.4 * dx
nx = 100
nt = 500

u = np.zeros((nx, nt))
pi = np.zeros((nx, nt))
xi = np.zeros((nx, nt))

x = utils.discretize(0, 0, nx, dx)
print(x)
exit()
t = utils.discretize(0, 0, nt, dt)

ampl = 8
sigma = 1/100
cntr = 0.8

sol = prepare_gaussian_pulse(x=x,
                             center=cntr,
                             amplitude=ampl,
                             sigma=sigma)

u[:,0], pi[:,0], xi[:,0] = sol

for i in range(1, nt):
    rhs_func = rhs(dx)
    sol = utils.rk4(rhs_func, sol, dt)
    u[:,i], pi[:,i], xi[:,i] = sol



# calculate_diagnostics()

# plt.plot(t, energy_density)
# plt.savefig("energy_density.png")
# plt.clf()

# plt.semilogy(t, mean_magnitude)
# plt.title("Mean Magnitude: Second Order BC")
# plt.savefig("mean_magnitude2.png")
# plt.clf()

plot = True
if plot:
    for i in range(0, nt):
        asol = analytic_sol(x=x,
                            center=cntr+t[i],
                            amplitude=ampl,
                            sigma=sigma)

        # plt.plot(x, asol, '--')
        plt.plot(x, pi[:,i], "blue")
        plt.title(f"t={t[i]:.2f}")
        plt.xlim(x[0],x[-1])
        # plt.ylim(-8,8)
        plt.tight_layout()
        plt.savefig(f"./results/{i}.png")
        plt.clf()

gif = False

if gif:
    os.chdir("./results")
    print("Converting to gif...")
    os.system(f"convert -delay 0.5 -loop 0 {{0..{nt-1}}}.png wave.gif")
    os.system("mv wave.gif ../wave_absorbing.gif")
