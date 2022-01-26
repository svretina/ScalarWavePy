#!/usr/bin/env python3

import os
import numpy as np
from scalarwavepy import wave
from scalarwavepy import grid
from scalarwavepy import utils
from scalarwavepy import globvars
import matplotlib.pyplot as plt

params = {
    "lines.linewidth": 2,
    "axes.labelsize": 20,
    "axes.linewidth": 1,
    "axes.titlesize": 30,
    "figure.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "xtick.major.size": 5,
    "text.usetex": True,
}
plt.rcParams.update(params)
dirname = os.path.dirname(__file__)
figpath = f"{dirname}/figures"
resultspath = f"{dirname}/results"


def plot_convergence(
    dxs, pis, xis, line1, line2, time, savefig=False
):
    mpi, bpi = line1
    mxi, bxi = line2

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].loglog(
        dxs,
        pis,
        "r-x",
        lw=2,
        label=r"$\Vert \pi_h - \pi \Vert_2$",
    )
    ax[1].loglog(
        dxs,
        xis,
        "b-o",
        lw=2,
        label=r"$\Vert \xi_h - \xi \Vert_2$",
    )

    ax[0].set_xlabel(r"$dx$")
    ax[1].set_xlabel(r"$dx$")

    ax[0].set_title(r"$\pi:=\partial_t u$")
    ax[1].set_title(r"$\xi:=\partial_x u$")

    ax[0].set_ylabel(r"$\Vert \pi_h - \pi \Vert_2$")
    ax[1].set_ylabel(r"$\Vert \xi_h - \xi \Vert_2$")

    ax[0].loglog(
        dxs,
        [i ** mpi * np.exp(0.9 * bpi) for i in dxs],
        "--",
        lw=2,
        label=r"$h^{%.1f}$" % mpi,
    )
    ax[1].loglog(
        dxs,
        [i ** mxi * np.exp(0.9 * bxi) for i in dxs],
        "--",
        lw=2,
        label=r"$h^{%.1f}$" % mxi,
    )

    ttitle = rf"$t = {time}$"

    plt.suptitle(ttitle)
    ax[0].legend(loc="upper left", fontsize=20)
    ax[1].legend(loc="upper left", fontsize=20)

    plt.tight_layout()

    if savefig:
        savename = f"{figpath}/L2_convergence.png"
        plt.savefig(savename)
        plt.clf()
        plt.close()
    else:
        plt.show()


def plot_convergence_over_time(
    time, pi_convergence, xi_convergence, savefig=False
):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].plot(
        time,
        pi_convergence,
        "r-x",
        lw=2,
        label=r"$\Vert \pi_h - \pi \Vert_2(t)$",
    )
    ax[1].plot(
        time,
        xi_convergence,
        "b-o",
        lw=2,
        label=r"$\Vert \xi_h - \xi \Vert_2(t)$",
    )

    ax[0].set_xlabel(r"$t$")
    ax[1].set_xlabel(r"$t$")

    ax[0].set_title(r"$\pi:=\partial_t u$")
    ax[1].set_title(r"$\xi:=\partial_x u$")

    ax[0].set_ylabel(r"$\Vert \pi_h - \pi \Vert_2(t)$")
    ax[1].set_ylabel(r"$\Vert \xi_h - \xi \Vert_2(t)$")

    ax[0].legend(fontsize=20)
    ax[1].legend(fontsize=20)

    plt.tight_layout()

    if savefig:
        savename = f"{figpath}/L2_convergence_over_time.png"
        plt.savefig(savename)
        plt.clf()
        plt.close()
    else:
        plt.show()


def plot_time_evolution(time, result, step=1, gif=False):
    n = result.shape[0]
    asol = result[0].func
    spatial_grid = result[0].state_vector[0].grid
    maxpi = np.max(asol.dt(spatial_grid, 0.5))
    for i in range(0, n, step):
        print(i)
        asolpi = asol.dt(spatial_grid.coords, time.coords[i])
        asolxi = asol.dx(spatial_grid.coords, time.coords[i])

        fig, [ax1, ax2] = plt.subplots(
            nrows=1, ncols=2, figsize=(20, 10)
        )

        ax1.plot(
            spatial_grid.coords, result[i].state_vector[1].values
        )
        ax1.plot(spatial_grid.coords, asolpi, "--")

        ax1.set_ylim(-1.1 * maxpi, 1.1 * maxpi)
        ax1.set_xlabel(r"$\rm x$")
        ax1.set_title(r"$\pi:=\partial_t u$")

        ax2.plot(
            spatial_grid.coords, result[i].state_vector[2].values
        )
        ax2.plot(spatial_grid.coords, asolxi, "--")
        ax2.set_xlabel(r"$\rm x$")
        ax2.set_title(r"$\xi:=\partial_x u$")
        ax2.set_ylim(-1.1 * maxpi, 1.1 * maxpi)

        ax1.set_xlim(spatial_grid.domain[0], spatial_grid.domain[1])
        ax2.set_xlim(spatial_grid.domain[0], spatial_grid.domain[1])

        plt.suptitle(rf"$\rm t={time.coords[i]:.2f}$")

        plt.tight_layout()
        plt.savefig(f"{resultspath}/{i}.png")
        plt.clf()
        plt.close()
    if gif:
        os.chdir("./results")
        print("Converting to gif...")
        os.system(
            f"convert -delay 0.5 -loop 0 {{0..{n-1}}}.png wave.gif"
        )
        os.system("mv wave.gif ../wave_absorbing.gif")


def plot_at_resolutions(
        initial_spacing,
        eval_time=1,
        depth=5,
        savefig=False
):
    results = {}
    spatial_domain = grid.SingleDomain([0, 1])
    time_domain = grid.SingleDomain([0, eval_time])
    ncells = utils.ncells_from_dx(0, 1, initial_spacing)
    spatial_grid = grid.SingleGrid(spatial_domain, ncells)
    time_grid = grid.TimeGrid_from_cfl(spatial_grid,
                                       time_domain)
    index = int(np.round(eval_time / time_grid.spacing))
    f = globvars.PULSE
    sv = grid.StateVector(func=f)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for i in range(1, depth):
        dx = initial_spacing / i
        ncells = utils.ncells_from_dx(0, 1, dx)
        spatial_grid = grid.SingleGrid(spatial_domain, ncells)
        sv.initialize(spatial_grid)
        time_grid = grid.TimeGrid_from_cfl(spatial_grid,
                                           time_domain)
        dt = time_grid.spacing
        index = int(np.round(eval_time / dt, 1))
        res = wave.evolve(sv, spatial_grid, time_grid)
        pi = res[index].state_vector[1]
        xi = res[index].state_vector[2]
        ax[0].plot(pi.grid.coords, pi.values,'-o')
        ax[1].plot(xi.grid.coords, xi.values,'-o')

    analytic_pi = f.dt(spatial_grid, time_grid.coords[index])
    analytic_xi = f.dx(spatial_grid, time_grid.coords[index])

    ax[0].plot(pi.grid.coords, analytic_pi,'r--',lw=2)
    ax[1].plot(xi.grid.coords, analytic_xi,'r--',lw=2)
    ax[0].set_title(r"$\pi:=\partial_t u$")
    ax[1].set_title(r"$\xi:=\partial_x u$")
    ax[0].set_xlim(spatial_domain.domain[0],
                   spatial_domain.domain[-1])
    ax[1].set_xlim(spatial_domain.domain[0],
                   spatial_domain.domain[-1])
    if savefig:
        savename = f"{figpath}/pixi_resolutions.png"
        plt.savefig(savename)
        plt.clf()
        plt.close()
    else:
        plt.show()


def plot_energy_density(t, energy_density, savefig=False):
    plt.plot(t, energy_density)
    plt.xlabel(r"$\rm time$")
    plt.title(r"$\rm Energy$ $\rm Density$")
    plt.tight_layout()

    if savefig:
        plt.savefig(f"{figpath}/energy_density.png")
    else:
        plt.show()
