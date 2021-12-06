#!/usr/bin/env python3

import os
import numpy as np
from scalarwavepy import wave
from scalarwavepy import utils
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


def plot_convergence(dxs, pis, xis, line1, line2, time, savefig=False):
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


def plot_time_evolution(w, asol, gif=False):
    # maxpi = max(abs(w.state_vector[1, :, 0]))
    # maxxi = max(abs(w.state_vector[2, :, 0]))
    maxpi = asol.A * np.sqrt(2 / np.exp(1)) / np.sqrt(asol.sigma)
    maxxi = asol.A * np.sqrt(2 / np.exp(1)) / np.sqrt(asol.sigma)
    for i in range(0, w.nt + 1):
        asolpi = asol.dt(w.x, w.t[i])
        asolxi = asol.dx(w.x, w.t[i])
        fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax1.plot(w.x, asolpi, "--")
        ax1.plot(w.x, w.state_vector[1, :, i])
        ax1.set_ylim(-1.1 * maxpi, 1.1 * maxpi)
        ax1.set_xlabel(r"$\rm x$")
        ax1.set_title(r"$\pi:=\partial_t u$")

        ax2.plot(w.x, asolxi, "--")
        ax2.plot(w.x, w.state_vector[2, :, i])
        ax2.set_xlabel(r"$\rm x$")
        ax2.set_title(r"$\xi:=\partial_x u$")
        ax2.set_ylim(-1.1 * maxpi, 1.1 * maxpi)

        ax1.set_xlim(w.x[0], w.x[-1])
        ax2.set_xlim(w.x[0], w.x[-1])

        plt.suptitle(rf"$\rm t={w.t[i]:.2f}$")
        # plt.ylim(-8,8)
        plt.tight_layout()
        plt.savefig(f"{resultspath}/{i}.png")
        plt.clf()
        plt.close()

    if gif:
        os.chdir("./results")
        print("Converting to gif...")
        os.system(f"convert -delay 0.5 -loop 0 {{0..{w.nt-1}}}.png wave.gif")
        os.system(f"convert -delay 0.5 -loop 0 {{0..{w.nt-1}}}.png wave.gif")
        os.system("mv wave.gif ../wave_absorbing.gif")


def plot_resolutions(result, pulse, savefig=False):
    w = result[1]
    t_eval = result[0]
    result.pop(0)
    result.pop(1)
    idx_eval = int(np.round(t_eval / w.dt, 1))
    x = utils.discretize(w.x0, w.xn, w.nx)
    analyticpi = pulse.dt(x, w.t[idx_eval])
    analyticxi = pulse.dx(x, w.t[idx_eval])
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # ax[0].plot(x, analyticpi, "--", lw=1, label=f"analytic")
    # ax[1].plot(x, analyticxi, "--", lw=1, label=f"analytic")

    for subresult in result.items():
        # i = subresult['factor']
        subresult = subresult[1]
        errorpi = (subresult["pi"] - subresult["api"]) / subresult["dx"] ** 2
        errorxi = (subresult["xi"] - subresult["axi"]) / subresult["dx"] ** 2

        ax[0].plot(
            subresult["x"],
            errorpi,
            "-o",
            lw=1,
            label=f"dx={subresult['dx']:.4f}",
        )
        ax[1].plot(
            subresult["x"],
            errorxi,
            "-o",
            lw=1,
            label=f"dx={subresult['dx']:.4f}",
        )

    ax[0].set_xlabel(r"$x$")
    ax[1].set_xlabel(r"$x$")

    ax[0].set_title(r"$\pi:=\partial_t u$")
    ax[1].set_title(r"$\xi:=\partial_x u$")

    ax[0].set_ylabel(r"$(\pi-\pi_a)/h^2$")
    ax[1].set_ylabel(r"$(\xi-\xi_a)/h^2$")
    plt.suptitle(rf"$\rm time = {w.t[idx_eval]}$")
    ax[0].legend()
    ax[1].legend()
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
