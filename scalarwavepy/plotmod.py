#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scalarwavepy import utils
from scalarwavepy import grids
from scalarwavepy import global_vars
from scalarwavepy import grid_functions as gf

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
    dxs,
    pis,
    xis,
    line1,
    line2,
    time,
    savefig=False,
):
    mpi, bpi = line1
    mxi, bxi = line2

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(20, 10),
    )

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
    ax[0].legend(
        loc="upper left",
        fontsize=20,
    )
    ax[1].legend(
        loc="upper left",
        fontsize=20,
    )

    plt.tight_layout()

    if savefig:
        savename = f"{figpath}/L2_convergence.png"
        plt.savefig(savename)
        plt.clf()
        plt.close()
    else:
        plt.show()


def plot_convergence_over_time(
    time,
    pi_convergence,
    xi_convergence,
    savefig=False,
):
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(20, 10),
    )

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


def plot_time_evolution(time, result, start=0, step=1, gif=False):
    print("Plottings time evolution")
    if result.ndomains == 1:
        n = result.shape[0]
        asol = result.vector[0].func
        spatial_grid = result.spatial_grid
        maxpi = np.max(asol.dt(spatial_grid.coords, 0.5))
        for i in range(0, n, step):
            print(f"Time[{i}]={time.coords[i]}")
            asolpi = asol.dt(spatial_grid.coords, time.coords[i])
            asolxi = asol.dx(spatial_grid.coords, time.coords[i])

            fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

            # pi
            ax1.plot(spatial_grid.coords, result.vector[i].state_vector[1].values)
            ax1.plot(spatial_grid.coords, asolpi, "--")
            ax1.set_ylim(-1.1 * maxpi, 1.1 * maxpi)
            ax1.set_xlabel(r"$\rm x$")
            ax1.set_title(r"$\pi:=\partial_t u$")
            ax1.set_xlim(spatial_grid.domain[0], spatial_grid.domain[1])

            # xi
            ax2.plot(spatial_grid.coords, result.vector[i].state_vector[2].values)
            ax2.plot(spatial_grid.coords, asolxi, "--")
            ax2.set_xlabel(r"$\rm x$")
            ax2.set_title(r"$\xi:=\partial_x u$")
            ax2.set_ylim(-1.1 * maxpi, 1.1 * maxpi)
            ax2.set_xlim(spatial_grid.domain[0], spatial_grid.domain[1])

            plt.suptitle(rf"$\rm t={time.coords[i]:.2f}$")
            plt.tight_layout()
            plt.savefig(f"{resultspath}/{i}.png")
            plt.clf()
            plt.close()

    elif result.ndomains == 2:
        n = int(
            np.ceil(
                (
                    result.tensor[1, 0].grid.coords[-1]
                    - result.tensor[0, 0].grid.coords[0]
                )
                / result.tensor[0, 0].grid.dx
            )
        )
        xsol = np.linspace(
            result.tensor[0, 0].grid.coords[0], result.tensor[1, 0].grid.coords[-1], n
        )
        asol = result.tensor[0, 0].func
        maxpi = np.max(asol.dt(xsol, 0.5))
        for i in range(start, time.npoints, step):
            print(f"Time[{i}]={time.coords[i]}")
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
            # analytical solution
            asolpi = asol.dt(xsol, time.coords[i])
            asolxi = asol.dx(xsol, time.coords[i])
            ax1.plot(xsol, asolpi, "--", label="analytical solution")
            ax2.plot(xsol, asolxi, "--")

            for j in range(result.shape[0]):
                x = result.tensor[j, i].grid.coords
                pi = result.tensor[j, i].pi.values
                xi = result.tensor[j, i].xi.values

                ax1.plot(x, pi)
                ax2.plot(x, xi)

            ax1.axvline(result.tensor[0, 0].grid.coords[-1])
            ax2.axvline(result.tensor[0, 0].grid.coords[-1])

            ax1.set_xlim(
                result.tensor[0, 0].grid.coords[0], result.tensor[1, 0].grid.coords[-1]
            )
            ax2.set_xlim(
                result.tensor[0, 0].grid.coords[0], result.tensor[1, 0].grid.coords[-1]
            )

            ax1.set_ylim(-1.1 * maxpi, 1.1 * maxpi)
            ax2.set_ylim(-1.1 * maxpi, 1.1 * maxpi)

            ax1.set_xlabel(r"$\rm x$")
            ax1.set_title(r"$\pi:=\partial_t u$")
            ax2.set_xlabel(r"$\rm x$")
            ax2.set_title(r"$\xi:=\partial_x u$")
            plt.suptitle(rf"$\rm t={time.coords[i]:.2f}$")
            fig.legend()
            plt.tight_layout()
            plt.savefig(f"{resultspath}/{i}.png")
            plt.clf()
            plt.close()
        if gif:
            os.chdir("./results")
            print("Converting to gif...")
            os.system(f"convert -delay 0.5 -loop 0 {{0..{n-1}}}.png wave.gif")
            os.system("mv wave.gif ../wave_absorbing.gif")


def plot_at_resolutions(
    initial_ncells=10,
    eval_time=1,
    depth=5,
    savefig=False,
):
    results = dict()
    args = dict(
        final_time=2,
        domain=[[0, 1], [1, 2]],
        alpha=3,
        ncells=[initial_ncells, initial_ncells],
    )
    # res = utils.run(**args)
    factor = 2 ** np.linspace(1, depth, depth, dtype=int)
    factor = np.concatenate(([1], factor))
    print(factor)
    raise SystemError
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for i in range(len(factor)):
        ncellsi = factor[i] * initial_ncells

        index = int(np.round(eval_time / dt, 1))
        utils.run(final_time, ncells, domain, args, kwargs)
        res = wave.evolve(
            sv,
            spatial_grid,
            time_grid,
        )
        pi = res[index].state_vector[1]
        xi = res[index].state_vector[2]
        ax[0].plot(
            pi.grid.coords,
            pi.values,
            "-o",
        )
        ax[1].plot(
            xi.grid.coords,
            xi.values,
            "-o",
        )

    analytic_pi = f.dt(
        spatial_grid,
        time_grid.coords[index],
    )
    analytic_xi = f.dx(
        spatial_grid,
        time_grid.coords[index],
    )

    ax[0].plot(
        pi.grid.coords,
        analytic_pi,
        "r--",
        lw=2,
    )
    ax[1].plot(
        xi.grid.coords,
        analytic_xi,
        "r--",
        lw=2,
    )
    ax[0].set_title(r"$\pi:=\partial_t u$")
    ax[1].set_title(r"$\xi:=\partial_x u$")
    ax[0].set_xlim(
        spatial_domain.domain[0],
        spatial_domain.domain[-1],
    )
    ax[1].set_xlim(
        spatial_domain.domain[0],
        spatial_domain.domain[-1],
    )
    if savefig:
        savename = f"{figpath}/pixi_resolutions.png"
        plt.savefig(savename)
        plt.clf()
        plt.close()
    else:
        plt.show()


def plot_energy(energy, append=None, savefig=False):
    print("Plotting energy over time.")

    if append is None or append[0] is None:
        fig, ax = plt.subplots(nrows=2, sharex=True)
    else:
        try:
            fig, ax = append
        except:
            pass

    n = energy.grid.ncells
    energydt = energy.differentiated
    ax[0].plot(energydt.grid.coords, energydt.values, "-x")
    col = ax[0].get_lines()[-1].get_color()
    ax[1].plot(energy.grid.coords, energy.values, "-x", color=col, label=f"N={n}")

    # plot the points that correspond to increasing energy
    indexes = np.where(energydt.values > 0)
    ax[0].plot(energydt.grid.coords[indexes], energydt.values[indexes], "rx")
    ax[1].plot(energy.grid.coords[indexes], energy.values[indexes], "ro")

    ax[1].set_xlabel(r"$\rm time$")
    ax[1].set_ylabel(r"$\rm Energy$")
    ax[0].set_ylabel(r"$\rm \partial_t Energy$")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_xlim(energy.grid.coords[0], energy.grid.coords[-1])
    plt.tight_layout()

    if append:
        return fig, ax

    if savefig:
        plt.savefig(f"{figpath}/energy.png")
    else:
        plt.show()


def plot_energy_convergence(mode, rrange, savefig=False, *args, **kwargs):
    start, stop, step = rrange
    fig, ax = None, None
    if mode == "ncells":
        a = list()
        for i in np.arange(start, stop, step):
            if np.asarray(kwargs["domain"]).ndim == 1:
                ncells = i
            else:
                ncells = [i, i]

            res = utils.run(
                kwargs["final_time"], ncells, kwargs["domain"], alpha=kwargs["alpha"]
            )
            fig, ax = plot_energy(res.energy, append=[fig, ax], savefig=False)
            h, l = ax[1].get_legend_handles_labels()
            a.append(f"ncells={i:.1f}")
            ax[1].legend(h, a)
    elif mode == "alpha":
        a = list()
        for i in np.arange(start, stop, step):
            res = utils.run(
                kwargs["final_time"], kwargs["ncells"], kwargs["domain"], alpha=i
            )
            fig, ax = plot_energy(res.energy, append=[fig, ax], savefig=False)
            h, l = ax[1].get_legend_handles_labels()
            a.append(f"a={i:.1f}")
            ax[1].legend(h, a, bbox_to_anchor=(1, 2.5))
            # plt.legend()

    if savefig:
        plt.savefig(f"{figpath}/energy_convergence_{mode}.png")
    else:
        plt.show()
