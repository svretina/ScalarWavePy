#!/usr/bin/env python3

# from scalarwavepy import utils
from scalarwavepy import grids
from scalarwavepy import utils
from scalarwavepy import plotmod as pltm


ft = 40
single_domain = False
if single_domain:
    domain = [0, 1]
    ncells = 80
else:
    ncells = [80, 80]
    domain = [[0, 1], [1, 2]]
alpha = 0.005 * 100
args = dict(final_time=ft, domain=domain, alpha=alpha, ncells=ncells)

# pltm.plot_at_resolutions(20)  # <---- work out the index according to the ncells
# pltm.plot_energy_convergence("ncells", [20, 140, 20], **args)
# pltm.plot_energy_convergence("alpha", [0.05, 1, 0.05], **args)


res = utils.run(**args)
# print(res.energy.max())
# pltm.plot_energy(res.energy)

args2 = dict(final_time=ft, domain=[0, 1], alpha=alpha, ncells=80)
res2 = utils.run(**args2)


args3 = dict(final_time=ft, domain=[1, 2], alpha=alpha, ncells=80)
res3 = utils.run(**args3)
energy2 = res2.energy + res3.energy

import matplotlib.pyplot as plt

plt.plot(res.energy.grid.coords, res.energy.values, label="connected")
plt.plot(energy2.grid.coords, energy2.values, label="[0, 1] + [1, 2] (disc)")
plt.plot(res2.energy.grid.coords, res2.energy.values, label="[0, 1]")
plt.plot(res3.energy.grid.coords, res3.energy.values, label="[1, 2]")
plt.legend()
plt.xlabel("time")
plt.ylabel("Energy")
plt.show()
# pltm.plot_time_evolution(res.time, res, start=0, gif=False)
