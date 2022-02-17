#!/usr/bin/env python3

import numpy as np
from scalarwavepy import utils
from scalarwavepy import domains
from scalarwavepy import global_vars

class SingleGrid:
    def __init__(self, single_domain, ncells):
        """Constructor of :py:class:`~.UniformGrid`.
        A cell is defined as the space between two grid points
        :param domain: Physical Domain of the grid
        :type domain: Domain class
        :param ncells: Number of cells for the grid.
        :type ncells: int
        """
        if not isinstance(single_domain, domains.SingleDomain):
            raise TypeError(
                f"The domain should be of class SingleDomain.\n\
                You provided a {type(single_domain)} type."
            )
        assert isinstance(ncells, (int, np.integer))
        self.domain = single_domain.domain
        self.ncells = ncells
        self.npoints = ncells + 1
        self.coords = utils.discretize(
            self.domain[0],
            self.domain[1],
            ncells,
        )
        self.dx = utils.spacing(
            self.domain[0],
            self.domain[1],
            ncells,
        )
        self.spacing = self.dx

    def __str__(self):
        return str(self.coords)

    def __len__(self):
        return len(self.coords)

    @property
    def shape(self):
        return self.coords.shape


class MultipleGrid:
    def __init__(self, domain, ncells):
        if not isinstance(ncells, np.ndarray):
            ncells = np.asarray(ncells, dtype=np.int)

        assert domain.domain_list.shape[0] == ncells.shape[0]
        assert isinstance(
            domain,
            (domains.SingleDomain, domains.MultipleDomain),
        )

        self.domain = domain.domain
        self.ndomains = self.domain.shape[0]
        self.ncells = ncells
        self.npoints = ncells + 1
        self.ugrids, self.dx = self.get_grids()

        assert self.dx.shape == self.ncells.shape

    def get_grids(self):
        num_subdomains = self.domain.shape[0]
        dxs = np.empty((num_subdomains))
        grid = np.empty((num_subdomains), dtype="object")

        for i in range(num_subdomains):
            subdomain = self.domain[i]
            grid[i] = SingleGrid(subdomain, self.ncells[i])
            dxs[i] = grid[i].dx
        return grid, dxs

    @property
    def shape(self):
        return (
            self.npoints.shape[0],
            tuple(self.npoints),
        )

    def __getitem__(self, index):
        return self.ugrids[index]

    def __str__(self):
        return str(self.ugrids)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        val = self.iter_index

        if val >= self.ndomains:
            raise StopIteration
        self.iter_index = val + 1
        return self.ugrids[val]


def TimeGrid_from_cfl(space_grid=None, time_domain=None):
    assert isinstance(time_domain, domains.SingleDomain)
    if isinstance(space_grid, SingleGrid):
        ncells_t = np.ceil(
            (1 / global_vars.CFL)
            * (
                (time_domain.domain[1] - time_domain.domain[0])
                / (space_grid.domain[1] - space_grid.domain[0])
            )
            * space_grid.ncells
        )

        time_grid = SingleGrid(time_domain, int(ncells_t))
    if isinstance(space_grid, MultipleGrid):
        ncells_t = list()
        time_domain = domains.MultipleDomain(
            np.repeat(
                time_domain.domain,
                space_grid.ndomains,
            )
            .reshape(-1, 2)
            .T
        )

        for i in range(space_grid.ndomains):
            ncells_t.append(
                int(
                    np.ceil(
                        (1 / global_vars.CFL)
                        * (
                            (
                                time_domain.domain[i].domain[1]
                                - time_domain.domain[i].domain[0]
                            )
                            / (
                                space_grid.domain[i].domain[1]
                                - space_grid.domain[i].domain[0]
                            )
                        )
                        * space_grid.ncells[i]
                    )
                )
            )
        time_grid = MultipleGrid(time_domain, ncells_t)
    return time_grid
