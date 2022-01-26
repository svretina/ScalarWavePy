#!/usr/bin/env python3

import numpy as np
from scalarwavepy import wave
from scalarwavepy import utils
from scalarwavepy import analytic
from scalarwavepy import globvars
from scalarwavepy.numerical import BaseNumerical


class SingleDomain:
    def __init__(self, physical_domain):
        """Constructor of :py:class:`~.SingleDomain`.
        :param physical_domain: a list of the physical domain
        eg [0,1]
        :type physical_domain: list or np.ndarray
        """
        if isinstance(physical_domain, list):
            self.domain = np.array(physical_domain)
        elif isinstance(physical_domain, np.ndarray):
            self.domain = physical_domain
        else:
            raise TypeError(
                f"Pass list or numpy ndarray.\
                You passed {type(physical_domain)}"
            )


class SingleGrid(BaseNumerical):
    def __init__(self, single_domain, ncells):
        """Constructor of :py:class:`~.UniformGrid`.
        A cell is defined as the space between two grid points
        :param domain: Physical Domain of the grid
        :type domain: Domain class
        :param ncells: Number of cells for the grid.
        :type ncells: int
        """
        if not isinstance(single_domain, SingleDomain):
            raise TypeError(
                "The domain should be of class SingleDomain."
            )
        assert isinstance(ncells, int)
        self.domain = single_domain.domain
        self.ncells = ncells
        self.npoints = ncells + 1
        self.coords = utils.discretize(
            self.domain[0], self.domain[1], ncells
        )
        self.dx = utils.spacing(
            self.domain[0], self.domain[1], ncells
        )
        self.spacing = self.dx

    def __str__(self):
        return str(self.coords)

    def __len__(self):
        return len(self.coords)

    @property
    def shape(self):
        return self.coords.shape

    def _apply_reduction(self, reduction, *args, **kwargs):
        return reduction(self.coords, *args, **kwargs)

    def _apply_unary(self, function, *args, **kwargs):
        return function(self.coords, *args, **kwargs)

    def _apply_binary(self, other, function, *args, **kwargs):
        # If it is a number
        if isinstance(other, (int, float, complex)):
            return function(self.coords, other, *args, **kwargs)
        # If we are here, its because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")


class GridFunction(BaseNumerical):
    def __init__(self, grid, values):
        self.grid = grid
        if callable(values):
            # y = f(x)
            self.values = values(grid)
        elif isinstance(values, (float, int)):
            # y = const
            self.values = np.repeat(values, len(grid))
            assert grid.shape == self.values.shape
        elif isinstance(values, np.ndarray):
            # direct assignment
            self.values = values
            assert grid.shape == values.shape
        else:
            raise TypeError(
                "values can be of type float or numpy.ndarray only."
            )

    def differentiate(self):
        f = self.values
        tmp = np.empty_like(self.values)
        tmp[1:-1] = (f[2:] - f[:-2]) / (2 * self.grid.spacing)
        tmp[0] = (f[1] - f[0]) / self.grid.spacing
        tmp[-1] = (f[-1] - f[-2]) / self.grid.spacing
        self.values = tmp

    def differentiated(self):
        f = self.values
        tmp = np.empty_like(self.values)
        tmp[1:-1] = (f[2:] - f[:-2]) / (2 * self.grid.spacing)
        tmp[0] = (f[1] - f[0]) / self.grid.spacing
        tmp[-1] = (f[-1] - f[-2]) / self.grid.spacing
        return type(self)(self.grid, tmp)

    def __len__(self):
        return len(self.grid)

    def __str__(self):
        tmp = f"x:    {self.grid}\nf(x): {self.values}"
        return tmp

    # def __repr__(self):
    #     tmp = f"{self.values}"
    #     return tmp

    # def __getitem__(self, index):
    #     return self.values[index]

    def _apply_reduction(self, reduction, *args, **kwargs):
        return reduction(self.values, *args, **kwargs)

    def _apply_unary(self, function, *args, **kwargs):
        return type(self)(
            self.grid, function(self.values, *args, **kwargs)
        )

    def _apply_binary(self, other, function, *args, **kwargs):
        # If it is a number
        if isinstance(other, (int, float, complex)):
            return type(self)(
                self.grid,
                function(self.values, other, *args, **kwargs),
            )

        if isinstance(other, type(self)):
            if not (
                self.grid.shape == other.grid.shape
                and np.allclose(
                    self.grid.domain[0],
                    other.grid.domain[0],
                    atol=1e-14,
                )
                and np.allclose(
                    self.grid.dx, other.grid.dx, atol=1e-14
                )
            ):
                raise ValueError(
                    "The objects do not have the same grid!"
                )
            return type(self)(
                self.grid,
                function(
                    self.values,
                    other.values,
                    *args,
                    **kwargs,
                ),
            )
        # If we are here, its because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")


class StateVector(BaseNumerical):
    def __init__(self, vector=None, func=None):
        self.state_vector = vector
        self.func = func

    def initialize(self, grid):
        if not callable(self.func):
            u = GridFunction(grid, 0)
            pi = GridFunction(grid, 0)
            xi = GridFunction(grid, 0)
        else:
            assert isinstance(self.func, analytic.Gaussian)
            u = GridFunction(grid, lambda s: self.func(s, 0))
            pi = GridFunction(grid, lambda s: self.func.dt(s, 0))
            xi = GridFunction(grid, lambda s: self.func.dx(s, 0))
        self.state_vector = np.array([u, pi, xi])

    # def __repr__(self):
    #     return f"{self.state_vector}"

    def _apply_reduction(self, reduction, *args, **kwargs):
        return reduction(self.state_vector, *args, **kwargs)

    def _apply_unary(self, function, *args, **kwargs):
        return type(self)(
            function(self.state_vector, *args, **kwargs)
        )

    def _apply_binary(self, other, function, *args, **kwargs):
        # If it is a number
        if isinstance(other, (int, float, complex, np.ndarray)):
            return type(self)(
                function(self.state_vector, other, *args, **kwargs)
            )

        if isinstance(other, type(self)):
            # Check if the number of variables is the same in both state vectors
            if not (
                self.state_vector.shape == other.state_vector.shape
            ):
                raise ValueError(
                    "The objects do not have the same grid!"
                )
            return type(self)(
                function(
                    self.state_vector,
                    other.state_vector,
                    *args,
                    **kwargs,
                ),
                self.func,
            )
        # If we are here, its because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")


def TimeGrid_from_cfl(space_grid=None, time_domain=None):
    assert isinstance(space_grid, SingleGrid)
    assert isinstance(time_domain, SingleDomain)

    ncells_t = np.ceil(
        (1 / globvars.CFL)
        * (
            (time_domain.domain[1] - time_domain.domain[0])
            / (space_grid.domain[1] - space_grid.domain[0])
        )
        * space_grid.ncells
    )
    # round up to next integer
    time_grid = SingleGrid(time_domain, int(ncells_t))
    return time_grid
