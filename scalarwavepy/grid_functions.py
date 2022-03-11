#!/usr/bin/env python3

import numpy as np
from scalarwavepy import utils
from scalarwavepy import grids
from scalarwavepy import analytic
from scalarwavepy import global_vars
from scalarwavepy.numerical import BaseNumerical
from scalarwavepy import domains


class GridFunction(BaseNumerical):
    def __init__(self, grid, values):
        self.grid = grid
        if callable(values):
            # y = f(x)
            self.function = values
            if isinstance(grid, grids.SingleGrid):
                self.values = values(grid.coords)
            else:
                raise TypeError(
                    f"Grid can only be of type SingleGrid. You passed a {type(grid)} object."
                )
        elif isinstance(values, (float, int)):
            # y = const
            self.values = np.full(grid.shape, values)
        elif isinstance(values, (np.ndarray, list)):
            if isinstance(values, list):
                values = np.asarray(values)
            # direct assignment
            if grid.shape == values.shape:
                self.values = values
            else:
                raise ValueError(
                    f"Grid's shape {grid.shape} and Values' shape {values.shape} dont match."
                )
        else:
            raise TypeError("Values can be of type float or numpy.ndarray only.")

    @property
    def shape(self):
        return self.values.shape

    def differentiate(self):
        f = self.values
        tmp = np.empty_like(self.values)
        tmp[1:-1] = (f[2:] - f[:-2]) / (2 * self.grid.spacing)
        tmp[0] = (f[1] - f[0]) / self.grid.spacing
        tmp[-1] = (f[-1] - f[-2]) / self.grid.spacing
        self.values = tmp

    @property
    def differentiated(self):
        f = self.values.copy()
        tmp = np.empty_like(self.values)
        tmp[1:-1] = (f[2:] - f[:-2]) / (2 * self.grid.spacing)
        tmp[0] = (f[1] - f[0]) / self.grid.spacing
        tmp[-1] = (f[-1] - f[-2]) / self.grid.spacing
        return type(self)(self.grid, tmp)

    @staticmethod
    def _trapezoidal_rule(vector, dx):
        tmp = 0.5 * (vector[0] + vector[-1]) + np.sum(vector[1:-1], axis=0)
        return dx * tmp

    def integrate(self):
        return self._trapezoidal_rule(self.values, self.grid.dx)

    def norm(self):
        return np.sqrt(self._trapezoidal_rule(self.values * self.values, self.grid.dx))

    def __len__(self):
        return len(self.grid)

    def __str__(self):
        tmp = f"x:    {self.grid}\nf(x): {self.values}"
        return tmp

    def _apply_reduction(self, reduction, *args, **kwargs):
        return reduction(self.values, *args, **kwargs)

    def _apply_unary(self, function, *args, **kwargs):
        return type(self)(self.grid, function(self.values, *args, **kwargs))

    def _apply_binary(self, other, function, *args, **kwargs):
        # If it is a number
        if isinstance(other, (int, float, complex)):
            return type(self)(self.grid, function(self.values, other, *args, **kwargs))

        if isinstance(other, type(self)):
            if not (
                self.grid.shape == other.grid.shape
                and np.allclose(self.grid.domain[0], other.grid.domain[0], atol=1e-14)
                and np.allclose(self.grid.dx, other.grid.dx, atol=1e-14)
            ):
                raise ValueError("The objects do not have the same grid!")
            return type(self)(
                self.grid, function(self.values, other.values, *args, **kwargs)
            )
        # If we are here, its because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")


class StateVector(BaseNumerical):
    def __init__(self, grid=None, vector=None, func=None):
        assert isinstance(grid, grids.SingleGrid)
        self.func = func
        self.grid = grid
        if vector is not None:
            self.state_vector = vector
            self.u = vector[0]
            self.pi = vector[1]
            self.xi = vector[2]
        elif vector is None:
            assert grid is not None
            if not callable(func):
                u = GridFunction(grid, 0)
                pi = GridFunction(grid, 0)
                xi = GridFunction(grid, 0)
            else:
                u = GridFunction(grid, lambda s: self.func(s, 0))
                pi = GridFunction(grid, lambda s: self.func.dt(s, 0))
                xi = GridFunction(grid, lambda s: self.func.dx(s, 0))
            self.u = u
            self.pi = pi
            self.xi = xi
            self.state_vector = np.array([u, pi, xi])
        else:
            raise ValueError("Cannot pass grid=None and vector=None")

    @property
    def energy(self):
        tmp = self.pi ** 2 + self.xi ** 2
        energy = tmp.integrate() / len(self.xi)
        return energy

    def _apply_reduction(self, reduction, *args, **kwargs):
        return reduction(self.state_vector, *args, **kwargs)

    def _apply_unary(self, function, *args, **kwargs):
        return type(self)(
            self.grid, function(self.state_vector, *args, **kwargs), self.func
        )

    def _apply_binary(self, other, function, *args, **kwargs):
        # If it is a number
        if isinstance(other, (int, float, complex, np.ndarray)):
            return type(self)(
                self.grid,
                function(self.state_vector, other, *args, **kwargs),
                self.func,
            )

        if isinstance(other, type(self)):
            # Check if the number of variables is
            # the same in both state vectors
            if not (self.state_vector.shape == other.state_vector.shape):
                raise ValueError("The objects do not have the same grid!")
            return type(self)(
                self.grid,
                function(self.state_vector, other.state_vector, *args, **kwargs),
                self.func,
            )
        # If we are here, its because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")


# I dont think the name is correct.
# Maybe StateVectorBundle?
class StateTensor:
    def __init__(self, grid=None, tensor=None, func=None):
        assert isinstance(grid, grids.MultipleGrid)
        self.func = func
        self.grid = grid
        if tensor is not None:
            self.state_tensor = tensor
        elif tensor is None:
            assert grid is not None
            self.state_tensor = np.asarray(
                [
                    StateVector(
                        grid.ugrids[i],
                        tensor,
                        func,
                    )
                    for i in range(grid.ndomains)
                ],
                dtype=object,
            )
        else:
            raise TypeError("Cannot pass grid=None and tensor=None")

    def _apply_reduction(self, reduction, *args, **kwargs):
        return reduction(self.state_tensor, *args, **kwargs)

    def _apply_unary(self, function, *args, **kwargs):
        return type(self)(
            self.grid, function(self.state_tensor, *args, **kwargs), self.func
        )

    def _apply_binary(self, other, function, *args, **kwargs):
        # If it is a number
        if isinstance(other, (int, float, complex, np.ndarray)):
            return type(self)(
                self.grid,
                function(self.state_tensor, other, *args, **kwargs),
                self.func,
            )

        if isinstance(other, type(self)):
            # Check if the number of variables is
            # the same in both state vectors
            if not (self.state_vector.shape == other.state_vector.shape):
                raise ValueError("The objects do not have the same grid!")
            return type(self)(
                self.grid,
                function(self.state_tensor, other.state_tensor, *args, **kwargs),
                self.func,
            )
        # If we are here, its because we cannot add the two objects
        raise TypeError("I don't know how to combine these objects")


class Result:
    def __init__(self, spatial_grid, time_grid):
        self.spatial_grid = spatial_grid
        self.time = time_grid

        if isinstance(spatial_grid, grids.MultipleGrid):
            self.ndomains = spatial_grid.ndomains
            self.tensor = np.empty((self.ndomains, time_grid.npoints), dtype=object)
        if isinstance(spatial_grid, grids.SingleGrid):
            self.vector = np.empty(time_grid.npoints, dtype=object)
            self.ndomains = 1

    def initialize(self, state):
        if isinstance(self.spatial_grid, grids.MultipleGrid):
            for i in range(self.ndomains):
                self.tensor[i, 0] = state.state_tensor[i]
        if isinstance(self.spatial_grid, grids.SingleGrid):
            self.vector[0] = state

    @property
    def energy(self):
        if isinstance(self.spatial_grid, grids.MultipleGrid):
            values = list()
            for i in range(self.time.npoints):
                tmp = 0
                for j in range(self.ndomains):
                    tmp = tmp + self.tensor[j, i].energy
                values.append(tmp)
            return GridFunction(self.time, values)
        if isinstance(self.spatial_grid, grids.SingleGrid):
            values = list()
            for i in range(self.time.npoints):
                values.append(self.vector[i].energy)
            return GridFunction(self.time, values)

    @property
    def shape(self):
        if isinstance(self.spatial_grid, grids.MultipleGrid):
            return self.tensor.shape
        if isinstance(self.spatial_grid, grids.SingleGrid):
            return self.vector.shape
