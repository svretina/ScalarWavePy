#!/usr/bin/env python3

import numpy as np
from scalarwavepy import wave
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
            if isinstance(grid, grids.MultipleGrid):

                def func1(x):
                    return list(map(values, x.coords))

                def func2(function, x):
                    return list(map(function, x.ugrids))

                self.values = np.array(
                    func2(func1, grid),
                    dtype=object,
                )
            elif isinstance(grid, grids.SingleGrid):
                self.values = values(grid.coords)
            else:
                raise TypeError(
                    f"Grid cannot be of type {type(grid)}"
                )
        elif isinstance(values, (float, int)):
            # y = const
            self.values = np.full(grid.shape, values)
        elif isinstance(values, np.ndarray):
            # direct assignment
            if grid.shape == values.shape:
                self.values = values
            else:
                raise ValueError(f"Grid shape {grid.shape} and Values shape {values.shape} dont match.")
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

    def _apply_reduction(self, reduction, *args, **kwargs):
        return reduction(self.values, *args, **kwargs)

    def _apply_unary(self, function, *args, **kwargs):
        return type(self)(
            self.grid,
            function(self.values, *args, **kwargs),
        )

    def _apply_binary(self, other, function, *args, **kwargs):
        # If it is a number
        if isinstance(other, (int, float, complex)):
            return type(self)(
                self.grid,
                function(
                    self.values,
                    other,
                    *args,
                    **kwargs,
                ),
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
                    self.grid.dx,
                    other.grid.dx,
                    atol=1e-14,
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
            u = GridFunction(grid, lambda s: self.func(s, 0))
            pi = GridFunction(grid, lambda s: self.func.dt(s, 0))
            xi = GridFunction(grid, lambda s: self.func.dx(s, 0))
        self.state_vector = np.array([u, pi, xi])

    def _apply_reduction(self, reduction, *args, **kwargs):
        return reduction(self.state_vector, *args, **kwargs)

    def _apply_unary(self, function, *args, **kwargs):
        return type(self)(
            function(
                self.state_vector,
                *args,
                **kwargs,
            )
        )

    def _apply_binary(self, other, function, *args, **kwargs):
        # If it is a number
        if isinstance(
            other,
            (int, float, complex, np.ndarray),
        ):
            return type(self)(
                function(
                    self.state_vector,
                    other,
                    *args,
                    **kwargs,
                )
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
