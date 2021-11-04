import pytest
from scalarwavepy import utils
import numpy as np

def test_discretize():
    pass


def test_spatial_derivative_constant():
    assert type(utils.spatial_derivative([1,1,1,1], 1)) is np.ndarray


def test_spatial_derivative_constant():
    assert sum(utils.spatial_derivative([1,1,1,1], 1)) == 0


@pytest.mark.parametrize('slope', [(3), (-2), (-1), (5), (100), (0)])
def test_spatial_derivative_linear(slope):
    x = np.linspace(-10, 10, 11)
    y = slope * x
    assert np.all(utils.spatial_derivative(y, x[2]-x[1]) == slope)


@pytest.mark.parametrize('deg', [(3), (-2), (-1), (5), (100), (0)])
def test_spatial_derivative_quad(deg):
    x = np.linspace(1, 10, 10)
    y = x ** deg
    interior = x[1:-1]
    answer_interior = deg * interior ** (deg-1)

    assert np.all(utils.spatial_derivative(y, x[2]-x[1])[1:-1] == answer_interior)
