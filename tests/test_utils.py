import pytest
from scalarwavepy import utils
import numpy as np


def test_discretize():
    pass


def test_spatial_derivative_constant():
    assert type(utils.spatial_derivative([1, 1, 1, 1], 1)) is np.ndarray


def test_spatial_derivative_constant():
    assert sum(utils.spatial_derivative([1, 1, 1, 1], 1)) == 0


@pytest.mark.parametrize("slope", [(3), (-2), (-1), (5), (100), (0)])
def test_spatial_derivative_linear(slope):
    x = np.linspace(-10, 10, 11)
    y = slope * x
    assert np.all(utils.spatial_derivative(y, x[2] - x[1]) == slope)


@pytest.mark.parametrize("deg", [(2)])
def test_spatial_derivative_quad(deg):
    h = 1
    x = np.arange(1, 10, h)
    y = x ** deg
    interior = x[1:-1]
    answer_interior = deg * interior ** (deg - 1)
    cdiff = utils.spatial_derivative(y, x[2] - x[1])[1:-1]
    # num_error = abs(cdiff-answer_interior)
    # th_error = (1/6) * deg*(deg-1)*(deg-2)*x**(deg-3)

    # print(cdiff,'\n', answer_interior)
    # assert np.all(num_error < th_error)
    assert np.all(answer_interior == cdiff)

@pytest.mark.parametrize("slope", [(1), (3), (5), (7), (100)])
def test_integrate_linear1(slope):
    x = np.linspace(-10, 10, 11)
    y = slope * x

    answer = 0
    tmp = utils.integrate(y, x[2] - x[1])
    assert np.all(utils.integrate(y, x[2] - x[1]) == answer)


@pytest.mark.parametrize("slope", [(1), (3), (5), (7), (100)])
def test_integrate_linear2(slope):
    x = np.linspace(0, 1, 11)
    y = slope * x

    answer = 0.5 * slope
    assert np.all(utils.integrate(y, x[2] - x[1]) == answer)


@pytest.mark.parametrize("slope", [(3), (-2), (-1), (5), (100), (0)])
def test_integrate_addition(slope):
    x = np.linspace(0, 1, 11)
    y1 = slope * x
    y2 = x
    answer = 0.5 * slope + 0.5
    assert np.all(utils.integrate(y1 + y2, x[2] - x[1]) == answer)


@pytest.mark.parametrize("power", [(1), (3), (5), (7), (9)])
def test_integrate_power(power):
    x = np.linspace(-1, 1, 11)
    y = x ** power

    answer = 0
    assert np.allclose(utils.integrate(y, x[2] - x[1]), answer, 1e-15)


@pytest.mark.parametrize("power", [(3), (5), (7), (8), (9)])
def test_integrate_power2(power):
    x = np.linspace(0, 1, 11)
    y = x ** power
    answer = 1 / (power + 1)
    tmp = utils.integrate(y, x[2] - x[1])

    error = abs(answer - tmp)
    th_error = (
        (x[-1] - x[0]) ** 3
        / (12 * 11 ** 2)
        * power
        * (power - 1)
        * np.max(x ** (power - 2))
    )
    assert np.all(error < th_error)


@pytest.mark.parametrize(
    "dx, xn", [(0.1, 1), (0.003, 1.2), (0.3, 1.2), (0.123, 1.23), (1 / 3, 1)]
)
def test_n_from_dx(dx, xn):
    ns = utils.n_from_dx(dx, xn)
    dx2 = xn / ns
    assert np.isclose(abs(dx - dx2), 0, 1e-14)


@pytest.mark.parametrize("n, xn", [(10, 10), (10, 1)])
def test_dx_from_n(n, xn):
    dx = utils.dx_from_n(n, xn)
    n2 = xn / dx + 1
    assert np.isclose(abs(n + 1 - n2), 0, 1e-14)
