import pytest
import numpy as np
from scalarwavepy import utils


def f(s):
    return s * s


@pytest.mark.parametrize(
    "ui, uf, nu", [(0, 1, 10), (1, 10, 10), (1, 100, 100)]
)
def test_discretize_length(ui, uf, nu):
    arr = utils.discretize(ui, uf, nu)
    assert len(arr) == nu + 1


@pytest.mark.parametrize(
    "ui, uf, nu", [(0, 1, 10), (1, 10, 10), (1, 100, 100)]
)
def test_discretize_type(ui, uf, nu):
    arr = utils.discretize(ui, uf, nu)
    assert type(arr) == np.ndarray


@pytest.mark.parametrize(
    "ui, uf, nu", [(0, 1, 10), (1, 10, 10), (0, 100, 100)]
)
def test_discretize(ui, uf, nu):
    arr = utils.discretize(ui, uf, nu)
    print(arr)
    assert arr[0] == ui and arr[-1] == uf


@pytest.mark.parametrize(
    "f, s, h",
    [
        (f, np.array([1, 1, 1]), 1),
        (f, np.array([np.ones(10), np.ones(10), np.ones(10)]), 1),
    ],
)
def test_rk4(f, s, h):
    sprime = utils.rk4(f, s, h)
    assert sprime.shape == s.shape


@pytest.mark.parametrize(
    "f, s, h",
    [
        (f, np.array([0, 0, 0]), 1),
        (f, np.array([np.zeros(10), np.zeros(10), np.zeros(10)]), 1),
    ],
)
def test_rk4_zeros(f, s, h):
    sprime = utils.rk4(f, s, h)
    assert np.sum(sprime) == 0


@pytest.mark.parametrize("vec, dx", [(np.ones(11), 1)])
def test_L2_norm(vec, dx):
    assert type(vec) == np.ndarray


@pytest.mark.parametrize("vec, dx", [(np.ones(11), 0.1)])
def test_L2_norm(vec, dx):
    norm = utils.L2_norm(vec, dx)
    assert norm == 1


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
    "xi, xn, dx",
    [(0, 1, 0.1), (0, 1.2, 0.003), (0.3, 1.2, 0.3), (0, 1, 1 / 3)],
)
def test_n_from_dx(xi, xn, dx):
    ns = utils.n_from_dx(xi, xn, dx)
    dx2 = utils.spacing(xi, xn, ns-1)
    assert np.isclose(abs(dx - dx2), 0, 1e-16)


@pytest.mark.parametrize("xi, xn, n", [(1, 10, 10), (0, 10, 11)])
def test_spacing(xi, xn, n):
    dx = utils.spacing(xi, xn, n)
    n2 = (xn - xi) / dx
    assert np.isclose(abs(n - n2), 0, 1e-16)


def test_check_monotonicity1():
    x = np.arange(1, 10, 1)
    answer = True
    assert utils.check_monotonicity(x) == answer


def test_check_monotonicity2():
    x = [1, 2, 3, 4, 5, 4, 3, 4, 5, 6, 7, 8, 9]
    answer = False
    assert utils.check_monotonicity(x) == answer
