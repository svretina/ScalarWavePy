import pytest
import numpy as np
from scalarwavepy import ode


@pytest.mark.parametrize(
    "s, dx", [(np.ones((3,11,11)), 0.1)]
)
def test_calculate_diagnostics(s, dx):
    enrgy_dnsty = ode.calculate_diagnostics(s, dx)
    assert np.all(enrgy_dnsty == np.ones(11))


@pytest.mark.parametrize(
    "s, dx", [(np.ones((3,11,11)), 0.1)]
)
def test_calculate_diagnostics(s, dx):
    enrgy_dnsty = ode.calculate_diagnostics(s, dx)
    assert type(enrgy_dnsty) is np.ndarray


# def test_rhs():
#     f = ode.rhs(0.1, 0, 0, 1)
#     assert type(f) == 'function'


@pytest.mark.parametrize(
    "s, dx", [(np.ones((3,11)), 0.1)]
)
def test_RHS1(s, dx):
    tmp = ode.RHS(s, dx, 1, 0, 0)
    assert tmp.shape == s.shape
