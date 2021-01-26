import pytest
from pylie.core import _LieIntegrator
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

f = lambda x: x
y0 = np.array([0])
h = 0.1
T = 2
verbose = True
solve_immediately = False


@pytest.fixture
def base_integrator():
    return _LieIntegrator(f, y0, h, T, verbose, solve_immediately)


def test_integrator_steplength_close_to_provided_steplength(base_integrator):
    assert_almost_equal(base_integrator.h, 0.1, decimal=2)


def test_integrator_time_axis_ends_at_specified_time(base_integrator):
    assert_equal(base_integrator.t[-1], T)


def test_invalid_y0_raises_type_error():
    with pytest.raises(TypeError):
        invalid_integrator = _LieIntegrator(f, 0, h, T, verbose, solve_immediately)