"""The comparator recovers a known discrete system and keeps future data separate."""
import numpy as np
import pytest
from realityforge.baselines import iris_recurrence


def independent_discrete_pendulum():
    t = np.arange(180)/30
    angle = np.empty(180)
    angle[0], angle[1] = .55, .54
    for i in range(2, len(t)):
        velocity = (angle[i-1]-angle[i-2])*30
        acceleration = -17.3*np.sin(angle[i-1])-.24*velocity
        angle[i] = angle[i-1]+velocity/30+acceleration/900
    return t, angle


def test_published_recurrence_adapter_recovers_discrete_parameters():
    t, truth = independent_discrete_pendulum()
    parameters, predicted = iris_recurrence(t[:117], truth[:117], t)
    assert parameters['k'] == pytest.approx(17.3, abs=1e-10)
    assert parameters['beta'] == pytest.approx(.24, abs=1e-10)
    np.testing.assert_allclose(predicted, truth, atol=1e-11)


def test_published_recurrence_has_no_access_to_forecast_observations():
    t, observed = independent_discrete_pendulum()
    altered = observed.copy()
    altered[117:] += 8*np.sin(t[117:]*3)
    first, prediction = iris_recurrence(t[:117], observed[:117], t)
    second, repeated = iris_recurrence(t[:117], altered[:117], t)
    assert first == second
    np.testing.assert_array_equal(prediction, repeated)
    assert np.max(np.abs(prediction[117:]-altered[117:])) > 1


@pytest.mark.parametrize('problem', ['nonuniform', 'nonfinite', 'short', 'constant'])
def test_published_recurrence_rejects_invalid_or_unidentifiable_inputs(problem):
    t, y = independent_discrete_pendulum()
    if problem == 'nonuniform': t[5] += .001
    if problem == 'nonfinite': y[5] = np.nan
    if problem == 'short': t, y = t[:12], y[:12]
    if problem == 'constant': y[:] = 0
    with pytest.raises(ValueError): iris_recurrence(t, y, t)
