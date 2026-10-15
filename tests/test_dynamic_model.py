"""Independent update/transition oracles for the dynamic bootstrap model."""
import numpy as np
import pytest
from scipy.special import softmax

from dwts.dynamic_model import DynamicPosterior
from dwts.forward_model import SequentialPosterior


def test_zero_transition_is_exact_static_filter_without_mh():
    dynamic = DynamicPosterior(4, particles=256, process_sd=0., drift=0., seed=18)
    static = SequentialPosterior(4, particles=256, mh_steps=0, seed=18)
    for week, indices, judges, outcome in [(1, [0, 1, 2, 3], [5., 6., 7., 8.], 3),
                                          (2, [0, 1, 2], [7., 5., 8.], 1)]:
        dynamic.advance(indices, judges, week)
        np.testing.assert_array_equal(dynamic.predict(indices, judges), static.predict(indices, judges))
        dynamic.observe(indices, judges, outcome)
        static.observe(indices, judges, outcome)
        np.testing.assert_array_equal(dynamic.particles, static.particles)
        np.testing.assert_array_equal(dynamic.weights, static.weights)


def test_score_drift_and_elapsed_week_noise_are_model_transitions():
    posterior = DynamicPosterior(3, particles=32, process_sd=0., drift=2., seed=4)
    initial = posterior.particles.copy()
    posterior.advance([0, 1, 2], [1., 1., 1.], 1)
    np.testing.assert_array_equal(posterior.particles, initial)
    posterior.advance([0, 1], [1., 3.], 3)
    expected = initial.copy()
    expected[:, :2] += [-1., 1.]
    np.testing.assert_allclose(posterior.particles, expected)
    with pytest.raises(ValueError, match="chronological"):
        posterior.advance([0, 1], [1., 3.], 3)
    # Verify known Gaussian propagation variance with many independent particles.
    noisy = DynamicPosterior(2, particles=100000, process_sd=.5, drift=0., seed=5)
    noisy.advance([0, 1], [1., 1.], 1)
    before = noisy.particles.copy()
    noisy.advance([0, 1], [1., 1.], 5)
    noise = noisy.particles-before
    assert abs(noise.mean()) < .01
    assert abs(noise.var()-1.) < .02


def test_normalized_weight_update_matches_direct_categorical_bayes():
    posterior = DynamicPosterior(3, particles=128, ess_fraction=.001, seed=9)
    judges = np.array([8., 5., 6.])
    shares = softmax(posterior.particles, axis=1)
    likelihood = softmax(-3*3*(.5*judges/judges.sum()+.5*shares), axis=1)
    expected = posterior.weights*likelihood[:, 1]
    expected /= expected.sum()
    particles = posterior.particles.copy()
    d = posterior.observe([0, 1, 2], judges, 1)
    np.testing.assert_allclose(posterior.weights, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(posterior.particles, particles)
    assert not d["resampled"]


def test_new_entrant_is_an_independent_prior_and_first_appearance_has_no_noise():
    posterior = DynamicPosterior(2, particles=2048, seed=11, process_sd=10., drift=10.)
    posterior.advance([0, 1], [6., 8.], 1)
    posterior.extend(1)
    new = posterior.particles[:, 2].copy()
    posterior.advance([0, 2], [6., 8.], 2)
    np.testing.assert_array_equal(posterior.particles[:, 2], new)
    assert abs(new.mean()) < .1 and abs(new.var()-1) < .1


@pytest.mark.parametrize("settings", [dict(process_sd=-1), dict(drift=np.inf),
    dict(temperature=0), dict(particles=1), dict(ess_fraction=0)])
def test_invalid_controls(settings):
    with pytest.raises(ValueError):
        DynamicPosterior(3, **settings)
