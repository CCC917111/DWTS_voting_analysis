"""Independent normalized-likelihood and small exact-posterior checks."""
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp, softmax

from dwts.forward_model import SequentialPosterior, exit_log_probs


def test_categorical_normalizer_and_roster_conditioning():
    alpha = np.array([[.2, -1., 3., .5], [2., 1., -3., 0.]])
    ids, judges = [0, 2, 3], np.array([7., 8., 6.])
    shares = softmax(alpha[:, ids], axis=1)
    merit = 3*(.5*judges/judges.sum()+.5*shares)
    expected = softmax(-3*merit, axis=1)
    np.testing.assert_allclose(np.exp(exit_log_probs(alpha, ids, judges)), expected)
    changed = alpha.copy()
    changed[:, 1] = 10000
    np.testing.assert_array_equal(exit_log_probs(alpha, ids, judges),
                                  exit_log_probs(changed, ids, judges))
    np.testing.assert_allclose(np.exp(exit_log_probs(alpha, ids, judges)).sum(axis=1), 1.)


def test_rank_uses_logit_order_without_share_underflow_ties():
    a = np.array([-10000., -10001., 0.])
    lp = exit_log_probs(a, [0, 1, 2], [5., 5., 5.], rule="rank", temperature=2.)
    assert lp[1] > lp[0] > lp[2]
    np.testing.assert_allclose(exit_log_probs(a+100., [0, 1, 2], [5.]*3, rule="rank"),
                               exit_log_probs(a, [0, 1, 2], [5.]*3, rule="rank"))


@pytest.mark.parametrize("bad", [dict(prior_sd=0), dict(particles=1), dict(rule="unknown"),
                                dict(ess_fraction=1.1), dict(mh_steps=-1)])
def test_invalid_engine_settings_rejected(bad):
    with pytest.raises(ValueError):
        SequentialPosterior(3, **bad)


def test_full_history_target_and_new_entrant_prior():
    posterior = SequentialPosterior(3, particles=32, prior_sd=1.7, mh_steps=0, seed=8)
    posterior.observe([0, 1, 2], [7., 6., 5.], 2)
    posterior.observe([0, 1], [5., 8.], 0)
    posterior.extend(1)
    candidates = np.array([[.1, .2, -.3, 3.], [-2., 1., .5, 0.]])
    expected = -.5*np.square(candidates/1.7).sum(axis=1)
    for idx, j, out in [([0, 1, 2], [7., 6., 5.], 2), ([0, 1], [5., 8.], 0)]:
        p = softmax(candidates[:, idx], axis=1)
        loghazard = -3*len(idx)*(.5*np.array(j)/sum(j)+.5*p)
        expected += loghazard[:, out]-logsumexp(loghazard, axis=1)
    np.testing.assert_allclose(posterior.log_target(candidates), expected, rtol=0, atol=1e-12)


def test_smc_matches_gauss_hermite_multi_event_posterior():
    # Common support shifts integrate out. Difference d~N(0,2) reduces the
    # two-person posterior to an independently integrated one-dimensional oracle.
    x, weights = hermgauss(100)
    difference = 2*x
    alpha = np.column_stack((difference/2, -difference/2))
    prior = weights/np.sqrt(np.pi)
    events = [([6., 5.], 1), ([4., 7.], 0), ([8., 6.], 1), ([5., 5.], 1)]
    target = prior.copy()
    for judges, outcome in events:
        p = softmax(alpha, axis=1)
        hazard = softmax(-3*2*(.5*np.array(judges)/sum(judges)+.5*p), axis=1)
        target *= hazard[:, outcome]
    target /= target.sum()
    expected = target @ softmax(-3*2*(.5*np.array([7., 6.])/13+.5*softmax(alpha, axis=1)), axis=1)
    actual = []
    for seed in (30, 71):
        posterior = SequentialPosterior(2, particles=32768, seed=seed, mh_steps=6)
        for judges, outcome in events:
            posterior.observe([0, 1], judges, outcome)
        actual.append(posterior.predict([0, 1], [7., 6.]))
    np.testing.assert_allclose(actual, np.tile(expected, (2, 1)), atol=.007, rtol=0)


def test_systematic_resampling_does_not_select_zero_weight_at_uniform_zero():
    posterior = SequentialPosterior(2, particles=16, mh_steps=0)
    posterior.particles[:, 0] = np.arange(16)
    posterior.weights[:] = 0.
    posterior.weights[-1] = 1.
    class ZeroUniform:
        def random(self):
            return 0.
    posterior.rng = ZeroUniform()
    posterior.observe([0, 1], [5., 5.], 0)
    np.testing.assert_array_equal(posterior.particles[:, 0], np.full(16, 15.))


def test_prediction_does_not_modify_state_or_depend_on_unobserved_exit():
    posterior = SequentialPosterior(3, particles=256, seed=9)
    particles, weights = posterior.particles.copy(), posterior.weights.copy()
    p = posterior.predict([0, 1, 2], [6., 5., 4.])
    assert np.isclose(p.sum(), 1.)
    np.testing.assert_array_equal(posterior.particles, particles)
    np.testing.assert_array_equal(posterior.weights, weights)
    assert posterior.history == []
