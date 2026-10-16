"""Protect release timing, independent mixture identities, and regret claims."""
from dataclasses import fields, replace
import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from dwts.expert_mixture import AdaptiveOnline, ForecastPacket, PRIOR, select_candidate


def packet(week=1, season=1, probabilities=None):
    if probabilities is None:
        probabilities = np.array([[.6, .3, .1], [.2, .5, .3],
                                  [.4, .1, .5], [1/3, 1/3, 1/3]])
    return ForecastPacket("toy", "a", season, week, ("a", "b", "c"), probabilities)


def test_packet_contains_no_outcome_and_requires_season_reset():
    assert not {"outcome", "observed_exit", "label"} & {field.name for field in fields(ForecastPacket)}
    with pytest.raises(ValueError, match="reset"):
        AdaptiveOnline().predict(packet())


def test_release_and_pending_guards_preserve_recoverable_state():
    model = AdaptiveOnline()
    model.reset("toy", 1)
    with pytest.raises(ValueError, match="No prediction"):
        model.release(0)
    model.predict(packet())
    with pytest.raises(ValueError, match="previous outcome"):
        model.predict(packet(2))
    with pytest.raises(ValueError, match="pending"):
        model.reset("toy", 2)
    for invalid in (-1, 3, .5, True, np.nan):
        with pytest.raises(ValueError, match="Invalid released"):
            model.release(invalid)
    model.release(0)
    with pytest.raises(ValueError, match="No prediction"):
        model.release(0)


def test_chronology_fold_consistency_and_cross_season_reset():
    model = AdaptiveOnline()
    model.reset("toy", 1)
    model.predict(packet(3))
    model.release(0)
    for week in (3, 2):
        with pytest.raises(ValueError, match="out-of-order"):
            model.predict(packet(week))
    with pytest.raises(ValueError, match="folds"):
        model.predict(replace(packet(4), fold="b"))
    with pytest.raises(ValueError, match="reset"):
        model.predict(packet(1, 2))
    model.reset("toy", 2)
    _, before = model.predict(packet(1, 2))
    np.testing.assert_allclose(before, PRIOR, atol=1e-15, rtol=0)


def test_input_and_returned_arrays_cannot_change_a_frozen_prediction():
    sample = packet()
    original = sample.probabilities.copy()
    custom_prior = PRIOR.copy()
    model = AdaptiveOnline(prior=custom_prior)
    custom_prior[:] = .25
    model.reset("toy", 1)
    prediction, before = model.predict(sample)
    np.testing.assert_allclose(prediction, PRIOR @ original, atol=1e-15, rtol=0)
    for returned in (prediction, before, model.logweights, model.log_weights):
        assert not returned.flags.writeable
        with pytest.raises(ValueError):
            returned[0] = .1
    sample.probabilities[:] = 1/3
    after = model.release(0)
    expected = PRIOR * original[:, 0]
    expected /= expected.sum()
    np.testing.assert_allclose(after, expected, atol=1e-15, rtol=0)


@pytest.mark.parametrize("eta,gamma", [(0, 0), (1, -.1), (1, 1), (np.nan, 0),
                                      (1, np.inf), ("one", 0), (True, 0), (1, None)])
def test_invalid_parameters(eta, gamma):
    with pytest.raises(ValueError):
        AdaptiveOnline(eta, gamma)


@pytest.mark.parametrize("prior", [None, [], [.5, .5], [.5, .2, .2, .2],
                                   [.5, .5, 0, 0], [.5, np.nan, .2, .1],
                                   [.5, np.inf, .2, .1], [[.5, .2, .2, .1]]])
def test_prior_must_cover_every_expert_with_finite_positive_mass(prior):
    with pytest.raises(ValueError, match="Prior"):
        AdaptiveOnline(prior=prior)


def test_invalid_probability_packets_leave_prediction_available():
    valid = packet()
    invalid = [replace(valid, ids=("a", "a", "c")),
               replace(valid, ids=("a", "b", "")),
               replace(valid, ids=("a", "b")),
               replace(valid, probabilities=np.ones((4, 3))),
               replace(valid, probabilities=np.ones((3, 3))/3),
               replace(valid, probabilities=np.full((4, 3), np.nan)),
               replace(valid, probabilities=np.full((4, 3), np.inf)),
               replace(valid, probabilities=np.array([[0., .5, .5]] * 4)),
               replace(valid, probabilities=None)]
    model = AdaptiveOnline()
    model.reset("toy", 1)
    for sample in invalid:
        with pytest.raises(ValueError):
            model.predict(sample)
    model.predict(valid)
    model.release(0)


@pytest.mark.parametrize("change", [{"week": 0}, {"week": 1.5}, {"week": True},
                                    {"season": np.nan}, {"phase": ""}, {"fold": ""}])
def test_packet_metadata_requires_an_explicit_valid_clock(change):
    model = AdaptiveOnline()
    model.reset("toy", 1)
    with pytest.raises(ValueError, match="phase, fold, season, or week"):
        model.predict(replace(packet(), **change))


def test_tempering_and_fixed_share_match_an_independent_probability_update():
    for eta, gamma in ((1, .2), (.5, 0), (2, 0)):
        model = AdaptiveOnline(eta, gamma)
        model.reset("toy", 1)
        sample = packet()
        model.predict(sample)
        after = model.release(1)
        expected = PRIOR * sample.probabilities[:, 1]**eta
        expected /= expected.sum()
        expected = (1-gamma) * expected + gamma * PRIOR
        np.testing.assert_allclose(after, expected, atol=1e-15, rtol=0)


def test_fixed_share_forward_filter_matches_exhaustive_hidden_expert_paths():
    samples = [packet(1), packet(2, probabilities=np.roll(packet().probabilities, 1, axis=1)),
               packet(3, probabilities=np.roll(packet().probabilities, 2, axis=1)), packet(4)]
    observed = [0, 2, 1, 2]
    for gamma in (0., .05, .2):
        model = AdaptiveOnline(eta=1, gamma=gamma)
        model.reset("toy", 1)
        joint = 1.
        for time, (sample, outcome) in enumerate(zip(samples, observed)):
            prediction, _ = model.predict(sample)
            joint *= prediction[outcome]
            model.release(outcome)
            enumeration = 0.
            for path in itertools.product(range(4), repeat=time+1):
                probability = PRIOR[path[0]] * samples[0].probabilities[path[0], observed[0]]
                for step in range(1, time+1):
                    transition = (1-gamma)*(path[step] == path[step-1]) + gamma*PRIOR[path[step]]
                    probability *= transition * samples[step].probabilities[path[step], observed[step]]
                enumeration += probability
            np.testing.assert_allclose(joint, enumeration, atol=1e-15, rtol=1e-12)


def test_bayesian_identity_and_valid_regret_bounds_hold_at_every_prefix():
    generator = np.random.default_rng(42)
    samples = [packet(week, probabilities=generator.dirichlet(np.ones(3), size=4))
               for week in range(1, 31)]
    outcomes = generator.integers(0, 3, size=len(samples))
    for eta, gamma in ((1, 0), (1, .05), (1, .2), (.5, 0)):
        model = AdaptiveOnline(eta, gamma)
        model.reset("toy", 1)
        expert_loss = np.zeros(4)
        mixture_loss = 0.
        for count, (sample, outcome) in enumerate(zip(samples, outcomes), start=1):
            prediction, _ = model.predict(sample)
            mixture_loss -= np.log(prediction[outcome])
            expert_loss -= np.log(sample.probabilities[:, outcome])
            model.release(outcome)
            allowance = (-np.log(PRIOR) - (count-1)*np.log1p(-gamma)
                         if eta == 1 else -np.log(PRIOR)/eta)
            assert np.all(mixture_loss <= expert_loss + allowance + 1e-11)
            if eta == 1 and gamma == 0:
                exact = -logsumexp(np.log(PRIOR)-expert_loss)
                assert mixture_loss == pytest.approx(exact, abs=1e-11)


def test_eta_above_one_does_not_inherit_the_bayesian_regret_bound():
    probabilities = np.array([[.9, .05, .05], [.05, .9, .05],
                              [.05, .9, .05], [.05, .9, .05]])
    model = AdaptiveOnline(eta=2)
    model.reset("toy", 1)
    mixture_loss = past_loss = 0.
    for week, outcome in enumerate((0, 1, 0), start=1):
        prediction, _ = model.predict(packet(week, probabilities=probabilities))
        mixture_loss -= np.log(prediction[outcome])
        past_loss -= np.log(probabilities[0, outcome])
        model.release(outcome)
    assert mixture_loss > past_loss - np.log(PRIOR[0]) + .5


def test_log_space_update_preserves_tiny_likelihoods_without_clipping():
    probabilities = np.array([[1e-250, .6, .4], [1e-240, .3, .7],
                              [1e-245, .5, .5], [1/3, 1/3, 1/3]])
    model = AdaptiveOnline()
    model.reset("toy", 1)
    model.predict(packet(probabilities=probabilities))
    model.release(0)
    expected = np.log(PRIOR) + np.log(probabilities[:, 0])
    expected -= logsumexp(expected)
    np.testing.assert_allclose(model.logweights, expected, atol=1e-12, rtol=0)
    assert model.logweights[0] < -500


def test_selection_uses_global_minimum_and_declared_candidate_order():
    tolerance = 1e-9
    losses = [2*tolerance, 1.1*tolerance, .2*tolerance]
    assert select_candidate(losses, tolerance) == 1
    assert select_candidate(losses, 0) == 2
    assert select_candidate([1, 1, 1]) == 0


@pytest.mark.parametrize("losses,tolerance", [([], 0), ([np.nan], 0), ([np.inf], 0),
                                             ([[1, 2]], 0), ([1], -1), ([1], np.nan)])
def test_selection_rejects_missing_or_nonfinite_scores(losses, tolerance):
    with pytest.raises(ValueError):
        select_candidate(losses, tolerance)
