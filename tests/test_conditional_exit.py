"""Test categorical fitting and the release clock with synthetic score panels."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dwts.conditional_exit import (Event, baseline_prob, basic_loss, build_events,
                                   current_features, fit, nll, objective, predictions)


def panel():
    rows = []
    for season in (3, 4):
        for week, scores, exit_id in ((1, (1., 2., 3., 4.), None),
                                     (2, (4., 3., 2., 1.), "D"),
                                     (3, (2., 3., 4.), "A")):
            for identifier, score in zip("ABCD", scores):
                rows.append(dict(season=season, week=week, contestant_id=identifier,
                    judge_mean=score, eliminated=identifier == exit_id,
                    withdrawal=False, is_final_week=week == 3,
                    placement=99, result_label="unused metadata"))
    return pd.DataFrame(rows)


def test_features_use_own_previous_scores_and_cold_start():
    frame = pd.DataFrame(dict(contestant_id=["C", "B", "A"], judge_mean=[3., 2., 1.]))
    ids, x, current, past = current_features(frame, {})
    assert ids == ("A", "B", "C")
    np.testing.assert_array_equal(past, current)
    np.testing.assert_array_equal(x[:, 2], 0.)
    ids, x, current, past = current_features(frame, {"A": [2., 1.], "B": [1.]})
    np.testing.assert_array_equal(past, [1.5, 1., 1.5])
    change = np.array([current[0]-1., current[1]-1., 0.])
    np.testing.assert_allclose(x[:, 2], change-change.mean(), atol=1e-15)
    np.testing.assert_allclose(x.mean(axis=0), 0., atol=1e-15)


def test_skipped_outcome_scores_enter_history_and_seasons_reset():
    events = build_events(panel())
    assert not events[0].update
    np.testing.assert_array_equal(events[1].past, events[0].current)
    np.testing.assert_array_equal(events[3].past, events[3].current)
    assert events[1].eligible
    assert events[2].update and not events[2].eligible


def test_future_labels_scores_rosters_and_placement_cannot_change_earlier_features():
    original = panel()
    changed = original.copy()
    later = changed.week.gt(1)
    changed.loc[later, "judge_mean"] *= 10.
    changed.loc[later, "contestant_id"] = "future:"+changed.loc[later, "contestant_id"]
    changed["eliminated"] = ~changed.eliminated
    changed["withdrawal"] = True
    changed["is_final_week"] = True
    changed["placement"] = -100
    changed["result_label"] = "arbitrary metadata"
    a = [event for event in build_events(original) if event.week == 1]
    b = [event for event in build_events(changed) if event.week == 1]
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left.x, right.x)
    label_only = original.copy()
    label_only["eliminated"] = ~label_only.eliminated
    for left, right in zip(build_events(original), build_events(label_only)):
        np.testing.assert_array_equal(left.x, right.x)


@pytest.mark.parametrize("dimension", [2, 3])
def test_analytic_gradient_matches_finite_difference(dimension):
    events = [event for event in build_events(panel()) if event.eligible]
    beta = np.array([-.9, -.3, .2])[:dimension]
    _, gradient = objective(beta, events, .01, dimension)
    step = 1e-6
    numerical = np.array([
        (objective(beta+np.eye(dimension)[i]*step, events, .01, dimension)[0]
         - objective(beta-np.eye(dimension)[i]*step, events, .01, dimension)[0])/(2*step)
        for i in range(dimension)])
    np.testing.assert_allclose(gradient, numerical, rtol=0, atol=1e-8)


def test_fit_learns_predictive_score_signal_and_passes_gradient_gate():
    frame = panel().query("week == 2").copy()
    frame["eliminated"] = frame.contestant_id.eq("D")
    frame["is_final_week"] = False
    events = build_events(frame)
    beta, diagnostic = fit(events, .01)
    assert nll(events, beta) < np.log(4)-.5
    assert diagnostic["projected_gradient_inf"] <= 1e-5
    rows, scores = predictions(events, "choice_two", beta=beta)
    assert len(rows) == 8 and len(scores) == 2
    assert all(score["top1"] == 1 for score in scores)


def test_likelihood_stays_finite_for_large_logits():
    event = [event for event in build_events(panel()) if event.eligible][0]
    value, gradient = objective(np.array([1000., -800.]), [event], 0., 2)
    assert np.isfinite(value) and np.isfinite(gradient).all()


def test_baselines_are_normalized_and_score_the_identical_target():
    event = [event for event in build_events(panel()) if event.eligible][0]
    for model in ("uniform", "current_judges", "past_judges"):
        p = baseline_prob(event, model, 3.)
        assert np.isclose(p.sum(), 1.)
        assert 0 <= basic_loss(event, p)["brier"] <= 2
    np.testing.assert_array_equal(baseline_prob(event, "past_judges", 3.),
                                  baseline_prob(event, "fixed_legacy_past12", 3.))
    with pytest.raises(ValueError, match="Unknown baseline"):
        baseline_prob(event, "typo", 3.)


@pytest.mark.parametrize("scores", [[-1., 2., 3.], [np.nan, 2., 3.],
                                    [np.inf, 2., 3.], [0., 0., 0.]])
def test_invalid_current_scores_are_rejected(scores):
    frame = pd.DataFrame(dict(contestant_id=list("ABC"), judge_mean=scores))
    with pytest.raises(ValueError, match="current scores"):
        current_features(frame, {})


def test_invalid_history_duplicate_ids_and_unreleased_label_types_are_rejected():
    frame = pd.DataFrame(dict(contestant_id=list("ABC"), judge_mean=[1., 2., 3.]))
    with pytest.raises(ValueError, match="histories"):
        current_features(frame, {"A": [-1.]})
    with pytest.raises(ValueError, match="unique"):
        current_features(frame.assign(contestant_id="A"), {})
    invalid = panel()
    invalid["eliminated"] = "False"
    with pytest.raises(ValueError, match="labels"):
        build_events(invalid)
    with pytest.raises(ValueError, match="Duplicate"):
        build_events(pd.concat([panel(), panel().iloc[[0]]]))


@pytest.mark.parametrize("probability", [[.5, .5], [0., .3, .3, .4],
                                         [.2, .2, .2, .2], [np.nan, .3, .3, .4]])
def test_invalid_predictions_are_rejected(probability):
    event = [event for event in build_events(panel()) if event.eligible][0]
    with pytest.raises(ValueError, match="normalized prediction"):
        basic_loss(event, probability)


def test_shape_target_and_fit_population_are_validated():
    event = build_events(panel())[0]
    with pytest.raises(ValueError, match="features"):
        replace(event, x=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="exactly one"):
        basic_loss(event, np.full(4, .25))
    with pytest.raises(ValueError, match="nonfinal"):
        fit([event], .01)
    eligible = [item for item in build_events(panel()) if item.eligible]
    for ridge in (-1., np.nan):
        with pytest.raises(ValueError, match="ridge"):
            fit(eligible, ridge)
    with pytest.raises(ValueError, match="dimension"):
        objective(np.ones(4), eligible, .01, 4)


def test_withdrawals_multiexits_and_finals_do_not_enter_fit_or_evaluation():
    frame = panel()
    frame.loc[frame.week.eq(2) & frame.contestant_id.eq("D"), "withdrawal"] = True
    events = build_events(frame)
    assert not any(event.eligible for event in events)
    assert predictions(events, "uniform") == ([], [])
    frame = panel()
    frame.loc[frame.week.eq(2) & frame.contestant_id.eq("A"), "eliminated"] = True
    assert not any(event.eligible for event in build_events(frame))
