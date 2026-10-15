"""Enforce the forecast release clock and same-row categorical comparisons."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dwts.forecasting import (MODELS, aggregate_scores, forecast_season,
                              paired_bootstrap, read_protocol, weekly_scores)


def panel():
    rows = []
    for week, ids, exit_id in [(1, "ABCDE", "E"), (2, "ABCD", "D"),
                                (3, "ABC", "C"), (4, "AB", None)]:
        for i, identifier in enumerate(ids):
            rows.append(dict(season=21, week=week, contestant_id=identifier,
                judge_mean=float(5+i+week/2), eliminated=identifier == exit_id,
                withdrawal=False, is_final_week=week == 4, placement=1,
                result_label="future information"))
    return pd.DataFrame(rows)


def predict(frame):
    return forecast_season(frame, particles=512, seed=17, temperature=6., mh_steps=3)


def probabilities(rows, up_to=None):
    if up_to is not None:
        rows = rows[rows.week.le(up_to)]
    return rows.sort_values(["week", "model", "contestant_id"]).probability.to_numpy()


def test_future_scores_outcomes_rosters_and_metadata_cannot_change_earlier_predictions():
    original = panel()
    changed = original.copy()
    later = changed.week.gt(2)
    changed.loc[later, "judge_mean"] *= 100
    changed.loc[later, "eliminated"] = ~changed.loc[later, "eliminated"]
    changed.loc[later, "contestant_id"] = "NEW:"+changed.loc[later, "contestant_id"]
    changed.loc[later, "placement"] = 9000
    changed.loc[later, "result_label"] = "Eliminated Week 1"
    changed.loc[later, "withdrawal"] = True
    changed.loc[later, "is_final_week"] = True
    a, _ = predict(original)
    b, _ = predict(changed)
    np.testing.assert_array_equal(probabilities(a, 2), probabilities(b, 2))


def test_current_outcome_released_only_after_all_current_predictions():
    original, changed = panel(), panel()
    changed.loc[changed.week.eq(1), "eliminated"] = changed.loc[changed.week.eq(1), "contestant_id"].eq("A")
    a, _ = predict(original)
    b, _ = predict(changed)
    np.testing.assert_array_equal(probabilities(a, 1), probabilities(b, 1))
    assert not np.array_equal(probabilities(a, 2), probabilities(b, 2))


def test_final_placement_and_result_labels_never_affect_inference():
    original, changed = panel(), panel()
    changed["is_final_week"] = True
    changed["placement"] = np.arange(len(changed))
    changed["result_label"] = "Eliminated Week 99"
    a, da = predict(original)
    b, db = predict(changed)
    np.testing.assert_array_equal(probabilities(a), probabilities(b))
    pd.testing.assert_frame_equal(da, db)
    assert a.eligible.any() and not b.eligible.any()


def test_released_withdrawal_and_multiexit_skip_inference_and_survival_baseline_updates():
    withdrawn, multi = panel(), panel()
    withdrawn.loc[withdrawn.week.eq(1) & withdrawn.contestant_id.eq("E"), "withdrawal"] = True
    multi.loc[multi.week.eq(1) & multi.contestant_id.eq("A"), "eliminated"] = True
    a, da = predict(withdrawn)
    b, db = predict(multi)
    np.testing.assert_array_equal(probabilities(a), probabilities(b))
    pd.testing.assert_frame_equal(da, db)
    assert not a[a.week.eq(1)].updated.any()
    assert 1 not in da.week.tolist()


def test_current_withdrawal_cannot_change_current_probabilities_but_changes_future_state():
    original, changed = panel(), panel()
    changed.loc[changed.week.eq(1) & changed.contestant_id.eq("E"), "withdrawal"] = True
    a, _ = predict(original)
    b, _ = predict(changed)
    np.testing.assert_array_equal(probabilities(a, 1), probabilities(b, 1))
    assert not np.array_equal(probabilities(a, 2), probabilities(b, 2))


def test_baselines_and_forward_model_are_normalized_and_score_identical_weeks():
    rows, _ = predict(panel())
    assert set(rows.model) == set(MODELS)
    sums = rows.groupby(["week", "model"]).probability.sum()
    np.testing.assert_allclose(sums, 1., atol=1e-12, rtol=0)
    scores = weekly_scores(rows)
    assert scores.groupby("model").size().nunique() == 1
    assert scores.groupby("model").size().iloc[0] == 3
    metrics, calibration = aggregate_scores(scores)
    assert metrics.log_loss.gt(0).all()
    assert metrics.brier.between(0, 2).all()
    assert calibration.groupby("model").n_weeks.sum().eq(3).all()


def test_past_judge_baseline_uses_only_own_prior_scores_and_declared_cold_start():
    rows, _ = forecast_season(panel(), particles=64, past_judge_temperature=2.)
    week1 = panel().query("week == 1").sort_values("contestant_id")
    relative = len(week1)*week1.judge_mean.to_numpy()/week1.judge_mean.sum()
    expected = np.exp(-2*relative[:4])
    expected /= expected.sum()
    actual = rows.query("week == 2 and model == 'past_judges'").sort_values("contestant_id").probability
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_paired_season_bootstrap_rejects_mismatched_rows_and_preserves_sign():
    rows, _ = predict(panel())
    scores = weekly_scores(rows)
    other = scores.copy()
    other["season"] = 22
    scores = pd.concat((scores, other), ignore_index=True)
    actual = paired_bootstrap(scores, draws=100, seed=4)
    assert len(actual) == 12
    assert actual.lower.le(actual.difference+1e-12).all()
    assert actual.upper.ge(actual.difference-1e-12).all()
    broken = scores.drop(scores[scores.model.eq("uniform")].index[0])
    with pytest.raises(ValueError, match="same weeks"):
        paired_bootstrap(broken, draws=10)


def test_committed_forecast_protocol_is_fixed_and_valid():
    cfg = read_protocol(Path(__file__).resolve().parents[1]/"configs/forecast_study.json")
    assert max(cfg["development_seasons"]) < min(cfg["evaluation_seasons"])
    assert max(cfg["evaluation_seasons"]) == 27
    assert len(cfg["prior_sd_grid"])*len(cfg["temperature_grid"]) == 9
    assert cfg["mc_thresholds"]["absolute_log_loss_difference"] == .025
