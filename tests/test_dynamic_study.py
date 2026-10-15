"""Fair-clock and common-synthetic-observation checks."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dwts.dynamic_study import (LATENT_MODELS, fit_synthetic, forecast_latent_season,
    paired_differences, read_protocol, synthetic_season, weighted_quantile)
from dwts.forecasting import weekly_scores

ROOT = Path(__file__).resolve().parents[1]


def panel():
    rows = []
    for week, ids, eliminated in [(1, "ABCDE", "E"), (2, "ABCD", "D"),
                                   (3, "ABC", "C"), (4, "AB", None)]:
        for i, identifier in enumerate(ids):
            rows.append(dict(season=21, week=week, contestant_id=identifier,
                judge_mean=5.+i+week/2, eliminated=identifier == eliminated,
                withdrawal=False, is_final_week=week == 4, placement=1))
    return pd.DataFrame(rows)


def predict(frame):
    selected = {model: dict(prior_sd=1., temperature=3., process_sd=.25 if
        model == "dynamic_bootstrap" else 0., drift=1. if model == "dynamic_bootstrap" else 0.)
        for model in LATENT_MODELS}
    return forecast_latent_season(frame, selected, dict(judge_weight=.5, ess_fraction=.7),
                                 particles=256, seed=19)


def values(rows, week):
    return rows[rows.week.le(week)].sort_values(["week", "model", "contestant_id"]).probability.to_numpy()


def test_outcome_current_withdrawal_and_future_rosters_never_enter_current_prediction():
    original, changed = panel(), panel()
    changed.loc[changed.week.eq(1), "eliminated"] = changed.loc[changed.week.eq(1), "contestant_id"].eq("A")
    changed.loc[changed.week.eq(1), "withdrawal"] = True
    changed.loc[changed.week.gt(1), "contestant_id"] = "FUTURE:"+changed.loc[changed.week.gt(1), "contestant_id"]
    changed.loc[changed.week.gt(1), "judge_mean"] *= 100
    a, _ = predict(original)
    b, _ = predict(changed)
    np.testing.assert_array_equal(values(a, 1), values(b, 1))


def test_final_flags_and_placement_only_filter_scoring():
    changed = panel()
    changed["is_final_week"], changed["placement"] = True, 999
    a, da = predict(panel())
    b, db = predict(changed)
    np.testing.assert_array_equal(values(a, 4), values(b, 4))
    pd.testing.assert_frame_equal(da, db)
    assert not b.eligible.any()
    assert weekly_scores(a).groupby("model").size().eq(3).all()


def test_both_models_score_identical_weeks_and_paired_cluster_bootstrap():
    a, _ = predict(panel())
    scores = weekly_scores(a)
    duplicate = scores.copy()
    duplicate["season"] = 22
    pairs = paired_differences(pd.concat([scores, duplicate]), draws=100)
    assert len(pairs) == 3
    assert pairs.lower.le(pairs.difference+1e-12).all()
    assert pairs.upper.ge(pairs.difference-1e-12).all()
    broken = pd.concat([scores, duplicate]).iloc[1:]
    with pytest.raises(ValueError, match="same scored weeks"):
        paired_differences(broken, draws=10)


def test_committed_protocol_has_equal_candidate_grid_and_fixed_transition():
    cfg = read_protocol(ROOT/"configs/dynamic_study.json")
    assert len(cfg["prior_sd_grid"])*len(cfg["temperature_grid"]) == 9
    assert cfg["development_seasons"][-1] < cfg["evaluation_seasons"][0]
    assert cfg["dynamic_process_sd"] == .25
    assert cfg["baseline_settings"]["past_judge_temperature"] == 12.


def test_shared_synthetic_stream_is_reproducible_and_filtering_metrics_are_valid():
    settings = read_protocol(ROOT/"configs/dynamic_study.json")["synthetic"]
    for condition in ("static", "drift", "shock"):
        events = synthetic_season(condition, 0., 0, settings)
        again = synthetic_season(condition, 0., 0, settings)
        for a, b in zip(events, again):
            np.testing.assert_array_equal(a["truth"], b["truth"])
            assert a["outcome"] == b["outcome"]
        rows = fit_synthetic(events, settings, particles=128, seed=5)
        assert set(rows.model) == set(LATENT_MODELS)
        assert rows.log_loss.gt(0).all()
        assert rows.coverage.between(0, 1).all()
        assert rows.interval_width.between(0, 1).all()
        assert rows.rank_concordance.between(0, 1).all()
        # Truth is not modified by either fit, nor are synthetic exits regenerated.
        for a, b in zip(events, again):
            np.testing.assert_array_equal(a["truth"], b["truth"])


def test_weighted_quantile_has_known_discrete_cdf_answer():
    values = np.array([[0., 3.], [1., 2.], [2., 1.]])
    np.testing.assert_array_equal(weighted_quantile(values, np.array([.1, .7, .2]),
                                                   probabilities=(.2, .9)), [[1., 1.], [2., 2.]])


def test_development_selection_has_equal_budget_and_never_requests_evaluation(monkeypatch):
    import dwts.dynamic_study as study
    cfg = read_protocol(ROOT/"configs/dynamic_study.json")
    cfg["prior_sd_grid"], cfg["temperature_grid"] = [.5, 1.], [3.]
    calls = []
    def fake_population(panel, seasons, selected, cfg, particles, seed, baselines):
        calls.append((tuple(seasons), particles, seed))
        rows = []
        for model in LATENT_MODELS:
            for name, exited in [("A", True), ("B", False)]:
                rows.append(dict(season=3, week=1, contestant_id=name, model=model,
                    probability=.5, observed_exit=exited, eligible=True, updated=True))
        return pd.DataFrame(rows), pd.DataFrame()
    monkeypatch.setattr(study, "run_population", fake_population)
    selected, trials = study.choose_development(panel(), cfg)
    assert len(calls) == 2*2*len(cfg["seeds"])
    assert all(x[0] == tuple(range(3, 21)) for x in calls)
    assert trials.groupby("model").size().eq(2).all()
    assert trials.development_repeats.eq(len(cfg["seeds"])).all()
    # A genuine tie uses declared candidate order, never evaluation performance.
    assert all(candidate["prior_sd"] == .5 for candidate in selected.values())


def test_future_score_changes_and_new_identities_cannot_change_two_earlier_weeks():
    original, changed = panel(), panel()
    later = changed.week.gt(2)
    changed.loc[later, "contestant_id"] = "UNKNOWN:"+changed.loc[later, "contestant_id"]
    changed.loc[later, "judge_mean"] *= 200
    changed.loc[later, "eliminated"] = ~changed.loc[later, "eliminated"]
    a, _ = predict(original)
    b, _ = predict(changed)
    np.testing.assert_array_equal(values(a, 2), values(b, 2))
