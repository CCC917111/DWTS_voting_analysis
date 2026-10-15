"""Evidence checks for nested, complete-roster retrospective benchmarks."""
import numpy as np
import pandas as pd
import pytest

from dwts import benchmarks as bm


def panel(seasons=6):
    rows = []
    for season in range(1, seasons+1):
        for week in range(1, 4):
            share = np.array([.12+season*.002+week*.02, .28-week*.01,
                              .31+season*.001, .29-season*.003-week*.01])
            for i in range(4):
                rows.append(dict(season=season, week=week, contestant_id=f"S{season}:C{i}",
                    fan_mean=share[i], n_active=4, judge_mean=5+i*.7+week*.2,
                    age=22+i*9, partner=f"P{i}", industry=["Acting", "Music", "Sports", "Other"][i]))
    return pd.DataFrame(rows)


def test_normalization_is_by_complete_season_week_without_reading_targets():
    d = panel(2)
    prediction = np.arange(1, len(d)+1, dtype=float)
    result = bm.normalize_weekly(d, prediction)
    assert np.allclose(pd.Series(result).groupby([d.season, d.week]).sum(), 1)
    changed = d.copy()
    changed["fan_mean"] = np.nan
    assert np.array_equal(bm.normalize_weekly(changed, prediction), result)
    with pytest.raises(ValueError, match="complete"):
        bm.normalize_weekly(d.iloc[:-1], prediction[:-1])
    with pytest.raises(ValueError, match="zero-total"):
        bm.normalize_weekly(d, np.zeros(len(d)))


def test_baselines_use_only_training_mean_and_own_earlier_weeks():
    d = panel(2)
    train, test = np.flatnonzero(d.season.eq(1)), np.flatnonzero(d.season.eq(2))
    before = bm.baseline_predictions(d, train, test)
    changed = d.copy()
    changed.loc[(changed.season.eq(2)) & changed.week.ge(2), "fan_mean"] += 10
    after = bm.baseline_predictions(changed, train, test)
    # First two weeks' inputs contain no week-two or future target values.
    first_two = d.iloc[test].week.le(2).to_numpy()
    for name in bm.BASELINES:
        assert np.array_equal(before[name][first_two], after[name][first_two])
    assert np.all(before["training_mean"] == d.iloc[train].fan_mean.mean())
    assert np.all(before["last_share"][:4] == .25)
    assert np.allclose(before["last_share"][4:8], d[(d.season.eq(2)) & d.week.eq(1)].fan_mean)
    assert np.allclose(before["history_mean_share"][8:12],
        d[(d.season.eq(2)) & d.week.le(2)].groupby("contestant_id").fan_mean.mean())


def test_naive_normalization_is_a_fair_complete_roster_control():
    d = panel(2)
    # Remove exited competitors from later weeks. Raw past shares then lose mass;
    # every normalized baseline must apply the same active-roster constraint as RF.
    d = d[~((d.week.ge(2) & d.contestant_id.str.endswith("C3")) |
            (d.week.ge(3) & d.contestant_id.str.endswith("C2")))].reset_index(drop=True)
    d["n_active"] = d.groupby(["season", "week"]).season.transform("size")
    d["fan_mean"] /= d.groupby(["season", "week"]).fan_mean.transform("sum")
    train, test = np.flatnonzero(d.season.eq(1)), np.flatnonzero(d.season.eq(2))
    values = bm.baseline_predictions(d, train, test)
    for name in ["last_share", "history_mean_share"]:
        weekly = pd.Series(values[name]).groupby(d.iloc[test].week.to_numpy()).sum()
        normalized = pd.Series(values[name+"_normalized"]).groupby(d.iloc[test].week.to_numpy()).sum()
        assert weekly.iloc[1] < 1
        assert np.allclose(normalized, 1)


def test_raw_and_normalized_select_their_own_inner_losses(monkeypatch):
    d = panel(6)
    # Weight 0 is perfectly proportional but poorly scaled; weight 1 has lower
    # raw error but a deliberately reversed ordering after normalization.
    def fake_predict(rows, train, test, weight, feature_set, protocol, seed, cache):
        actual = rows.iloc[test].fan_mean.to_numpy()
        return actual*2 if weight == 0 else .25+.01*(.25-actual)
    monkeypatch.setattr(bm, "_predict", fake_predict)
    protocol = bm.Protocol(trees=2, outer_folds=3, inner_folds=2,
                           history_grid=(0., 1.), bootstrap_draws=10)
    train = np.flatnonzero(d.season.le(4))
    chosen, trials = bm.select_weights(d, train, protocol, 0)
    assert chosen == {"raw": 1., "normalized": 0.}
    assert set(trials.method) == {"raw", "normalized"}
    for row in trials.itertuples():
        assert set(row.train_seasons.split(",")).isdisjoint(row.validation_seasons.split(","))
        assert not ({"5", "6"} & set(row.train_seasons.split(",")))


def test_fresh_nested_fits_keep_seasons_separate_and_cover_each_row(monkeypatch):
    d = panel(6)
    original = bm._designs
    calls = []
    def checked_designs(rows, train, test, *args, **kwargs):
        assert set(rows.iloc[train].season).isdisjoint(rows.iloc[test].season)
        assert rows.iloc[test].groupby(["season", "week"]).size().eq(4).all()
        calls.append((tuple(train), tuple(test)))
        return original(rows, train, test, *args, **kwargs)
    monkeypatch.setattr(bm, "_designs", checked_designs)
    protocol = bm.Protocol(trees=3, outer_folds=3, inner_folds=2,
                           history_grid=(0., 1.), min_samples_leaf=1, bootstrap_draws=20)
    result = bm.evaluate(d, protocol)
    assert calls
    oof = result["oof_predictions"]
    assert len(oof) == len(d)*len(bm.MODELS)
    assert not oof.duplicated(["model"]+bm.KEYS).any()
    assert oof.groupby("season").fold.nunique().eq(1).all()
    normalized = oof[oof.model.str.endswith("_normalized")]
    assert np.allclose(normalized.groupby(["model", "season", "week"]).prediction.sum(), 1)
    assert set(result["metrics"].model) == set(bm.MODELS)
    selection = result["weight_selection"]
    assert selection[selection.model.str.contains("pair_history")].history_weight.eq(0).all()


def test_bootstrap_resamples_whole_paired_seasons_deterministically():
    d = panel(4)
    y = d.fan_mean.to_numpy()
    predictions = {"rf_tuned_raw": y+.03, "rf_tuned_normalized": y.copy()}
    a = bm.paired_season_bootstrap(d, predictions, 100, 123)
    b = bm.paired_season_bootstrap(d, predictions, 100, 123)
    pd.testing.assert_frame_equal(a, b)
    rmse = a[a.metric.eq("rmse")].iloc[0]
    assert rmse.difference == pytest.approx(-.03)
    assert rmse.lower_95 == pytest.approx(-.03)
    assert rmse.upper_95 == pytest.approx(-.03)
    assert rmse.season_clusters == 4
    assert a.resampling_unit.eq("season").all()
    with pytest.raises(ValueError, match="two seasons"):
        bm.paired_season_bootstrap(d[d.season.eq(1)], {"rf_tuned_raw": y[:12]}, 10, 123)


def test_invalid_protocol_and_partial_panel_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="inner_folds"):
        bm.Protocol(inner_folds=1)
    with pytest.raises(ValueError, match="history_grid"):
        bm.Protocol(history_grid=(.5, 0.))
    config = tmp_path/"config.json"
    config.write_text('{"paper_target_rmse": 0.01}')
    with pytest.raises(ValueError, match="Unknown"):
        bm.read_protocol(config)
    with pytest.raises(ValueError, match="complete"):
        bm.validate_panel(panel().iloc[:-1])
    source, out = tmp_path/"source", tmp_path/"out"
    source.mkdir()
    out.mkdir()
    with pytest.raises(ValueError, match="never overwrite"):
        bm.run(source, config, out, tmp_path/"data.csv")
