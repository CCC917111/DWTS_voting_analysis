"""End-to-end clock and publication checks using invented scores and exits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dwts.exit_study import default_config, main, read_config, run_study, synthetic_panel, PUBLIC_FILES
from scripts.package_repository import selected_files


@pytest.fixture(scope="module")
def experiment():
    cfg = default_config()
    cfg["bootstrap_draws"] = 100
    return cfg, run_study(synthetic_panel(), cfg)


def test_every_model_scores_the_same_events_and_complete_roster(experiment):
    _, (tables, selection, fits) = experiment
    scores, probabilities = tables["private_event_scores"], tables["private_predictions"]
    assert len(fits) == 21
    assert len(selection["online"]) == 8
    keys = [set(zip(g.phase, g.season, g.week)) for _, g in scores.groupby("model")]
    assert all(k == keys[0] for k in keys)
    assert set(scores.model) == {"past_judges", "current_judges", "choice_two", "uniform",
                                 "online_bayes", "selected_fixed_share", "selected_tempered"}
    grouped = probabilities.groupby(["phase", "season", "week", "model"])
    np.testing.assert_allclose(grouped.probability.sum(), 1., atol=1e-12)
    assert grouped.observed_exit.sum().eq(1).all()
    for item in selection["online"]:
        assert all(end < item["train_end"] for end in item["inner_train_ends"])
        assert item["selected"]["n_events"] > 0


def test_legacy_labels_cannot_change_any_training_or_selection(experiment):
    cfg, (before, selection, _) = experiment
    panel = synthetic_panel()
    for (_, _), g in panel[panel.season.ge(21)].groupby(["season", "week"]):
        old = int(np.flatnonzero(g.eliminated)[0])
        panel.loc[g.index, "eliminated"] = False
        panel.loc[g.index[(old+1) % len(g)], "eliminated"] = True
    after, changed_selection, _ = run_study(panel, cfg)
    assert selection == changed_selection
    left = before["private_event_scores"]
    right = after["private_event_scores"]
    pd.testing.assert_frame_equal(left[left.phase.eq("nested_development")].reset_index(drop=True),
                                  right[right.phase.eq("nested_development")].reset_index(drop=True))
    assert not left[left.phase.eq("legacy_exploratory")].equals(right[right.phase.eq("legacy_exploratory")])


@pytest.mark.parametrize("key,value", [("initial_weights", [.4,.2,.2,.2]),
    ("share_gamma_grid", [0,.9]), ("bootstrap_draws", 0), ("tie_tolerance", float("nan")),
    ("ridge_grid", []), ("temperature_grid", [0]), ("startup_ridge", -1)])
def test_invalid_configuration_rejected(tmp_path, key, value):
    cfg = default_config(); cfg[key] = value
    path = tmp_path/"config.json"; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        read_config(path)


def test_existing_output_is_rejected_before_work(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["exit_study", "--demo", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and not list(tmp_path.iterdir())


def test_release_includes_only_approved_aggregates(tmp_path):
    folder = tmp_path/"results/exit_forecast_study"
    folder.mkdir(parents=True)
    for name in [*PUBLIC_FILES, "private_predictions.csv", "private_event_scores.csv",
                 "private_weights.csv", "posterior.nc"]:
        (folder/name).write_text("fixture")
    published = {p.name for p in selected_files(tmp_path)}
    assert published == set(PUBLIC_FILES)
