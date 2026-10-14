"""Differential-CV controls use deterministic fake estimators, not real RF fits."""
import json
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

from dwts import attribution
from scripts import run_identity_cv as cv


class FakeForest:
    def __init__(self, kind, trees, seed, leaf):
        self.kind, self.seed = kind, seed

    def fit(self, x, y):
        self.mean = float(np.mean(y))
        self.coeff = np.arange(1, x.shape[1] + 1) * .01 + (self.seed % 17) * .0001
        self.feature_importances_ = self.coeff / self.coeff.sum()
        self.classes_ = np.unique(y)
        return self

    def _signal(self, x):
        return np.tanh(x @ self.coeff)

    def predict(self, x):
        if self.kind == "regression":
            return self.mean + .3 * self._signal(x)
        return (self.predict_proba(x)[:, -1] > .5).astype(int)

    def predict_proba(self, x):
        p = np.clip(.1 + .75 * self.mean + .1 * self._signal(x), .001, .999)
        return np.column_stack([1 - p, p])


class FakeExplainer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, x, check_additivity):
        assert check_additivity
        return x * self.model.coeff


def panel():
    data = []
    for season in range(1, 7):
        for week in (1, 2):
            for person in range(3):
                data.append(dict(season=season, week=week, contestant_id=f"S{season}:C{person}",
                                 partner=f"P{person}", industry="Acting", age=30. + person,
                                 judge_mean=5. + person + .1 * season + .2 * week,
                                 fan_mean=[.2, .3, .5][person], n_active=3))
    return pd.DataFrame(data)


def reference_run(d, out, protocol):
    attribution.rf_analysis(d, out, seed=protocol.seed, trees=protocol.trees, folds=protocol.folds,
                            history_grid=protocol.weights, inner_folds=protocol.inner_folds,
                            min_samples_leaf=protocol.min_samples_leaf, judge_target=protocol.judge_target,
                            fan_scale=protocol.fan_scale, with_shap=protocol.with_shap)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(attribution, "_forest", FakeForest)
    monkeypatch.setattr(cv, "_forest", FakeForest)
    monkeypatch.setitem(__import__("sys").modules, "shap", types.SimpleNamespace(TreeExplainer=FakeExplainer))
    p = cv.Protocol(folds=3, inner_folds=2, weights=(0., .5, 1.))
    d = panel()
    source = tmp_path / "baseline"
    reference_run(d, source, p)
    return d, cv.Baseline(source, d, p), p


def compare_result_tables(left, right):
    for name in ("rf_metrics", "rf_predictions", "feature_importance", "pcir", "history_weight_trials",
                 "history_weight_selection", "rf_fold_metrics"):
        a = pd.read_csv(left / f"{name}.csv", float_precision="round_trip")
        b = pd.read_csv(right / f"{name}.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=name)


def test_exact_design_guard_rejects_tiny_change_and_name_change():
    a = (np.array([[1., 2.]]), np.array([[3., 4.]]), np.array(["a", "b"], dtype=object))
    assert cv.compare_designs(a, tuple(x.copy() for x in a))["design_equal"]
    b = tuple(x.copy() for x in a)
    b[0][0, 0] = np.nextafter(1., 2.)
    assert not cv.compare_designs(a, b)["design_equal"]
    b = tuple(x.copy() for x in a)
    b[2][0] = "different"
    assert not cv.compare_designs(a, b)["design_equal"]


def test_unchanged_inputs_reuse_everything_without_fit(frozen, tmp_path, monkeypatch):
    d, baseline, p = frozen
    def forbidden(*args, **kwargs):
        raise AssertionError("No estimator may be constructed for identical inputs")
    monkeypatch.setattr(cv, "_forest", forbidden)
    plan = cv.build_plan(d, d.copy(), baseline, p)
    assert plan["inner"].design_equal.all()
    assert plan["outer_bounds"].selection_known_without_fitting.all()
    assert plan["outer_bounds"].outer_refits_max.sum() == 0
    out = tmp_path / "differential"
    out.mkdir()
    ledger = cv.execute_plan(d, d.copy(), baseline, plan, out, p)
    assert ledger.action.eq("reuse_baseline").all()
    compare_result_tables(baseline.directory, out)


def test_changed_inputs_match_full_original_protocol(frozen, tmp_path):
    d, baseline, p = frozen
    canonical = d.copy()
    canonical.loc[canonical.season.eq(4) & canonical.partner.eq("P0"), "partner"] = "Replacement"
    plan = cv.build_plan(d, canonical, baseline, p)
    assert (~plan["inner"].design_equal).any()
    assert plan["inner"].design_equal.any()
    out = tmp_path / "differential"
    out.mkdir()
    ledger = cv.execute_plan(d, canonical, baseline, plan, out, p)
    full = tmp_path / "full"
    reference_run(canonical, full, p)
    compare_result_tables(full, out)
    assert ledger.loc[ledger.stage.eq("inner"), "action"].eq("refit").sum() == (~plan["inner"].design_equal).sum()
    expected_seeds = p.seed + 1000 * (ledger.outer_fold + 1) + ledger.inner_fold
    inner = ledger.stage.eq("inner")
    np.testing.assert_array_equal(ledger.loc[inner, "seed"], expected_seeds[inner])
    np.testing.assert_array_equal(ledger.loc[~inner, "seed"], (p.seed + ledger.outer_fold)[~inner])


def test_changed_selection_forbids_outer_reuse_even_when_designs_match(frozen, tmp_path, monkeypatch):
    d, baseline, p = frozen
    plan = cv.build_plan(d, d.copy(), baseline, p)
    baseline_weights = set(x["history_weight"] for x in baseline.selections.values())
    forced = next((w for w in p.weights if w not in baseline_weights), None)
    assert forced is not None
    monkeypatch.setattr(cv, "choose_weight", lambda trials: forced)
    out = tmp_path / "forced_guard_test"
    out.mkdir()
    ledger = cv.execute_plan(d, d.copy(), baseline, plan, out, p)
    outer = ledger.loc[ledger.stage.eq("outer")]
    assert outer.design_equal.all()
    assert outer.action.eq("refit").all()
    assert outer.reason.eq("selected_weight_changed").all()


def test_changed_target_rejected_before_planning(frozen):
    d, baseline, p = frozen
    changed = d.copy()
    changed.loc[0, "judge_mean"] += .01
    with pytest.raises(ValueError, match="Target changed"):
        cv.build_plan(d, changed, baseline, p)


def test_tie_break_smallest_weight():
    trials = [dict(history_weight=w, squared_error_sum=2., validation_n=3) for w in (1., .5, 0.)]
    assert cv.choose_weight(trials) == 0.


def test_nonempty_output_fails_before_reading_inputs(tmp_path):
    out = tmp_path / "used"
    out.mkdir()
    (out / "preserve").write_text("existing result")
    with pytest.raises(ValueError, match="new or empty"):
        cv.run("does-not-exist", "does-not-exist", out)
    assert (out / "preserve").read_text() == "existing result"


def test_canonical_roundtrip_exact_target_and_provenance_guards(tmp_path):
    raw = panel()
    raw.loc[0, "fan_mean"] = 0.12345678901234567
    source = tmp_path / "source"
    source.mkdir()
    raw.to_csv(source / "votes.csv", index=False)
    raw = pd.read_csv(source / "votes.csv")
    canonical = raw.copy()
    canonical["partner_raw"] = raw.partner
    canonical["partner_guest_annotation"] = ""
    canonical["partner_annotation_week"] = np.nan
    canonical["partner_annotation_this_week"] = False
    canonical["partner_identity_changed"] = False
    canonical["partner_id_policy"] = cv.POLICY
    path = tmp_path / "canonical.csv"
    manifest = tmp_path / "manifest.json"
    def write():
        canonical.to_csv(path, index=False)
        manifest.write_text(json.dumps(dict(source_sha256=cv.digest(source / "votes.csv"), canonical_votes_sha256=cv.digest(path),
                                           policy=cv.POLICY, module_sha256=cv.digest(cv.ROOT / "dwts/partner_identity.py"))))
    write()
    a, b, _ = cv.load_identity(source, path, manifest)
    pd.testing.assert_frame_equal(a, b[a.columns], check_exact=True)
    canonical.loc[0, "fan_mean"] = np.nextafter(canonical.loc[0, "fan_mean"], 1.)
    write()
    with pytest.raises(AssertionError):
        cv.load_identity(source, path, manifest)


def test_saved_loss_must_justify_saved_weight(frozen):
    d, baseline, p = frozen
    path = baseline.directory / "history_weight_selection.csv"
    selection = pd.read_csv(path)
    selection.loc[0, "history_weight"] = next(w for w in p.weights if w != selection.loc[0, "history_weight"])
    selection.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not minimize"):
        cv.Baseline(baseline.directory, d, p)


@pytest.mark.parametrize("subdirectory", ["baseline/new_run", "identity/new_run"])
def test_output_cannot_nest_inside_input_directory(tmp_path, subdirectory):
    with pytest.raises(ValueError, match="separate"):
        cv.run(tmp_path / "baseline", tmp_path / "identity/votes.csv", tmp_path / subdirectory)


@pytest.mark.parametrize("table,column", [("rf_fold_metrics", "rmse"), ("pcir", "pcir"), ("rf_metrics", "r2")])
def test_saved_metrics_must_match_underlying_saved_results(frozen, table, column):
    d, baseline, p = frozen
    path = baseline.directory / f"{table}.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, column] += 1.
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent"):
        cv.Baseline(baseline.directory, d, p)
