import numpy as np
import pandas as pd
import pytest

from dwts.ols_unit_audit import (CANDIDATES, NEW_CANDIDATES, score_unit_features,
    graphic_width_ratio_bounds, standardization_invariance_audit)
from dwts.report_ols import fit_literal_ols


def panel():
    rng = np.random.default_rng(61519)
    rows = []
    industries = ["Acting", "Sports", "RealityTV", "Music", "Model", "Other"]
    for season in range(1, 7):
        ages = rng.integers(20, 65, 12)
        for week in range(1, 5):
            scores = 5+rng.uniform(0, 4, 12)
            fan = rng.dirichlet(np.ones(12))
            for person in range(12):
                rows.append(dict(season=season, week=week, contestant_id=str(person),
                    judge_mean=scores[person], n_active=12, age=ages[person],
                    fan_mean=fan[person], placement=person+1,
                    industry=industries[person % 6]))
    return normalized(pd.DataFrame(rows))


def normalized(d):
    d = d.copy()
    group = d.groupby(["season", "week"]).judge_mean
    d["judge_share"] = d.judge_mean/group.transform("sum")
    d["judge_z"] = (d.judge_mean-group.transform("mean"))/group.transform(lambda x: x.std(ddof=0)).replace(0, 1)
    return d


def test_closed_candidate_set_and_exact_history_values():
    d = panel()
    f, weeks = score_unit_features(d)
    assert CANDIDATES == ("lagged_judge_share", "lagged_raw_judge_mean",
                          "lagged_weekly_relative_mean", "lagged_within_week_z")
    assert NEW_CANDIDATES == CANDIDATES[1:]
    assert len(weeks) == 24
    assert f.loc[f.week == 1, list(CANDIDATES)].isna().all().all()
    assert f.loc[f.week > 1, list(CANDIDATES)].notna().all().all()
    source_columns = ["judge_share", "judge_mean", "judge_share", "judge_z"]
    g = d[(d.season == 1) & (d.contestant_id == "0")].sort_values("week")
    for candidate, source in zip(CANDIDATES, source_columns):
        actual = f[(f.season == 1) & (f.contestant_id == "0")].sort_values("week")[candidate]
        values = g[source] * (g.n_active if candidate == "lagged_weekly_relative_mean" else 1)
        np.testing.assert_allclose(actual, [np.nan, values.iloc[0], values.iloc[:2].mean(),
                                           values.iloc[:3].mean()], equal_nan=True)


def test_score_quantities_are_past_only_and_season_local():
    d = panel()
    before, _ = score_unit_features(d)
    changed = d.copy()
    changed.loc[(changed.season == 2) & (changed.week >= 2), "judge_mean"] *= 1.5
    changed.loc[changed.season == 1, "judge_mean"] += 3
    after, _ = score_unit_features(normalized(changed))
    unaffected = (d.season >= 3) | ((d.season == 2) & (d.week <= 2))
    np.testing.assert_allclose(before.loc[unaffected, list(CANDIDATES)],
                               after.loc[unaffected, list(CANDIDATES)], equal_nan=True)


@pytest.mark.parametrize("bad_column,message", [("judge_share", "Frozen Q"),
                                                ("judge_z", "Frozen z"),
                                                ("n_active", "n_active")])
def test_frozen_quantities_cannot_silently_change(bad_column, message):
    d = panel()
    d[bad_column] = d[bad_column].astype(float)
    d.loc[0, bad_column] += .001
    with pytest.raises(ValueError, match=message):
        score_unit_features(d)


def test_features_reject_missing_weeks():
    with pytest.raises(ValueError, match="contiguous"):
        score_unit_features(panel().query("week != 2"))


def test_positive_affine_scaling_does_not_change_standardized_model():
    f, _ = score_unit_features(panel())
    before, before_rows, _ = fit_literal_ols(f, "judge_mean")
    changed = f.copy()
    changed["partner_score_literal"] = 17+100*changed.partner_score_literal
    changed["judge_mean"] = 13+3*changed.judge_mean
    after, after_rows, _ = fit_literal_ols(changed, "judge_mean")
    np.testing.assert_allclose(before[["coefficient", "lower_95", "upper_95"]],
                               after[["coefficient", "lower_95", "upper_95"]], atol=1e-12)
    pd.testing.assert_frame_equal(before_rows[["season", "contestant_id", "week"]],
                                  after_rows[["season", "contestant_id", "week"]])
    checks = standardization_invariance_audit(f)
    assert checks.maximum_absolute_design_difference.max() < 1e-12
    assert not checks.extra_ols_fit.any()


def test_graphic_width_ratios_keep_uncertainty_and_shared_design_condition():
    rows = []
    for term, width in [("a", .1), ("b", .2)]:
        for outcome, ratio in [("Judge Score", 1), ("Fan Votes", 2), ("Placement (1=Best)", 1.5)]:
            rows.append(dict(outcome=outcome, term=term, ci95_low_approx=-width*ratio,
                             ci95_high_approx=width*ratio, absolute_graphic_tolerance=.01))
    bounds = graphic_width_ratio_bounds(pd.DataFrame(rows))
    assert bounds.graphically_feasible_common_ratio.all()
    for target, expected in [("fan_mean", 2), ("placement", 1.5)]:
        b = bounds[bounds.target == target].iloc[0]
        assert b.all_terms_common_ratio_lower < expected < b.all_terms_common_ratio_upper


def test_weekly_population_scaling_is_not_a_global_unit_conversion():
    d = panel()
    # Retain contiguous histories but change the weekly cast size.
    d = d[d.contestant_id.astype(int) >= d.week-1].copy()
    d["n_active"] = d.groupby(["season", "week"]).judge_mean.transform("size")
    f, _ = score_unit_features(normalized(d))
    eligible = f[f.feature_eligible]
    assert eligible.lagged_judge_share.corr(eligible.lagged_weekly_relative_mean) < .99
    assert np.isfinite(eligible[list(CANDIDATES)]).all().all()


def test_saved_audit_has_complete_fixed_comparison_coverage():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = root/"outputs/v5_ols_units"
    if not (out/"manifest.json").is_file():
        pytest.skip("Saved v5 result not present; run the fixed audit separately")
    coefficients = pd.read_csv(out/"ols_coefficients.csv")
    comparisons = pd.read_csv(out/"all_figure5_triplet_comparisons.csv")
    assert len(coefficients) == 4*3*8*2
    assert len(comparisons) == 4*3*7*3*2
    assert coefficients.n.eq(2356).all()
    assert coefficients.design_rank.eq(8).all()
    assert not comparisons.exact_author_match_claim.any()
    assert not comparisons.protocol_selected.any()
    assert comparisons.groupby(["candidate", "covariance", "metric"]).size().eq(21).all()
    baseline = pd.read_csv(root/"outputs/report_ols_literal/ols_coefficients.csv")
    baseline = baseline[baseline.branch == "literal_drop_contestant_week"].reset_index(drop=True)
    reused = coefficients[coefficients.candidate == "lagged_judge_share"].reset_index(drop=True)
    np.testing.assert_array_equal(reused[["coefficient", "lower_95", "upper_95"]],
                                   baseline[["coefficient", "lower_95", "upper_95"]])
