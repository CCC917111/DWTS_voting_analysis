import numpy as np
import pandas as pd
import pytest

from dwts.report_ols import literal_partner_score, fit_literal_ols


def small_panel():
    rows = []
    for season in [1, 2, 3]:
        for week, share in enumerate([.2, .3, .4], start=1):
            for person in [0, 1]:
                q = share if person == 0 else 1-share
                rows.append(dict(season=season, week=week, contestant_id=str(person),
                    judge_share=q, n_active=2, age=20+person*10+season,
                    judge_mean=4+q*5+season/10, fan_mean=q,
                    placement=1+person, industry="Acting" if person == 0 else "Sports"))
    return pd.DataFrame(rows)


def test_literal_q_mean_not_relative_rf_feature():
    d = small_panel()
    f = literal_partner_score(d)
    g = f[(f.season == 1) & (f.contestant_id == "0")]
    np.testing.assert_allclose(g.partner_score_literal, [np.nan, .2, .25], equal_nan=True)
    assert (~f.feature_eligible).sum() == 6
    neutral = literal_partner_score(d, "neutral_share")
    np.testing.assert_allclose(neutral.loc[neutral.week == 1, "partner_score_literal"], .5)
    assert neutral.feature_eligible.all()


def test_literal_feature_is_past_only_and_contestant_season_local():
    d = small_panel()
    before = literal_partner_score(d)
    changed = d.copy()
    changed.loc[(changed.season == 2) & (changed.week >= 2), "judge_share"] = .9
    changed.loc[changed.season == 1, "judge_share"] = .8
    changed["pq_judge"] = 1000
    changed["history_weight"] = 1
    after = literal_partner_score(changed)
    unaffected = (d.season == 3) | ((d.season == 2) & (d.week <= 2))
    np.testing.assert_allclose(before.loc[unaffected, "partner_score_literal"],
                               after.loc[unaffected, "partner_score_literal"], equal_nan=True)


def test_literal_feature_rejects_missing_calendar_weeks_and_duplicates():
    d = small_panel()
    with pytest.raises(ValueError, match="contiguous"):
        literal_partner_score(d[d.week != 2])
    with pytest.raises(ValueError, match="Duplicate"):
        literal_partner_score(pd.concat([d, d.iloc[[0]]], ignore_index=True))


def test_covariance_supplement_keeps_point_estimates_and_units_explicit():
    rng = np.random.default_rng(6021)
    d = []
    industries = ["Acting", "Sports", "RealityTV", "Music", "Model", "Other"]
    for season in range(1, 7):
        ages = rng.integers(20, 65, 12)
        for week in range(1, 5):
            shares = rng.dirichlet(np.ones(12))
            for person in range(12):
                d.append(dict(season=season, week=week, contestant_id=str(person),
                    judge_share=shares[person], n_active=12, age=ages[person],
                    judge_mean=7+rng.normal(), fan_mean=shares[person],
                    placement=person+1, industry=industries[person % 6]))
    f = literal_partner_score(pd.DataFrame(d))
    coefs, rows, norm = fit_literal_ols(f, "judge_mean")
    a = coefs[coefs.covariance == "conventional_OLS"]
    b = coefs[coefs.covariance == "season_cluster_t"]
    np.testing.assert_allclose(a.coefficient, b.coefficient)
    assert len(rows) == 216
    assert coefs.design_rank.eq(8).all()
    assert np.isfinite(coefs[["coefficient", "lower_95", "upper_95", "pvalue"]]).all().all()
    assert set(norm.ddof) == {1}
    _, placement, _ = fit_literal_ols(f, "placement", "contestant_season")
    assert len(placement) == 72
    assert placement.source_week_count.eq(3).all()
    assert placement.first_source_week.eq(2).all()
