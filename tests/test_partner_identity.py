import numpy as np
import pandas as pd
import pytest

from dwts.features import partner_features
from dwts.partner_identity import canonical_primary_partners, PARTNER_ANNOTATIONS, feature_endpoints_fast


def rows():
    return pd.DataFrame([
        dict(season=33, week=1, partner="Alan Bersten", judge_mean=8., fan_mean=.6, n_active=2),
        dict(season=33, week=1, partner="Other Pro", judge_mean=6., fan_mean=.4, n_active=2),
        dict(season=34, week=1, partner="Alan Bersten (Rashad Jennings week 9)", judge_mean=7., fan_mean=.7, n_active=2),
        dict(season=34, week=1, partner="Other Pro", judge_mean=5., fan_mean=.3, n_active=2),
        dict(season=34, week=9, partner="Alan Bersten (Rashad Jennings week 9)", judge_mean=9., fan_mean=.8, n_active=2),
        dict(season=34, week=9, partner="Other Pro", judge_mean=8., fan_mean=.2, n_active=2),
    ])


def test_primary_mapping_preserves_raw_strings_outcomes_order_and_week_flag():
    d = rows()
    snapshot = d.copy(deep=True)
    result = canonical_primary_partners(d)
    pd.testing.assert_frame_equal(d, snapshot)
    pd.testing.assert_frame_equal(d.drop(columns="partner"), result[d.columns].drop(columns="partner"))
    assert result.partner_raw.tolist() == d.partner.tolist()
    assert result.partner.tolist() == ["Alan Bersten", "Other Pro"] * 3
    assert result.partner_annotation_this_week.tolist() == [False, False, False, False, True, False]
    assert result.loc[4, "partner_guest_annotation"] == "Rashad Jennings"


def test_all_six_reviewed_labels_and_aliases_left_distinct():
    names = list(PARTNER_ANNOTATIONS) + ["Val Chmerkovskiy", "Valentin Chmerkovskiy", "Kym Johnson", "Kym Herjavec"]
    d = pd.DataFrame({"season": 34, "week": 1, "partner": names})
    result = canonical_primary_partners(d)
    assert result.partner.tolist()[:6] == [v[0] for v in PARTNER_ANNOTATIONS.values()]
    assert result.partner.tolist()[6:] == names[6:]
    assert result.partner_identity_changed.sum() == 6


def test_unreviewed_annotation_and_wrong_season_fail_closed():
    d = rows()
    d.loc[2, "partner"] = "Alan Bersten (Unknown Guest week 9)"
    with pytest.raises(ValueError, match="Unreviewed"):
        canonical_primary_partners(d)
    d = rows()
    d.loc[2, "season"] = 32
    with pytest.raises(ValueError, match="season 34"):
        canonical_primary_partners(d)


def test_reapplication_cannot_erase_provenance():
    with pytest.raises(ValueError, match="provenance"):
        canonical_primary_partners(canonical_primary_partners(rows()))


def test_verified_alias_stage_is_explicit_and_preserves_both_names():
    d = pd.DataFrame({"season": [33, 34, 24], "week": [1, 9, 1],
                      "partner": ["Val Chmerkovskiy", "Val Chmerkovskiy (Joey Graziadei week 9)", "Kym Herjavec"]})
    result = canonical_primary_partners(d, include_verified_aliases=True)
    assert result.partner.tolist() == ["Valentin Chmerkovskiy", "Valentin Chmerkovskiy", "Kym Johnson"]
    assert result.partner_raw.tolist() == d.partner.tolist()
    assert result.partner_annotation_this_week.tolist() == [False, True, False]


def test_fast_audit_reference_matches_feature_endpoints():
    for d in (rows(), canonical_primary_partners(rows())):
        for target, reference in ((d, d), (d[d.season.eq(34)], d[d.season.eq(33)])):
            zero, one = feature_endpoints_fast(target, reference)
            for weight in (0., .5, 1.):
                actual = partner_features(target, reference=reference, history_weight=weight)
                np.testing.assert_allclose(actual[["pq_judge", "pq_vote"]],
                                           (1-weight)*zero + weight*one, rtol=0, atol=1e-12)


def test_identity_restores_prior_only_when_history_weight_nonzero():
    d = rows()
    canonical = canonical_primary_partners(d)
    for weight in (0., .5, 1.):
        a = partner_features(d, history_weight=weight)
        b = partner_features(canonical, history_weight=weight)
        assert np.isclose(b.loc[2, "pq_judge"] - a.loc[2, "pq_judge"], weight*(8/7-1))
        assert np.isclose(b.loc[2, "pq_vote"] - a.loc[2, "pq_vote"], weight*(1.2-1))
        if weight == 0:
            np.testing.assert_array_equal(a[["pq_judge", "pq_vote"]], b[["pq_judge", "pq_vote"]])


def test_heldout_history_remains_excluded_after_canonicalization():
    d = canonical_primary_partners(rows())
    target = d[d.season.eq(34)]
    no_prior = partner_features(target, reference=target, history_weight=1.)
    assert no_prior.loc[2, "pq_judge"] == 1.
    assert no_prior.loc[2, "pq_vote"] == 1.
    observed_prior = partner_features(target, reference=d[d.season.eq(33)], history_weight=1.)
    assert np.isclose(observed_prior.loc[2, "pq_judge"], 8/7)
    assert np.isclose(observed_prior.loc[2, "pq_vote"], 1.2)
