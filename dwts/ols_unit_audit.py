"""Bounded score-quantity audit; no changes to the literal OLS implementation.

The closed candidate set changes only the quantity averaged over earlier weeks.
All alternatives are semantic hypotheses, never selected by graphic proximity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .report_ols import literal_partner_score

CANDIDATES = (
    "lagged_judge_share",
    "lagged_raw_judge_mean",
    "lagged_weekly_relative_mean",
    "lagged_within_week_z",
)
NEW_CANDIDATES = CANDIDATES[1:]
TARGETS = ("judge_mean", "fan_mean", "placement")
KEYS = ["season", "contestant_id", "week"]
OUTCOMES = {"Judge Score": "judge_mean", "Fan Votes": "fan_mean",
            "Placement (1=Best)": "placement"}
TERMS = {"Partner Score (std)": "partner_score_literal", "Age (std)": "age",
         **{f"Industry: {x}": f"industry_{x}" for x in
            ["Other", "Model", "Music", "RealityTV", "Sports"]}}


def score_unit_features(votes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all fixed features with identical eligibility and strictly past data.

    Frozen Q and within-week z inputs are checked against J. No population or
    normalization is silently recomputed after omitting first-week model rows.
    """
    required = set(KEYS + ["judge_mean", "judge_share", "judge_z", "n_active"])
    if required - set(votes):
        raise ValueError(f"Missing score-unit inputs: {sorted(required-set(votes))}")
    if not np.isfinite(votes[["judge_mean", "judge_share", "judge_z", "n_active"]]).all().all():
        raise ValueError("All score quantities must be finite")
    d = literal_partner_score(votes, "drop").sort_values(KEYS)
    group = d.groupby(["season", "week"]).judge_mean
    weekly_mean = group.transform("mean")
    weekly_total = group.transform("sum")
    weekly_sd = group.transform(lambda x: x.std(ddof=0))
    weekly_n = group.transform("size")
    if (weekly_total <= 0).any():
        raise ValueError("Weekly judge total must be positive")
    if not np.array_equal(d.n_active.to_numpy(), weekly_n.to_numpy()):
        raise ValueError("n_active does not describe the frozen weekly population")
    expected_q = d.judge_mean / weekly_total
    expected_z = (d.judge_mean-weekly_mean) / weekly_sd.replace(0, 1)
    if not np.allclose(d.judge_share, expected_q, rtol=1e-12, atol=1e-12):
        raise ValueError("Frozen Q differs from J / weekly sum of J")
    if not np.allclose(d.judge_z, expected_z, rtol=1e-12, atol=1e-12):
        raise ValueError("Frozen z differs from the declared within-week ddof=0 normalization")
    d["source_weekly_relative_mean"] = d.n_active*d.judge_share
    if not np.allclose(d.source_weekly_relative_mean, d.judge_mean/weekly_mean,
                       rtol=1e-12, atol=1e-12):
        raise ValueError("N*Q does not equal the weekly-relative judge mean")
    sources = dict(zip(CANDIDATES, ["judge_share", "judge_mean",
                                   "source_weekly_relative_mean", "judge_z"]))
    for candidate, source in sources.items():
        d[candidate] = np.nan
        for _, g in d.groupby(["season", "contestant_id"], sort=False):
            d.loc[g.index, candidate] = (g[source].shift(1).cumsum()
                                         / (g.week-1).replace(0, np.nan))
        if not d[candidate].notna().equals(d.feature_eligible):
            raise ValueError("Candidate changed the fixed eligible sample")
    np.testing.assert_allclose(d.lagged_judge_share, d.partner_score_literal,
                               equal_nan=True, atol=1e-14)
    weeks = pd.DataFrame({"season": d.season, "week": d.week,
                          "n_active": weekly_n, "judge_mean_weekly": weekly_mean,
                          "judge_sum_weekly": weekly_total, "judge_sd_weekly_ddof0": weekly_sd})
    weeks = weeks.drop_duplicates(["season", "week"]).sort_values(["season", "week"])
    return d.sort_index(), weeks.reset_index(drop=True)


def compare_graphic_triplets(coefficients: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Export every point/endpoint, preserving graphical uncertainty."""
    if len(targets) != 21 or targets.duplicated(["outcome", "term"]).any():
        raise ValueError("Expected exactly 21 unique graphical coefficient triplets")
    rows = []
    for g in targets.to_dict("records"):
        models = coefficients[(coefficients.target == OUTCOMES[g["outcome"]]) &
                              (coefficients.feature == TERMS[g["term"]])]
        if set(models.candidate) != set(CANDIDATES):
            raise ValueError("Graphical comparison lacks a predeclared candidate")
        if models.duplicated(["candidate", "covariance"]).any():
            raise ValueError("Duplicate candidate/covariance coefficient")
        for m in models.to_dict("records"):
            for metric, column in [("coefficient", "coefficient_approx"),
                                   ("lower_95", "ci95_low_approx"),
                                   ("upper_95", "ci95_high_approx")]:
                value, tolerance = g[column], g["absolute_graphic_tolerance"]
                error = abs(m[metric]-value)
                rows.append(dict(candidate=m["candidate"], target=m["target"],
                    feature=m["feature"], covariance=m["covariance"], metric=metric,
                    computed=m[metric], graphic_approx=value, absolute_error=error,
                    graphic_tolerance=tolerance,
                    within_graphic_tolerance=bool(error <= tolerance+1e-12),
                    exact_author_match_claim=False, protocol_selected=False))
    return pd.DataFrame(rows)


def graphic_width_ratio_bounds(targets: pd.DataFrame) -> pd.DataFrame:
    """A common interval-width ratio is necessary for a shared classical design.

    The endpoint digitization tolerance makes each plotted half-width uncertain
    by +/- tolerance. Bounds intentionally do not assume exact digitized values.
    """
    rows = []
    base = targets[targets.outcome == "Judge Score"].set_index("term")
    for outcome in ["Fan Votes", "Placement (1=Best)"]:
        other = targets[targets.outcome == outcome].set_index("term")
        for term in base.index:
            a, b = base.loc[term], other.loc[term]
            wa = (a.ci95_high_approx-a.ci95_low_approx)/2
            wb = (b.ci95_high_approx-b.ci95_low_approx)/2
            ta, tb = a.absolute_graphic_tolerance, b.absolute_graphic_tolerance
            if wa <= ta or wb <= tb:
                raise ValueError("Graphical tolerance obscures an interval's positive width")
            rows.append(dict(outcome=outcome, target=OUTCOMES[outcome], term=term,
                central_width_ratio=wb/wa, ratio_lower=(wb-tb)/(wa+ta),
                ratio_upper=(wb+tb)/(wa-ta),
                interpretation="Necessary, not sufficient, common-design classical-CI condition"))
    result = pd.DataFrame(rows)
    for target, group in result.groupby("target"):
        common_low, common_high = group.ratio_lower.max(), group.ratio_upper.min()
        result.loc[group.index, "all_terms_common_ratio_lower"] = common_low
        result.loc[group.index, "all_terms_common_ratio_upper"] = common_high
        result.loc[group.index, "graphically_feasible_common_ratio"] = bool(common_low <= common_high)
    return result


def standardization_invariance_audit(features: pd.DataFrame) -> pd.DataFrame:
    """Check the affine invariance on design columns, without extra OLS fits."""
    d = features[features.feature_eligible]
    rows = []
    for column in ["age", *CANDIDATES, *TARGETS]:
        raw = d[column].astype(float)
        transformed = 17+100*raw
        z = (raw-raw.mean())/raw.std(ddof=1)
        transformed_z = (transformed-transformed.mean())/transformed.std(ddof=1)
        rows.append(dict(variable=column, positive_multiplier=100, additive_constant=17,
                         maximum_absolute_design_difference=float(abs(z-transformed_z).max()),
                         extra_ols_fit=False))
    return pd.DataFrame(rows)
