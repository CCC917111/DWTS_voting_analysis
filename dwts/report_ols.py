"""Separate, literal §6.1 OLS audit. Does not change the RF feature pipeline.

The report's p15 Partner_Score equation uses Q (judge share, notation p4),
not the relative-score and cross-season blend introduced later for RF.
Cold starts, observation units and covariance estimators remain declared choices.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import INDUSTRIES


def literal_partner_score(votes, cold_start="drop"):
    """Compute sum(Q_i,1:x-1)/(x-1) within a contestant-season.

    Refuse missing calendar weeks rather than silently replacing x-1 by a row
    count. The frozen 2,777-row panel has contiguous weeks starting at one.
    """
    if cold_start not in {"drop", "neutral_share"}:
        raise ValueError("cold_start must be drop or neutral_share")
    needed = {"season", "contestant_id", "week", "judge_share", "n_active"}
    if needed - set(votes):
        raise ValueError("Missing literal OLS inputs: " + str(sorted(needed-set(votes))))
    if votes.duplicated(["season", "contestant_id", "week"]).any():
        raise ValueError("Duplicate contestant-season-week")
    d = votes.sort_values(["season", "contestant_id", "week"]).copy()
    if not np.isfinite(d.judge_share).all() or not d.judge_share.between(0, 1).all():
        raise ValueError("judge_share must be finite and in [0,1]")
    if not np.isfinite(d.n_active).all() or (d.n_active < 1).any():
        raise ValueError("n_active must be positive")
    d["partner_score_literal"] = np.nan
    for _, g in d.groupby(["season", "contestant_id"], sort=False):
        if not np.array_equal(g.week.to_numpy(), np.arange(1, len(g)+1)):
            raise ValueError("Literal denominator needs contiguous calendar weeks starting at 1")
        # Shift before summing: current/future values never enter this feature.
        past_sum = g.judge_share.shift(1).cumsum()
        d.loc[g.index, "partner_score_literal"] = past_sum / (g.week-1).replace(0, np.nan)
    first = d.week.eq(1)
    if cold_start == "neutral_share":
        d.loc[first, "partner_score_literal"] = 1.0/d.loc[first, "n_active"]
    d["cold_start"] = cold_start
    d["feature_last_observed_week"] = d.week-1
    d["feature_eligible"] = d.partner_score_literal.notna()
    d["feature_exclusion_reason"] = np.where(d.feature_eligible, "", "undefined_week_one_0_over_0")
    return d.sort_index()


def _analysis_rows(features, target, observation_unit):
    d = features[features.feature_eligible].copy()
    if observation_unit == "contestant_season":
        if target != "placement":
            raise ValueError("Contestant-season sensitivity is defined for placement only")
        d = d.groupby(["season", "contestant_id"], as_index=False).agg(
            age=("age", "first"), industry=("industry", "first"),
            placement=("placement", "first"),
            partner_score_literal=("partner_score_literal", "mean"),
            source_week_count=("week", "size"), first_source_week=("week", "min"),
            last_source_week=("week", "max"))
    elif observation_unit != "contestant_week":
        raise ValueError("Unknown observation unit")
    return d.reset_index(drop=True)


def fit_literal_ols(features, target, observation_unit="contestant_week"):
    """Same point estimate, conventional OLS and season-cluster t-based CIs."""
    import statsmodels.api as sm

    d = _analysis_rows(features, target, observation_unit)
    numeric_columns = ["age", "partner_score_literal", target]
    if not np.isfinite(d[numeric_columns].to_numpy(float)).all():
        raise ValueError("Non-finite OLS inputs; no undisclosed imputation")
    if not d.industry.isin(INDUSTRIES).all():
        raise ValueError("Unexpected industry category")
    numeric = d[["age", "partner_score_literal"]].astype(float)
    means, scales = numeric.mean(), numeric.std(ddof=1)
    if (scales <= 0).any() or d[target].std(ddof=1) <= 0:
        raise ValueError("Cannot z-standardize a constant continuous variable")
    numeric = (numeric-means)/scales
    cats = pd.get_dummies(pd.Categorical(d.industry, categories=INDUSTRIES),
                          prefix="industry", dtype=float).drop(columns="industry_Acting")
    x = sm.add_constant(pd.concat([numeric, cats], axis=1), has_constant="add")
    y_mean, y_sd = d[target].mean(), d[target].std(ddof=1)
    y = (d[target].to_numpy(float)-y_mean)/y_sd
    fit = sm.OLS(y, x).fit()
    if d.season.nunique() < 2:
        raise ValueError("Season-cluster supplement requires at least two seasons")
    clustered = fit.get_robustcov_results(cov_type="cluster", groups=d.season,
                                         use_correction=True, use_t=True)
    rows = []
    for covariance, result in [("conventional_OLS", fit), ("season_cluster_t", clustered)]:
        ci = np.asarray(result.conf_int())
        params, pvalues = np.asarray(result.params), np.asarray(result.pvalues)
        for i, name in enumerate(x):
            rows.append(dict(target=target, feature=name, coefficient=float(params[i]),
                lower_95=ci[i, 0], upper_95=ci[i, 1], pvalue=float(pvalues[i]),
                covariance=covariance, ci_reference="Student_t", n=len(d),
                season_clusters=d.season.nunique(), r2_in_sample=fit.rsquared,
                design_rank=np.linalg.matrix_rank(x), design_columns=x.shape[1],
                residual_df=fit.df_resid,
                ci_df=getattr(result, "df_resid_inference", result.df_resid),
                observation_unit=observation_unit))
    standardization = [dict(variable=name, mean=means[name], sd=scales[name], ddof=1)
                       for name in means.index]
    standardization.append(dict(variable=target, mean=y_mean, sd=y_sd, ddof=1))
    d["outcome_raw"] = d[target]
    d["outcome_z"] = y
    for name in x:
        d["design_"+name] = x[name].to_numpy()
    return pd.DataFrame(rows), d, pd.DataFrame(standardization)


def run_literal_ols(votes, outdir):
    """Predeclared 2 cold starts x weekly/placement-only unit sensitivities."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    coefficients, design_rows, normalizations, choices = [], [], [], []
    for cold in ["drop", "neutral_share"]:
        f = literal_partner_score(votes, cold)
        f.to_csv(out/f"feature_rows_{cold}.csv", index=False)
        for unit, targets in [("contestant_week", ["judge_mean", "fan_mean", "placement"]),
                              ("contestant_season", ["placement"])]:
            branch = f"literal_{cold}_{unit}"
            for target in targets:
                coef, rows, normalization = fit_literal_ols(f, target, unit)
                for frame in [coef, rows, normalization]:
                    frame["branch"] = branch
                    frame["target"] = target
                    frame["cold_start"] = cold
                coefficients.append(coef)
                design_rows.append(rows)
                normalizations.append(normalization)
                choices.append(dict(branch=branch, target=target, cold_start=cold,
                    observation_unit=unit, primary=cold == "drop" and unit == "contestant_week",
                    input_rows=len(votes), undefined_feature_rows=int((~f.feature_eligible).sum()),
                    model_rows=len(rows), industry_reference="Acting", z_score_ddof=1,
                    feature="sum of own prior-week judge shares / (week-1); no cross-season or RF blend",
                    placement_aggregation="mean of eligible lagged features" if unit == "contestant_season" else "none"))
    coef = pd.concat(coefficients, ignore_index=True)
    coef.to_csv(out/"ols_coefficients.csv", index=False)
    pd.concat(design_rows, ignore_index=True).to_csv(out/"model_rows.csv", index=False)
    pd.concat(normalizations, ignore_index=True).to_csv(out/"standardization.csv", index=False)
    pd.DataFrame(choices).to_csv(out/"branch_choices.csv", index=False)
    comparison = coef[(coef.feature == "partner_score_literal") &
                      coef.target.isin(["judge_mean", "fan_mean"]) &
                      coef.covariance.eq("conventional_OLS")].copy()
    comparison["report_approximate_beta"] = comparison.target.map({"judge_mean": .93, "fan_mean": .48})
    comparison["difference_from_approximate_text"] = comparison.coefficient-comparison.report_approximate_beta
    comparison["report_source"] = "printed p19, section 6.3; beta approximately .93/.48"
    comparison["verdict"] = "approximate_reference_only_no_numeric_match_gate"
    comparison.to_csv(out/"approximate_text_comparison.csv", index=False)
    return coef


def plot_literal_ols(coefficients, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = coefficients[coefficients.branch.eq("literal_drop_contestant_week")]
    names = ["partner_score_literal", "age"]+["industry_"+x for x in INDUSTRIES if x != "Acting"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for ax, target in zip(axes, ["judge_mean", "fan_mean", "placement"]):
        for covariance, shift, color in [("conventional_OLS", -.09, "#376b88"),
                                          ("season_cluster_t", .09, "#d28a37")]:
            g = d[d.target.eq(target) & d.covariance.eq(covariance)].set_index("feature").loc[names]
            ax.errorbar(g.coefficient, np.arange(len(names))+shift,
                        xerr=np.vstack([g.coefficient-g.lower_95, g.upper_95-g.coefficient]),
                        fmt="o", color=color, capsize=3, label=covariance)
        ax.axvline(0, color="gray", linewidth=.8)
        ax.set_yticks(np.arange(len(names)), [n.replace("industry_", "") for n in names])
        ax.invert_yaxis()
        ax.set(title=target, xlabel="Standardized coefficient and 95% CI")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 5 equation-literal branch: prior judge shares; week 1 omitted\n"
                 "All three outcomes use contestant-week rows; author unit and CI convention unknown")
    fig.savefig(Path(outdir)/"figure5_literal_ols.png", dpi=180)
    plt.close(fig)
