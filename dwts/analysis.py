"""Analyze voting rules and associations from diagnosed posterior means.

Exports explicit analysis populations, lagged-share OLS and separately normalized
controversy summaries from a diagnosed fit. Historical reference checks are optional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, spearmanr, kendalltau

from .data import prepare
from .full_report import (correlation_audit, mechanisms_audit, literal_spearman, kendall_a)
from .report_ols import fit_literal_ols, literal_partner_score, run_literal_ols
from .rules import controversy, eliminate, merit, order_worst

DATA_SHA256 = "ea99caec6ea243bdb450a1971a95ba8a95701a93be7ff29f0ba3c57d72ddff52"
REPORT_SHA256 = "c731e44d92c84d6b667ce76b3a4a73085b555c5c12db7fd2afadbdfb5787999a"
KEYS = ["season", "week", "contestant_id"]
TABLE5 = {
    "S1-2": (11, "none", {"rank": 9, "percentage": 8, "agreement": 7}),
    "S3-27": (192, "none", {"rank": 139, "percentage": 189, "agreement": 138}),
    "S28-34": (56, "judge_save", {"rank": 44, "percentage": 40, "agreement": 28}),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_votes(votes, expected_seasons=None):
    required = set(KEYS + ["judge_mean", "fan_mean", "eliminated", "withdrawal", "is_final_week"])
    if required - set(votes):
        raise ValueError(f"Missing frozen vote columns: {sorted(required-set(votes))}")
    if votes.empty or votes[KEYS].isna().any().any() or votes.duplicated(KEYS).any():
        raise ValueError("Frozen votes must have nonempty, unique season/week/contestant keys")
    if expected_seasons is not None and set(votes.season.unique()) != set(expected_seasons):
        raise ValueError("Frozen votes do not contain the required seasons")
    for column in ["eliminated", "withdrawal", "is_final_week"]:
        if votes[column].isna().any() or not pd.api.types.is_bool_dtype(votes[column]):
            raise ValueError(f"Frozen {column} must contain booleans")
    if not np.isfinite(votes[["judge_mean", "fan_mean"]].to_numpy(float)).all():
        raise ValueError("Frozen scores and fan means must be finite")
    if not votes.fan_mean.between(0, 1).all() or (votes.judge_mean < 0).any():
        raise ValueError("Invalid score or fan share range")
    totals = votes.groupby(["season", "week"]).fan_mean.sum()
    if not np.allclose(totals, 1, atol=1e-10, rtol=0):
        raise ValueError("Frozen fan means must sum to one in every week")


def check_output_path(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new output directory; never overwrite an analysis run")
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Analysis output must be separate from the frozen source")


def rule_outcomes(votes):
    """Evaluate declared regime mechanisms on observed weekly rosters.

    These are posterior-mean consistency checks against observed eliminations;
    they do not measure accuracy against unobserved audience vote totals.
    """
    weekly = []
    for (season, week), group in votes.groupby(["season", "week"], sort=True):
        actual = set(np.flatnonzero(group.eliminated.to_numpy(bool)))
        if not actual or len(actual) == len(group):
            continue
        scenario = "S1-2" if season <= 2 else ("S3-27" if season <= 27 else "S28-34")
        mechanism = "judge_save" if season >= 28 else "none"
        supported = mechanism == "none" or len(actual) == 1
        row = dict(season=int(season), week=int(week), scenario=scenario,
                   mechanism=mechanism, actual_exits=len(actual), supported=supported)
        predictions = {}
        for rule in ["rank", "percentage"]:
            predicted = (set(eliminate(group.judge_mean.to_numpy(), group.fan_mean.to_numpy(),
                                      group.contestant_id.to_numpy(), rule,
                                      mechanism=mechanism, count=len(actual))) if supported else None)
            predictions[rule] = predicted
            row[rule] = predicted == actual if supported else None
        row["agreement"] = predictions["rank"] == predictions["percentage"] if supported else None
        weekly.append(row)
    columns = ["season", "week", "scenario", "mechanism", "actual_exits", "supported",
               "rank", "percentage", "agreement"]
    detail = pd.DataFrame(weekly, columns=columns)
    return detail


def table5_tables(votes):
    """Optional historical comparison; unsupported exits are never counted as hits."""
    detail = rule_outcomes(votes)
    comparisons = []
    for scenario, (reported_weeks, mechanism, targets) in TABLE5.items():
        group = detail[detail.scenario.eq(scenario)]
        supported = group[group.supported.eq(True)]
        undefined = len(group) - len(supported)
        status = ("undefined_multi_exit_save" if undefined else
                  "denominator_mismatch" if len(group) != reported_weeks else
                  "conditional_same_mechanism_and_denominator")
        comparable = undefined == 0 and len(group) == reported_weeks
        for metric, target in targets.items():
            hits = int(supported[metric].eq(True).sum())
            comparisons.append(dict(scenario=scenario, mechanism=mechanism, metric=metric,
                reported_correct=target, reported_weeks=reported_weeks,
                reconstructed_weeks=len(group), supported_weeks=len(supported),
                undefined_weeks=undefined, correct_supported_only=hits,
                accuracy_supported_only=hits / len(supported) if len(supported) else np.nan,
                comparison_status=status, numerical_match=hits == target if comparable else None,
                author_tie_policy_identified=False))
    return detail, pd.DataFrame(comparisons)


def eda_summary(panel, protocol, include_mixed_withdrawal_weeks):
    """Exact score ties; explicitly declare the population and moment convention."""
    lowest = saves = weeks = 0
    for _, group in panel.groupby(["season", "week"]):
        if not group.eliminated.any() or (group.withdrawal.any() and not include_mixed_withdrawal_weeks):
            continue
        weeks += 1
        mask = group.judge_mean.eq(group.judge_mean.min())
        lowest += int(mask.sum())
        saves += int((mask & ~group.eliminated).sum())
    y = panel.judge_mean.to_numpy(float)
    q1, q3 = np.quantile(y, [.25, .75])
    return dict(protocol=protocol, contestant_weeks=len(panel), unique_people=panel.name.nunique(),
                contestant_seasons=panel.contestant_id.nunique(), elimination_weeks=weeks,
                include_mixed_withdrawal_weeks=include_mixed_withdrawal_weeks, tie_policy="exact",
                lowest_scorers=lowest, audience_saves=saves,
                audience_save_fraction=saves / lowest if lowest else np.nan,
                cv=np.std(y, ddof=1) / np.mean(y), iqr_coefficient=(q3-q1)/(q3+q1),
                skew=skew(y, bias=False), excess_kurtosis=kurtosis(y, bias=False),
                moment_convention="sample_sd_and_unbiased_shape")


def controversy_rates(weekly):
    """A fraction of controversial weeks is distinct from events per week/row."""
    keys = ["season", "week", "judge_weight", "mechanism"]
    if weekly.empty or weekly.duplicated(keys).any():
        raise ValueError("Mechanism records must have unique, nonempty weekly keys")
    numeric = weekly[["type_a", "type_b", "controversies", "n", "exit_count"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric < 0).any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Mechanism counts must be finite nonnegative integers")
    if ((weekly.n < 3) | (weekly.exit_count < 1) | (weekly.exit_count >= weekly.n)).any():
        raise ValueError("Invalid roster or exit count")
    if not weekly.controversies.eq(weekly.type_a + weekly.type_b).all() or (weekly.controversies > weekly.exit_count).any():
        raise ValueError("Controversy events disagree with event types or exit count")
    rows = []
    for scope, frame in [("all_weeks_n_ge_3", weekly),
                         ("nonfinal_weeks", weekly[~weekly.is_final]),
                         ("elimination_weeks", weekly[weekly.has_exit])]:
        for (weight, mechanism), group in frame.groupby(["judge_weight", "mechanism"], sort=True):
            events = int(group.controversies.sum())
            controversial_weeks = int(group.controversies.gt(0).sum())
            rows.append(dict(scope=scope, judge_weight=weight, mechanism=mechanism,
                weeks=len(group), contestant_rows=int(group.n.sum()), event_count=events,
                controversial_weeks=controversial_weeks,
                controversial_week_fraction=controversial_weeks / len(group),
                controversy_events_per_week=events / len(group),
                controversy_events_per_contestant_row=events / group.n.sum(),
                count_definition="type_a_plus_type_b"))
    return pd.DataFrame(rows)


def load_source(source, data):
    """Verify source diagnostics, declared fitted seasons and preprocessing identity."""
    source, data = Path(source), Path(data)
    manifest = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "run_config.json").read_text())
    statuses = json.loads((source / "sampling_status.json").read_text())
    seasons = config.get("seasons")
    if (not isinstance(seasons, list) or not seasons or
            any(type(season) is not int or not 1 <= season <= 34 for season in seasons) or
            len(set(seasons)) != len(seasons)):
        raise ValueError("Source run_config must declare unique fitted seasons in 1..34")
    if digest(data) != DATA_SHA256 or manifest.get("input_sha256") != DATA_SHA256:
        raise ValueError("Raw data checksum must match the pinned data and frozen source")
    if manifest.get("status") != "completed" or manifest.get("all_sampling_diagnostics_pass") is not True:
        raise ValueError("Source must be completed with all sampling diagnostics passed")
    if (len(statuses) != len(seasons) or any(type(s["season"]) is not int for s in statuses) or
            {s["season"] for s in statuses} != set(seasons)):
        raise ValueError("Source diagnostic records must match the declared fitted seasons")
    for status in statuses:
        values = [status["max_rhat"], status["min_ess_bulk"], status["min_ess_tail"]]
        if (status["sampling_diagnostics_pass"] is not True or not np.isfinite(values).all()
                or values[0] > 1.01 or min(values[1:]) < 400):
            raise ValueError("Frozen sampling diagnostic values do not pass the gate")
    votes = pd.read_csv(source / "votes.csv", float_precision="round_trip")
    validate_votes(votes, expected_seasons=seasons)
    panel, _ = prepare(data, **config.get("data", {}))
    panel = panel[panel.season.isin(seasons)].reset_index(drop=True)
    columns = KEYS + ["judge_mean", "judge_share", "judge_z", "eliminated", "withdrawal",
                      "is_final_week", "rule", "partner", "age", "industry", "placement", "n_active"]
    try:
        pd.testing.assert_frame_equal(
            panel.sort_values(KEYS)[columns].reset_index(drop=True),
            votes.sort_values(KEYS)[columns].reset_index(drop=True),
            check_exact=False, atol=1e-14, rtol=0)
    except AssertionError as error:
        raise ValueError("Frozen votes disagree with the declared raw-data preprocessing") from error
    return votes, panel, manifest, config


def rf_reference_comparison(metrics):
    """Compare saved numbers only; unknown author protocols never become a pass."""
    references = {
        "A1_judge": {"r2": .146, "rmse": 5.320},
        "A2_fan": {"r2": .614, "rmse": .072},
        "A3_top_half_judge": {"f1": .766, "auc": .778},
        "A4_bottom_two_judge": {"f1": .488, "auc": .755},
        "A3_top_half_fan": {"f1": .803, "auc": .879},
        "A4_bottom_two_fan": {"f1": .547, "auc": .775},
    }
    if metrics.task.duplicated().any() or set(metrics.task) != set(references):
        raise ValueError("Saved RF metrics must contain exactly the six declared tasks")
    rows = []
    for item in metrics.to_dict("records"):
        for metric, reference in references[item["task"]].items():
            value = float(item[metric])
            if not np.isfinite(value):
                raise ValueError("Saved RF metric is not finite")
            rows.append(dict(task=item["task"], metric=metric, saved_value=value,
                             report_reference=reference, difference=value-reference,
                             within_printed_tolerance=abs(value-reference) <= .0005+1e-12,
                             protocol_comparison="numeric_only_author_protocol_unidentified"))
    return pd.DataFrame(rows)


def summarize_rules(detail):
    rows = []
    for (scenario, mechanism), group in detail.groupby(["scenario", "mechanism"], sort=True):
        supported = group[group.supported]
        for metric in ["rank", "percentage", "agreement"]:
            hits = int(supported[metric].eq(True).sum())
            rows.append(dict(scenario=scenario, mechanism=mechanism, metric=metric,
                eligible_weeks=len(group), supported_weeks=len(supported),
                undefined_weeks=len(group)-len(supported), consistent_weeks=hits,
                consistency_supported_only=hits/len(supported) if len(supported) else np.nan))
    return pd.DataFrame(rows, columns=["scenario", "mechanism", "metric", "eligible_weeks",
        "supported_weeks", "undefined_weeks", "consistent_weeks", "consistency_supported_only"])


def correlation_analysis(votes, out):
    """Export weekly tie-aware correlations and explicitly named aggregations."""
    out.mkdir()
    rows = []
    for (season, week), group in votes.groupby(["season", "week"]):
        judge, fan = group.judge_mean.to_numpy(), group.fan_mean.to_numpy()
        rank, percentage = merit(judge, fan, "rank"), merit(judge, fan, "percentage")
        for name, (x, y) in {"percentage_vs_rank": (percentage, rank),
                "percentage_vs_fan": (percentage, fan), "rank_vs_fan": (rank, fan),
                "percentage_vs_judge": (percentage, judge), "rank_vs_judge": (rank, judge)}.items():
            constant = np.ptp(x) == 0 or np.ptp(y) == 0
            sp = (np.nan, np.nan) if constant else spearmanr(x, y)
            kt = (np.nan, np.nan) if constant else kendalltau(x, y, variant="b")
            rows.append(dict(season=season, week=week, n=len(group), comparison=name,
                is_final=bool(group.is_final_week.any()), has_exit=bool(group.eliminated.any()),
                spearman_tie_aware=sp[0], spearman_weekly_pvalue=sp[1],
                kendall_tau_b=kt[0], kendall_weekly_pvalue=kt[1],
                spearman_rank_difference=literal_spearman(x, y), kendall_tau_a=kendall_a(x, y)))
    weekly = pd.DataFrame(rows)
    weekly.to_csv(out / "weekly_correlations.csv", index=False)
    summary = []
    for scope, frame in [("all_weeks", weekly), ("nonfinal_weeks", weekly[~weekly.is_final]),
                          ("elimination_weeks", weekly[weekly.has_exit])]:
        for name, group in frame.groupby("comparison"):
            for method in ["spearman_tie_aware", "kendall_tau_b", "spearman_rank_difference", "kendall_tau_a"]:
                for aggregation in ["equal_week_mean", "equal_season_mean"]:
                    values = (group[method].dropna() if aggregation == "equal_week_mean" else
                              group.groupby("season")[method].mean().dropna())
                    summary.append(dict(scope=scope, comparison=name, method=method,
                        aggregation=aggregation, units=len(values), coefficient=values.mean(),
                        inference="descriptive_coefficient_no_pooled_pvalue"))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    return weekly


def mechanism_analysis(votes, out):
    """Simulate fixed weekly rosters on a predeclared weight/mechanism grid."""
    out.mkdir()
    rows = []
    for (season, week), group in votes.groupby(["season", "week"]):
        if len(group) < 3:
            continue
        judge, fan, ids = group.judge_mean.to_numpy(), group.fan_mean.to_numpy(), group.contestant_id.to_numpy()
        for weight in (k / 10 for k in range(11)):
            for mechanism in ["none", "judge_save", "fan_save", "bottom3_judge_save", "bottom3_save_other_two_exit"]:
                if mechanism == "bottom3_save_other_two_exit":
                    pool = order_worst(judge, fan, ids, "rank", weight)[:3]
                    saved = max(pool, key=lambda i: (judge[i], fan[i], str(ids[i])))
                    exits = np.array([i for i in pool if i != saved])
                else:
                    exits = eliminate(judge, fan, ids, "rank", weight, mechanism)
                type_a, type_b = controversy(judge, fan, exits)
                rows.append(dict(season=season, week=week, judge_weight=round(weight, 1),
                    mechanism=mechanism, n=len(group), is_final=bool(group.is_final_week.any()),
                    has_exit=bool(group.eliminated.any()), type_a=type_a, type_b=type_b,
                    controversies=type_a+type_b, exit_count=len(exits)))
    columns = ["season", "week", "judge_weight", "mechanism", "n", "is_final", "has_exit",
               "type_a", "type_b", "controversies", "exit_count"]
    weekly = pd.DataFrame(rows, columns=columns)
    weekly.to_csv(out / "weekly_scenarios.csv", index=False)
    return weekly


def association_analysis(votes, out):
    """Fit the existing lagged-share models without any historical target comparison."""
    out.mkdir()
    if votes.season.nunique() < 2:
        write_json(out / "status.json", dict(status="unavailable", reason="Season-cluster intervals require at least two seasons"))
        return "unavailable_fewer_than_two_seasons"
    coefficients, designs, normalizations, choices = [], [], [], []
    for cold in ["drop", "neutral_share"]:
        features = literal_partner_score(votes, cold)
        features.to_csv(out / f"feature_rows_{cold}.csv", index=False)
        for unit, targets in [("contestant_week", ["judge_mean", "fan_mean", "placement"]),
                              ("contestant_season", ["placement"])]:
            branch = f"literal_{cold}_{unit}"
            for target in targets:
                coef, rows, normalization = fit_literal_ols(features, target, unit)
                for frame in [coef, rows, normalization]:
                    frame["branch"], frame["target"], frame["cold_start"] = branch, target, cold
                coefficients.append(coef)
                designs.append(rows)
                normalizations.append(normalization)
                choices.append(dict(branch=branch, target=target, cold_start=cold,
                    observation_unit=unit, primary=cold == "drop" and unit == "contestant_week",
                    input_rows=len(votes), undefined_feature_rows=int((~features.feature_eligible).sum()),
                    model_rows=len(rows), industry_reference="Acting", z_score_ddof=1,
                    feature="sum of own prior-week judge shares / (week-1); no cross-season or RF blend",
                    placement_aggregation="mean of eligible lagged features" if unit == "contestant_season" else "none"))
    pd.concat(coefficients, ignore_index=True).to_csv(out / "ols_coefficients.csv", index=False)
    pd.concat(designs, ignore_index=True).to_csv(out / "model_rows.csv", index=False)
    pd.concat(normalizations, ignore_index=True).to_csv(out / "standardization.csv", index=False)
    pd.DataFrame(choices).to_csv(out / "branch_choices.csv", index=False)
    return "completed"


def validate_rf_metrics(metrics, n):
    """Validate saved task metrics without consulting any published target value."""
    tasks = {"A1_judge": ["r2", "rmse"], "A2_fan": ["r2", "rmse"],
             "A3_top_half_judge": ["f1", "auc"], "A4_bottom_two_judge": ["f1", "auc"],
             "A3_top_half_fan": ["f1", "auc"], "A4_bottom_two_fan": ["f1", "auc"]}
    if "task" not in metrics or metrics.task.duplicated().any() or set(metrics.task) != set(tasks):
        raise ValueError("Saved RF metrics must contain exactly the six declared tasks")
    if "n" not in metrics or not metrics.n.eq(n).all():
        raise ValueError("Saved RF metric row counts disagree with frozen votes")
    for row in metrics.to_dict("records"):
        for name in tasks[row["task"]]:
            value = float(row[name])
            if not np.isfinite(value) or (name == "rmse" and value < 0) or (name in {"f1", "auc"} and not 0 <= value <= 1):
                raise ValueError("Saved RF metrics contain invalid values")


def reference_comparison(votes, source, out, report_pdf):
    """Run optional historical checks; unidentified protocols cannot establish agreement."""
    out.mkdir()
    _, comparison = table5_tables(votes)
    comparison.to_csv(out / "table5_comparison.csv", index=False)
    correlations = out / "correlations"
    correlations.mkdir()
    correlation_audit(votes, correlations)
    mechanisms = out / "mechanisms"
    mechanisms.mkdir()
    mechanisms_audit(votes, mechanisms)
    if votes.season.nunique() >= 2:
        run_literal_ols(votes, out / "ols")
    if (source / "rf_metrics.csv").exists():
        rf_reference_comparison(pd.read_csv(source / "rf_metrics.csv")).to_csv(out / "rf_numeric_comparison.csv", index=False)
    result = dict(execution_completed=True, comparison_status="unresolved_reference_protocols",
        numerical_agreement_established=False,
        report_pdf_sha256=digest(report_pdf) if report_pdf else None,
        reference_identity="verified_local_pdf" if report_pdf else "recorded_historical_reference_values",
        unresolved=["Historical multi-exit save and tie rules", "Historical analysis populations and denominators",
                    "Historical target units, folds and aggregation", "Historical sampler and RF settings"],
        parameter_selection="No refitting or parameter selection against historical values", exit_code=2)
    write_json(out / "status.json", result)
    return result


def run(source, data, out, report_pdf=None, *, compare_reference=False):
    source, data, out = Path(source), Path(data), Path(out)
    check_output_path(source, out)
    if report_pdf is not None and not compare_reference:
        raise ValueError("--report-pdf requires the explicit --compare-reference option")
    if report_pdf is not None and digest(report_pdf) != REPORT_SHA256:
        raise ValueError("Report PDF differs from the recorded historical source; no output written")
    votes, panel, source_manifest, config = load_source(source, data)
    metrics = None
    if (source / "rf_metrics.csv").exists():
        metrics = pd.read_csv(source / "rf_metrics.csv")
        validate_rf_metrics(metrics, len(votes))
    out.mkdir(parents=True)
    write_json(out / "status.json", {"execution_completed": False})
    detail = rule_outcomes(votes)
    detail.to_csv(out / "rule_weekly.csv", index=False)
    summarize_rules(detail).to_csv(out / "rule_summary.csv", index=False)

    eda_data = dict(outcome_policy="label_first", include_final_week_eliminations=True,
                    exclude_withdrawn_contestants=True)
    candidate, _ = prepare(data, **eda_data)
    candidate = candidate[candidate.season.isin(config["seasons"])].reset_index(drop=True)
    mixed = config.get("fit", {}).get("withdrawal_policy", "skip_week") == "include_scored"
    pd.DataFrame([eda_summary(panel, "fitted_model_population", mixed),
                  eda_summary(candidate, "label_first_exclude_withdrawn_include_final_exits", True)]).to_csv(
                      out / "eda_scopes.csv", index=False)
    ols_status = association_analysis(votes, out / "ols")

    weekly = correlation_analysis(votes, out / "correlations")
    gaps = []
    for rule in ["rank", "percentage"]:
        paired = weekly[weekly.comparison.isin([f"{rule}_vs_fan", f"{rule}_vs_judge"])].pivot(
            index=["season", "week"], columns="comparison", values="spearman_tie_aware").dropna()
        fan, judge = paired[f"{rule}_vs_fan"], paired[f"{rule}_vs_judge"]
        gaps.append(dict(rule=rule, paired_valid_weeks=len(paired),
                         absolute_difference_of_weekly_means=abs(fan.mean()-judge.mean()),
                         mean_of_weekly_absolute_differences=(fan-judge).abs().mean()))
    pd.DataFrame(gaps).to_csv(out / "influence_gaps.csv", index=False)

    mechanism_weekly = mechanism_analysis(votes, out / "mechanisms")
    if mechanism_weekly.empty:
        rates = pd.DataFrame(columns=["scope", "judge_weight", "mechanism", "weeks", "contestant_rows",
            "event_count", "controversial_weeks", "controversial_week_fraction",
            "controversy_events_per_week", "controversy_events_per_contestant_row", "count_definition"])
    else:
        rates = controversy_rates(mechanism_weekly)
    rates.to_csv(out / "controversy_rates.csv", index=False)
    rates.to_csv(out / "mechanisms" / "summary.csv", index=False)
    rf_status = "unavailable_not_fitted_by_this_analysis"
    if metrics is not None:
        metrics.to_csv(out / "rf_metrics.csv", index=False)
        write_json(out / "rf_protocol.json", {key: value for key, value in config.items()
                   if key.startswith("rf_") or key in ["history_weight", "history_grid", "partner_identity_mode"]})
        rf_status = "validated_saved_metrics_no_refit"

    scopes = dict(fitted_seasons=config["seasons"], posterior_population=config.get("data", {}),
        rule_mechanisms={"S1-2": "none", "S3-27": "none", "S28-34": "judge_save"},
        multi_exit_judge_save="undefined; excluded from consistency denominator",
        rule_consistency="retrospective in-sample consistency: posterior fan shares were fitted using observed elimination constraints",
        rule_ties="aggregate ties lose on lower fan share, then lower judge score, then lexical contestant ID",
        eda_alternative=dict(data=eda_data, tie_policy="exact"),
        ols=dict(feature="own_prior_judge_share_mean", continuous_standardization="ddof=1 X and Y",
                 cold_starts=["drop", "neutral_share"], covariance=["conventional_OLS", "season_cluster_t"],
                 placement_units=["contestant_week", "contestant_season"]),
        influence_gap="Both aggregation orders exported on identical paired valid weeks",
        controversy="Weekly fraction, events per week, and events per contestant row exported separately",
        mechanism_scenarios=dict(rule="rank", judge_weights=[k/10 for k in range(11)],
            mechanisms=["none", "judge_save", "fan_save", "bottom3_judge_save", "bottom3_save_other_two_exit"],
            eligible_rosters="at least three contestants", trajectory="fixed observed weekly roster; no season rollout"),
        diagnostics_gate=dict(max_rhat=1.01, min_ess_bulk=400, min_ess_tail=400),
        parameter_selection="Declared scenarios; no refitting against historical results")
    write_json(out / "analysis_scope.json", scopes)
    status = dict(execution_completed=True,
        source=str(source.resolve()), source_manifest=source_manifest,
        frozen_votes_sha256=digest(source / "votes.csv"), raw_data_sha256=digest(data),
        analysis_module_sha256=digest(Path(__file__)), fitted_seasons=config["seasons"],
        new_posterior_fits=0, new_rf_fits=0, rf_status=rf_status, ols_status=ols_status,
        undefined_rule_weeks=int((~detail.supported).sum()),
        research_limitations=["Fan shares are inferred latent quantities, not observed audience vote totals",
            "Rule consistency is retrospective and in-sample because observed eliminations constrain fitted fan shares",
            "Mechanism scenarios use fixed observed rosters and do not forecast full counterfactual seasons",
            "Multi-exit judge-save rounds have no defined save rule and remain unscored",
            "OLS associations are descriptive; no causal effect is established"],
        exit_code=0, exit_meaning="Analysis completed with validated inputs and declared scenario scopes")
    if compare_reference:
        comparison = reference_comparison(votes, source, out / "reference_comparison", report_pdf)
        status["reference_comparison"] = comparison
        status["exit_code"] = comparison["exit_code"]
        status["exit_meaning"] = "Analysis completed; optional historical reference comparisons remain unresolved"
    write_json(out / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data", default=Path("data/raw/2026_MCM_Problem_C_Data.csv"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compare-reference", action="store_true", help="Optionally compare recorded historical report values; unresolved protocols return exit 2")
    parser.add_argument("--report-pdf", type=Path, help="With --compare-reference, verify a local PDF identity without copying it")
    args = parser.parse_args()
    result = run(args.source, args.data, args.out, args.report_pdf, compare_reference=args.compare_reference)
    print(json.dumps({key: result[key] for key in ["execution_completed", "undefined_rule_weeks",
          "exit_code", "exit_meaning"]}, indent=2), flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
