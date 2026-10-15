"""Analyze voting rules and associations from diagnosed posterior means.

Exports explicit chapter populations, equation-literal OLS and separately
normalized controversy summaries without selecting parameters by report proximity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .data import prepare
from .full_report import correlation_audit, mechanisms_audit
from .report_ols import run_literal_ols
from .rules import eliminate

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


def validate_votes(votes, expected_seasons=frozenset(range(1, 35))):
    required = set(KEYS + ["judge_mean", "fan_mean", "eliminated", "withdrawal", "is_final_week"])
    if required - set(votes):
        raise ValueError(f"Missing frozen vote columns: {sorted(required-set(votes))}")
    if votes.empty or votes[KEYS].isna().any().any() or votes.duplicated(KEYS).any():
        raise ValueError("Frozen votes must have nonempty, unique season/week/contestant keys")
    if set(votes.season.unique()) != set(expected_seasons):
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


def table5_tables(votes):
    """Use the report heading's mechanism; undefined save exits remain missing."""
    weekly = []
    for (season, week), group in votes.groupby(["season", "week"], sort=True):
        actual = set(np.flatnonzero(group.eliminated.to_numpy(bool)))
        if not actual or len(actual) == len(group):
            continue
        scenario = "S1-2" if season <= 2 else ("S3-27" if season <= 27 else "S28-34")
        _, mechanism, _ = TABLE5[scenario]
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
                moment_convention="sample_sd_and_unbiased_shape", author_scope_identified=False)


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
                author_scope_identified=False))
    return pd.DataFrame(rows)


def load_source(source, data):
    source, data = Path(source), Path(data)
    manifest = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "run_config.json").read_text())
    statuses = json.loads((source / "sampling_status.json").read_text())
    if digest(data) != DATA_SHA256 or manifest.get("input_sha256") != DATA_SHA256:
        raise ValueError("Raw data checksum must match the pinned data and frozen source")
    if manifest.get("status") != "completed" or not manifest.get("all_sampling_diagnostics_pass"):
        raise ValueError("Source must be completed with all sampling diagnostics passed")
    if len(statuses) != 34 or {s["season"] for s in statuses} != set(range(1, 35)):
        raise ValueError("Source requires exactly 34 season diagnostic records")
    for status in statuses:
        values = [status["max_rhat"], status["min_ess_bulk"], status["min_ess_tail"]]
        if (not status["sampling_diagnostics_pass"] or not np.isfinite(values).all()
                or values[0] > 1.01 or min(values[1:]) < 400):
            raise ValueError("Frozen sampling diagnostic values do not pass the gate")
    votes = pd.read_csv(source / "votes.csv", float_precision="round_trip")
    validate_votes(votes)
    panel, _ = prepare(data, **config.get("data", {}))
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


def run(source, data, out, report_pdf=None):
    source, data, out = Path(source), Path(data), Path(out)
    check_output_path(source, out)
    if report_pdf is not None and digest(report_pdf) != REPORT_SHA256:
        raise ValueError("Report PDF differs from the directly audited source; no output written")
    votes, panel, source_manifest, config = load_source(source, data)
    out.mkdir(parents=True)
    write_json(out / "status.json", {"execution_completed": False, "numerical_replication_established": False})
    detail, comparison = table5_tables(votes)
    detail.to_csv(out / "table5_weekly.csv", index=False)
    comparison.to_csv(out / "table5_comparison.csv", index=False)

    eda_data = dict(outcome_policy="label_first", include_final_week_eliminations=True,
                    exclude_withdrawn_contestants=True)
    candidate, _ = prepare(data, **eda_data)
    mixed = config.get("fit", {}).get("withdrawal_policy", "skip_week") == "include_scored"
    pd.DataFrame([eda_summary(panel, "frozen_model_population", mixed),
                  eda_summary(candidate, "eda_recovery_candidate_not_author_identified", True)]).to_csv(
                      out / "eda_scopes.csv", index=False)
    run_literal_ols(votes, out / "literal_ols")

    correlations = out / "correlations"
    correlations.mkdir()
    weekly = correlation_audit(votes, correlations)
    gaps = []
    for rule in ["rank", "percentage"]:
        paired = weekly[weekly.comparison.isin([f"{rule}_vs_fan", f"{rule}_vs_judge"])].pivot(
            index=["season", "week"], columns="comparison", values="spearman_tie_aware").dropna()
        fan, judge = paired[f"{rule}_vs_fan"], paired[f"{rule}_vs_judge"]
        gaps.append(dict(rule=rule, paired_valid_weeks=len(paired),
                         absolute_difference_of_weekly_means=abs(fan.mean()-judge.mean()),
                         mean_of_weekly_absolute_differences=(fan-judge).abs().mean(),
                         author_aggregation_identified=False))
    pd.DataFrame(gaps).to_csv(out / "influence_gap_definitions.csv", index=False)

    mechanisms = out / "mechanism_sensitivities"
    mechanisms.mkdir()
    mechanisms_audit(votes, mechanisms)
    rates = controversy_rates(pd.read_csv(mechanisms / "figure6_weekly_protocols.csv"))
    rates.to_csv(out / "controversy_rates.csv", index=False)
    rf_status = "unavailable_not_fitted_by_this_analysis"
    if (source / "rf_metrics.csv").exists():
        metrics = pd.read_csv(source / "rf_metrics.csv")
        if not metrics.n.eq(len(votes)).all():
            raise ValueError("Saved RF metric row counts disagree with frozen votes")
        rf_reference_comparison(metrics).to_csv(out / "rf_numeric_comparison.csv", index=False)
        write_json(out / "rf_protocol.json", {key: value for key, value in config.items()
                   if key.startswith("rf_") or key in ["history_weight", "history_grid", "partner_identity_mode"]})
        rf_status = "saved_numbers_compared_no_refit_author_protocol_unidentified"

    scopes = dict(posterior_population=config.get("data", {}),
        table5_mechanisms={key: value[1] for key, value in TABLE5.items()},
        table5_multi_exit_save="undefined; never assigned invented exits or zero hits",
        eda_recovery_candidate=dict(data=eda_data, tie_policy="exact", author_scope_identified=False),
        ols=dict(feature="own_prior_judge_share_mean", continuous_standardization="ddof=1 X and Y",
                 cold_starts=["drop", "neutral_share"], author_cold_start_identified=False,
                 placement_units=["contestant_week", "contestant_season"]),
        influence_gap="Both aggregation orders exported on identical paired valid weeks",
        controversy="Weekly fraction, events per week, and events per contestant row exported separately",
        original_mechanisms=["none", "judge_save"],
        bottom_three_exit_interpretations=["one_exit", "other_two_exit"],
        parameter_selection="No refitting or selection against report numbers")
    write_json(out / "chapter_scopes.json", scopes)
    status = dict(execution_completed=True, numerical_replication_established=False,
        source=str(source.resolve()), source_manifest=source_manifest,
        frozen_votes_sha256=digest(source / "votes.csv"), raw_data_sha256=digest(data),
        report_pdf_sha256=digest(report_pdf) if report_pdf else None,
        references="Directly audited report identity verified" if report_pdf else "Repository-recorded report references",
        analysis_module_sha256=digest(Path(__file__)), new_posterior_fits=0, new_rf_fits=0,
        rf_comparison_status=rf_status,
        table5_undefined_weeks=int(detail.loc[~detail.supported, ["season", "week"]].drop_duplicates().shape[0]),
        scope_candidates_selected_against_report=False,
        unresolved=["Author multi-exit save and tie rules", "EDA and Figure 6 populations/denominators",
                    "Table 9 target units, folds and aggregation", "Author sampler and RF settings"],
        exit_code=2, exit_meaning="Analysis completed; reference comparisons remain unresolved")
    write_json(out / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data", default=Path("data/raw/2026_MCM_Problem_C_Data.csv"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-pdf", type=Path, help="Optional local PDF; verify audited SHA-256, never copy it")
    args = parser.parse_args()
    result = run(args.source, args.data, args.out, args.report_pdf)
    print(json.dumps({key: result[key] for key in ["execution_completed", "numerical_replication_established",
          "table5_undefined_weeks", "exit_code", "exit_meaning"]}, indent=2), flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
