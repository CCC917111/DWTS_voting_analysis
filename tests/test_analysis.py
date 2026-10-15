"""Regression checks for genuine mechanism, denominator and rate ambiguities."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dwts.analysis import (check_output_path, controversy_rates, eda_summary,
                               rf_reference_comparison, table5_tables, validate_votes)


def week(season=28, exits=(1,)):
    return pd.DataFrame(dict(season=[season]*3, week=[1]*3,
        contestant_id=["A", "B", "C"], name=["A", "B", "C"],
        judge_mean=[10., 9., 1.], fan_mean=[.01, .49, .50],
        eliminated=[i in exits for i in range(3)], withdrawal=[False]*3,
        is_final_week=[False]*3))


def test_late_heading_uses_judge_save_instead_of_no_save():
    # Aggregate tie loses A; judge-save instead loses lower-scored B.
    detail, comparison = table5_tables(week())
    assert detail.iloc[0].mechanism == "judge_save"
    assert bool(detail.iloc[0]["rank"]) is True
    assert comparison[comparison.scenario.eq("S28-34")].numerical_match.isna().all()
    assert comparison[comparison.scenario.eq("S28-34")].comparison_status.eq("denominator_mismatch").all()


def test_multi_exit_save_is_missing_not_an_invented_loss_or_perfect_match():
    detail, comparison = table5_tables(week(exits=(0, 1)))
    assert not detail.iloc[0].supported
    assert detail[["rank", "percentage", "agreement"]].isna().all().all()
    late = comparison[comparison.scenario.eq("S28-34")]
    assert late.undefined_weeks.eq(1).all() and late.supported_weeks.eq(0).all()
    assert late.accuracy_supported_only.isna().all()
    assert late.numerical_match.isna().all()
    assert late.comparison_status.eq("undefined_multi_exit_save").all()


def test_early_multi_exit_set_remains_supported_without_a_save():
    detail, comparison = table5_tables(week(season=3, exits=(0, 1)))
    assert detail.iloc[0].supported and detail.iloc[0].mechanism == "none"
    assert comparison[comparison.scenario.eq("S3-27")].undefined_weeks.eq(0).all()


def test_empty_elimination_scope_cannot_claim_numeric_agreement():
    detail, comparison = table5_tables(week(exits=()))
    assert detail.empty and len(comparison) == 9
    assert comparison.numerical_match.isna().all()
    assert comparison.comparison_status.eq("denominator_mismatch").all()


def test_eda_does_not_merge_distinct_nearby_scores():
    data = week(exits=(2,))
    data["judge_mean"] = [6.3333, 19/3, 9.]
    result = eda_summary(data, "synthetic_exact_tie_control", True)
    assert result["lowest_scorers"] == 1 and result["audience_saves"] == 1
    assert result["tie_policy"] == "exact"


def mechanism_weeks():
    return pd.DataFrame(dict(season=[1, 1], week=[1, 2], judge_weight=[.5, .5],
        mechanism=["bottom3_save_other_two_exit"]*2, n=[5, 5], exit_count=[2, 2],
        type_a=[1, 0], type_b=[1, 0], controversies=[2, 0],
        is_final=[False, False], has_exit=[True, True]))


def test_two_events_in_one_week_are_not_two_controversial_weeks():
    rates = controversy_rates(mechanism_weeks())
    row = rates[rates.scope.eq("nonfinal_weeks")].iloc[0]
    assert row.weeks == 2 and row.controversial_weeks == 1 and row.event_count == 2
    assert row.controversial_week_fraction == .5
    assert row.controversy_events_per_week == 1
    assert row.controversy_events_per_contestant_row == .2


def test_duplicate_mechanism_weeks_refused():
    data = mechanism_weeks()
    with pytest.raises(ValueError, match="unique"):
        controversy_rates(pd.concat([data, data.iloc[:1]]))


def test_inconsistent_event_count_refused():
    data = mechanism_weeks()
    data.loc[0, "controversies"] = 1
    with pytest.raises(ValueError, match="disagree"):
        controversy_rates(data)


@pytest.mark.parametrize("change", ["duplicate", "nan", "unnormalized", "non_boolean"])
def test_invalid_frozen_votes_rejected(change):
    data = week()
    if change == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif change == "nan":
        data.loc[0, "fan_mean"] = np.nan
    elif change == "unnormalized":
        data.loc[0, "fan_mean"] = .02
    else:
        data["eliminated"] = data.eliminated.astype(str)
    with pytest.raises(ValueError):
        validate_votes(data, expected_seasons={28})


def test_analysis_cannot_overwrite_or_write_inside_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="never overwrite"):
        check_output_path(source, source)
    with pytest.raises(ValueError, match="separate"):
        check_output_path(source, source / "new-audit")
    check_output_path(source, tmp_path / "new-audit")
    assert not (tmp_path / "new-audit").exists()


def test_matching_rf_numbers_do_not_establish_protocol_equivalence():
    tasks = ["A1_judge", "A2_fan", "A3_top_half_judge", "A4_bottom_two_judge",
             "A3_top_half_fan", "A4_bottom_two_fan"]
    data = pd.DataFrame(dict(task=tasks, r2=[.146, .614, np.nan, np.nan, np.nan, np.nan],
        rmse=[5.320, .072, np.nan, np.nan, np.nan, np.nan],
        f1=[np.nan, np.nan, .766, .488, .803, .547],
        auc=[np.nan, np.nan, .778, .755, .879, .775]))
    result = rf_reference_comparison(data)
    assert len(result) == 12 and result.within_printed_tolerance.all()
    assert result.protocol_comparison.eq("numeric_only_author_protocol_unidentified").all()
    with pytest.raises(ValueError, match="six"):
        rf_reference_comparison(data.iloc[:1])


@pytest.fixture
def fitted_source(tmp_path, monkeypatch):
    """A diagnosed two-season source, with bytes and panel identity checked end to end."""
    import json
    import dwts.analysis as analysis
    from dwts.data import prepare

    raw = []
    for season in [1, 3]:
        for index, name in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"]):
            raw.append(dict(celebrity_name=name, ballroom_partner=f"P{index}",
                celebrity_industry=["actor/actress", "athlete", "singer/rapper", "model", "tv personality", "engineer", "actor/actress", "athlete"][index],
                season=season, results=["Eliminated Week 1", "Eliminated Week 2", "1st Place", "2nd Place", "3rd Place", "4th Place", "5th Place", "6th Place"][index],
                placement=[8, 7, 1, 2, 3, 4, 5, 6][index], celebrity_age_during_season=20+index*7+season*(index%3+1),
                week1_judge1_score=5+index+season*(index%2+1)/10,
                week2_judge1_score=0 if index == 0 else 6+index+season*(index%3+1)/10,
                week3_judge1_score=0 if index < 2 else 7+index+season*(index%4+1)/10))
    data = tmp_path / "raw.csv"
    pd.DataFrame(raw).to_csv(data, index=False)
    monkeypatch.setattr(analysis, "DATA_SHA256", analysis.digest(data))
    config = dict(seasons=[1, 3], data=dict(outcome_policy="label_first", include_final_week_eliminations=False),
                  fit=dict(withdrawal_policy="include_scored"))
    panel, _ = prepare(data, **config["data"])
    panel["fan_mean"] = panel.judge_share
    source = tmp_path / "source"
    source.mkdir()
    panel.to_csv(source / "votes.csv", index=False)
    (source / "manifest.json").write_text(json.dumps(dict(input_sha256=analysis.digest(data),
        status="completed", all_sampling_diagnostics_pass=True)))
    (source / "run_config.json").write_text(json.dumps(config))
    statuses = [dict(season=season, max_rhat=1.005, min_ess_bulk=500, min_ess_tail=600,
                     sampling_diagnostics_pass=True) for season in config["seasons"]]
    (source / "sampling_status.json").write_text(json.dumps(statuses))
    return source, data


def test_standalone_analysis_completes_without_reference_targets(fitted_source, tmp_path, monkeypatch):
    import json
    import dwts.analysis as analysis

    def unexpected_reference(*args, **kwargs):
        raise AssertionError("A standalone run must not invoke historical target comparison")

    monkeypatch.setattr(analysis, "reference_comparison", unexpected_reference)
    source, data = fitted_source
    out = tmp_path / "standalone"
    result = analysis.run(source, data, out)
    assert result["execution_completed"] and result["exit_code"] == 0
    assert result["fitted_seasons"] == [1, 3]
    assert result["ols_status"] == "completed"
    assert "numerical_replication_established" not in result
    assert "reference_comparison" not in result and not (out / "reference_comparison").exists()
    assert (out / "ols" / "ols_coefficients.csv").exists()
    summary = pd.read_csv(out / "rule_summary.csv")
    assert summary.supported_weeks.eq(2).all() and summary.undefined_weeks.eq(0).all()
    assert "report_reference" not in summary and "numerical_match" not in summary
    scope = json.loads((out / "analysis_scope.json").read_text())
    assert scope["mechanism_scenarios"]["trajectory"].startswith("fixed observed")
    assert scope["diagnostics_gate"]["min_ess_bulk"] == 400
    assert any("not observed audience" in text for text in result["research_limitations"])


def test_reference_comparison_is_explicit_and_remains_unresolved(fitted_source, tmp_path):
    import json
    from dwts.analysis import run

    source, data = fitted_source
    out = tmp_path / "with-reference"
    result = run(source, data, out, compare_reference=True)
    assert result["execution_completed"] and result["exit_code"] == 2
    reference = json.loads((out / "reference_comparison" / "status.json").read_text())
    assert not reference["numerical_agreement_established"]
    assert reference["comparison_status"] == "unresolved_reference_protocols"
    comparisons = pd.read_csv(out / "reference_comparison" / "table5_comparison.csv")
    # Keep the published target intact despite this deliberately different population.
    row = comparisons[comparisons.scenario.eq("S3-27") & comparisons.metric.eq("percentage")].iloc[0]
    assert row.reported_correct == 189 and row.reported_weeks == 192
    assert row.comparison_status == "denominator_mismatch" and pd.isna(row.numerical_match)


@pytest.mark.parametrize("corruption", ["false_pass", "high_rhat", "low_ess", "wrong_season", "duplicate_season"])
def test_invalid_source_diagnostics_rejected_before_outputs(fitted_source, tmp_path, corruption):
    import json
    from dwts.analysis import run

    source, data = fitted_source
    path = source / "sampling_status.json"
    statuses = json.loads(path.read_text())
    if corruption == "false_pass":
        statuses[0]["sampling_diagnostics_pass"] = False
    elif corruption == "high_rhat":
        statuses[0]["max_rhat"] = 1.02
    elif corruption == "low_ess":
        statuses[0]["min_ess_tail"] = 399
    elif corruption == "wrong_season":
        statuses[0]["season"] = 2
    else:
        statuses[0]["season"] = statuses[1]["season"]
    path.write_text(json.dumps(statuses))
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="diagnostic"):
        run(source, data, out)
    assert not out.exists()


def test_changed_preprocessing_rejected_before_outputs(fitted_source, tmp_path):
    import json
    from dwts.analysis import run

    source, data = fitted_source
    path = source / "run_config.json"
    config = json.loads(path.read_text())
    config["data"]["exclude_withdrawn_contestants"] = True
    # Modify a frozen score to remain numerically valid but violate raw-data identity.
    votes = pd.read_csv(source / "votes.csv")
    votes.loc[0, "judge_mean"] += .001
    votes.to_csv(source / "votes.csv", index=False)
    path.write_text(json.dumps(config))
    out = tmp_path / "wrong-panel"
    with pytest.raises(ValueError, match="preprocessing"):
        run(source, data, out)
    assert not out.exists()


def test_local_pdf_requires_explicit_reference_opt_in(fitted_source, tmp_path):
    from dwts.analysis import run

    source, data = fitted_source
    out = tmp_path / "implicit-reference"
    with pytest.raises(ValueError, match="requires.*--compare-reference"):
        run(source, data, out, report_pdf=tmp_path / "never-read.pdf")
    assert not out.exists()


def test_single_declared_season_is_valid_without_invented_cluster_inference(fitted_source, tmp_path):
    import json
    from dwts.analysis import run

    source, data = fitted_source
    config = json.loads((source / "run_config.json").read_text())
    config["seasons"] = [3]
    (source / "run_config.json").write_text(json.dumps(config))
    statuses = json.loads((source / "sampling_status.json").read_text())
    (source / "sampling_status.json").write_text(json.dumps([row for row in statuses if row["season"] == 3]))
    votes = pd.read_csv(source / "votes.csv", float_precision="round_trip")
    votes[votes.season.eq(3)].to_csv(source / "votes.csv", index=False)
    result = run(source, data, tmp_path / "one-season")
    assert result["execution_completed"] and result["exit_code"] == 0
    assert result["fitted_seasons"] == [3]
    assert result["ols_status"] == "unavailable_fewer_than_two_seasons"
