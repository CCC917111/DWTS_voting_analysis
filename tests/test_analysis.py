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
    assert result["author_scope_identified"] is False


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
