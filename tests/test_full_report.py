import numpy as np
import pandas as pd
from dwts.verification import compare_numbers
from dwts.full_report import literal_spearman, kendall_a, compatible_denominators, table5_count_match


def test_unreported_denominators_never_count_as_validation():
    x=pd.DataFrame({'metric':['table8_rank_none_rounds'], 'report_value':[301.],
                    'reconstruction_value':[301.]})
    y=compare_numbers(x).iloc[0]
    assert y.verdict=='scope_assumption_not_validation'
    assert y.reference_class=='inferred_scope_not_reported'
    assert compatible_denominators()==[300,301]
    twice=compare_numbers(compare_numbers(x))
    assert twice.legacy_assumed_reference_value.iloc[0]==301
    assert pd.isna(twice.report_value.iloc[0])


def test_literal_correlations_are_labeled_distinct_from_tie_aware():
    assert literal_spearman([1,2,3],[1,2,3])==1
    assert literal_spearman([1,2,3],[3,2,1])==-1
    assert kendall_a([1,2,3],[1,2,3])==1
    assert kendall_a([1,2,3],[3,2,1])==-1
    assert np.isclose(kendall_a([1,1,2],[1,2,3]),2/3)


def test_literal_rank_shortcut_does_not_pretend_to_handle_constant_ties():
    # This explicitly reproduces the printed shortcut, which is not
    # tie-corrected Spearman; it can be nonzero for a constant input.
    assert np.isclose(literal_spearman([1,1,1],[1,2,3]),.5)


def test_unsupported_extra_week_cannot_create_table5_pass():
    assert table5_count_match(True,56,56,56,44,44)
    assert not table5_count_match(True,56,57,56,44,44)
    assert not table5_count_match(True,47,56,56,32,44)
    assert not table5_count_match(False,56,56,56,44,44)
