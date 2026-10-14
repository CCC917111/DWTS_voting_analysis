"""Keep the default suite self-contained; opt in explicitly to external evidence.

Original v5 test files are unchanged. Marks are applied by exact node name so
missing research inputs never silently masquerade as executed validation.
"""
import pytest

OFFICIAL_DATA = {
    "test_official_data_counts_and_normalization",
    "test_recovery_inclusion_counts_match_report_denominators",
    "test_resume_mismatch_does_not_mutate_prior_results",
    "test_full_reconstruction_v5_config_is_accepted_without_fitting",
    "test_default_option_preserves_legacy_panel_and_audit",
    "test_zero_first_changes_only_s18_diana_constraint",
    "test_exclusion_removes_ten_contestants_in_nine_seasons_before_normalization",
    "test_exact_reuse_comparison_rejects_one_ulp_change",
    "test_preflight_accepts_frozen_original_season_inputs",
    "test_preflight_refuses_modified_raw",
    "test_preflight_refuses_changed_original_vote_input",
}
ARCHIVE = {
    "test_explicit_identity_config_routes_both_models_and_retains_source",
    "test_current_audit_certificates_and_weighting_counterexample",
    "test_saved_audit_has_complete_fixed_comparison_coverage",
    "test_preflight_accepts_frozen_original_season_inputs",
    "test_preflight_refuses_modified_raw",
    "test_preflight_refuses_changed_original_vote_input",
}

def pytest_collection_modifyitems(items):
    for item in items:
        name = item.originalname or item.name
        if name in OFFICIAL_DATA:
            item.add_marker(pytest.mark.official_data)
        if name in ARCHIVE:
            item.add_marker(pytest.mark.archive)
