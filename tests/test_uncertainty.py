"""Joint-draw counterfactual checks independent of official data or chains."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from dwts.rules import eliminate
from dwts.uncertainty import (Policy, SeasonResult, aggregate_tables, evaluate_season,
                              included_weeks, policies_from_protocol, policy_draws,
                              load_season, read_protocol, result_groups, run, stratified_draw_indices,
                              summarize, validate_sensitivity, validate_source_implementation)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "uncertainty_study.json"
POLICIES = policies_from_protocol(read_protocol(CONFIG))


def frame():
    return pd.DataFrame({"season": [3]*6, "week": [1]*3+[2]*3,
                         "contestant_id": ["A", "B", "C"]*2,
                         "judge_mean": [10., 9., 1.]*2,
                         "eliminated": [False, False, True]*2,
                         "withdrawal": [False]*6, "is_final_week": [False]*6})


def test_draw_selection_reproducible_stratified_and_without_replacement():
    selected = stratified_draw_indices(4000, 512, 7, 3, 0)
    assert np.array_equal(selected, stratified_draw_indices(4000, 512, 7, 3, 0))
    assert len(set(selected)) == 512 and (np.diff(selected) > 0).all()
    bounds = np.floor(np.linspace(0, 4000, 513)).astype(int)
    assert ((bounds[:-1] <= selected) & (selected < bounds[1:])).all()
    assert not np.array_equal(selected, stratified_draw_indices(4000, 512, 7, 3, 1))


def test_season_loader_rejects_requested_draw_count_above_stored_draws(tmp_path):
    import xarray as xr

    source = tmp_path / "source"
    season_dir = source / "season_03"
    season_dir.mkdir(parents=True)
    data = frame().assign(fan_mean=1/3, judge_z=0.)
    data.to_csv(season_dir / "votes.csv", index=False)
    config = {"fit": {"chains": 2, "draws": 4}}
    status = {"season": 3, "config": config["fit"], "sampling_diagnostics_pass": True,
              "max_rhat": 1.001, "min_ess_bulk": 500., "min_ess_tail": 600.}
    (season_dir / "fit_status.json").write_text(json.dumps(status))
    parameters = [f"alpha[{name}]" for name in ["A", "B", "C"]] + ["beta", "sigma"] + [
        f"fan_share[{i}]" for i in range(len(data))]
    pd.DataFrame({"r_hat": 1.001, "ess_bulk": 500., "ess_tail": 600.},
                 index=parameters).to_csv(season_dir / "diagnostics.csv", index_label="parameter")
    posterior = xr.Dataset(
        {"fan_share": (("chain", "draw", "observation"), np.full((2, 4, len(data)), 1/3))},
        coords={"chain": np.arange(2), "draw": np.arange(4),
                "observation": np.arange(len(data)), "contestant": ["A", "B", "C"]})
    posterior.to_netcdf(season_dir / "posterior.nc", group="posterior", engine="h5netcdf")
    # This exercises the actual source loader boundary. The stratifier's own
    # range check cannot protect this path if the loader silently caps first.
    with pytest.raises(ValueError, match="Requested draws per chain exceed stored"):
        load_season(source, data, 3, config, seed=7, count=8)


def test_fixed_protocol_does_not_allow_outcome_selected_weights(tmp_path):
    cfg = json.loads(CONFIG.read_text())
    cfg["judge_weights"] = [.4, .5, .6]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="fixed 18 policies"):
        read_protocol(path)


def test_vectorized_policies_agree_with_scalar_rules_including_exact_rank_ties():
    rng = np.random.default_rng(321)
    for n in [3, 5, 9]:
        j = rng.integers(1, 11, size=n).astype(float)
        p = rng.dirichlet(np.ones(n), size=12)
        z = np.log(p)
        ids = np.array([f"C{i:02d}" for i in range(n)])
        policies = POLICIES + (Policy("rank", .4, "none"), Policy("rank", .6, "judge_save"))
        actual, _ = policy_draws(j, p, z, ids, policies)
        for draw, shares in enumerate(p):
            for k, policy in enumerate(policies):
                expected = eliminate(j, shares, ids, policy.rule, policy.judge_weight,
                                     policy.mechanism)[0]
                assert actual[draw, k] == expected


def test_latent_order_survives_softmax_underflow():
    # Numeric shares merge A/B at zero. Their distinct latent fan ranks must
    # survive: all combined rank sums tie, then A loses on fewer fan votes.
    z = np.tile([-1000., -999., 0.], (4, 1))
    p = softmax(z, axis=1)
    losers, _ = policy_draws([10., 9., 1.], p, z, ["A", "B", "C"],
                             (Policy("rank", .5, "none"),))
    assert (p[:, :2] == 0).all()
    assert (losers == 0).all()


def test_percentage_judge_ties_use_latent_vote_oracle_after_underflow():
    # A/Z lexical order reverses the true latent order. Numeric zero shares
    # alone would use the lexical tie-break and remove the wrong contestant.
    j, ids = np.array([10., 10., 10.]), np.array(["Z", "A", "C"])
    z = np.tile([-1000., -999., 0.], (4, 1))
    p = softmax(z, axis=1)
    policies = tuple(Policy("percentage", .5, mechanism)
                     for mechanism in ["none", "judge_save", "fan_save"])
    losers, _ = policy_draws(j, p, z, ids, policies)
    for draw in range(len(z)):
        combined = .5 * j / j.sum() + .5 * p[draw]
        pool = sorted(range(3), key=lambda i: (combined[i], z[draw, i], j[i], ids[i]))[:2]
        oracle = [pool[0], min(pool, key=lambda i: (j[i], z[draw, i], ids[i])),
                  min(pool, key=lambda i: (z[draw, i], j[i], ids[i]))]
        assert np.array_equal(losers[draw], oracle)
        assert (losers[draw] == 0).all()
        assert eliminate(j, p[draw], ids, "percentage")[0] == 1


def test_one_exit_does_not_depend_on_observed_elimination_count():
    data = frame()
    z = np.tile([-2., 0., 1., -2., 0., 1.], (8, 1))
    p = np.concatenate([softmax(z[:, :3], axis=1), softmax(z[:, 3:], axis=1)], axis=1)
    first = evaluate_season(data, p, z, POLICIES)
    data["eliminated"] = [True]*3+[False]*3
    second = evaluate_season(data, p, z, POLICIES)
    assert np.array_equal(first.counts, second.counts)


def test_scope_excludes_final_and_withdrawal_weeks_without_redefining_labels():
    data = pd.concat([frame(), frame().iloc[:3].assign(week=3)], ignore_index=True)
    data.loc[data.week.eq(2), "is_final_week"] = True
    data.loc[(data.week.eq(3)) & (data.contestant_id.eq("A")), "withdrawal"] = True
    assert included_weeks(data) == (1,)


def test_intervals_preserve_joint_week_dependence_and_paired_differences():
    # In half the posterior draws both weeks are controversial; in the other
    # half neither is. Treating weeks independently would narrow the interval.
    counts = np.zeros((8, len(POLICIES), 3), dtype=int)
    counts[:4, :, 0] = 2
    saving = POLICIES.index(Policy("rank", .5, "judge_save"))
    counts[:, saving, 0] = 0
    result = SeasonResult(3, (1, 2), 6, counts, POLICIES)
    summary, contrast = aggregate_tables([result], "synthetic")
    row = summary[(summary.level == "overall") & (summary.rule == "rank") &
                  (summary.judge_weight == .5) & (summary.mechanism == "none") &
                  (summary.metric == "controversial_week_fraction")].iloc[0]
    assert row["mean"] == .5 and row.lower == 0 and row.upper == 1
    comparison = contrast[(contrast.level == "overall") &
                          (contrast.target == "rank:w0.5:judge_save") &
                          (contrast.contrast == "save_vs_none")].iloc[0]
    assert comparison["mean"] == -.5
    assert comparison.posterior_probability_lower == .5
    assert comparison.posterior_probability_equal == .5
    assert comparison.posterior_probability_higher == 0


def test_overall_aggregation_weights_weeks_and_never_rows():
    counts1 = np.zeros((8, len(POLICIES), 3), int)
    counts1[:, :, 0] = 1
    counts2 = np.zeros_like(counts1)
    r1 = SeasonResult(1, (1,), 10, counts1, POLICIES)
    r2 = SeasonResult(3, (1, 2, 3), 12, counts2, POLICIES)
    summary, _ = aggregate_tables([r1, r2], "synthetic")
    overall = summary[(summary.level == "overall") &
                      (summary.metric == "controversial_week_fraction")]
    assert overall["mean"].eq(.25).all() and overall.weeks.eq(4).all()
    assert overall.contestant_rows.eq(22).all()


def test_summary_and_contrast_exports_contain_no_person_or_draw_identifiers():
    counts = np.zeros((8, len(POLICIES), 3), int)
    result = SeasonResult(3, (1, 2), 6, counts, POLICIES)
    summary, contrast = aggregate_tables([result], "synthetic")
    banned = {"contestant_id", "name", "eliminated_id", "draw", "chain", "fan_share"}
    assert not banned & set(summary) and not banned & set(contrast)
    for row in contrast.itertuples():
        assert row.posterior_probability_lower + row.posterior_probability_equal + row.posterior_probability_higher == 1


@pytest.mark.parametrize("change", ["shares", "nan", "negative_score", "duplicate_id"])
def test_invalid_policy_inputs_fail_before_any_summary(change):
    j, p, z, ids = np.array([10., 9., 1.]), np.tile([.1, .2, .7], (4, 1)), np.tile([-2., -1., 0.], (4, 1)), ["A", "B", "C"]
    if change == "shares":
        p[0, 0] = .5
    elif change == "nan":
        z[0, 0] = np.nan
    elif change == "negative_score":
        j[0] = -1
    else:
        ids[0] = ids[1]
    with pytest.raises(ValueError, match="Invalid"):
        policy_draws(j, p, z, ids, POLICIES)


def test_matched_sensitivity_rejects_unreported_additional_changes():
    base = {"data": {"outcome_policy": "label_first"}, "fit": {"penalty_lambda": 1., "likelihood": "report", "seed": 3}, "seasons": [3, 4]}
    alt = {"data": base["data"], "fit": dict(base["fit"], penalty_lambda=.1), "seasons": [3]}
    declared = {"changed_fit_field": "penalty_lambda", "changed_fit_value": .1}
    validate_sensitivity(base, alt, declared)
    alt["fit"]["seed"] = 4
    with pytest.raises(ValueError, match="additional"):
        validate_sensitivity(base, alt, declared)


@pytest.mark.parametrize("field", ["input_sha256", "code_sha256", "python", "packages"])
def test_matched_sensitivity_requires_recorded_same_model_implementation(field):
    original = {"input_sha256": "data", "code_sha256": "model", "python": "3.12", "packages": {"numpy": "2.3"}}
    alternative = dict(original)
    validate_source_implementation(original, alternative)
    alternative[field] = "changed"
    with pytest.raises(ValueError, match="implementation/environment"):
        validate_source_implementation(original, alternative)


def test_existing_output_is_never_overwritten_even_before_input_loading(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="never overwrite"):
        run(source, CONFIG, out, tmp_path / "missing.csv")
    assert sentinel.read_text() == "preserve"


def test_output_nested_in_source_is_rejected_without_creating_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    out = source / "new"
    with pytest.raises(ValueError, match="separate"):
        run(source, CONFIG, out, tmp_path / "missing.csv")
    assert not out.exists()


def test_results_require_matching_draws_and_unique_seasons():
    result = SeasonResult(3, (1,), 3, np.zeros((8, len(POLICIES), 3), int), POLICIES)
    with pytest.raises(ValueError, match="unique"):
        list(result_groups([result, result]))
    with pytest.raises(ValueError, match="joint draw count"):
        list(result_groups([result, replace(result, season=4, counts=result.counts[:4])]))


def test_finite_sample_summary_uses_equal_tailed_quantiles():
    summary = summarize(np.array([0., 0., 1., 1.]))
    assert summary == {"mean": .5, "lower": 0., "upper": 1.}
    with pytest.raises(ValueError, match="finite"):
        summarize(np.array([0., 1., 2., np.nan]))
