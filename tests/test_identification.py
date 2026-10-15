"""Independent algebra, joint-policy witnesses, and aggregate-only study exports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

import dwts.identification as identification
from dwts.identification import (InfeasibleRegionError, PRIVATE_ARTIFACTS, PUBLIC_ARTIFACTS,
                                 analytical_coordinate_bounds, attainable_unique_exit,
                                 main, percentage_region, read_protocol, run_study,
                                 sharp_coordinate_bounds)


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return read_protocol(ROOT / "configs/identification_study.json")


def test_two_person_algebra_is_not_n_ge_three_survivor_formula():
    # B exits iff .5*.4+.5*(1-pA)<=.5*.6+.5*pA, hence pA>=.4.
    answer = sharp_coordinate_bounds([.6, .4], 1)
    np.testing.assert_allclose(answer.lower, [.4, 0.], atol=1e-12)
    np.testing.assert_allclose(answer.upper, [1., .6], atol=1e-12)
    assert answer.interior_margin > 0
    assert answer.lp_optimizations == 5
    assert answer.max_analytic_endpoint_difference < 1e-12


def test_three_person_exact_bounds_and_separate_minimizers():
    # pB>=pA+1/5 and pC>=pA+3/10; 3*pA+1/2<=1.
    answer = sharp_coordinate_bounds([.5, .3, .2], 0)
    np.testing.assert_allclose(answer.lower, [0., 1/5, 3/10], atol=1e-12)
    np.testing.assert_allclose(answer.upper, [1/6, 7/10, 4/5], atol=1e-12)
    # Coordinate extrema are individually feasible and NOT a common vector.
    assert answer.lower.sum() == pytest.approx(.5)
    assert answer.upper.sum() > 1
    for witnesses, target in [(answer.lower_witnesses, answer.lower),
                              (answer.upper_witnesses, answer.upper)]:
        for k, p in enumerate(witnesses):
            assert p.sum() == pytest.approx(1)
            assert p.min() >= -1e-12
            merit = .5 * np.array([.5, .3, .2]) + .5 * p
            assert merit[0] <= merit[1:].min() + 1e-12
            assert p[k] == pytest.approx(target[k])
    # Mixing each boundary optimizer with the strict interior approaches the
    # endpoint while giving all-positive shares and a unique observed exit.
    for boundary in np.vstack([answer.lower_witnesses, answer.upper_witnesses]):
        p = (1 - 1e-5) * boundary + 1e-5 * answer.interior_witness
        merit = .5 * np.array([.5, .3, .2]) + .5 * p
        assert p.min() > 0
        assert merit[0] < merit[1:].min()


def test_random_lp_endpoints_against_independent_formulas_and_monotone_root():
    rng = np.random.default_rng(826177)
    infeasible = 0
    for n in [2, 3, 5, 9]:
        for _ in range(8):
            j = rng.dirichlet(np.ones(n))
            e = int(rng.integers(n))
            w = float(rng.uniform(.05, .95))
            survivors = np.flatnonzero(np.arange(n) != e)
            # This oracle does not call the production formula or LP builder.
            offsets = [(w * (j[e] - j[k])) / (1 - w) for k in survivors]
            required_mass = sum(max(0., d) for d in offsets)
            if required_mass > 1:
                infeasible += 1
                with pytest.raises(InfeasibleRegionError):
                    sharp_coordinate_bounds(j, e, w)
                continue
            root = brentq(lambda t: t + sum(max(0., t + d) for d in offsets) - 1., 0., 1.)
            answer = sharp_coordinate_bounds(j, e, w)
            assert answer.lower[e] == pytest.approx(0., abs=1e-10)
            assert answer.upper[e] == pytest.approx(root, abs=1e-10)
            for k, offset in zip(survivors, offsets):
                expected_lower = max(0., offset) if n >= 3 else 1 - root
                assert answer.lower[k] == pytest.approx(expected_lower, abs=1e-10)
                assert answer.upper[k] == pytest.approx(1 - required_mass + max(0., offset), abs=1e-10)
            if n >= 3:
                np.testing.assert_allclose((answer.upper - answer.lower)[survivors], 1 - required_mass, atol=1e-10)
    assert infeasible > 0


def test_infeasible_and_boundary_only_region_are_distinguished():
    with pytest.raises(InfeasibleRegionError):
        sharp_coordinate_bounds([1., 0., 0.], 0, .9)
    with pytest.raises(InfeasibleRegionError):
        analytical_coordinate_bounds([1., 0., 0.], 0, .9)
    # The only configuration is p=(0,.4,.6), and all combined merits tie.
    boundary = sharp_coordinate_bounds([.5, .3, .2], 0, 2/3)
    np.testing.assert_allclose(boundary.lower, [0., .4, .6], atol=1e-12)
    np.testing.assert_allclose(boundary.upper, boundary.lower, atol=1e-12)
    assert boundary.interior_margin < 1e-12
    unique = attainable_unique_exit([.5, .3, .2], 0, 0, 2/3, observed_weight=2/3)
    assert not unique.attainable
    assert unique.status in {"boundary_only", "below_strict_margin_tolerance"}


def test_counterfactual_support_by_hand_and_unchanged_rule_oracle():
    j = [.6, .4]
    # Observed B requires pA>.4. At w=.25 candidate A requires
    # pA<7/15, so the joint OPEN interval is nonempty.
    result = attainable_unique_exit(j, 1, 0, .25)
    assert result.attainable
    assert .4 < result.witness[0] < 7/15
    assert result.margin > 1e-8
    # At w=.75 candidate A instead requires pA<.2, impossible jointly.
    impossible = attainable_unique_exit(j, 1, 0, .75)
    assert not impossible.attainable
    assert impossible.status == "weakly_infeasible"
    assert impossible.witness is None
    assert not attainable_unique_exit(j, 1, 0, .5).attainable
    assert attainable_unique_exit(j, 1, 1, .5).attainable


def test_policy_witness_reuses_one_vector_for_both_rules():
    j = np.array([.5, .3, .2])
    for new_weight in [.25, .5, .75]:
        support = []
        for candidate in range(3):
            result = attainable_unique_exit(j, 0, candidate, new_weight)
            if not result.attainable:
                continue
            support.append(candidate)
            p = result.witness
            assert p.sum() == pytest.approx(1)
            assert np.min(p) > 0
            old = .5 * j + .5 * p
            new = new_weight * j + (1 - new_weight) * p
            assert old[0] < old[1:].min()
            assert new[candidate] < np.min(np.delete(new, candidate))
        if new_weight == .5:
            assert support == [0]


def test_permutation_equivariance_of_bounds_and_policy_support():
    j = np.array([.1, .4, .2, .3])
    permutation = np.array([3, 0, 1, 2])
    original = sharp_coordinate_bounds(j, 1)
    e = int(np.flatnonzero(permutation == 1)[0])
    permuted = sharp_coordinate_bounds(j[permutation], e)
    np.testing.assert_allclose(permuted.lower, original.lower[permutation], atol=1e-10)
    np.testing.assert_allclose(permuted.upper, original.upper[permutation], atol=1e-10)
    for candidate in range(len(j)):
        old = attainable_unique_exit(j, 1, candidate, .25)
        new = attainable_unique_exit(j[permutation], e, int(np.flatnonzero(permutation == candidate)[0]), .25)
        assert old.attainable == new.attainable
        if old.margin is not None:
            assert old.margin == pytest.approx(new.margin, abs=1e-10)


@pytest.mark.parametrize("j,e,w", [([.5], 0, .5), ([[.5, .5]], 0, .5), ([.5, .6], 0, .5),
    ([-.1, 1.1], 0, .5), ([np.nan, .5], 0, .5), ([np.inf, 0.], 0, .5),
    ([.5, .5], True, .5), ([.5, .5], .5, .5), ([.5, .5], 2, .5), ([.5, .5], -1, .5),
    ([.5, .5], 0, 0), ([.5, .5], 0, 1), ([.5, .5], 0, np.nan), ([.5, .5], 0, True),
    ([.5, .5], 0, ".5")])
def test_invalid_math_inputs_fail_before_optimization(j, e, w):
    with pytest.raises(ValueError):
        sharp_coordinate_bounds(j, e, w)
    with pytest.raises(ValueError):
        percentage_region(j, e, w)


@pytest.mark.parametrize("kwargs", [{"candidate": True}, {"candidate": 3}, {"counterfactual_weight": 0},
                                   {"tolerance": np.nan}, {"strict_margin_tolerance": 0}, {"tolerance": "1e-8"}])
def test_invalid_policy_controls(kwargs):
    inputs = {"judge_shares": [.5, .3, .2], "observed_eliminated": 0,
              "candidate": 1, "counterfactual_weight": .25, **kwargs}
    with pytest.raises(ValueError):
        attainable_unique_exit(**inputs)


@pytest.mark.parametrize("update", [{"schema_version": True}, {"candidate_seasons": [2, 3]},
    {"candidate_seasons": [3, 3]}, {"candidate_seasons": [4, 3]}, {"candidate_seasons": []},
    {"minimum_active": 2}, {"minimum_active": True}, {"outcome_policy": "zero_first"},
    {"include_final_week_eliminations": True}, {"exclude_withdrawal_weeks": False},
    {"ordinary_exit_count": 2}, {"counterfactual_judge_weights": [.25, .75]},
    {"counterfactual_judge_weights": [.5, .5]}, {"counterfactual_judge_weights": [".5"]},
    {"feasibility_tolerance": 0}, {"strict_margin_tolerance": np.inf}, {"extra": 1}])
def test_protocol_rejects_unsupported_or_ambiguous_inputs(tmp_path, update):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({**protocol(), **update}))
    with pytest.raises(ValueError):
        read_protocol(path)


def fixture_data(path):
    rows = []
    for index, (name, result, placement, scores) in enumerate([
        ("Alpha", "1st Place", 1, [8, 8, 8]),
        ("Beta", "2nd Place", 2, [7, 7, 7]),
        ("Gamma", "Eliminated Week 2", 3, [6, 6, 0]),
        ("Delta", "Eliminated Week 1", 4, [5, 0, 0])]):
        rows.append({"celebrity_name": name, "ballroom_partner": f"Partner {index}",
            "celebrity_industry": "actor/actress", "celebrity_age_during_season": 30,
            "season": 3, "results": result, "placement": placement,
            **{f"week{week}_judge1_score": score for week, score in enumerate(scores, 1)}})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_small_study_source_integrity_aggregate_allowlist_and_no_overwrite(tmp_path, monkeypatch):
    data, config, out = tmp_path / "source.csv", tmp_path / "config.json", tmp_path / "result"
    fixture_data(data)
    config.write_text(json.dumps({**protocol(), "candidate_seasons": [3]}))
    monkeypatch.chdir(tmp_path)
    assert run_study(data, config, out) == out
    status = json.loads((out / "status.json").read_text())
    assert status["eligible_weeks"] == 2
    assert status["eligible_contestant_weeks"] == 7
    assert status["public_artifacts"] == [*PUBLIC_ARTIFACTS, "status.json"]
    assert status["private_artifacts"] == list(PRIVATE_ARTIFACTS)
    assert set(status["artifacts_sha256"]) == set(PUBLIC_ARTIFACTS)
    for filename in PUBLIC_ARTIFACTS:
        assert hashlib.sha256((out / filename).read_bytes()).hexdigest() == status["artifacts_sha256"][filename]
        text = (out / filename).read_text()
        assert "Alpha" not in text and "Gamma" not in text
        assert "witness_id_order" not in text
    for filename in PRIVATE_ARTIFACTS:
        assert (out / filename).stat().st_mode & 0o777 == 0o600
    summary = pd.read_csv(out / "width_summary.csv")
    assert set(summary.population) == {"all", "survivor", "observed_exit", "all_mean_width"}
    eligibility = pd.read_csv(out / "eligibility_summary.csv")
    assert eligibility.loc[eligibility.reason.eq("candidate_weeks"), "count"].item() == 3
    policy = pd.read_csv(out / "policy_ambiguity.csv")
    assert policy.loc[policy.counterfactual_judge_weight.eq(.5), "singleton_weeks"].item() == 2
    checks = json.loads((out / "numerical_checks.json").read_text())
    assert checks["max_independent_analytic_endpoint_difference"] < 1e-10
    assert checks["strict_observed_region_certified_weeks"] == 2
    with pytest.raises(ValueError, match="fresh output"):
        run_study(data, config, out)


def test_cli_reuses_explicit_argv_without_cwd_assumptions(tmp_path, monkeypatch):
    data, config, out = tmp_path / "data.csv", tmp_path / "config.json", tmp_path / "run"
    fixture_data(data)
    config.write_text(json.dumps({**protocol(), "candidate_seasons": [3]}))
    monkeypatch.chdir(tmp_path)
    main(["--data", str(data), "--config", str(config), "--out", str(out)])
    assert (out / "status.json").is_file()


def test_source_change_during_input_preparation_cannot_mark_run_complete(tmp_path, monkeypatch):
    data, config, out = tmp_path / "data.csv", tmp_path / "config.json", tmp_path / "run"
    fixture_data(data)
    config.write_text(json.dumps({**protocol(), "candidate_seasons": [3]}))
    original_prepare = identification.prepare
    def changed_source(*args, **kwargs):
        config.write_text(config.read_text() + "\n")
        return original_prepare(*args, **kwargs)
    monkeypatch.setattr(identification, "prepare", changed_source)
    with pytest.raises(RuntimeError, match="source changed.*config"):
        run_study(data, config, out)
    assert not (out / "status.json").exists()


@pytest.mark.official_data
def test_official_population_bounds_and_policy_reproduce_private_pilot(tmp_path):
    data = ROOT / "data/raw/2026_MCM_Problem_C_Data.csv"
    if not data.exists():
        pytest.skip("Separately obtained COMAP data is unavailable")
    out = run_study(data, ROOT / "configs/identification_study.json", tmp_path / "official")
    status = json.loads((out / "status.json").read_text())
    assert status["eligible_weeks"] == 164
    assert status["eligible_contestant_weeks"] == 1343
    widths = pd.read_csv(out / "width_summary.csv").set_index(["population", "weighting"])
    assert widths.loc[("survivor", "contestant_week"), "median"] == pytest.approx(.996303, abs=1e-6)
    assert widths.loc[("observed_exit", "contestant_week"), "median"] == pytest.approx(.136029, abs=1e-6)
    ambiguity = pd.read_csv(out / "policy_ambiguity.csv").set_index("counterfactual_judge_weight")
    assert ambiguity.ambiguous_weeks.to_dict() == {.25: 155, .5: 0, .75: 87}
    assert ambiguity.borderline_candidate_count.sum() == 0
