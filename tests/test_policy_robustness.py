"""Independent endpoint, pairing and finite-scenario policy oracles."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from scipy.stats import rankdata

from dwts.rules import eliminate
from dwts.prior_sensitivity import transport_draws
from dwts.policy_robustness import (
    METRICS, REFERENCE, SeasonMetrics, aggregate, evaluate_season, pareto_relation,
    posterior_dominance_support, read_protocol, scenario_tables,
    validate_matched_frames, weekly_metrics,
)
from dwts.uncertainty import Policy, policies_from_protocol

CONFIG = Path(__file__).resolve().parents[1] / "configs/policy_robustness_study.json"
POLICIES = policies_from_protocol(read_protocol(CONFIG))


def frame():
    return pd.DataFrame(dict(season=[28]*8, week=[1]*4+[2]*4,
        contestant_id=["A", "B", "C", "D"]*2, judge_mean=[10., 9., 9., 1.]*2,
        eliminated=[False, False, False, True]*2, withdrawal=[False]*8,
        is_final_week=[False]*8, rule=["rank"]*8))


def test_six_metrics_match_scalar_elimination_and_average_rank_oracle():
    rng = np.random.default_rng(7352)
    for n in [3, 4, 7]:
        judges = rng.integers(1, 9, n).astype(float)
        logits = rng.integers(-3, 4, (8, n)).astype(float)
        shares = softmax(logits, axis=1)
        ids = np.array([f"P{i}" for i in range(n)])
        actual = weekly_metrics(judges, shares, logits, ids, POLICIES)
        jr = rankdata(-judges, method="average")
        for draw in range(len(shares)):
            fr = rankdata(-logits[draw], method="average")
            for k, policy in enumerate(POLICIES):
                loser = eliminate(judges, shares[draw], ids, policy.rule,
                                  policy.judge_weight, policy.mechanism)[0]
                oracle = [jr[loser] <= 2 and fr[loser] > n/2,
                    jr[loser] > n-2 and fr[loser] <= n/2,
                    (jr[loser]-1)/(n-1), (fr[loser]-1)/(n-1),
                    jr[loser] <= n/2, fr[loser] <= n/2]
                assert np.array_equal(actual[draw, k], oracle)


def test_fractional_judge_ties_and_latent_underflow_are_not_rounded():
    # D is removed by fan-save; its tied judge rank is 2.5, outside top half.
    judges = [10., 9., 1., 9.]
    logits = np.tile([0., -998., -999., -1000.], (4, 1))
    policies = (Policy("percentage", .25, "fan_save"),)
    values = weekly_metrics(judges, softmax(logits, axis=1), logits,
                            ["A", "B", "C", "D"], policies)
    assert np.all(values[:, 0, 2] == .5)
    assert np.all(values[:, 0, 3] == 1)
    assert np.all(values[:, 0, 4:] == 0)


def test_joint_weeks_preserved_and_final_withdrawal_scope_excluded():
    data = frame()
    logits = np.array([[-3., 0., 0., 3.], [3., 0., 0., -3.]]*4)
    shares = softmax(logits, axis=1)
    logits, shares = np.tile(logits, (1, 2)), np.tile(shares, (1, 2))
    result = evaluate_season(data, shares, logits, POLICIES)
    assert result.weeks == (1, 2)
    first_week = weekly_metrics(data.judge_mean[:4], shares[:, :4], logits[:, :4],
                                data.contestant_id[:4], POLICIES)
    assert np.array_equal(result.totals, 2 * first_week)
    rates, population = aggregate([result])
    assert np.array_equal(rates, first_week)
    assert population["weeks"] == 2
    changed = data.copy()
    changed.loc[changed.week.eq(2), "is_final_week"] = True
    assert evaluate_season(changed, shares, logits, POLICIES).weeks == (1,)
    changed.loc[changed.week.eq(1) & changed.contestant_id.eq("A"), "withdrawal"] = True
    with pytest.raises(ValueError, match="eligible"):
        evaluate_season(changed, shares, logits, POLICIES)


def test_rank_endpoint_invariance_and_scale_one_exact_identity():
    data = frame()
    rng = np.random.default_rng(937)
    logits = rng.normal(size=(12, len(data)))
    shares = np.concatenate([softmax(logits[:, :4], axis=1),
                             softmax(logits[:, 4:], axis=1)], axis=1)
    baseline = evaluate_season(data, shares, logits, POLICIES)
    rank_columns = [k for k, policy in enumerate(POLICIES) if policy.rule == "rank"]
    for scale in [.5, 1., 2.]:
        p, z = transport_draws(data, shares, logits, scale)
        alternative = evaluate_season(data, p, z, POLICIES)
        assert np.array_equal(alternative.totals[:, rank_columns], baseline.totals[:, rank_columns])
        if scale == 1:
            assert np.array_equal(alternative.totals, baseline.totals)


@pytest.mark.parametrize("delta,expected", [
    ([-.1, 0, 0, 0, 0, 0], "target_dominates"),
    ([.1, 0, 0, 0, 0, 0], "reference_dominates"),
    ([-.1, .1, 0, 0, 0, 0], "tradeoff"),
    ([0, 0, .1, 0, 0, 0], "target_dominates"),
    ([0, 0, 0, 0, 0, 0], "equal"),
    ([1e-13, 0, 0, 0, 0, 0], "equal"),
])
def test_pareto_normative_directions_and_tolerance(delta, expected):
    baseline = np.full(6, .5)
    assert pareto_relation(baseline + delta, baseline) == expected


def result(values):
    # Two policies, all six endpoints, eight draws, one observed week.
    values = np.asarray(values, float)
    return SeasonMetrics(28, (1,), 4, np.tile(values, (8, 1, 1)),
                         (REFERENCE, Policy("percentage", .5, "none")))


def test_scenario_denominators_and_no_probability_weighted_robustness():
    baseline = np.full(6, .5)
    better = baseline + [-.1, 0, 0, 0, 0, 0]
    worse = baseline + [.1, 0, 0, 0, 0, 0]
    policies = (REFERENCE, Policy("percentage", .5, "none"))
    scenarios = [("matched", "first", {}, [result([baseline, better])]),
                 ("matched", "second", {}, [result([baseline, worse])])]
    metrics, contrasts, stability, worst = scenario_tables(scenarios, policies, .95, 1e-12)
    row = stability.iloc[0]
    assert row.scenario_count == 2
    assert row.target_dominates_scenarios == row.reference_dominates_scenarios == 1
    assert row.tradeoff_scenarios == row.equal_scenarios == 0
    assert not row.robust_target_dominance and not row.robust_reference_dominance
    selected = worst[worst.policy.eq(policies[1].label) & worst.metric.eq(METRICS[0][0])].iloc[0]
    assert selected.worst_scenario == "second"
    assert selected.worst_conditional_mean == pytest.approx(.6)
    assert selected.best_conditional_mean == pytest.approx(.4)
    assert len(metrics) == len(contrasts) == 2 * 2 * 6
    # Reference vs itself has exactly zero paired differences, no fake uncertainty.
    self_contrasts = contrasts[contrasts.target.eq(REFERENCE.label)]
    assert (self_contrasts[["mean_difference_target_minus_reference", "lower", "upper"]] == 0).all().all()
    assert (self_contrasts.posterior_probability_equal == 1).all()


def test_robust_dominance_requires_each_scenario_not_just_one():
    baseline = np.full(6, .5)
    better = baseline + [-.1, 0, 0, 0, 0, 0]
    policies = (REFERENCE, Policy("percentage", .5, "none"))
    scenarios = [("matched", "first", {}, [result([better, baseline])]),
                 ("matched", "second", {}, [result([better, baseline])])]
    stability = scenario_tables(scenarios, policies, .95, 1e-12)[2].iloc[0]
    assert stability.target_dominates_scenarios == 2
    assert stability.robust_target_dominance
    scenarios[1] = ("matched", "second", {}, [result([baseline, baseline])])
    stability = scenario_tables(scenarios, policies, .95, 1e-12)[2].iloc[0]
    assert not stability.robust_target_dominance
    assert stability.equal_scenarios == 1


def test_posterior_joint_support_is_not_mean_endpoint_dominance():
    baseline = np.full((8, 6), .5)
    alternative = baseline.copy()
    alternative[:4, 0] -= .2
    alternative[4:, 0] += .1
    assert pareto_relation(alternative.mean(axis=0), baseline.mean(axis=0)) == "target_dominates"
    assert posterior_dominance_support(alternative, baseline, 1e-12) == .5


def test_mismatched_cohort_denominators_rejected():
    baseline = np.full(6, .5)
    first = result([baseline, baseline])
    second = result([baseline, baseline])
    second.weeks = (1, 2)
    policies = first.policies
    with pytest.raises(ValueError, match="identical matched denominators"):
        scenario_tables([("same", "a", {}, [first]), ("same", "b", {}, [second])], policies, .95, 1e-12)


def test_matched_rows_scores_and_labels_exact_but_posterior_summaries_may_differ():
    baseline = frame().assign(fan_mean=.25)
    alternative = baseline.copy()
    alternative["fan_mean"] = .1
    validate_matched_frames(baseline, alternative)
    for column, value in [("judge_mean", 9.99), ("eliminated", True),
                          ("contestant_id", "somebody else"), ("rule", "percentage")]:
        changed = alternative.copy()
        changed.loc[0, column] = value
        with pytest.raises(ValueError, match="identical rosters"):
            validate_matched_frames(baseline, changed)


def test_outcome_selected_metrics_directions_and_scenarios_rejected(tmp_path):
    config = json.loads(CONFIG.read_text())
    for key, value in [("pareto_tolerance", .01), ("late_joint_prior_scales", [1., 2.]),
                       ("reference_policy", "percentage:w0.5:none")]:
        altered = {**config, key: value}
        path = tmp_path / "protocol.json"
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            read_protocol(path)
    config["metrics"][0]["direction"] = "higher"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="directions"):
        read_protocol(path)


def test_run_rejects_source_mutation_before_any_output_write(tmp_path, monkeypatch):
    import dwts.policy_robustness as study

    cfg = json.loads(CONFIG.read_text())
    source = tmp_path / "baseline"
    roots = {"baseline": (source, list(range(1, 35)), "report", 1.)}
    for declaration in cfg["sensitivity_sources"]:
        root = tmp_path / declaration["label"]
        declaration["source"] = str(root)
        field, value = declaration["changed_fit_field"], declaration["changed_fit_value"]
        roots[declaration["label"]] = (root, declaration["seasons"],
            value if field == "likelihood" else "report", value if field == "penalty_lambda" else 1.)
    configs = {}
    for root, seasons, likelihood, penalty in roots.values():
        root.mkdir()
        for name in study.SOURCE_FILES:
            (root / name).write_text("original source artifact")
        for season in seasons:
            directory = root / f"season_{season:02d}"
            directory.mkdir()
            for name in study.SEASON_FILES:
                (directory / name).write_text("original season artifact")
        configs[root] = {"seasons": seasons, "data": {}, "fit": {
            "parameterization": "noncentered", "likelihood": likelihood, "penalty_lambda": penalty}}
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(cfg))
    data = tmp_path / "data.csv"
    data.write_text("private synthetic fixture")
    manifest = {"input_sha256": "same-input", "code_sha256": "same-code",
                "python": "same-python", "packages": {"numpy": "same-version"}}
    monkeypatch.setattr(study, "load_source", lambda root, data: (None, None, manifest, configs[root]))

    def fake_load(root, votes, season, fit_config, seed, count):
        observations = frame().assign(season=season)
        logits = np.tile([-3., -1., 1., 3.]*2, (4, 1))
        shares = np.tile(softmax(logits[:, :4], axis=1), (1, 2))
        if root == source and season == 34:
            (root / "votes.csv").write_text("mutation after initial source hashing")
        return observations, shares, logits, {}

    monkeypatch.setattr(study, "load_season", fake_load)
    out = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match="Frozen posterior source changed"):
        study.run(source, protocol, out, data)
    assert not out.exists()
