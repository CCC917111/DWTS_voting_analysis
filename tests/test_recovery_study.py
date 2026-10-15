"""Known-truth checks use independent calculations, not a fitted-output snapshot."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import expit, softmax
from scipy.stats import norm

from dwts.forward_model import SequentialPosterior, exit_log_probs
from dwts.recovery_study import (aggregate_recovery, generated_season, read_protocol,
                                 recovery_metrics, run_study, weighted_interval,
                                 wilson_interval)


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return read_protocol(ROOT / "configs/recovery_study.json")


def test_exact_normalized_likelihood_and_rank_scale_invariance():
    z = np.array([.3, -.8, 1.1])
    judges = np.array([7., 9., 8.])
    active = np.arange(3)
    # Hand-calculated judge/fan orders, best-oriented normalized rank merits.
    merit = .5 * np.array([0., 2., 1.]) / 2 + .5 * np.array([1., 0., 2.]) / 2
    np.testing.assert_allclose(np.exp(exit_log_probs(z, active, judges, rule="rank")), softmax(-3 * merit))
    np.testing.assert_allclose(exit_log_probs(z, active, judges, rule="rank"),
                               exit_log_probs(17 * z, active, judges, rule="rank"))
    p = softmax(z)
    expected = softmax(-3 * 3 * (.5 * judges / judges.sum() + .5 * p))
    np.testing.assert_allclose(np.exp(exit_log_probs(z, active, judges)), expected)
    assert not np.allclose(exit_log_probs(z, active, judges), exit_log_probs(17 * z, active, judges))


def test_particle_posterior_moment_against_independent_gaussian_integral():
    # With two equal-scored contestants, delta=alpha0-alpha1~N(0,2).
    # Percentage merit difference is 2*sigmoid(delta)-1. An observed exit0
    # has likelihood sigmoid(-3*merit_difference). Integrate this one-
    # dimensional posterior directly, without the shared likelihood function.
    def joint(delta):
        return norm.pdf(delta, scale=np.sqrt(2)) * expit(-3 * (2 * expit(delta) - 1))
    denominator = quad(joint, -12, 12, epsabs=1e-11)[0]
    expected_mean = quad(lambda d: expit(d) * joint(d), -12, 12, epsabs=1e-11)[0] / denominator
    posterior = SequentialPosterior(2, particles=65536, seed=816,
                                    ess_fraction=.0001, mh_steps=0)
    diagnostic = posterior.observe(np.arange(2), np.ones(2), 0)
    assert not diagnostic["resampled"]
    estimated_mean = posterior.weights @ posterior.posterior_fan_shares(np.arange(2))[:, 0]
    assert estimated_mean == pytest.approx(expected_mean, abs=.004)
    # This posterior mean is lower after exit0, independently checking sign.
    assert estimated_mean < .4


def test_generated_rosters_follow_drawn_exits_and_paired_priors_share_season():
    cfg = protocol()
    a, b = generated_season(cfg, "rank", 7), generated_season(cfg, "rank", 7)
    for previous, current in zip(a, a[1:]):
        np.testing.assert_array_equal(current["indices"], np.delete(previous["indices"], previous["outcome"]))
    for x, y in zip(a, b):
        assert x["outcome"] == y["outcome"]
        np.testing.assert_array_equal(x["truth_shares"], y["truth_shares"])
        np.testing.assert_array_equal(x["judges"], y["judges"])
    assert sum(len(x["indices"]) for x in a) == 33
    assert sum(len(x["indices"]) * (len(x["indices"]) - 1) // 2 for x in a) == 83


def test_future_truth_does_not_select_judge_schedule():
    cfg = protocol()
    a = generated_season(cfg, "percentage", 13)
    changed = {**cfg, "truth_prior_sd": 4.}
    b = generated_season(changed, "percentage", 13)
    # Every score trajectory was generated before outcomes; shared contestants
    # retain identical judge scores even if exit outcomes/rosters differ.
    for aa, bb in zip(a, b):
        for contestant in set(aa["indices"]) & set(bb["indices"]):
            assert aa["judges"][np.flatnonzero(aa["indices"] == contestant)[0]] == bb["judges"][np.flatnonzero(bb["indices"] == contestant)[0]]


def test_known_truth_metrics_and_tie_denominators():
    z = np.array([1., 0., -1.])
    obs = [{"week": 0, "indices": np.arange(3), "truth_logits": z, "truth_shares": softmax(z)}]
    perfect, means = recovery_metrics(obs, np.tile(z, (4, 1)), np.ones(4) / 4)
    assert perfect["share_mse"] == pytest.approx(0)
    assert perfect["coverage_95"] == 1
    assert perfect["interval_width_95"] == pytest.approx(0)
    assert perfect["point_rank_concordance"] == 1
    assert perfect["posterior_rank_concordance"] == 1
    assert perfect["contestant_weeks"] == 3
    assert perfect["pair_comparisons"] == 3
    np.testing.assert_allclose(means, softmax(z))
    tied, _ = recovery_metrics(obs, np.zeros((4, 3)), np.ones(4) / 4)
    assert tied["point_rank_concordance"] == .5
    assert tied["posterior_rank_concordance"] == .5
    reversed_result, _ = recovery_metrics(obs, np.tile(-z, (4, 1)), np.ones(4) / 4)
    assert reversed_result["point_rank_concordance"] == 0


def test_weighted_intervals_respect_weight_and_zero_mass():
    values = np.array([[0., 1.], [.4, .6], [1., 0.]])
    mean, lo, hi = weighted_interval(values, np.array([0., 1., 0.]))
    np.testing.assert_array_equal(mean, [.4, .6])
    np.testing.assert_array_equal(lo, mean)
    np.testing.assert_array_equal(hi, mean)
    with pytest.raises(ValueError):
        weighted_interval(values, np.zeros(3))


def test_prior_point_baseline_uses_analytic_uniform_not_particle_noise():
    truth = np.array([1., 0., -1.])
    obs = [{"week": 0, "indices": np.arange(3), "truth_logits": truth,
            "truth_shares": softmax(truth)}]
    particles = np.tile(truth, (4, 1))
    result, means = recovery_metrics(obs, particles, np.ones(4) / 4, prior_only=True)
    np.testing.assert_array_equal(means, np.ones(3) / 3)
    assert result["point_rank_concordance"] == .5
    assert result["share_mse"] == pytest.approx(np.mean((softmax(truth) - 1 / 3) ** 2))


def test_coverage_probe_is_fixed_and_wilson_not_spuriously_precise():
    lo, hi = wilson_interval(46, 48)
    assert lo < .9 < hi
    assert hi > .98
    assert wilson_interval(0, 48)[0] == 0
    assert wilson_interval(48, 48)[1] == pytest.approx(1)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_aggregation_clusters_seasons_and_transforms_rmse_after_pooling():
    cfg = {**protocol(), "replications_per_rule": 2, "bootstrap_draws": 500}
    rows = [{"rule": "rank", "scenario": "matched", "method": "posterior", "replicate": i,
             "share_mse": mse, "point_rank_concordance": .5, "posterior_rank_concordance": .5,
             "coverage_95": coverage, "interval_width_95": .4,
             "contestant_weeks": 33, "pair_comparisons": 83, "probe_covered": i}
            for i, (mse, coverage) in enumerate([(.01, 0.), (.09, 1.)])]
    summary, probe = aggregate_recovery(pd.DataFrame(rows), cfg)
    rmse = summary.loc[summary.metric.eq("share_rmse")].iloc[0]
    assert rmse.estimate == pytest.approx(np.sqrt(.05))
    assert rmse.n_seasons == 2
    assert rmse.n_contestant_weeks == 66
    assert probe.iloc[0].total == 2
    incomplete = pd.DataFrame(rows[:1])
    with pytest.raises(ValueError, match="all independent"):
        aggregate_recovery(incomplete, cfg)


def test_full_small_study_freezes_protocol_and_separates_private_rows(tmp_path, monkeypatch):
    cfg = {**protocol(), "replications_per_rule": 2, "contestants": 4, "weeks": 2,
           "particles": 128, "approximation_replications": 1,
           "approximation_particles": 256, "bootstrap_draws": 100, "mh_steps": 1}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(cfg))
    # Installed users may run outside a checkout; source integrity must not
    # assume that cwd contains a dwts/ directory.
    monkeypatch.chdir(tmp_path)
    out = run_study(path, tmp_path / "fresh")
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["main_inference_fits"] == 16
    assert status["independent_generated_seasons"] == 8
    assert status["approximation_fits"] == 8
    assert not any("private" in x for x in status["public_artifacts"])
    assert len(pd.read_csv(out / "recovery_summary.csv")) == 80
    assert len(pd.read_csv(out / "coverage_probe.csv")) == 16
    assert all((out / x).exists() for x in status["private_artifacts"])
    with pytest.raises(ValueError, match="fresh"):
        run_study(path, out)
