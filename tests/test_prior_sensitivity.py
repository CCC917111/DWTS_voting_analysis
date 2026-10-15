"""Independent density, Jacobian and fixed-roster checks for analytic transport."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from scipy.stats import halfcauchy, norm, rankdata

from dwts.fast_mh import log_target, pack
from dwts.model import FitConfig
from dwts.prior_sensitivity import (RANK_SEASONS, SCALES, log_transport_jacobian,
    prior_scales, read_protocol, run, sensitivity_means, transport_draws, validate_rank_population)
from dwts.uncertainty import aggregate_tables, evaluate_season, policies_from_protocol

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "prior_sensitivity_study.json"
PROTOCOL = read_protocol(CONFIG)
POLICIES = policies_from_protocol(PROTOCOL)


def frame():
    rows = []
    for week, judges in [(1, [10., 9., 8., 7.]), (2, [8., 8., 6., 4.])]:
        for i, score in enumerate(judges):
            rows.append(dict(season=28, week=week, contestant_id=list("ABCD")[i],
                judge_mean=score, judge_z=(score-np.mean(judges))/np.std(judges),
                judge_share=score/np.sum(judges), eliminated=i == 0,
                withdrawal=False, is_final_week=False, rule="rank"))
    return pd.DataFrame(rows)


def independent_prior(alpha, beta, sigma, epsilon, scale):
    return (norm.logpdf(alpha, scale=2*scale).sum() + norm.logpdf(beta, scale=scale) +
            halfcauchy.logpdf(sigma, scale=scale) + norm.logpdf(epsilon).sum())


def independent_rank_log_likelihood(data, logits, cfg):
    total = 0.
    for _, group in data.groupby("week"):
        exits = group.eliminated.to_numpy(bool)
        if not exits.any() or exits.all() or (group.withdrawal.any() and cfg.withdrawal_policy == "skip_week"):
            continue
        combined = -2*(cfg.judge_weight*rankdata(-group.judge_mean, method="average") +
            (1-cfg.judge_weight)*rankdata(-logits[group.index], method="average"))
        boundary = (np.sort(combined)[1] if cfg.likelihood == "bottom_two" and exits.sum() == 1
                    else combined[~exits].min())
        total -= 100*cfg.penalty_lambda*max(0., combined[exits].max()-boundary)
    return total


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("likelihood", ["report", "bottom_two"])
def test_density_change_of_variables_matches_independent_priors_and_actual_target(scale, likelihood):
    data = frame()
    cfg = FitConfig(sampler="fast_mh", penalty_lambda=.7, likelihood=likelihood)
    _, _, _, _, args = pack(data, cfg)
    rng = np.random.default_rng(163)
    n, obs = data.contestant_id.nunique(), len(data)
    offsets = []
    for _ in range(8):
        alpha, beta, sigma, epsilon = rng.normal(size=n), rng.normal(), np.exp(rng.normal()), rng.normal(size=obs)
        ids = np.searchsorted(sorted(data.contestant_id.unique()), data.contestant_id)
        logits = alpha[ids]+beta*data.judge_z.to_numpy()+sigma*epsilon
        base_likelihood = independent_rank_log_likelihood(data, logits, cfg)
        changed_likelihood = independent_rank_log_likelihood(data, scale*logits, cfg)
        assert changed_likelihood == base_likelihood
        base = independent_prior(alpha, beta, sigma, epsilon, 1.) + base_likelihood
        changed = independent_prior(scale*alpha, scale*beta, scale*sigma, epsilon, scale) + changed_likelihood
        assert changed + log_transport_jacobian(n, scale) == pytest.approx(base, abs=1e-10)
        # The actual sampler stores log(sigma), whose Jacobian removes one c.
        changed_log_sigma_density = changed + np.log(scale*sigma)
        assert (changed_log_sigma_density + log_transport_jacobian(n, scale, coordinates="log_sigma")
                == pytest.approx(base+np.log(sigma), abs=1e-10))
        state = np.r_[alpha, beta, np.log(sigma), epsilon]
        # Independent normalized priors and rank penalties differ from the
        # compiled target only by fixed prior normalizing constants.
        offsets.append(base+np.log(sigma)-log_target(state, args))
    np.testing.assert_allclose(offsets, offsets[0], rtol=0, atol=1e-10)


@pytest.mark.parametrize("coordinates", ["sigma", "log_sigma"])
def test_jacobian_matches_direct_diagonal_determinant(coordinates):
    n, obs, scale = 7, 11, 2.
    diagonal = [scale]*(n+1) + [scale if coordinates == "sigma" else 1.] + [1.]*obs
    sign, logdet = np.linalg.slogdet(np.diag(diagonal))
    assert sign == 1
    assert log_transport_jacobian(n, scale, coordinates=coordinates) == pytest.approx(logdet)


def test_transport_preserves_joint_week_structure_and_baseline_numeric_shares():
    data = frame()
    logits = np.random.default_rng(62).normal(size=(8, len(data)))
    shares = np.empty_like(logits)
    for _, group in data.groupby("week"):
        shares[:, group.index] = softmax(logits[:, group.index], axis=1)
    unchanged, unchanged_logits = transport_draws(data, shares, logits, 1.)
    np.testing.assert_array_equal(unchanged, shares)
    np.testing.assert_array_equal(unchanged_logits, logits)
    for scale in [.5, 2.]:
        actual, changed_logits = transport_draws(data, shares, logits, scale)
        np.testing.assert_array_equal(changed_logits, scale*logits)
        for _, group in data.groupby("week"):
            np.testing.assert_array_equal(actual[:, group.index], softmax(scale*logits[:, group.index], axis=1))


def test_exact_rank_policies_invariant_but_percentage_decisions_can_change():
    data = frame().iloc[:4].copy()
    logits = np.tile(np.array([0., 1, 2, 3]), (8, 1))
    shares = softmax(logits, axis=1)
    results = {}
    for scale in [.1, 1., 2.]:
        p, z = transport_draws(data, shares, logits, scale)
        results[scale] = evaluate_season(data, p, z, POLICIES)
    rank = [i for i, p in enumerate(POLICIES) if p.rule == "rank"]
    percentage = [i for i, p in enumerate(POLICIES) if p.rule == "percentage"]
    np.testing.assert_array_equal(results[.1].counts[:, rank], results[1.].counts[:, rank])
    np.testing.assert_array_equal(results[2.].counts[:, rank], results[1.].counts[:, rank])
    assert not np.array_equal(results[.1].counts[:, percentage], results[1.].counts[:, percentage])


def test_percentage_likelihood_invalidates_transport_even_in_excluded_week():
    data = frame()
    data.loc[data.week.eq(2), ["rule", "is_final_week"]] = ["percentage", True]
    with pytest.raises(ValueError, match="rank-only"):
        validate_rank_population(data)
    with pytest.raises(ValueError, match="rank-only"):
        transport_draws(data, np.ones((4, 8))/4, np.zeros((4, 8)), 2.)


def test_numerical_scale_that_merges_distinct_logits_is_rejected():
    data = frame().iloc[:4].copy()
    logits = np.tile([0., np.nextafter(0., 1.), 1., 2.], (4, 1))
    with pytest.raises(ValueError, match="precision changed a rank"):
        transport_draws(data, softmax(logits, axis=1), logits, .5)


@pytest.mark.parametrize("scale", [0., -1., np.nan, np.inf, np.finfo(float).max])
def test_nonpositive_or_nonfinite_prior_scales_fail(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        prior_scales(scale)


def test_sensitivity_exports_conditional_mean_differences_without_cross_prior_intervals():
    data = frame()
    logits = np.random.default_rng(8).normal(size=(8, 8))
    shares = np.concatenate([softmax(logits[:, :4], axis=1), softmax(logits[:, 4:], axis=1)], axis=1)
    tables = []
    for scale in SCALES:
        p, z = transport_draws(data, shares, logits, scale)
        summary, _ = aggregate_tables([evaluate_season(data, p, z, POLICIES)], f"scale_{scale}")
        summary["joint_prior_scale"] = scale
        tables.append(summary)
    result = sensitivity_means(pd.concat(tables, ignore_index=True))
    assert set(result.joint_prior_scale) == {.5, 2.}
    assert not {"lower", "upper", "posterior_probability_lower"} & set(result)
    assert result[result.rule.eq("rank")].mean_difference_alternative_minus_baseline.eq(0).all()
    assert not {"name", "contestant_id", "draw", "chain", "fan_share"} & set(result)


def test_protocol_rejects_percentage_season_and_unreported_prior_family(tmp_path):
    for key, value in [("seasons", list(RANK_SEASONS)+[3]), ("joint_prior_scales", [.5, 1., 3.])]:
        cfg = dict(PROTOCOL)
        cfg[key] = value
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="nine rank seasons"):
            read_protocol(path)


def test_existing_output_is_protected_before_reading_sources(tmp_path):
    source, out = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    out.mkdir()
    (out / "keep").write_text("preserve")
    with pytest.raises(ValueError, match="never overwrite"):
        run(source, CONFIG, out, tmp_path / "missing.csv")
    assert (out / "keep").read_text() == "preserve"


def test_selected_mixed_likelihood_source_rejected_before_loading_chains_or_writing(tmp_path, monkeypatch):
    import dwts.prior_sensitivity as module
    source, out = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    data = pd.concat([frame().assign(season=s) for s in RANK_SEASONS], ignore_index=True)
    data.loc[data.season.eq(34), "rule"] = "percentage"
    cfg = {"seasons": list(RANK_SEASONS), "fit": {"parameterization": "noncentered", "likelihood": "report"}}
    monkeypatch.setattr(module, "load_source", lambda *args: (data, None, {}, cfg))
    monkeypatch.setattr(module, "load_season", lambda *args: pytest.fail("Must reject mixed target before loading chains"))
    with pytest.raises(ValueError, match="rank-only"):
        run(source, CONFIG, out, tmp_path / "data.csv")
    assert not out.exists()
