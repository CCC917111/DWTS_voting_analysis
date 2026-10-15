"""Exact joint-prior-scale posterior transport for rank-only fitted seasons.

For c > 0, change alpha's prior SD from 2 to 2c, beta's SD from 1 to c,
and sigma's Half-Cauchy scale from 1 to c, leaving epsilon ~ Normal(0, 1).
The map (alpha, beta, sigma, epsilon) -> (c alpha, c beta, c sigma, epsilon)
gives z -> cz and preserves every rank-based likelihood term. Its Jacobian
is c**(n_contestants + 2) in sigma coordinates, so the alternative posterior
is exactly the pushforward of the baseline posterior. In log-sigma sampling
coordinates the Jacobian is c**(n_contestants + 1). No MCMC refit is needed.

This identity does not apply to percentage likelihoods, unrelated prior
changes, or independently varying the three prior scales. Numerical summaries
remain Monte Carlo approximations from the validated stored baseline draws.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata

from .analysis import check_output_path, digest, load_source, write_json
from .uncertainty import (SCOPE, aggregate_tables, evaluate_season, load_season,
                          policies_from_protocol)

RANK_SEASONS = (1, 2, 28, 29, 30, 31, 32, 33, 34)
SCALES = (.5, 1., 2.)


def prior_scales(scale):
    """The jointly changed family; these three scales cannot vary separately."""
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0 or scale > np.finfo(float).max / 2:
        raise ValueError("Joint prior scale must be finite and positive")
    return dict(alpha_normal_sd=2 * scale, beta_normal_sd=scale,
                sigma_half_cauchy_scale=scale, epsilon_normal_sd=1.)


def log_transport_jacobian(n_contestants, scale, *, coordinates="sigma"):
    """Log absolute Jacobian in the stated noncentered sampling coordinates."""
    scale = prior_scales(scale)["beta_normal_sd"]
    if type(n_contestants) is not int or n_contestants < 2:
        raise ValueError("Need at least two contestant intercepts")
    if coordinates not in {"sigma", "log_sigma"}:
        raise ValueError("Coordinates must use sigma or log_sigma")
    exponent = n_contestants + (2 if coordinates == "sigma" else 1)
    return float(exponent * np.log(scale))


def read_protocol(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"schema_version", "seed", "draws_per_chain", "scope",
               "counterfactual_exits_per_week", "seasons", "joint_prior_scales",
               "rules", "judge_weights", "mechanisms", "interval_probability"}
    if set(cfg) != allowed or cfg["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete joint-prior-scale protocol")
    if (type(cfg["seed"]) is not int or cfg["seed"] < 0 or
            type(cfg["draws_per_chain"]) is not int or cfg["draws_per_chain"] < 4):
        raise ValueError("Need a nonnegative seed and at least four draws per chain")
    if (cfg["seasons"] != list(RANK_SEASONS) or
            cfg["joint_prior_scales"] != list(SCALES)):
        raise ValueError("Use the declared nine rank seasons and joint scales 0.5, 1, 2")
    if (cfg["scope"] != SCOPE or cfg["counterfactual_exits_per_week"] != 1 or
            cfg["rules"] != ["rank", "percentage"] or
            cfg["judge_weights"] != [.25, .5, .75] or
            cfg["mechanisms"] != ["none", "judge_save", "fan_save"] or
            cfg["interval_probability"] != .95):
        raise ValueError("Use the fixed 18 policies, one-exit scope and 95% interval")
    return cfg


def validate_rank_population(frame):
    """Every fitted week must have a rank likelihood, including excluded weeks."""
    if (frame.empty or "rule" not in frame or frame.season.nunique() != 1 or
            not frame.rule.eq("rank").all()):
        raise ValueError("Analytic prior transport requires a rank-only fitted season")


def transport_draws(frame, shares, logits, scale):
    """Push each joint draw forward and recompute its within-week fan shares."""
    prior_scales(scale)
    validate_rank_population(frame)
    shares, logits = np.asarray(shares, float), np.asarray(logits, float)
    if (shares.ndim != 2 or shares.shape != logits.shape or shares.shape[1] != len(frame)
            or not np.array_equal(frame.index, np.arange(len(frame))) or
            not np.isfinite(shares).all() or not np.isfinite(logits).all() or (shares < 0).any()):
        raise ValueError("Need finite, canonically aligned share and logit draws")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        changed_logits = float(scale) * logits
    if not np.isfinite(changed_logits).all():
        raise ValueError("Joint scale exceeds finite logit arithmetic")
    changed_shares = np.empty_like(shares)
    for _, group in frame.groupby("week", sort=True):
        ix = group.index.to_numpy()
        if not np.allclose(shares[:, ix].sum(axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError("Baseline shares must sum to one within every fitted week")
        if not np.array_equal(rankdata(-logits[:, ix], axis=1, method="average"),
                              rankdata(-changed_logits[:, ix], axis=1, method="average")):
            raise ValueError("Finite precision changed a rank during analytic transport")
        # Preserve the baseline's exact stored percentage arithmetic at c=1.
        changed_shares[:, ix] = (shares[:, ix] if scale == 1 else
                                  softmax(changed_logits[:, ix], axis=1))
    return changed_shares, changed_logits


def sensitivity_means(summaries):
    """Compare conditional means; no coupling-dependent cross-prior intervals."""
    keys = ["scope", "level", "group", "rule", "judge_weight", "mechanism", "metric"]
    baseline = summaries[summaries.joint_prior_scale.eq(1.)]
    rows = []
    for scale in SCALES:
        if scale == 1:
            continue
        alternative = summaries[summaries.joint_prior_scale.eq(scale)]
        merged = alternative.merge(baseline, on=keys, suffixes=("_alternative", "_baseline"),
                                   validate="one_to_one")
        for row in merged.to_dict("records"):
            if row["weeks_alternative"] != row["weeks_baseline"]:
                raise ValueError("Joint-prior comparisons must retain identical observed weeks")
            rows.append({**{key: row[key] for key in keys}, "joint_prior_scale": scale,
                "baseline_joint_prior_scale": 1., "weeks": row["weeks_baseline"],
                "baseline_mean": row["mean_baseline"], "alternative_mean": row["mean_alternative"],
                "mean_difference_alternative_minus_baseline": row["mean_alternative"] - row["mean_baseline"],
                "comparison": "difference_of_conditional_means_under_exact_joint_prior_scale_transport"})
    return pd.DataFrame(rows)


def run(source, protocol_path, out, data):
    source, out, data = Path(source).resolve(), Path(out).resolve(), Path(data).resolve()
    check_output_path(source, out)
    cfg = read_protocol(protocol_path)
    policies = policies_from_protocol(cfg)
    votes, _, manifest, fit_cfg = load_source(source, data)
    if (not set(cfg["seasons"]) <= set(fit_cfg["seasons"]) or
            fit_cfg["fit"].get("parameterization") != "noncentered" or
            fit_cfg["fit"].get("likelihood") not in {"report", "bottom_two"}):
        raise ValueError("Source must contain all selected rank seasons with the supported noncentered target")
    # A full source may also contain percentage seasons; they are never transported.
    for season in cfg["seasons"]:
        validate_rank_population(votes[votes.season.eq(season)])
    results_by_scale = {scale: [] for scale in SCALES}
    source_integrity = []
    for season in cfg["seasons"]:
        print(f"joint-prior transport: validating season {season}", flush=True)
        frame, shares, logits, integrity = load_season(source, votes, season, fit_cfg,
                                                      cfg["seed"], cfg["draws_per_chain"])
        validate_rank_population(frame)
        by_scale = {}
        for scale in SCALES:
            changed_shares, changed_logits = transport_draws(frame, shares, logits, scale)
            by_scale[scale] = evaluate_season(frame, changed_shares, changed_logits, policies)
        rank_columns = [k for k, policy in enumerate(policies) if policy.rule == "rank"]
        for scale in SCALES:
            if not np.array_equal(by_scale[scale].counts[:, rank_columns], by_scale[1.].counts[:, rank_columns]):
                raise ValueError("A rank-policy result changed under invariant joint-prior transport")
            results_by_scale[scale].append(by_scale[scale])
        n = int(frame.contestant_id.nunique())
        integrity["log_transport_jacobian_by_scale"] = {str(scale): dict(
            sigma_coordinates=log_transport_jacobian(n, scale),
            log_sigma_coordinates=log_transport_jacobian(n, scale, coordinates="log_sigma"))
            for scale in SCALES}
        source_integrity.append(integrity)
    summaries, contrasts = [], []
    for scale in SCALES:
        label = f"analytic_joint_prior_scale_{scale:g}"
        summary, contrast = aggregate_tables(results_by_scale[scale], label, cfg["interval_probability"])
        summary["joint_prior_scale"], contrast["joint_prior_scale"] = scale, scale
        summaries.append(summary)
        contrasts.append(contrast)
    summary, contrast = pd.concat(summaries, ignore_index=True), pd.concat(contrasts, ignore_index=True)
    sensitivity = sensitivity_means(summary)
    # All validation and analytic-invariance checks precede the first output write.
    out.mkdir(parents=True)
    summary.to_csv(out / "policy_summary.csv", index=False)
    contrast.to_csv(out / "policy_contrasts.csv", index=False)
    sensitivity.to_csv(out / "prior_scale_sensitivity.csv", index=False)
    write_json(out / "protocol.json", dict(cfg,
        method="analytic_posterior_transport_of_joint_prior_scales", new_posterior_fits=0,
        changed_priors_by_scale={str(scale): prior_scales(scale) for scale in SCALES},
        unchanged_assumptions=["Observed scores and elimination labels", "Penalty lambda and judge weight",
                               "Rank likelihood and withdrawal policy", "Independent season fits", "Standard Normal epsilon prior"],
        proof=dict(map="T_c(alpha,beta,sigma,epsilon)=(c*alpha,c*beta,c*sigma,epsilon); z_c=c*z",
                   rank_likelihood="L(T_c(theta))=L(theta) for every rank-only fitted week",
                   jacobian_sigma="abs(det(DT_c))=c**(n_contestants+2)",
                   jacobian_log_sigma="abs(det(DT_c))=c**(n_contestants+1)",
                   prior_density="pi_c(T_c(theta))*abs(det(DT_c))=pi_1(theta)",
                   posterior="p_c = pushforward(T_c, p_1); normalizing constants are equal"),
        posterior_joint_draws={str(scale): len(results[0].counts) for scale, results in results_by_scale.items()},
        intervals="Equal-tailed 95% model-conditional posterior credible intervals; Monte Carlo approximations from baseline chains.",
        seasonal_combination="Independent seeded seasonal permutations; whole-season joint draws and paired policies preserved.",
        fan_ordering="Latent logits define fan ranks and fan tie-break/save order for both policy families.",
        percentage_merit="Numeric softmax(c*z) shares; original stored numeric shares retained exactly at c=1.",
        rank_policy_check="All retained joint-draw rank-policy counts equal c=1 exactly for every scale and season.",
        limitations=["Transport is restricted to selected rank-likelihood seasons 1,2,28..34; not percentage-likelihood seasons 3..27.",
                     "Only the specified joint Normal/Normal/Half-Cauchy scale family is covered; this is not sensitivity to all prior choices.",
                     "This is exact analytic posterior transport, not new MCMC fitting; numerical summaries retain finite-chain approximation error.",
                     "Original baseline diagnostics were validated; diagnostics and effective sample sizes of transformed fan shares and policy functions were not recomputed.",
                     "Credible intervals are conditional on each prior specification; no cross-prior difference probability or interval is claimed.",
                     "Observed elimination labels informed the posterior; no independent true-vote or held-out forecast validation is established.",
                     "One-exit policies act on fixed observed nonfinal rosters with at least three contestants; all withdrawal weeks are excluded.",
                     "No whole-season counterfactual trajectories or causal policy effects are identified."]))
    write_json(out / "source_integrity.json", dict(
        source_directory_name=source.name, source_manifest=manifest, source_run_config=fit_cfg,
        run_files_sha256={name: digest(source / name) for name in
                          ["manifest.json", "run_config.json", "sampling_status.json", "votes.csv"]},
        transported_seasons=source_integrity))
    generated = ["policy_summary.csv", "policy_contrasts.csv", "prior_scale_sensitivity.csv",
                 "protocol.json", "source_integrity.json"]
    write_json(out / "status.json", dict(execution_completed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        method="analytic_posterior_transport", new_posterior_fits=0,
        seasons=len(cfg["seasons"]), eligible_weeks=sum(len(r.weeks) for r in results_by_scale[1.]),
        rank_policy_invariance_verified=True, all_source_diagnostics_pass=True,
        independent_true_vote_validation=False, code_sha256=digest(Path(__file__)),
        protocol_sha256=digest(protocol_path),
        analysis_code_sha256={name: digest(Path(__file__).with_name(name)) for name in
                              ["prior_sensitivity.py", "uncertainty.py", "rules.py", "analysis.py", "data.py", "model.py", "fast_mh.py"]},
        output_sha256={name: digest(out / name) for name in generated}))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Completed diagnosed baseline fit directory")
    parser.add_argument("--config", type=Path, default=Path("configs/prior_sensitivity_study.json"))
    parser.add_argument("--out", type=Path, required=True, help="New aggregate-only output directory")
    parser.add_argument("--data", type=Path, default=Path("data/raw/2026_MCM_Problem_C_Data.csv"))
    args = parser.parse_args(argv)
    print(f"Joint-prior-scale transport completed: {run(args.source, args.config, args.out, args.data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
