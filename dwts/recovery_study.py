"""Known-truth, retrospective recovery for the normalized forward model.

The generated exits determine subsequent rosters. All observed exits in a
synthetic season enter its posterior before any recovery metric is computed.
This is a simulation of retrospective inference, not prospective prediction or
validation against actual audience votes. Only aggregate exports are public.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.special import softmax
from scipy.stats import norm

from .forward_model import SequentialPosterior, exit_log_probs


METRICS = ("share_rmse", "point_rank_concordance", "posterior_rank_concordance",
           "coverage_95", "interval_width_95")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8")


def read_protocol(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema_version", "seed", "replications_per_rule", "contestants", "weeks",
                "rules", "scenarios", "truth_prior_sd", "drift_slope_sd", "judge_weight",
                "temperature", "particles", "mh_steps", "proposal_scale", "ess_fraction",
                "interval_probability", "bootstrap_draws", "approximation_replications",
                "approximation_particles", "probe_contestant", "probe_week"}
    if set(cfg) != expected or cfg["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete recovery protocol")
    for name in ("seed", "replications_per_rule", "contestants", "weeks", "particles",
                 "mh_steps", "bootstrap_draws", "approximation_replications",
                 "approximation_particles", "probe_contestant", "probe_week"):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if (cfg["replications_per_rule"] < 2 or cfg["contestants"] < 3 or cfg["weeks"] < 1 or
            cfg["weeks"] > cfg["contestants"] - 2 or cfg["particles"] < 64 or
            cfg["mh_steps"] < 1 or cfg["bootstrap_draws"] < 100 or
            not 1 <= cfg["approximation_replications"] <= cfg["replications_per_rule"] or
            cfg["approximation_particles"] < cfg["particles"] or
            cfg["probe_contestant"] != 0 or cfg["probe_week"] != 0):
        raise ValueError("Invalid replication, roster, inference or fixed-probe settings")
    if cfg["rules"] != ["rank", "percentage"] or cfg["interval_probability"] != .95:
        raise ValueError("Both rules and the fixed 95% interval are required")
    expected_scenarios = [
        {"name": "matched", "truth": "static", "inference_prior_sd": 1.},
        {"name": "narrow_prior", "truth": "static", "inference_prior_sd": .5},
        {"name": "wide_prior", "truth": "static", "inference_prior_sd": 2.},
        {"name": "gradual_drift", "truth": "drift", "inference_prior_sd": 1.}]
    if cfg["scenarios"] != expected_scenarios:
        raise ValueError("Use the declared paired prior and drift scenarios")
    if (cfg["truth_prior_sd"] != 1. or cfg["drift_slope_sd"] != .7 or
            not 0 < cfg["judge_weight"] < 1 or cfg["temperature"] <= 0 or
            cfg["proposal_scale"] <= 0 or not 0 < cfg["ess_fraction"] <= 1):
        raise ValueError("Invalid data-generation or inference settings")
    return cfg


def generated_season(cfg, rule, replicate, truth_kind="static"):
    """Draw truth, all judges, and exit uniforms before processing any exits."""
    if rule not in cfg["rules"] or truth_kind not in {"static", "drift"}:
        raise ValueError("Unknown rule or truth scenario")
    rule_index = cfg["rules"].index(rule)
    # Matched/narrow/wide scenarios share exactly the same generated season.
    rng = np.random.default_rng(np.random.SeedSequence(
        [cfg["seed"], rule_index, replicate, 0 if truth_kind == "static" else 1, 711]))
    n, weeks = cfg["contestants"], cfg["weeks"]
    initial = rng.normal(0, cfg["truth_prior_sd"], n)
    slopes = rng.normal(0, cfg["drift_slope_sd"], n)
    skill = rng.normal(0, .7, n)
    trend = rng.normal(.12, .08, n)
    scores = np.clip(7 + skill[None, :] + np.arange(weeks)[:, None] * trend[None, :]
                     + rng.normal(0, .4, (weeks, n)), 1, 10)
    uniforms = rng.random(weeks)
    active = list(range(n))
    observations = []
    for week in range(weeks):
        z = initial if truth_kind == "static" else initial + week / max(weeks - 1, 1) * slopes
        indices = np.array(active, dtype=int)
        judges = scores[week, indices]
        probabilities = np.exp(exit_log_probs(z, indices, judges, rule=rule,
                            judge_weight=cfg["judge_weight"], temperature=cfg["temperature"]))
        outcome = min(int(np.searchsorted(np.cumsum(probabilities), uniforms[week])), len(active) - 1)
        observations.append({"week": week, "indices": indices, "judges": judges,
                             "outcome": outcome, "truth_logits": z[indices].copy(),
                             "truth_shares": softmax(z[indices])})
        del active[outcome]
    return observations


def weighted_interval(values, weights, probability=.95):
    """Equal-tailed particle interval, with midpoint-CDF interpolation."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    if (values.ndim != 2 or weights.shape != (len(values),) or not np.isfinite(values).all()
            or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError("Invalid weighted posterior sample")
    weights = weights / weights.sum()
    mean = weights @ values
    bounds = np.empty((2, values.shape[1]))
    positive = weights > 0
    for j in range(values.shape[1]):
        order = np.argsort(values[positive, j], kind="stable")
        x, w = values[positive, j][order], weights[positive][order]
        cdf = np.cumsum(w) - .5 * w
        bounds[:, j] = np.interp([(1 - probability) / 2, (1 + probability) / 2], cdf, x)
    return mean, bounds[0], bounds[1]


def recovery_metrics(observations, particles, weights, prior_only=False):
    """Aggregate each season first; contestants/weeks are not independent trials."""
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    squared_error = covered = width = rows = 0.
    point_correct = expected_correct = pairs = 0.
    point_means = []
    probe_covered = None
    for observation in observations:
        indices, truth = observation["indices"], observation["truth_shares"]
        z = particles[:, indices]
        shares = softmax(z, axis=1)
        mean, lo, hi = weighted_interval(shares, weights)
        if prior_only:
            # Symmetry gives this mean exactly. Finite-particle fluctuations
            # must not manufacture a ranked prior point estimate.
            mean = np.full(len(indices), 1 / len(indices))
        hit = (truth >= lo) & (truth <= hi)
        if observation["week"] == 0:
            # Contestant 0 is fixed before truth, judges and exits are drawn.
            probe_covered = int(hit[np.flatnonzero(indices == 0)[0]])
        squared_error += float(np.sum((mean - truth) ** 2))
        covered += int(hit.sum())
        width += float(np.sum(hi - lo))
        rows += len(indices)
        a, b = np.triu_indices(len(indices), k=1)
        truth_sign = np.sign(observation["truth_logits"][a] - observation["truth_logits"][b])
        point_sign = np.sign(mean[a] - mean[b])
        signs = np.sign(z[:, a] - z[:, b])
        point_correct += float(np.sum(np.where(point_sign * truth_sign > 0, 1.,
                                               np.where(point_sign * truth_sign == 0, .5, 0.))))
        products = signs * truth_sign[None, :]
        expected_correct += float(np.sum(weights @ np.where(products > 0, 1.,
                                                           np.where(products == 0, .5, 0.))))
        pairs += len(a)
        point_means.extend(mean)
    if not rows or not pairs or probe_covered is None:
        raise ValueError("A generated season with a fixed first-week probe is required")
    return {"share_mse": squared_error / rows,
            "point_rank_concordance": point_correct / pairs,
            "posterior_rank_concordance": expected_correct / pairs,
            "coverage_95": covered / rows, "interval_width_95": width / rows,
            "contestant_weeks": int(rows), "pair_comparisons": int(pairs),
            "probe_covered": probe_covered}, np.asarray(point_means)


def fit_season(cfg, rule, replicate, scenario, observations, particles=None, reference=False):
    rule_index = cfg["rules"].index(rule)
    scenario_index = [x["name"] for x in cfg["scenarios"]].index(scenario["name"])
    seed = int(np.random.SeedSequence([cfg["seed"], rule_index, replicate,
                                      scenario_index, 992 if reference else 991]).generate_state(1)[0])
    fit = SequentialPosterior(cfg["contestants"], prior_sd=scenario["inference_prior_sd"],
                              rule=rule, judge_weight=cfg["judge_weight"],
                              temperature=cfg["temperature"], particles=particles or cfg["particles"],
                              mh_steps=cfg["mh_steps"], proposal_scale=cfg["proposal_scale"],
                              ess_fraction=cfg["ess_fraction"], seed=seed)
    prior_particles, prior_weights = fit.particles.copy(), fit.weights.copy()
    diagnostic_rows = []
    for observation in observations:
        diagnostic = fit.observe(observation["indices"], observation["judges"], observation["outcome"])
        diagnostic_rows.append({"week": observation["week"], **diagnostic})
    posterior, means = recovery_metrics(observations, fit.particles, fit.weights)
    prior, _ = recovery_metrics(observations, prior_particles, prior_weights, prior_only=True)
    return posterior, prior, means, diagnostic_rows


def wilson_interval(successes, trials, probability=.95):
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Need 0 <= successes <= positive trials")
    z = norm.ppf(.5 + probability / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (0. if successes == 0 else float(max(0., center - radius)),
            1. if successes == trials else float(min(1., center + radius)))


def aggregate_recovery(frame, cfg):
    summaries, probes = [], []
    for group_index, ((rule, scenario, method), group) in enumerate(
            frame.groupby(["rule", "scenario", "method"], sort=True)):
        if len(group) != cfg["replications_per_rule"] or group.replicate.duplicated().any():
            raise ValueError("Each aggregate requires all independent seasonal replications")
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], group_index, 1232]))
        selection = rng.integers(0, len(group), (cfg["bootstrap_draws"], len(group)))
        common = {"rule": rule, "scenario": scenario, "method": method,
                  "n_seasons": len(group),
                  "n_contestant_weeks": int(group.contestant_weeks.sum()),
                  "n_pair_comparisons": int(group.pair_comparisons.sum())}
        for metric in METRICS:
            values = group["share_mse" if metric == "share_rmse" else metric].to_numpy(float)
            estimate, draws = values.mean(), values[selection].mean(axis=1)
            if metric == "share_rmse":
                estimate, draws = np.sqrt(estimate), np.sqrt(draws)
            lo, hi = np.quantile(draws, [.025, .975])
            summaries.append({**common, "metric": metric, "estimate": float(estimate),
                              "ci_lower": float(lo), "ci_upper": float(hi),
                              "uncertainty": "95% season-cluster percentile bootstrap"})
        successes = int(group.probe_covered.sum())
        lo, hi = wilson_interval(successes, len(group))
        probes.append({"rule": rule, "scenario": scenario, "method": method,
                       "covered": successes, "total": len(group), "coverage": successes / len(group),
                       "wilson_lower": lo, "wilson_upper": hi})
    return pd.DataFrame(summaries), pd.DataFrame(probes)


def run_study(config_path, out):
    cfg = read_protocol(config_path)
    out = Path(out)
    if out.exists():
        raise ValueError("Use a fresh output directory; recovery results are never overwritten")
    out.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    source_paths = {"dwts/recovery_study.py": Path(__file__).resolve(),
                    "dwts/forward_model.py": Path(__file__).with_name("forward_model.py").resolve(),
                    str(Path(config_path).as_posix()): Path(config_path).resolve()}
    sources = {name: digest(path) for name, path in source_paths.items()}
    protocol = {"schema_version": 1, "config": cfg, "config_sha256": digest(config_path),
                "source_sha256": sources, "started_at": started,
                "python": platform.python_version(),
                "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
                "independent_generated_seasons": len(cfg["rules"]) * cfg["replications_per_rule"] * 2,
                "main_inference_fits": len(cfg["rules"]) * cfg["replications_per_rule"] * len(cfg["scenarios"]),
                "scope": "retrospective complete generated season; all observed exits used before recovery",
                "truth": "iid Normal(0,1) initial logits; drift slopes iid Normal(0,0.7)",
                "judge_generation": "pregenerated independent score trajectories clipped to [1,10]",
                "likelihood": "normalized forward categorical exit model; softmax(-temperature*merit)",
                "paired_scenarios": "matched/narrow/wide use identical generated seasons; drift is independent",
                "prior_baseline": "unconditioned prior particles transformed on the same observed rosters; analytic uniform mean for point errors and rank ties",
                "denominators": {"contestant_weeks_per_season": sum(cfg["contestants"] - w for w in range(cfg["weeks"])),
                                 "rank_pairs_per_season": sum((cfg["contestants"] - w) * (cfg["contestants"] - w - 1) // 2 for w in range(cfg["weeks"]))},
                "rank_concordance": "pairwise correct order; exact ties count 0.5",
                "coverage": "equal-tailed 95% credible intervals, averaged within season then across seasons",
                "coverage_probe": "contestant 0 at week 0 fixed before generation; Wilson interval across independent seasons",
                "bootstrap": f"{cfg['bootstrap_draws']} season-cluster draws; intervals reflect finite simulation replications, not SMC error",
                "identification": "rank likelihood invariant to positive logit scale; magnitude remains prior-dependent",
                "inference": "SMC with ESS-triggered resampling and invariant full-history random-walk Metropolis moves",
                "approximation_check": f"first {cfg['approximation_replications']} replications per cell independently repeated at {cfg['approximation_particles']} particles; no MCMC Rhat claim",
                "limits": ["Simulations validate this new normalized model, not actual unobserved votes or the historical soft-constraint model.",
                           "Matched-case coverage is prior-predictive marginal coverage; it does not imply fixed-parameter or real-data calibration.",
                           "Forty-eight independent coverage probes percell give wide binomial uncertainty.",
                           "Independent particle rerun agreement is an approximation diagnostic, not proof of exact posterior sampling."]}
    write_json(out / "protocol.json", protocol)
    rows, diagnostics, checks = [], [], []
    for rule in cfg["rules"]:
        for replicate in range(cfg["replications_per_rule"]):
            simulated = {kind: generated_season(cfg, rule, replicate, kind) for kind in ("static", "drift")}
            for scenario in cfg["scenarios"]:
                observations = simulated[scenario["truth"]]
                posterior, prior, mean, fit_diagnostics = fit_season(cfg, rule, replicate, scenario, observations)
                key = {"rule": rule, "scenario": scenario["name"], "replicate": replicate}
                rows.extend([{**key, "method": "posterior", **posterior},
                             {**key, "method": "prior_only", **prior}])
                diagnostics.extend({**key, **item} for item in fit_diagnostics)
                if replicate < cfg["approximation_replications"]:
                    repeated, _, repeated_mean, _ = fit_season(cfg, rule, replicate, scenario, observations,
                        particles=cfg["approximation_particles"], reference=True)
                    checks.append({**key, "posterior_mean_share_rmse_difference": float(np.sqrt(np.mean((mean - repeated_mean) ** 2))),
                                   "coverage_difference": posterior["coverage_95"] - repeated["coverage_95"],
                                   "point_rank_difference": posterior["point_rank_concordance"] - repeated["point_rank_concordance"],
                                   "posterior_rank_difference": posterior["posterior_rank_concordance"] - repeated["posterior_rank_concordance"],
                                   "interval_width_difference": posterior["interval_width_95"] - repeated["interval_width_95"]})
            if (replicate + 1) % 8 == 0:
                print(f"{rule}: {replicate + 1}/{cfg['replications_per_rule']} independent static/drift seasons", flush=True)
    frame, diagnostic_frame, check_frame = pd.DataFrame(rows), pd.DataFrame(diagnostics), pd.DataFrame(checks)
    # Per-replication rows are local diagnostic products and excluded from the public package.
    frame.to_csv(out / "private_replicates.csv", index=False)
    diagnostic_frame.to_csv(out / "private_particle_diagnostics.csv", index=False)
    check_frame.to_csv(out / "private_approximation_replicates.csv", index=False)
    summary, probes = aggregate_recovery(frame, cfg)
    summary.to_csv(out / "recovery_summary.csv", index=False)
    probes.to_csv(out / "coverage_probe.csv", index=False)
    diagnostic_columns = [c for c in diagnostic_frame.select_dtypes(include=["number", "bool"]).columns
                          if c not in {"replicate", "week"}]
    diagnostic_summary = diagnostic_frame.groupby(["rule", "scenario"])[diagnostic_columns].agg(["min", "mean", "max"])
    diagnostic_summary.columns = [f"{a}_{b}" for a, b in diagnostic_summary.columns]
    diagnostic_summary.reset_index().to_csv(out / "inference_checks.csv", index=False)
    check_columns = [c for c in check_frame.columns if c not in {"rule", "scenario", "replicate"}]
    approximation_rows = []
    for (rule, scenario), g in check_frame.groupby(["rule", "scenario"]):
        for metric in check_columns:
            values = g[metric].to_numpy(float)
            approximation_rows.append({"rule": rule, "scenario": scenario, "metric": metric,
                "n_seasons": len(g), "mean_difference": float(values.mean()),
                "mean_absolute_difference": float(np.abs(values).mean()),
                "max_absolute_difference": float(np.abs(values).max()),
                "main_particles": cfg["particles"], "reference_particles": cfg["approximation_particles"]})
    pd.DataFrame(approximation_rows).to_csv(out / "approximation_checks.csv", index=False)
    for name, old in sources.items():
        if digest(source_paths[name]) != old:
            raise RuntimeError(f"Study source changed during the run: {name}")
    public_files = ["protocol.json", "recovery_summary.csv", "coverage_probe.csv", "inference_checks.csv", "approximation_checks.csv"]
    write_json(out / "status.json", {"schema_version": 1, "status": "completed",
               "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
               "source_sha256": sources, "config_sha256": digest(config_path),
               "main_inference_fits": protocol["main_inference_fits"],
               "independent_generated_seasons": protocol["independent_generated_seasons"],
               "approximation_fits": len(check_frame),
               "artifacts_sha256": {name: digest(out / name) for name in public_files},
               "public_artifacts": public_files,
               "private_artifacts": ["private_replicates.csv", "private_particle_diagnostics.csv", "private_approximation_replicates.csv"]})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/recovery_study.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run_study(args.config, args.out)


if __name__ == "__main__":
    main()
