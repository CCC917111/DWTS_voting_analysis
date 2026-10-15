"""Multiple declared policy objectives across matched posterior scenarios.

Policies act once on each eligible fixed observed roster. Scenario comparisons
are a finite sensitivity analysis, not a posterior distribution over models.
Only policy-vs-policy comparisons within the same posterior are draw-paired.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .analysis import check_output_path, digest, load_source, write_json
from .prior_sensitivity import transport_draws
from .uncertainty import (Policy, SCOPE, included_weeks, load_season,
                          policies_from_protocol, policy_draws, summarize,
                          validate_sensitivity, validate_source_implementation)

METRICS = (
    ("type_a_exit_rate", "lower"),
    ("type_b_exit_rate", "lower"),
    ("eliminated_judge_rank_percentile", "higher"),
    ("eliminated_fan_rank_percentile", "higher"),
    ("judge_top_half_exit_rate", "lower"),
    ("fan_top_half_exit_rate", "lower"),
)
LOSS_SIGNS = np.array([1. if direction == "lower" else -1. for _, direction in METRICS])
REFERENCE = Policy("rank", .5, "none")
SOURCE_FILES = ("manifest.json", "run_config.json", "sampling_status.json", "votes.csv")
SEASON_FILES = ("fit_status.json", "diagnostics.csv", "votes.csv", "posterior.nc")
CODE_FILES = ("policy_robustness.py", "uncertainty.py", "prior_sensitivity.py", "rules.py",
              "analysis.py", "data.py", "model.py", "fast_mh.py")


@dataclass
class SeasonMetrics:
    season: int
    weeks: tuple[int, ...]
    contestant_rows: int
    # Joint draw x policy x metric sums across retained weeks.
    totals: np.ndarray
    policies: tuple[Policy, ...]


def read_protocol(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = {"schema_version", "seed", "draws_per_chain", "scope",
            "counterfactual_exits_per_week", "rules", "judge_weights", "mechanisms",
            "interval_probability", "reference_policy", "metrics", "pareto_tolerance",
            "late_seasons", "late_joint_prior_scales", "sensitivity_sources"}
    if set(cfg) != keys or cfg["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete multi-metric policy protocol")
    if (type(cfg["seed"]) is not int or cfg["seed"] < 0 or
            type(cfg["draws_per_chain"]) is not int or cfg["draws_per_chain"] < 4):
        raise ValueError("Need a nonnegative seed and at least four draws per chain")
    if (cfg["scope"] != SCOPE or cfg["counterfactual_exits_per_week"] != 1 or
            cfg["rules"] != ["rank", "percentage"] or
            cfg["judge_weights"] != [.25, .5, .75] or
            cfg["mechanisms"] != ["none", "judge_save", "fan_save"] or
            cfg["interval_probability"] != .95 or cfg["reference_policy"] != REFERENCE.label):
        raise ValueError("Use fixed 18 policies, one-exit scope, reference and 95% intervals")
    if (cfg["metrics"] != [{"metric": metric, "direction": direction} for metric, direction in METRICS]
            or cfg["pareto_tolerance"] != 1e-12 or
            cfg["late_seasons"] != list(range(28, 35)) or
            cfg["late_joint_prior_scales"] != [.5, 1., 2.]):
        raise ValueError("Metric directions, tolerance and matched scenarios must be prespecified")
    expected = [("late_bottom_two", "likelihood", "bottom_two", list(range(28, 35))),
                ("season3_penalty_0p1", "penalty_lambda", .1, [3]),
                ("season3_penalty_10", "penalty_lambda", 10., [3])]
    declarations = cfg["sensitivity_sources"]
    if len(declarations) != len(expected):
        raise ValueError("All three matched source declarations are required")
    for declaration, (label, field, value, seasons) in zip(declarations, expected):
        if (set(declaration) != {"label", "source", "changed_fit_field", "changed_fit_value", "seasons"}
                or declaration["label"] != label or declaration["changed_fit_field"] != field
                or declaration["changed_fit_value"] != value or declaration["seasons"] != seasons):
            raise ValueError("Sensitivity declarations must match the fixed scenario grid")
    return cfg


def weekly_metrics(judges, shares, logits, ids, policies):
    """Return six bounded endpoints for each policy and the same joint draw.

    Average ranks use 1 for best. Rank percentile is (rank-1)/(n-1), so a
    larger eliminated rank percentile removes a weaker-rated contestant.
    Top-half means average rank <= n/2; no rounding or random tie breaking.
    """
    losers, types = policy_draws(judges, shares, logits, ids, policies)
    n = len(judges)
    jr = rankdata(-np.asarray(judges), method="average")
    fr = rankdata(-np.asarray(logits), axis=1, method="average")
    fan_exit = fr[np.arange(len(fr))[:, None], losers]
    judge_exit = jr[losers]
    values = np.stack((types[:, :, 0], types[:, :, 1],
                       (judge_exit - 1) / (n - 1), (fan_exit - 1) / (n - 1),
                       judge_exit <= n / 2, fan_exit <= n / 2), axis=2).astype(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Policy metrics must be finite bounded endpoints")
    return values


def evaluate_season(frame, shares, logits, policies):
    if (not np.array_equal(frame.index, np.arange(len(frame))) or
            shares.shape != logits.shape or shares.shape[1] != len(frame)):
        raise ValueError("Posterior arrays require canonically aligned observation rows")
    weeks = included_weeks(frame)
    if not weeks:
        raise ValueError("Each scenario season needs eligible observed weeks")
    total = np.zeros((len(shares), len(policies), len(METRICS)))
    roster_rows = 0
    for week in weeks:
        g = frame[frame.week.eq(week)]
        ix = g.index.to_numpy()
        total += weekly_metrics(g.judge_mean.to_numpy(), shares[:, ix], logits[:, ix],
                                g.contestant_id.to_numpy(), policies)
        roster_rows += len(g)
    return SeasonMetrics(int(frame.season.iloc[0]), weeks, roster_rows, total, tuple(policies))


def aggregate(results):
    if not results or len({r.season for r in results}) != len(results):
        raise ValueError("Need unique seasons in each matched cohort")
    first = results[0]
    if any(r.totals.shape != first.totals.shape or r.policies != first.policies for r in results):
        raise ValueError("Matched joint draws and policy declarations must agree")
    weeks = sum(len(r.weeks) for r in results)
    return np.sum([r.totals for r in results], axis=0) / weeks, dict(
        seasons=len(results), weeks=weeks, contestant_rows=sum(r.contestant_rows for r in results),
        posterior_draws=len(first.totals))


def pareto_relation(target, reference, tolerance=1e-12):
    """Pointwise relation of conditional mean endpoints, with explicit losses."""
    delta = (np.asarray(target, float) - np.asarray(reference, float)) * LOSS_SIGNS
    if delta.shape != (len(METRICS),) or not np.isfinite(delta).all() or tolerance < 0:
        raise ValueError("Pareto endpoints must be six finite means and a valid tolerance")
    better, worse = delta < -tolerance, delta > tolerance
    if better.any() and not worse.any():
        return "target_dominates"
    if worse.any() and not better.any():
        return "reference_dominates"
    if better.any() and worse.any():
        return "tradeoff"
    return "equal"


def posterior_dominance_support(target_draws, reference_draws, tolerance):
    """Within-model paired-draw probability; distinct from mean Pareto relation."""
    delta = (np.asarray(target_draws) - np.asarray(reference_draws)) * LOSS_SIGNS
    if delta.ndim != 2 or delta.shape[1] != len(METRICS) or not np.isfinite(delta).all():
        raise ValueError("Need finite paired draw-by-metric endpoints")
    return float(((delta <= tolerance).all(axis=1) & (delta < -tolerance).any(axis=1)).mean())


def scenario_tables(scenarios, policies, probability, tolerance):
    metrics, contrasts = [], []
    ref = policies.index(REFERENCE)
    grouped = {}
    for cohort, scenario, assumptions, results in scenarios:
        values, population = aggregate(results)
        info = dict(cohort=cohort, scenario=scenario, **assumptions, **population,
                    interval_probability=probability)
        grouped.setdefault(cohort, []).append((scenario, values, population))
        for k, policy in enumerate(policies):
            policy_info = dict(rule=policy.rule, judge_weight=policy.judge_weight,
                               mechanism=policy.mechanism, policy=policy.label)
            for m, (metric, direction) in enumerate(METRICS):
                metrics.append(dict(info, **policy_info, metric=metric, direction=direction,
                                    **summarize(values[:, k, m], probability)))
                delta = values[:, k, m] - values[:, ref, m]
                loss_delta = delta * LOSS_SIGNS[m]
                summary = summarize(delta, probability)
                contrasts.append(dict(info, target=policy.label, reference=REFERENCE.label,
                    metric=metric, direction=direction,
                    mean_difference_target_minus_reference=summary["mean"],
                    lower=summary["lower"], upper=summary["upper"],
                    posterior_probability_improvement=float((loss_delta < -tolerance).mean()),
                    posterior_probability_equal=float((np.abs(loss_delta) <= tolerance).mean()),
                    posterior_probability_worsening=float((loss_delta > tolerance).mean())))
    stability, worst_case = [], []
    for cohort, items in grouped.items():
        populations = [population for _, _, population in items]
        if any(population != populations[0] for population in populations):
            raise ValueError("Every robustness scenario must retain identical matched denominators")
        scenario_labels = "|".join(name for name, _, _ in items)
        means = np.stack([value.mean(axis=0) for _, value, _ in items])
        for target, reference in itertools.combinations(range(len(policies)), 2):
            relations = [pareto_relation(row[target], row[reference], tolerance) for row in means]
            forward = [posterior_dominance_support(value[:, target], value[:, reference], tolerance)
                       for _, value, _ in items]
            reverse = [posterior_dominance_support(value[:, reference], value[:, target], tolerance)
                       for _, value, _ in items]
            stability.append(dict(cohort=cohort, target=policies[target].label,
                reference=policies[reference].label, scenario_count=len(items),
                scenario_labels=scenario_labels, weeks=populations[0]["weeks"],
                target_dominates_scenarios=relations.count("target_dominates"),
                reference_dominates_scenarios=relations.count("reference_dominates"),
                tradeoff_scenarios=relations.count("tradeoff"), equal_scenarios=relations.count("equal"),
                robust_target_dominance=all(r == "target_dominates" for r in relations),
                robust_reference_dominance=all(r == "reference_dominates" for r in relations),
                stable_tradeoff=all(r == "tradeoff" for r in relations),
                minimum_conditional_posterior_target_dominance_support=min(forward),
                maximum_conditional_posterior_target_dominance_support=max(forward),
                minimum_conditional_posterior_reference_dominance_support=min(reverse),
                maximum_conditional_posterior_reference_dominance_support=max(reverse)))
        for k, policy in enumerate(policies):
            for m, (metric, direction) in enumerate(METRICS):
                losses = means[:, k, m] * LOSS_SIGNS[m]
                worst, best = int(np.argmax(losses)), int(np.argmin(losses))
                worst_case.append(dict(cohort=cohort, policy=policy.label, metric=metric,
                    direction=direction, scenario_count=len(items), weeks=populations[0]["weeks"],
                    worst_conditional_mean=means[worst, k, m], worst_scenario=items[worst][0],
                    best_conditional_mean=means[best, k, m], best_scenario=items[best][0],
                    scenario_mean_range=np.ptp(means[:, k, m]),
                    worst_case_definition="endpointwise_worst_of_finite_prespecified_conditional_means"))
    return tuple(pd.DataFrame(rows) for rows in (metrics, contrasts, stability, worst_case))


def validate_matched_frames(base, alternative):
    # Posterior summaries vary legitimately; every other prepared input must agree.
    posterior_columns = {"fan_mean", "fan_lower", "fan_upper", "relative_uncertainty"}
    columns = [name for name in base if name not in posterior_columns]
    if set(columns) != set(alternative) - posterior_columns:
        raise ValueError("Matched scenarios have different prepared input columns")
    try:
        pd.testing.assert_frame_equal(base[columns], alternative[columns], check_exact=True)
    except AssertionError as error:
        raise ValueError("Matched scenarios need identical rosters, scores and observed inputs") from error
    if included_weeks(base) != included_weeks(alternative):
        raise ValueError("Matched scenario eligibility differs")


def freeze_source(root, seasons):
    paths = [root / name for name in SOURCE_FILES]
    paths += [root / f"season_{season:02d}" / name for season in seasons for name in SEASON_FILES]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def run(source, protocol_path, out, data):
    source, out, data = Path(source).resolve(), Path(out).resolve(), Path(data).resolve()
    protocol_path = Path(protocol_path).resolve()
    cfg = read_protocol(protocol_path)
    initial_config_hash = digest(protocol_path)
    initial_code_hashes = {name: digest(Path(__file__).with_name(name)) for name in CODE_FILES}
    initial_data_hash = digest(data)
    policies = policies_from_protocol(cfg)
    specs = [("baseline", source, list(range(1, 35)), None)] + [
        (decl["label"], Path(decl["source"]).resolve(), decl["seasons"], decl)
        for decl in cfg["sensitivity_sources"]]
    sources = {}
    for label, root, seasons, declaration in specs:
        check_output_path(root, out)
        frozen = freeze_source(root, seasons)
        votes, _, manifest, fit_cfg = load_source(root, data)
        if set(fit_cfg["seasons"]) != set(seasons):
            raise ValueError("Source seasons differ from the declared matched cohort")
        if declaration is None:
            if (fit_cfg["fit"].get("likelihood") != "report" or
                    fit_cfg["fit"].get("penalty_lambda") != 1.):
                raise ValueError("Baseline must use the fixed report likelihood and penalty one")
        else:
            validate_sensitivity(sources["baseline"]["config"], fit_cfg, declaration)
            validate_source_implementation(sources["baseline"]["manifest"], manifest)
        sources[label] = dict(root=root, votes=votes, manifest=manifest, config=fit_cfg,
                              frozen=frozen, season_integrity=[])
    frames, full_results = {}, []
    late_results = {(likelihood, scale): [] for likelihood in ["report", "bottom_two"]
                    for scale in cfg["late_joint_prior_scales"]}
    penalty_results = {scale: [] for scale in [.1, 1., 10.]}
    rank_columns = [k for k, policy in enumerate(policies) if policy.rule == "rank"]
    invariant_checks = 0
    for label, _, seasons, _ in specs:
        entry = sources[label]
        for season in seasons:
            print(f"multi-metric policy study: {label} season {season}", flush=True)
            frame, p, z, integrity = load_season(entry["root"], entry["votes"], season,
                entry["config"], cfg["seed"], cfg["draws_per_chain"])
            entry["season_integrity"].append(integrity)
            if label == "baseline":
                frames[season] = frame
            else:
                validate_matched_frames(frames[season], frame)
            baseline_result = evaluate_season(frame, p, z, policies)
            if label == "baseline":
                full_results.append(baseline_result)
                if season == 3:
                    penalty_results[1.].append(baseline_result)
            elif season == 3:
                penalty_results[float(entry["config"]["fit"]["penalty_lambda"])].append(baseline_result)
            if season in cfg["late_seasons"]:
                if (entry["config"]["fit"].get("parameterization") != "noncentered" or
                        entry["config"]["fit"].get("likelihood") not in {"report", "bottom_two"}):
                    raise ValueError("Transport requires a supported noncentered rank target")
                likelihood = entry["config"]["fit"]["likelihood"]
                for scale in cfg["late_joint_prior_scales"]:
                    changed_p, changed_z = transport_draws(frame, p, z, scale)
                    result = evaluate_season(frame, changed_p, changed_z, policies)
                    if scale == 1 and (not np.array_equal(p, changed_p) or
                            not np.array_equal(z, changed_z) or
                            not np.array_equal(result.totals, baseline_result.totals)):
                        raise ValueError("Scale one must preserve all original draw metrics exactly")
                    if not np.array_equal(result.totals[:, rank_columns], baseline_result.totals[:, rank_columns]):
                        raise ValueError("Rank-policy metrics changed under invariant prior transport")
                    invariant_checks += 1
                    late_results[(likelihood, scale)].append(result)
    scenarios = [("baseline_full34", "report_lambda1_scale1", dict(
        likelihood="report", penalty_lambda=1., joint_prior_scale=1.), full_results)]
    scenarios += [("late7_likelihood_prior", f"{likelihood}_scale{scale:g}", dict(
        likelihood=likelihood, penalty_lambda=1., joint_prior_scale=scale), results)
        for (likelihood, scale), results in late_results.items()]
    scenarios += [("season3_penalty", f"lambda{penalty:g}", dict(
        likelihood="report", penalty_lambda=penalty, joint_prior_scale=1.), results)
        for penalty, results in penalty_results.items()]
    tables = scenario_tables(scenarios, policies, cfg["interval_probability"], cfg["pareto_tolerance"])
    if digest(protocol_path) != initial_config_hash or digest(data) != initial_data_hash:
        raise ValueError("Study protocol or raw input changed during execution")
    if any(digest(Path(__file__).with_name(name)) != value for name, value in initial_code_hashes.items()):
        raise ValueError("Study implementation changed during execution")
    for label, _, seasons, _ in specs:
        entry = sources[label]
        if freeze_source(entry["root"], seasons) != entry["frozen"]:
            raise ValueError("Frozen posterior source changed during execution")
    out.mkdir(parents=True)
    filenames = ["policy_metrics.csv", "policy_contrasts.csv", "pareto_stability.csv", "worst_case_metrics.csv"]
    for name, table in zip(filenames, tables):
        table.to_csv(out / name, index=False)
    write_json(out / "protocol.json", augmented_protocol(cfg))
    write_json(out / "source_integrity.json", {label: dict(
        source_directory_name=entry["root"].name, manifest=entry["manifest"],
        run_config=entry["config"], files_sha256=entry["frozen"],
        season_integrity=entry["season_integrity"], pre_post_source_equality_verified=True)
        for label, entry in sources.items()})
    generated = filenames + ["protocol.json", "source_integrity.json"]
    write_json(out / "status.json", dict(execution_completed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=initial_config_hash,
        code_sha256=initial_code_hashes, raw_data_sha256=initial_data_hash,
        baseline_seasons=len(full_results), baseline_weeks=sum(len(r.weeks) for r in full_results),
        scenarios=len(scenarios), policy_count=len(policies), metric_count=len(METRICS),
        rank_policy_transport_invariance_checks=invariant_checks,
        scale_one_exact_invariance_verified=True, matched_inputs_verified=True,
        pre_post_code_protocol_data_source_equality_verified=True,
        all_original_source_diagnostics_pass=True, independent_true_vote_validation=False,
        output_sha256={name: digest(out / name) for name in generated}))
    return out


def augmented_protocol(cfg):
    return dict(cfg,
        scenario_counts={"baseline_full34": 1, "late7_likelihood_prior": 6, "season3_penalty": 3},
        metrics_definition={
            "type_a_exit_rate": "Eliminated judge average rank <=2 and latent fan average rank >n/2.",
            "type_b_exit_rate": "Eliminated judge average rank >n-2 and latent fan average rank <=n/2.",
            "eliminated_judge_rank_percentile": "(Average descending judge rank -1)/(n-1); 0 best, 1 worst.",
            "eliminated_fan_rank_percentile": "(Average descending latent-logit fan rank -1)/(n-1); 0 best, 1 worst.",
            "judge_top_half_exit_rate": "Eliminated average descending judge rank <=n/2, including fractional ties.",
            "fan_top_half_exit_rate": "Eliminated average descending latent fan rank <=n/2, including fractional ties."},
        normative_interpretation="The six partly related declared preferences are contestable protection/alignment objectives, not a universal definition of fairness. No scalar fairness score is used.",
        reference="Fixed rank:w0.5:none in each same-scenario posterior; every policy including self is retained.",
        contrast_intervals="95% equal-tailed conditional posterior credible intervals for paired policy metric differences within the same posterior only.",
        dominance_definition="Pointwise Pareto dominance of conditional posterior mean endpoints: no loss difference > tolerance and at least one < -tolerance. Robust dominance requires dominance in EVERY enumerated scenario in that cohort.",
        posterior_dominance_support="Separate within-scenario paired-draw probability that all six oriented draw endpoints are weakly improved and at least one strictly improved; extrema across scenarios are descriptive extrema, not scenario probabilities.",
        scenario_interpretation="Finite prespecified sensitivity scenarios have no assigned probability or weight. Full34, late7 and season3 cohorts are never pooled. No draw pairing, interval or improvement probability is formed between independently fitted sources.",
        worst_case="Each endpoint's worst conditional mean across its own finite matched scenario grid; different endpoints can attain extrema in different scenarios.",
        joint_prior_transport="Exact alpha,beta,sigma co-scaling with epsilon unchanged, z->c*z, only on late rank-only seasons for both rank likelihoods; no additional MCMC fitting.",
        joint_dependency="Seeded independent seasonal permutations; all weeks and all policies retain the same draw within each season.",
        fan_ordering="Latent logits define fan ranks and fan tie-break/save order, protecting against numeric softmax underflow; percentage merits use numeric shares.",
        limitations=["Retrospective posterior conditioned on observed eliminations; not held-out predictive accuracy or validated true audience votes.",
            "One exit per eligible fixed observed roster; no changing survivor roster or whole-season causal counterfactual is simulated.",
            "Sensitivity covers only the enumerated likelihood, penalty and jointly co-scaled prior family, not all plausible assumptions.",
            "Original full-chain diagnostics were validated; transformed policy-function effective sample sizes were not recomputed.",
            "Finite-chain posterior summaries and scenario-conditioned Pareto relations are approximate and can be sensitive near tolerance boundaries.",
            "A model-conditional posterior dominance probability is not a universal fairness or policy guarantee."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/policy_robustness_study.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/raw/2026_MCM_Problem_C_Data.csv"))
    args = parser.parse_args(argv)
    print(f"Multi-metric robustness study completed: {run(args.source, args.config, args.out, args.data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
