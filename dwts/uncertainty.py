"""Propagate jointly sampled fan-support uncertainty into weekly rule decisions.

The estimand is a one-exit policy applied to fixed observed, nonfinal weekly
rosters. It is conditional on the fitted model, observed eliminations and the
declared controversy definition. It is neither a forecast of a counterfactual
season nor validation against unobserved audience vote totals. Only aggregate
season/phase/overall tables are exported; no person-level draws or chains.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata

from .analysis import check_output_path, digest, load_source, write_json
from .rules import weighted_rank_key

SCOPE = "nonfinal_observed_rosters_n_ge_3_excluding_withdrawal_weeks"


@dataclass(frozen=True)
class Policy:
    rule: str
    judge_weight: float
    mechanism: str

    @property
    def label(self):
        return f"{self.rule}:w{self.judge_weight:g}:{self.mechanism}"


@dataclass
class SeasonResult:
    season: int
    weeks: tuple[int, ...]
    contestant_rows: int
    # Arrays retain the same joint draw for every included week in a season.
    # Axis order is posterior draw, policy, metric (type A, type B, changed exit).
    counts: np.ndarray
    policies: tuple[Policy, ...]


def read_protocol(path):
    """Validate a fixed study declaration, never select settings by outcomes."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"schema_version", "seed", "draws_per_chain", "scope",
               "counterfactual_exits_per_week", "rules", "judge_weights",
               "mechanisms", "interval_probability", "sensitivity_sources"}
    if set(cfg) != allowed or cfg["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete uncertainty protocol")
    if (type(cfg["seed"]) is not int or cfg["seed"] < 0 or
            type(cfg["draws_per_chain"]) is not int or cfg["draws_per_chain"] < 4):
        raise ValueError("A nonnegative seed and at least four draws per chain are required")
    if cfg["scope"] != SCOPE or cfg["counterfactual_exits_per_week"] != 1:
        raise ValueError("Only the declared one-exit, fixed-roster scope is supported")
    if (cfg["rules"] != ["rank", "percentage"] or
            cfg["judge_weights"] != [.25, .5, .75] or
            cfg["mechanisms"] != ["none", "judge_save", "fan_save"] or
            cfg["interval_probability"] != .95):
        raise ValueError("Use the fixed 18 policies and 95% posterior interval")
    labels = set()
    for item in cfg["sensitivity_sources"]:
        if set(item) != {"label", "source", "changed_fit_field", "changed_fit_value"}:
            raise ValueError("Every sensitivity must declare its source and changed assumption")
        if (not isinstance(item["label"], str) or not item["label"] or
                item["label"] in labels or item["label"] == "baseline"):
            raise ValueError("Sensitivity labels must be unique and nonempty")
        labels.add(item["label"])
        if item["changed_fit_field"] not in {"likelihood", "penalty_lambda"}:
            raise ValueError("Only named likelihood/penalty sensitivities are supported")
    return cfg


def policies_from_protocol(cfg):
    return tuple(Policy(rule, float(weight), mechanism)
                 for rule in cfg["rules"] for weight in cfg["judge_weights"]
                 for mechanism in cfg["mechanisms"])


def stratified_draw_indices(draws, count, seed, season, chain):
    """One seeded index per equal-width stratum, without replacement."""
    if draws < 4 or count < 4 or count > draws:
        raise ValueError("Subsample requires 4 <= count <= stored draws")
    bounds = np.floor(np.linspace(0, draws, count + 1)).astype(int)
    rng = np.random.default_rng(np.random.SeedSequence([seed, season, chain, 101]))
    return np.array([rng.integers(a, b) for a, b in zip(bounds[:-1], bounds[1:])])


def included_weeks(frame):
    """Observed labels affect fitted inputs, never counterfactual exit counts."""
    return tuple(int(week) for week, g in frame.groupby("week", sort=True)
                 if len(g) >= 3 and not g.is_final_week.any() and not g.withdrawal.any())


def _validate_inputs(judges, shares, logits, ids):
    judges, shares, logits = (np.asarray(x, dtype=float) for x in (judges, shares, logits))
    ids = np.asarray(ids, str)
    if (judges.ndim != 1 or shares.ndim != 2 or shares.shape != logits.shape or
            shares.shape[1] != len(judges) or ids.shape != judges.shape or len(judges) < 3):
        raise ValueError("Need aligned judges, identifiers and draw-by-contestant arrays")
    if (len(set(ids)) != len(ids) or not np.isfinite(judges).all() or
            not np.isfinite(shares).all() or not np.isfinite(logits).all() or
            (judges < 0).any() or judges.sum() <= 0 or (shares < 0).any() or
            not np.allclose(shares.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Invalid score, share, latent score or identifier")
    return judges, shares, logits, ids


def policy_draws(judges, shares, logits, ids, policies):
    """Vectorized exact-rank ordering and controversy for each joint draw.

    Rank comparisons use latent logits so floating-point softmax underflow
    cannot merge different fan ranks. Percentage merits use numeric shares;
    latent logits resolve fan-order ties, as in posterior_eliminations.
    """
    j, p, z, ids = _validate_inputs(judges, shares, logits, ids)
    draws, n = p.shape
    judge_ranks = rankdata(-j, method="average")
    fan_ranks = rankdata(-z, axis=1, method="average")
    a = (judge_ranks[None, :] <= 2) & (fan_ranks > n / 2)
    b = (judge_ranks[None, :] > n - 2) & (fan_ranks <= n / 2)
    eliminated = np.empty((draws, len(policies)), dtype=np.int64)
    counts = np.empty((draws, len(policies), 2), dtype=np.int64)
    draw_index = np.arange(draws)
    id_key, judge_key = np.broadcast_to(ids, p.shape), np.broadcast_to(j, p.shape)
    for k, policy in enumerate(policies):
        if policy.rule == "rank":
            key = weighted_rank_key(judge_ranks, fan_ranks, policy.judge_weight)
        elif policy.rule == "percentage":
            key = policy.judge_weight * j / j.sum() + (1 - policy.judge_weight) * p
        else:
            raise ValueError("Unknown policy rule")
        order = np.lexsort((id_key, judge_key, z, key), axis=1)
        if policy.mechanism == "none":
            loser = order[:, 0]
        elif policy.mechanism in {"judge_save", "fan_save"}:
            pool = order[:, :2]
            jj, zz, ii = j[pool], np.take_along_axis(z, pool, axis=1), ids[pool]
            keys = (ii, zz, jj) if policy.mechanism == "judge_save" else (ii, jj, zz)
            loser = pool[draw_index, np.lexsort(keys, axis=1)[:, 0]]
        else:
            raise ValueError("Unknown policy mechanism")
        eliminated[:, k] = loser
        counts[:, k, 0] = a[draw_index, loser]
        counts[:, k, 1] = b[draw_index, loser]
    return eliminated, counts


def evaluate_season(frame, shares, logits, policies, weeks=None):
    """Aggregate whole-season joint draws before forming posterior intervals."""
    if not np.array_equal(frame.index, np.arange(len(frame))):
        raise ValueError("Season observations must have contiguous canonical indices")
    if shares.shape != logits.shape or shares.shape[1] != len(frame):
        raise ValueError("Posterior observations disagree with the canonical season rows")
    weeks = included_weeks(frame) if weeks is None else tuple(weeks)
    if not weeks or len(set(weeks)) != len(weeks) or not set(weeks) <= set(included_weeks(frame)):
        raise ValueError("Matched scope must contain unique eligible weeks")
    total = np.zeros((len(shares), len(policies), 3), dtype=np.int64)
    roster_rows = 0
    for week in weeks:
        g = frame[frame.week.eq(week)]
        ix = g.index.to_numpy()
        losers, controversy = policy_draws(g.judge_mean.to_numpy(), shares[:, ix],
                                         logits[:, ix], g.contestant_id.to_numpy(), policies)
        total[:, :, :2] += controversy
        for k, policy in enumerate(policies):
            paired_rank = policies.index(Policy("rank", policy.judge_weight, policy.mechanism))
            total[:, k, 2] += losers[:, k] != losers[:, paired_rank]
        roster_rows += len(g)
    return SeasonResult(int(frame.season.iloc[0]), weeks, roster_rows, total, tuple(policies))


def summarize(values, probability=.95):
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("Summary needs at least four finite joint draws")
    tail = (1 - probability) / 2
    return {"mean": float(values.mean()), "lower": float(np.quantile(values, tail)),
            "upper": float(np.quantile(values, 1 - tail))}


def result_groups(results):
    """Combine independent seasons by matching seeded draw permutations.

    Within a season all weeks and all policies retain the same posterior draw.
    Across seasons fitted independently, matching already shuffled draws forms
    Monte Carlo draws from the product posterior, not weekly independent draws.
    """
    if not results or len({r.season for r in results}) != len(results):
        raise ValueError("Need unique, nonempty season results")
    shape = results[0].counts.shape
    if any(r.counts.shape != shape or r.policies != results[0].policies for r in results):
        raise ValueError("Every season must share joint draw count and declared policies")
    yield "overall", "all", results
    for name, predicate in [("S1-2", lambda s: s <= 2), ("S3-27", lambda s: 3 <= s <= 27),
                            ("S28-34", lambda s: s >= 28)]:
        subset = [r for r in results if predicate(r.season)]
        if subset:
            yield "phase", name, subset
    for result in results:
        yield "season", str(result.season), [result]


def aggregate_tables(results, label, probability=.95):
    rows, contrasts = [], []
    policies = results[0].policies
    pairs = []
    for k, policy in enumerate(policies):
        if policy.mechanism != "none":
            pairs.append(("save_vs_none", k,
                          policies.index(Policy(policy.rule, policy.judge_weight, "none"))))
        if policy.rule == "percentage":
            pairs.append(("percentage_vs_rank", k,
                          policies.index(Policy("rank", policy.judge_weight, policy.mechanism))))
        if policy.judge_weight != .5:
            pairs.append(("weight_vs_half", k,
                          policies.index(Policy(policy.rule, .5, policy.mechanism))))
    for level, group, subset in result_groups(results):
        weeks = sum(len(r.weeks) for r in subset)
        counts = np.sum([r.counts for r in subset], axis=0)
        rate = (counts[:, :, 0] + counts[:, :, 1]) / weeks
        info = dict(source=label, scope=SCOPE, level=level, group=group,
                    seasons=len(subset), weeks=weeks,
                    contestant_rows=sum(r.contestant_rows for r in subset),
                    posterior_draws=len(rate), interval_probability=probability)
        for k, policy in enumerate(policies):
            for metric, values in [("controversial_week_fraction", rate[:, k]),
                                   ("type_a_week_fraction", counts[:, k, 0] / weeks),
                                   ("type_b_week_fraction", counts[:, k, 1] / weeks),
                                   ("exit_change_fraction_vs_rank", counts[:, k, 2] / weeks)]:
                rows.append(dict(info, rule=policy.rule, judge_weight=policy.judge_weight,
                                 mechanism=policy.mechanism, metric=metric,
                                 **summarize(values, probability)))
        for kind, target, reference in pairs:
            delta = rate[:, target] - rate[:, reference]
            contrasts.append(dict(info, contrast=kind, target=policies[target].label,
                reference=policies[reference].label,
                metric="controversial_week_fraction_difference_target_minus_reference",
                **summarize(delta, probability), posterior_probability_lower=float((delta < 0).mean()),
                posterior_probability_equal=float((delta == 0).mean()),
                posterior_probability_higher=float((delta > 0).mean())))
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def validate_sensitivity(base_config, alternative_config, declaration):
    """Check that only the named modeling assumption differs in matched fits."""
    if alternative_config.get("data", {}) != base_config.get("data", {}):
        raise ValueError("Sensitivity preprocessing differs from the baseline")
    changed = declaration["changed_fit_field"]
    base, alternative = base_config["fit"], alternative_config["fit"]
    if alternative.get(changed) != declaration["changed_fit_value"]:
        raise ValueError("Sensitivity does not use its declared changed assumption")
    if {k: v for k, v in base.items() if k != changed} != {
            k: v for k, v in alternative.items() if k != changed}:
        raise ValueError("Sensitivity changes additional fitted assumptions")
    if not set(alternative_config["seasons"]) <= set(base_config["seasons"]):
        raise ValueError("Sensitivity seasons must be a subset of the baseline")


def validate_source_implementation(base_manifest, alternative_manifest):
    """Matched assumptions require the same recorded model code and environment."""
    for field in ["input_sha256", "code_sha256", "python", "packages"]:
        if not base_manifest.get(field) or base_manifest[field] != alternative_manifest.get(field):
            raise ValueError(f"Sensitivity source implementation/environment differs: {field}")


def load_season(source, votes, season, cfg, seed, count):
    """Validate chain coordinates/diagnostics and recover latent fan ordering."""
    import xarray as xr
    root = Path(source) / f"season_{season:02d}"
    status = json.loads((root / "fit_status.json").read_text())
    if status.get("season") != season or status.get("config") != cfg["fit"]:
        raise ValueError("Season configuration does not match the source declaration")
    if status.get("sampling_diagnostics_pass") is not True:
        raise ValueError("Season sampling diagnostics did not pass")
    diag = pd.read_csv(root / "diagnostics.csv", index_col=0)
    numeric = diag[["r_hat", "ess_bulk", "ess_tail"]].to_numpy(float)
    if (not np.isfinite(numeric).all() or (numeric[:, 0] > 1.01).any() or
            (numeric[:, 1:] < 400).any()):
        raise ValueError("Original full-chain diagnostics do not pass")
    for field, actual in [("max_rhat", numeric[:, 0].max()),
                          ("min_ess_bulk", numeric[:, 1].min()),
                          ("min_ess_tail", numeric[:, 2].min())]:
        if not np.isclose(status[field], actual, rtol=0, atol=1e-10):
            raise ValueError("Diagnostic summary disagrees with original diagnostic rows")
    frame = votes[votes.season.eq(season)].sort_values(["week", "contestant_id"]).reset_index(drop=True)
    expected_parameters = {f"alpha[{name}]" for name in frame.contestant_id.unique()} | {
        "beta", "sigma"} | {f"fan_share[{i}]" for i in range(len(frame))}
    if diag.index.has_duplicates or set(diag.index) != expected_parameters:
        raise ValueError("Original diagnostic rows do not cover every declared model parameter")
    stored = pd.read_csv(root / "votes.csv", float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(stored, frame, check_exact=True)
    except AssertionError as error:
        raise ValueError("Per-season rows disagree with pooled canonical source rows") from error
    posterior_path = root / "posterior.nc"
    with xr.open_dataset(posterior_path, group="posterior", engine="h5netcdf") as posterior:
        nchain, ndraw = posterior.sizes["chain"], posterior.sizes["draw"]
        if (nchain != cfg["fit"]["chains"] or ndraw != cfg["fit"]["draws"] or
                posterior.sizes["observation"] != len(frame) or
                not np.array_equal(posterior.observation, np.arange(len(frame))) or
                not np.array_equal(posterior.chain, np.arange(nchain)) or
                not np.array_equal(posterior.draw, np.arange(ndraw)) or
                list(posterior.contestant.values.astype(str)) != sorted(frame.contestant_id.unique())):
            raise ValueError("Posterior coordinates disagree with declared observation/contestant identity")
        if tuple(posterior.fan_share.dims) != ("chain", "draw", "observation"):
            raise ValueError("Unexpected posterior fan-share dimension order")
        if count > ndraw:
            raise ValueError("Requested draws per chain exceed stored posterior draws; use a sufficiently long fit")
        full_mean = posterior.fan_share.mean(dim=["chain", "draw"]).values
        if not np.allclose(full_mean, frame.fan_mean, atol=1e-12, rtol=0):
            raise ValueError("Posterior mean shares disagree with frozen observation rows")
        actual_count = count
        indices = [stratified_draw_indices(ndraw, actual_count, seed, season, chain)
                   for chain in range(nchain)]
        selected = [posterior.isel(chain=chain, draw=ix).load()
                    for chain, ix in enumerate(indices)]
        p = np.concatenate([part.fan_share.values for part in selected])
        if "z" in posterior:
            z = np.concatenate([part.z.values for part in selected])
        else:
            contestant_index = np.searchsorted(sorted(frame.contestant_id.unique()), frame.contestant_id)
            z = np.concatenate([part.alpha.values[:, contestant_index] +
                                part.beta.values[:, None] * frame.judge_z.to_numpy() +
                                part.sigma.values[:, None] * part.epsilon.values
                                for part in selected])
        for week, g in frame.groupby("week", sort=True):
            ix = g.index.to_numpy()
            if not np.allclose(softmax(z[:, ix], axis=1), p[:, ix], rtol=0, atol=2e-13):
                raise ValueError(f"Latent scores and stored shares disagree in season {season}, week {week}")
        # Independent seasonal permutations form product-posterior joint draws.
        permutation = np.random.default_rng(np.random.SeedSequence([seed, season, 202])).permutation(len(p))
        p, z = p[permutation], z[permutation]
    files = ["fit_status.json", "diagnostics.csv", "votes.csv", "posterior.nc"]
    integrity = dict(season=season, hashes={name: digest(root / name) for name in files},
        stored_chains=nchain, stored_draws_per_chain=ndraw, selected_draws_per_chain=actual_count,
        selection_indices_sha256=hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
        permutation_sha256=hashlib.sha256(permutation.astype("<i8").tobytes()).hexdigest(),
        diagnostics={field: status[field] for field in ["max_rhat", "min_ess_bulk", "min_ess_tail"]})
    return frame, p, z, integrity


def run(source, protocol_path, out, data):
    """Validate all sources before creating a fresh aggregate-only output run."""
    source, out, data = Path(source).resolve(), Path(out).resolve(), Path(data).resolve()
    check_output_path(source, out)
    cfg = read_protocol(protocol_path)
    policies = policies_from_protocol(cfg)
    base_votes, _, base_manifest, base_cfg = load_source(source, data)
    sources = [("baseline", source, base_votes, base_manifest, base_cfg, None)]
    for declaration in cfg["sensitivity_sources"]:
        alternate = Path(declaration["source"]).resolve()
        check_output_path(alternate, out)
        votes, _, manifest, fit_cfg = load_source(alternate, data)
        validate_sensitivity(base_cfg, fit_cfg, declaration)
        validate_source_implementation(base_manifest, manifest)
        sources.append((declaration["label"], alternate, votes, manifest, fit_cfg, declaration))
    results_by_source, integrity_by_source, frames_by_source = {}, {}, {}
    for label, root, votes, manifest, fit_cfg, declaration in sources:
        results, integrity, frames = [], [], {}
        for season in sorted(fit_cfg["seasons"]):
            print(f"{label}: validating and evaluating season {season}", flush=True)
            frame, p, z, season_integrity = load_season(root, votes, season, fit_cfg,
                                                      cfg["seed"], cfg["draws_per_chain"])
            results.append(evaluate_season(frame, p, z, policies))
            frames[season] = frame
            integrity.append(season_integrity)
        results_by_source[label], frames_by_source[label] = results, frames
        integrity_by_source[label] = dict(
            source_directory_name=root.name, run_config=fit_cfg, manifest=manifest,
            run_files_sha256={name: digest(root / name) for name in
                              ["manifest.json", "run_config.json", "sampling_status.json", "votes.csv"]},
            seasons=integrity, declared_sensitivity=declaration)
    summaries, contrasts, sensitivity = [], [], []
    for label, results in results_by_source.items():
        summary, contrast = aggregate_tables(results, label, cfg["interval_probability"])
        summaries.append(summary)
        contrasts.append(contrast)
        if label != "baseline":
            alternative = {r.season: r for r in results}
            baseline = [r for r in results_by_source["baseline"] if r.season in alternative]
            for base in baseline:
                alt = alternative[base.season]
                if base.weeks != alt.weeks:
                    raise ValueError("Sensitivity must use identical eligible observed weeks")
                columns = ["season", "week", "contestant_id", "judge_mean", "judge_z",
                           "eliminated", "withdrawal", "is_final_week"]
                try:
                    pd.testing.assert_frame_equal(frames_by_source["baseline"][base.season][columns],
                                                  frames_by_source[label][base.season][columns], check_exact=True)
                except AssertionError as error:
                    raise ValueError("Sensitivity must use identical rosters, scores and observed labels") from error
            matched_label = f"baseline_matched_to_{label}"
            base_summary, base_contrast = aggregate_tables(baseline, matched_label, cfg["interval_probability"])
            summaries.append(base_summary)
            contrasts.append(base_contrast)
            # Differences of conditional means are sensitivity descriptions.
            # Independently fitted model posteriors have no natural draw pairing.
            keys = ["level", "group", "rule", "judge_weight", "mechanism", "metric"]
            merged = summary.merge(base_summary, on=keys, suffixes=("_alternative", "_baseline"), validate="one_to_one")
            for _, row in merged.iterrows():
                sensitivity.append({**{key: row[key] for key in keys}, "source": label,
                    "matched_baseline": matched_label, "weeks": int(row.weeks_baseline),
                    "baseline_mean": row.mean_baseline, "alternative_mean": row.mean_alternative,
                    "mean_difference_alternative_minus_baseline": row.mean_alternative - row.mean_baseline,
                    "comparison": "matched_rosters_difference_of_conditional_posterior_means"})
    # Nothing has been written before all checks and matched populations pass.
    out.mkdir(parents=True)
    pd.concat(summaries, ignore_index=True).to_csv(out / "policy_summary.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(out / "policy_contrasts.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out / "sensitivity_summary.csv", index=False)
    write_json(out / "protocol.json", dict(cfg,
        quantile_method="numpy_linear_equal_tailed_empirical",
        posterior_joint_draws_per_source={label: len(results[0].counts)
                                          for label, results in results_by_source.items()},
        fan_ordering="Latent logits define fan ranks and fan tie-break/save order for both rule families; numeric softmax underflow is not a genuine vote tie.",
        percentage_merit="Weighted stored numeric fan shares and judge shares; no repair or clipping of softmax underflow.",
        exact_rank_merit="Integer weighted doubled-average-rank key from exact decimal judge weights.",
        seasonal_combination="independent_seeded_permutations_preserving_joint_weeks_and_paired_policies",
        intervals="Conditional posterior credible intervals; not frequentist confidence intervals or p-values.",
        conditioning="Observed elimination labels informed the posterior; these are not independent held-out predictions.",
        limitations=["No unobserved true audience votes are available for accuracy validation.",
                     "Observed weekly rosters are held fixed; no full-season counterfactual path is simulated.",
                     "One exit per week is a standardized policy even in observed no-exit or multi-exit weeks.",
                     "All withdrawal weeks are excluded; other observed labels do not determine policy exit counts.",
                     "Intervals reflect model-conditional posterior uncertainty, not all model/specification uncertainty.",
                     "Original full-chain diagnostics apply before stratified subsampling; Monte Carlo summaries remain approximate.",
                     "Sensitivity fits vary penalty strength and late-season likelihood only; prior scales were not refitted.",
                     "A high posterior probability is conditional support under fitted assumptions, not a causal or universal policy guarantee."]))
    write_json(out / "source_integrity.json", integrity_by_source)
    generated = ["policy_summary.csv", "policy_contrasts.csv", "sensitivity_summary.csv",
                 "protocol.json", "source_integrity.json"]
    write_json(out / "status.json", dict(
        execution_completed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        sources=list(results_by_source), baseline_seasons=len(results_by_source["baseline"]),
        baseline_weeks=sum(len(r.weeks) for r in results_by_source["baseline"]),
        code_sha256=digest(Path(__file__)), protocol_sha256=digest(protocol_path),
        analysis_code_sha256={name: digest(Path(__file__).with_name(name))
                              for name in ["uncertainty.py", "rules.py", "analysis.py", "data.py"]},
        output_sha256={name: digest(out / name) for name in generated},
        all_source_diagnostics_pass=True, independent_true_vote_validation=False))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Completed diagnosed fit directory")
    parser.add_argument("--config", type=Path, default=Path("configs/uncertainty_study.json"))
    parser.add_argument("--out", type=Path, required=True, help="New directory for aggregate-only results")
    parser.add_argument("--data", type=Path, default=Path("data/raw/2026_MCM_Problem_C_Data.csv"))
    args = parser.parse_args(argv)
    print(f"Aggregate uncertainty study completed: {run(args.source, args.config, args.out, args.data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
