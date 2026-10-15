"""Partial identification under an additional deterministic percentage rule.

These are sharp feasible-set bounds, not posterior credible intervals. Neither
the historical soft-constraint likelihood nor the normalized stochastic exit
model implies the hard constraints used here. Only aggregate study exports are
intended for publication; individual bounds and constructive witnesses are private.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq, linprog

from .data import prepare


DEFAULT_TOLERANCE = 1e-8
PUBLIC_ARTIFACTS = ("protocol.json", "width_summary.csv", "eligibility_summary.csv",
                    "policy_ambiguity.csv", "policy_candidate_counts.csv", "numerical_checks.json")
PRIVATE_ARTIFACTS = ("private_bounds.csv", "private_weekly.csv", "private_policy_candidates.csv")


class InfeasibleRegionError(ValueError):
    """The observed exit has no shares consistent with the declared hard rule."""


@dataclass(frozen=True)
class CoordinateBounds:
    lower: np.ndarray
    upper: np.ndarray
    lower_witnesses: np.ndarray
    upper_witnesses: np.ndarray
    interior_margin: float
    interior_witness: np.ndarray
    max_feasibility_residual: float
    max_analytic_endpoint_difference: float
    lp_optimizations: int


@dataclass(frozen=True)
class ExitAttainability:
    attainable: bool
    status: str
    margin: float | None
    witness: np.ndarray | None
    max_feasibility_residual: float | None


def _weight(value, name="judge_weight"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number strictly between zero and one")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between zero and one")
    return value


def _tolerance(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("Tolerance must be a finite number in [1e-10, 1e-5]")
    value = float(value)
    if not np.isfinite(value) or not 1e-10 <= value <= 1e-5:
        raise ValueError("Tolerance must be a finite number in [1e-10, 1e-5]")
    return value


def _index(value, n, name="eliminated"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < n:
        raise ValueError(f"{name} must be an integer index within the roster")
    return int(value)


def _inputs(judge_shares, eliminated, judge_weight):
    j = np.asarray(judge_shares, dtype=float)
    if (j.ndim != 1 or len(j) < 2 or not np.isfinite(j).all() or
            (j < 0).any() or not np.isclose(j.sum(), 1., atol=1e-10, rtol=0)):
        raise ValueError("judge_shares must be a finite nonnegative vector summing to one, with n>=2")
    return j, _index(eliminated, len(j)), _weight(judge_weight)


def percentage_region(judge_shares, eliminated, judge_weight=.5):
    """Return A,b for A@p<=b; separately impose p>=0 and sum(p)=1.

    Rows use combined-merit units, w*j_e+(1-w)*p_e <=
    w*j_k+(1-w)*p_k. The closed region includes zeros and ties.
    """
    j, e, w = _inputs(judge_shares, eliminated, judge_weight)
    survivors = np.flatnonzero(np.arange(len(j)) != e)
    a = np.zeros((len(j) - 1, len(j)))
    a[:, e] = 1 - w
    a[np.arange(len(survivors)), survivors] = -(1 - w)
    return a, w * (j[survivors] - j[e])


def analytical_coordinate_bounds(judge_shares, eliminated, judge_weight=.5):
    """Independent monotone-root/formula solution, also supporting n=2.

    For survivors s, write d_s=w/(1-w)*(j_e-j_s). Feasibility is
    sum(max(0,d_s))<=1. The eliminated upper endpoint is the unique root
    t+sum(max(0,t+d_s))=1. Survivor lower formulas require n>=3;
    with n=2 the sole survivor instead has lower endpoint 1-upper_e.
    """
    j, e, w = _inputs(judge_shares, eliminated, judge_weight)
    survivors = np.flatnonzero(np.arange(len(j)) != e)
    d = w / (1 - w) * (j[e] - j[survivors])
    minimum = np.maximum(0., d)
    excess = float(minimum.sum() - 1)
    if excess > 1e-12:
        raise InfeasibleRegionError("No simplex shares satisfy the declared observed-exit rule")
    # A tiny negative residual at zero can be rounded by arithmetic. Never
    # pass a non-bracketed interval to the independent scalar root finder.
    if excess >= 0:
        upper_e = 0.
    else:
        upper_e = brentq(lambda t: t + np.maximum(0., t + d).sum() - 1.,
                         0., 1., xtol=5e-15, rtol=1e-14)
    lower, upper = np.zeros(len(j)), np.ones(len(j))
    upper[e] = upper_e
    lower[survivors] = minimum if len(j) >= 3 else 1 - upper_e
    upper[survivors] = 1 - minimum.sum() + minimum
    return lower, upper


def _solve(c, a, b, n, *, margin=False):
    result = linprog(c, A_ub=a, b_ub=b, A_eq=[np.r_[np.ones(n), 0.] if margin else np.ones(n)],
                     b_eq=[1.], bounds=[(0., None)] * len(c), method="highs",
                     options={"primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f"Linear optimization failed (status={result.status}): {result.message}")
    return result.x


def _residual(x, a, b, n, tolerance):
    residual = max(float(np.max(a @ x - b)), abs(float(x[:n].sum()) - 1.),
                   float(np.max(-x)), 0.)
    if residual > tolerance:
        raise RuntimeError(f"Constructive LP witness violates constraints by {residual:g}")
    return residual


def _margin_system(regions, n):
    # One common margin is measured in actual combined-merit and share units.
    a = np.vstack([np.column_stack([matrix, np.ones(len(matrix))]) for matrix, _ in regions] +
                  [np.column_stack([-np.eye(n), np.ones(n)])])
    b = np.concatenate([rhs for _, rhs in regions] + [np.zeros(n)])
    return a, b


def sharp_coordinate_bounds(judge_shares, eliminated, judge_weight=.5, *, tolerance=DEFAULT_TOLERANCE):
    """Optimize each coordinate separately and return constructive endpoints.

    Separate extrema usually use different full vectors. Their Cartesian box
    is not the jointly feasible region. Positive interior_margin certifies that
    the closed endpoints are also infima/suprema over positive shares and a
    unique observed exit, by mixing each endpoint with the interior witness.
    A zero margin makes no such strict-region claim.
    """
    j, e, w = _inputs(judge_shares, eliminated, judge_weight)
    tolerance = _tolerance(tolerance)
    analytic_lower, analytic_upper = analytical_coordinate_bounds(j, e, w)
    a, b = percentage_region(j, e, w)
    n = len(j)
    lower, upper = np.empty(n), np.empty(n)
    lows, highs = np.empty((n, n)), np.empty((n, n))
    residual = 0.
    for k in range(n):
        objective = np.eye(n)[k]
        lo, hi = _solve(objective, a, b, n), _solve(-objective, a, b, n)
        if lo is None or hi is None:
            raise InfeasibleRegionError("No shares satisfy the declared observed-exit rule")
        residual = max(residual, _residual(lo, a, b, n, tolerance), _residual(hi, a, b, n, tolerance))
        lower[k], upper[k], lows[k], highs[k] = lo[k], hi[k], lo, hi
    ma, mb = _margin_system([(a, b)], n)
    interior = _solve(np.r_[np.zeros(n), -1.], ma, mb, n, margin=True)
    if interior is None:
        raise RuntimeError("Closed feasible region unexpectedly has no nonnegative-margin witness")
    residual = max(residual, _residual(interior, ma, mb, n, tolerance))
    difference = float(max(np.max(np.abs(lower - analytic_lower)), np.max(np.abs(upper - analytic_upper))))
    if difference > tolerance:
        raise RuntimeError(f"LP endpoints disagree with independent analytic bounds by {difference:g}")
    return CoordinateBounds(lower, upper, lows, highs, float(interior[-1]), interior[:n],
                            residual, difference, 2 * n + 1)


def attainable_unique_exit(judge_shares, observed_eliminated, candidate, counterfactual_weight,
                          *, observed_weight=.5, tolerance=DEFAULT_TOLERANCE,
                          strict_margin_tolerance=DEFAULT_TOLERANCE):
    """Test one candidate using a single jointly feasible positive share vector.

    Maximize common delta subject to the simplex, p_i>=delta, and strict
    observed/counterfactual elimination gaps >=delta. A margin above the
    declared numerical threshold certifies attainability. Zero-margin ties
    and positive margins below that threshold are not counted. This is set
    membership, not a posterior probability or an entire-season rollout.
    """
    j, e, w = _inputs(judge_shares, observed_eliminated, observed_weight)
    candidate = _index(candidate, len(j), "candidate")
    new_weight = _weight(counterfactual_weight, "counterfactual_weight")
    tolerance, threshold = _tolerance(tolerance), _tolerance(strict_margin_tolerance)
    a, b = _margin_system([percentage_region(j, e, w), percentage_region(j, candidate, new_weight)], len(j))
    x = _solve(np.r_[np.zeros(len(j)), -1.], a, b, len(j), margin=True)
    if x is None:
        return ExitAttainability(False, "weakly_infeasible", None, None, None)
    residual = _residual(x, a, b, len(j), tolerance)
    margin = max(0., float(x[-1]))
    status = "attainable" if margin > threshold else ("boundary_only" if margin == 0 else "below_strict_margin_tolerance")
    if status == "attainable":
        # Independently evaluate merit instead of merely rechecking A@x.
        p = x[:-1]
        old_merit, new_merit = w * j + (1 - w) * p, new_weight * j + (1 - new_weight) * p
        for merits, loser in [(old_merit, e), (new_merit, candidate)]:
            if not merits[loser] < np.min(np.delete(merits, loser)) or not (p > 0).all():
                raise RuntimeError("Candidate witness does not independently establish strict unique exits")
    return ExitAttainability(status == "attainable", status, margin, x[:-1], residual)


def read_protocol(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema_version", "candidate_seasons", "judge_weight", "counterfactual_judge_weights",
                "minimum_active", "outcome_policy", "include_final_week_eliminations",
                "exclude_withdrawal_weeks", "ordinary_exit_count", "feasibility_tolerance",
                "strict_margin_tolerance"}
    if not isinstance(cfg, dict) or set(cfg) != expected or type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete identification protocol")
    seasons = cfg["candidate_seasons"]
    if (not isinstance(seasons, list) or not seasons or any(type(s) is not int or not 3 <= s <= 27 for s in seasons)
            or seasons != sorted(set(seasons))):
        raise ValueError("candidate_seasons must be a sorted unique subset of seasons 3–27")
    if (type(cfg["minimum_active"]) is not int or cfg["minimum_active"] < 3 or
            type(cfg["ordinary_exit_count"]) is not int or cfg["ordinary_exit_count"] != 1 or
            cfg["outcome_policy"] != "label_first" or cfg["include_final_week_eliminations"] is not False or
            cfg["exclude_withdrawal_weeks"] is not True):
        raise ValueError("Study requires n>=3, label_first, one exit, nonfinal weeks and no withdrawals")
    _weight(cfg["judge_weight"])
    weights = cfg["counterfactual_judge_weights"]
    if not isinstance(weights, list) or not weights:
        raise ValueError("counterfactual_judge_weights must be a nonempty list")
    values = [_weight(w, "counterfactual_judge_weights") for w in weights]
    if values != sorted(set(values)) or cfg["judge_weight"] not in values:
        raise ValueError("Counterfactual weights must be sorted, unique and include the observed weight oracle")
    _tolerance(cfg["feasibility_tolerance"])
    _tolerance(cfg["strict_margin_tolerance"])
    return cfg


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _summary(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "min": float(a.min()), "p05": float(np.quantile(a, .05)),
            "p25": float(np.quantile(a, .25)), "median": float(np.median(a)),
            "p75": float(np.quantile(a, .75)), "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def run_study(data_path, config_path, out):
    """Run the declared fixed-roster study, refusing to overwrite any output.

    public_artifacts in status.json is an explicit aggregate-only publication
    allowlist. The runner never copies results into the tracked results tree.
    """
    data_path, config_path, out = Path(data_path), Path(config_path), Path(out)
    if out.exists():
        raise ValueError("Use a fresh output directory; identification results are never overwritten")
    if not data_path.is_file():
        raise ValueError("Supply the separately obtained COMAP CSV with --data")
    source_paths = {"dwts/identification.py": Path(__file__).resolve(),
                    "dwts/data.py": Path(__file__).with_name("data.py"),
                    "config": config_path, "data": data_path}
    sources = {name: _digest(path) for name, path in source_paths.items()}
    cfg = read_protocol(config_path)
    panel, audit = prepare(data_path, outcome_policy=cfg["outcome_policy"], include_final_week_eliminations=False)
    candidate_panel = panel[panel.season.isin(cfg["candidate_seasons"])]
    if candidate_panel.empty:
        raise ValueError("No candidate weeks in the selected seasons")
    started = datetime.now(timezone.utc).isoformat()
    out.mkdir(parents=True)
    protocol = {"schema_version": 1, "config": cfg, "config_sha256": sources["config"],
        "source_sha256": sources, "started_at": started, "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "assumption": "Additional deterministic percentage rule; lowest combined merit exits, with no save",
        "merit": "judge_weight * normalized_judge_score + (1-judge_weight) * fan_share",
        "judge_normalization": "weekly mean available judge scores divided by their sum across active contestants",
        "domain": "closed simplex p>=0, sum(p)=1; observed exit weakly below each survivor",
        "ties": "Coordinate bounds use the closed weak-inequality relaxation; no boundary tie-break convention is imposed",
        "strict_region": "A positive common margin certifies that closed bounds are infima/suprema over positive shares and a unique observed exit",
        "policy_support": "For each candidate separately, maximize one common strict merit/share margin on a single joint share vector",
        "rosters": "Observed weekly rosters fixed; no later-roster or entire-season counterfactual simulation",
        "filter_order": ["final_week", "n_active_below_minimum", "withdrawal_week", "ordinary_exit_count_not_one"],
        "interpretation": "Finite-sample, assumption-conditional feasible-set descriptions; no posterior or true-vote calibration claim",
        "regime_limit": "Seasons3–27 are a declared percentage-rule study scope; the problem statement describes the return to ranks as uncertain and assumes season28, rather than establishing the exact historical transition",
        "not": ["bounds implied by either probabilistic model", "posterior credible intervals", "actual vote recovery",
                "policy probabilities", "historical rule verification", "rank/save/multi-exit/final analysis"]}
    _write_json(out / "protocol.json", protocol)
    bounds_rows, weekly_rows, candidate_rows, policy_rows = [], [], [], []
    exclusion_flags, exclusions = Counter(), Counter()
    candidate_week_count = 0
    policy_status = Counter()
    feasibility_tolerance = cfg["feasibility_tolerance"]
    for (season, week), frame in candidate_panel.groupby(["season", "week"], sort=True):
        candidate_week_count += 1
        reasons = []
        if frame.is_final_week.any(): reasons.append("final_week")
        if len(frame) < cfg["minimum_active"]: reasons.append("n_active_below_minimum")
        if frame.withdrawal.any(): reasons.append("withdrawal_week")
        if int(frame.eliminated.sum()) != 1: reasons.append("ordinary_exit_count_not_one")
        exclusion_flags.update(reasons)
        if reasons:
            exclusions[reasons[0]] += 1
            continue
        group = frame.sort_values("contestant_id").reset_index(drop=True)
        j = group.judge_share.to_numpy()
        observed = int(np.flatnonzero(group.eliminated.to_numpy())[0])
        answer = sharp_coordinate_bounds(j, observed, cfg["judge_weight"], tolerance=feasibility_tolerance)
        widths = answer.upper - answer.lower
        weekly_rows.append({"season": int(season), "week": int(week), "n_active": len(group),
            "observed_eliminated_id": group.iloc[observed].contestant_id,
            "mean_width": float(widths.mean()), "interior_margin": answer.interior_margin,
            "strict_region_certified": answer.interior_margin > cfg["strict_margin_tolerance"],
            "max_feasibility_residual": answer.max_feasibility_residual,
            "max_analytic_endpoint_difference": answer.max_analytic_endpoint_difference,
            "lp_optimizations": answer.lp_optimizations,
            "witness_id_order": json.dumps(group.contestant_id.tolist()),
            "interior_witness": json.dumps(answer.interior_witness.tolist())})
        for k, row in group.iterrows():
            bounds_rows.append({"season": int(season), "week": int(week), "n_active": len(group),
                "contestant_id": row.contestant_id, "observed_exit": bool(row.eliminated),
                "judge_share": float(j[k]), "lower": answer.lower[k], "upper": answer.upper[k], "width": widths[k],
                "lower_witness": json.dumps(answer.lower_witnesses[k].tolist()),
                "upper_witness": json.dumps(answer.upper_witnesses[k].tolist())})
        for weight in cfg["counterfactual_judge_weights"]:
            attainable = []
            borderline = 0
            for k in range(len(group)):
                result = attainable_unique_exit(j, observed, k, weight, observed_weight=cfg["judge_weight"],
                    tolerance=feasibility_tolerance, strict_margin_tolerance=cfg["strict_margin_tolerance"])
                policy_status[result.status] += 1
                borderline += result.status == "below_strict_margin_tolerance"
                if result.attainable:
                    attainable.append(k)
                candidate_rows.append({"season": int(season), "week": int(week), "n_active": len(group),
                    "counterfactual_judge_weight": weight, "candidate_id": group.iloc[k].contestant_id,
                    "status": result.status, "attainable": result.attainable, "margin": result.margin,
                    "max_feasibility_residual": result.max_feasibility_residual,
                    "witness_id_order": json.dumps(group.contestant_id.tolist()),
                    "witness": None if result.witness is None else json.dumps(result.witness.tolist())})
            if weight == cfg["judge_weight"] and answer.interior_margin > cfg["strict_margin_tolerance"] and attainable != [observed]:
                raise RuntimeError("Unchanged-weight strict-exit oracle failed")
            policy_rows.append({"season": int(season), "week": int(week), "n_active": len(group),
                "counterfactual_judge_weight": weight, "n_attainable_exits": len(attainable),
                "observed_exit_attainable": observed in attainable,
                "changed_exit_attainable": any(k != observed for k in attainable), "borderline_candidates": borderline})
    if not weekly_rows:
        raise ValueError("No eligible weeks under the declared identification protocol")
    bounds, weekly, candidates, policy = map(pd.DataFrame, [bounds_rows, weekly_rows, candidate_rows, policy_rows])
    width_summary = [{"population": name, "weighting": "contestant_week", **_summary(values)} for name, values in
        [("all", bounds.width), ("survivor", bounds.loc[~bounds.observed_exit, "width"]),
         ("observed_exit", bounds.loc[bounds.observed_exit, "width"])]]
    width_summary += [{"population": "all_mean_width", "weighting": "equal_week", **_summary(weekly.mean_width)},
                      {"population": "all_mean_width", "weighting": "equal_season", **_summary(weekly.groupby("season").mean_width.mean())}]
    pd.DataFrame(width_summary).to_csv(out / "width_summary.csv", index=False)
    eligibility = [{"category": "population", "reason": name, "count": count} for name, count in
        [("candidate_seasons", candidate_panel.season.nunique()), ("candidate_weeks", candidate_week_count),
         ("candidate_contestant_weeks", len(candidate_panel)), ("eligible_seasons", weekly.season.nunique()),
         ("eligible_weeks", len(weekly)), ("eligible_contestant_weeks", len(bounds)),
         ("preparation_audit_rows_full_input", len(audit))]]
    for kind, counts in [("overlapping_exclusion", exclusion_flags), ("exclusive_exclusion_in_filter_order", exclusions)]:
        eligibility.extend({"category": kind, "reason": reason, "count": counts[reason]} for reason in protocol["filter_order"])
    pd.DataFrame(eligibility).to_csv(out / "eligibility_summary.csv", index=False)
    ambiguity_rows, distributions = [], []
    for weight, group in policy.groupby("counterfactual_judge_weight", sort=True):
        ambiguity_rows.append({"counterfactual_judge_weight": weight, "weeks": len(group),
            "ambiguous_weeks": int((group.n_attainable_exits > 1).sum()),
            "ambiguous_fraction": float((group.n_attainable_exits > 1).mean()),
            "singleton_weeks": int((group.n_attainable_exits == 1).sum()),
            "no_certified_candidate_weeks": int((group.n_attainable_exits == 0).sum()),
            "observed_exit_attainable_weeks": int(group.observed_exit_attainable.sum()),
            "changed_exit_attainable_weeks": int(group.changed_exit_attainable.sum()),
            "all_active_attainable_weeks": int((group.n_attainable_exits == group.n_active).sum()),
            "mean_attainable_exits": float(group.n_attainable_exits.mean()),
            "median_attainable_exits": float(group.n_attainable_exits.median()),
            "borderline_candidate_count": int(group.borderline_candidates.sum())})
        distributions.extend({"counterfactual_judge_weight": weight, "n_attainable_exits": int(count), "weeks": int(n)}
                             for count, n in group.n_attainable_exits.value_counts().sort_index().items())
    pd.DataFrame(ambiguity_rows).to_csv(out / "policy_ambiguity.csv", index=False)
    pd.DataFrame(distributions).to_csv(out / "policy_candidate_counts.csv", index=False)
    for name, table in zip(PRIVATE_ARTIFACTS, [bounds, weekly, candidates]):
        table.to_csv(out / name, index=False)
        (out / name).chmod(0o600)
    positive = candidates.loc[candidates.attainable, "margin"]
    _write_json(out / "numerical_checks.json", {"schema_version": 1,
        "coordinate_lp_optimizations": int(weekly.lp_optimizations.sum()), "coordinate_endpoint_witnesses": 2 * len(bounds),
        "policy_lp_optimizations": len(candidates), "policy_status_counts": dict(policy_status),
        "strict_policy_witnesses": int(candidates.attainable.sum()),
        "minimum_certified_policy_margin": None if positive.empty else float(positive.min()),
        "max_policy_feasibility_residual": float(candidates.max_feasibility_residual.max()),
        "max_coordinate_feasibility_residual": float(weekly.max_feasibility_residual.max()),
        "max_independent_analytic_endpoint_difference": float(weekly.max_analytic_endpoint_difference.max()),
        "strict_observed_region_certified_weeks": int(weekly.strict_region_certified.sum()),
        "minimum_observed_interior_margin": float(weekly.interior_margin.min()),
        "unchanged_weight_oracle": "Only the observed exit is attainable on every strict-certified week",
        "strict_margin_tolerance": cfg["strict_margin_tolerance"], "solver": "scipy.optimize.linprog(method=highs)",
        "independent_endpoint_check": "Analytic survivor formulas and monotone scalar root for eliminated upper bound"})
    for name, old in sources.items():
        if _digest(source_paths[name]) != old:
            raise RuntimeError(f"Study source changed during execution: {name}")
    _write_json(out / "status.json", {"schema_version": 1, "status": "completed", "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(), "source_sha256": sources,
        "eligible_weeks": len(weekly), "eligible_contestant_weeks": len(bounds),
        "public_artifacts": [*PUBLIC_ARTIFACTS, "status.json"], "private_artifacts": list(PRIVATE_ARTIFACTS),
        "artifacts_sha256": {name: _digest(out / name) for name in PUBLIC_ARTIFACTS}})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Separately obtained COMAP CSV; never published by this command")
    parser.add_argument("--config", default="configs/identification_study.json")
    parser.add_argument("--out", required=True, help="Fresh local output directory; use ignored outputs/ in a checkout")
    args = parser.parse_args(argv)
    run_study(args.data, args.config, args.out)


if __name__ == "__main__":
    main()
