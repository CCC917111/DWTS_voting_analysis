"""No-fit certificates for Table 9 metric/target identifiability.

Reads only saved OOF predictions, panel targets, fold summaries and posterior means.
Writes independent audit artifacts; does not alter any saved modelling result.
The interval calculations allow arbitrary real-valued predictions. A non-exclusion
is mathematical feasibility, never evidence that the reported model produced it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["season", "week", "contestant_id"]
SCOPES = {
    "baseline": "report_verified",
    "zero_first": "candidate_zero_verified",
    "exclude_withdrawn": "candidate_withdrawn_verified",
}
REPORT = {"judge_mean": (0.146, 5.320), "judge_total": (0.146, 5.320), "fan_mean": (0.614, 0.072)}
HALF_UNIT = 0.0005


def implied_sd_interval(r2: float, rmse: float) -> tuple[float, float]:
    """Conservative closed interval enclosing every three-decimal rounding case."""
    return ((rmse - HALF_UNIT) / np.sqrt(1 - r2 + HALF_UNIT),
            (rmse + HALF_UNIT) / np.sqrt(1 - r2 - HALF_UNIT))


def overlaps(lo: float, hi: float, report_rmse: float) -> bool:
    return bool(lo <= report_rmse + HALF_UNIT + 1e-12 and hi >= report_rmse - HALF_UNIT - 1e-12)


def feasible_rmse_interval(n, variance, pooled_variance, r2_mode, rmse_mode, r2):
    """Exact interval over nonnegative fold MSEs and the rounded R2 interval.

    Write D=1-R2=sum(c_k*MSE_k). A mean of fold RMSEs is
    sum(b_k*sqrt(MSE_k)); its extrema at fixed D are
    sqrt(D)*min(b/sqrt(c)) and sqrt(D*sum(b*b/c)).
    For sqrt(sum(b_k*MSE_k)), use min/max(b/c) instead.
    """
    n = np.asarray(n, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if np.any(n <= 0) or np.any(variance <= 0) or pooled_variance <= 0:
        raise ValueError("All folds must be nonempty and have positive target variance")
    q = n / n.sum()
    equal = np.full(len(n), 1 / len(n))
    c = {"pooled": q / pooled_variance,
         "fold_equal": equal / variance,
         "fold_size_weighted": q / variance}[r2_mode]
    if rmse_mode == "pooled":
        low_coefficient, high_coefficient = np.min(q / c), np.max(q / c)
    else:
        b = {"fold_equal": equal, "fold_size_weighted": q}[rmse_mode]
        low_coefficient = np.min(b * b / c)
        high_coefficient = np.sum(b * b / c)
    return (float(np.sqrt((1 - r2 - HALF_UNIT) * low_coefficient)),
            float(np.sqrt((1 - r2 + HALF_UNIT) * high_coefficient)))


def natural_weights(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """A disclosed catalog of interpretable weights, not optimized against metrics."""
    season_n = d.groupby("season").actual.transform("size").to_numpy()
    week_n = d.groupby(["season", "week"]).actual.transform("size").to_numpy()
    contestant_n = d.groupby("contestant_id").actual.transform("size").to_numpy()
    season_weeks = d[["season", "week"]].drop_duplicates().groupby("season").size()
    sw = d.season.map(season_weeks).to_numpy()
    raw = {"row_equal": np.ones(len(d)), "season_equal": 1 / season_n,
           "week_equal": 1 / week_n, "contestant_season_equal": 1 / contestant_n,
           "season_then_week_equal": 1 / (sw * week_n)}
    return {name: w / w.sum() for name, w in raw.items()}


def run(out: Path) -> None:
    out = out.resolve()
    for folder in SCOPES.values():
        source = (ROOT / "outputs" / folder).resolve()
        if source == out or source in out.parents or out in source.parents:
            raise ValueError("Audit output must be separate from all saved input directories")
    tables = {name: [] for name in ["target_rounding", "mixed_aggregation_bounds", "arbitrary_fold_weight_bounds",
                                   "range_certificates", "row_weight_protocols", "saved_prediction_checks",
                                   "fold_target_statistics", "normalization_checks"]}
    inputs = {}
    for scope, folder in SCOPES.items():
        source = ROOT / "outputs" / folder
        frames = {}
        for filename in ["rf_predictions.csv", "rf_fold_metrics.csv", "panel_all_seasons.csv", "votes.csv"]:
            path = source / filename
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            frames[filename] = pd.read_csv(path)
        pred, panel, votes, saved_folds = (frames[x] for x in ["rf_predictions.csv", "panel_all_seasons.csv", "votes.csv", "rf_fold_metrics.csv"])
        assert not panel.duplicated(KEYS).any()
        assert not votes.duplicated(KEYS).any()
        for target, (r2, rmse) in REPORT.items():
            task = "A2_fan" if target == "fan_mean" else "A1_judge"
            d = pred[pred.task == task].copy()
            assert not d.duplicated(KEYS).any()
            assert d.groupby("season").fold.nunique().max() == 1
            reference = votes[KEYS + ["fan_mean"]] if target == "fan_mean" else panel[KEYS + [target]]
            d = d.merge(reference, on=KEYS, how="left", validate="one_to_one")
            assert d[target].notna().all() and len(d) == len(reference)
            saved_prediction_available = target != "judge_total"
            if saved_prediction_available:
                assert np.allclose(d.actual, d[target], atol=1e-14, rtol=0)
            d["actual"] = d[target]
            y = d.actual.to_numpy()
            variance, sd = float(np.var(y)), float(np.std(y))
            lo_sd, hi_sd = implied_sd_interval(r2, rmse)
            f = d.groupby("fold").actual.agg(n="size", mean="mean", variance=lambda x: float(np.var(x)))
            assert (f.variance > 0).all()
            for fold, row in f.iterrows():
                tables["fold_target_statistics"].append(dict(scope=scope, target=target, fold=int(fold), **row.to_dict()))
            tables["target_rounding"].append(dict(
                scope=scope, target=target, n=len(y), population_sd=sd,
                sample_sd=float(np.std(y, ddof=1)), report_r2=r2, report_rmse=rmse,
                report_r2_low=r2-HALF_UNIT, report_r2_high=r2+HALF_UNIT,
                report_rmse_low=rmse-HALF_UNIT, report_rmse_high=rmse+HALF_UNIT,
                required_population_sd_low=lo_sd, required_population_sd_high=hi_sd,
                required_absolute_affine_scale_low=lo_sd/sd, required_absolute_affine_scale_high=hi_sd/sd,
                fixed_row_pooled_pair_excluded=not (lo_sd <= sd <= hi_sd)))
            for rmode in ["pooled", "fold_equal", "fold_size_weighted"]:
                for emode in ["pooled", "fold_equal", "fold_size_weighted"]:
                    lo, hi = feasible_rmse_interval(f.n, f.variance, variance, rmode, emode, r2)
                    tables["mixed_aggregation_bounds"].append(dict(
                        scope=scope, target=target, r2_aggregation=rmode, rmse_aggregation=emode,
                        feasible_rmse_low=lo, feasible_rmse_high=hi,
                        report_rmse_low=rmse-HALF_UNIT, report_rmse_high=rmse+HALF_UNIT,
                        report_pair_excluded=not overlaps(lo, hi, rmse)))
            max_fold_var = float(f.variance.max())
            for mode in ["mean_fold_rmse", "root_mean_fold_mse"]:
                lo = 0.0 if mode == "mean_fold_rmse" else np.sqrt((1-r2-HALF_UNIT)*float(f.variance.min()))
                hi = np.sqrt((1-r2+HALF_UNIT)*max_fold_var)
                tables["arbitrary_fold_weight_bounds"].append(dict(
                    scope=scope, target=target, aggregation=mode,
                    max_fold_sd=np.sqrt(max_fold_var), bound_rmse_low=lo, bound_rmse_high=hi,
                    report_pair_excluded=not overlaps(lo, hi, rmse),
                    condition="identical nonnegative normalized fold weights for both metrics; fixed within-fold row weights"))
            range_sd = float((y.max()-y.min())/2)
            range_rmse = float(range_sd*np.sqrt(1-r2+HALF_UNIT))
            tables["range_certificates"].append(dict(
                scope=scope, target=target, minimum_target=float(y.min()), maximum_target=float(y.max()),
                max_possible_population_sd_any_nonnegative_row_weights=range_sd,
                max_possible_rmse_at_report_r2=range_rmse, min_report_rmse=rmse-HALF_UNIT,
                any_subset_or_row_weights_excluded=bool(rmse-HALF_UNIT > range_rmse),
                condition="fixed target values; pooled or common-weight fold R2 and mean RMSE / root mean MSE; all contributing R2 denominators positive"))
            for protocol, w in natural_weights(d).items():
                mean = float(np.sum(w*y)); v = float(np.sum(w*(y-mean)**2)); weighted_sd = np.sqrt(v)
                entry = dict(scope=scope, target=target, protocol=protocol, population_sd=weighted_sd,
                             rmse_at_report_r2_center=np.sqrt((1-r2)*v),
                             report_pair_excluded_for_any_predictions=not (lo_sd <= weighted_sd <= hi_sd),
                             saved_prediction_available=saved_prediction_available)
                if saved_prediction_available:
                    mse = float(np.sum(w*(y-d.prediction.to_numpy())**2))
                    entry.update(saved_prediction_r2=1-mse/v, saved_prediction_rmse=np.sqrt(mse),
                                 saved_prediction_matches_both=bool(abs(1-mse/v-r2) <= HALF_UNIT and abs(np.sqrt(mse)-rmse) <= HALF_UNIT))
                tables["row_weight_protocols"].append(entry)
            if saved_prediction_available:
                actual_folds = []
                for fold, g in d.groupby("fold"):
                    mse = float(np.mean((g.actual-g.prediction)**2)); v = float(np.var(g.actual))
                    check = saved_folds[(saved_folds.task == task) & (saved_folds.fold == fold)].iloc[0]
                    assert np.isclose(1-mse/v, check.r2, atol=1e-12, rtol=0)
                    assert np.isclose(np.sqrt(mse), check.rmse, atol=1e-12, rtol=0)
                    assert np.isclose(np.sqrt(v), check.target_sd, atol=1e-12, rtol=0)
                    actual_folds.append(1-mse/v)
                mse = float(np.mean((d.actual-d.prediction)**2))
                tables["saved_prediction_checks"].append(dict(
                    scope=scope, target=target, pooled_r2=1-mse/variance, pooled_rmse=np.sqrt(mse),
                    minimum_saved_fold_r2=min(actual_folds), maximum_saved_fold_r2=max(actual_folds),
                    any_convex_fold_r2_average_excluded=bool(max(actual_folds) < r2-HALF_UNIT or min(actual_folds) > r2+HALF_UNIT),
                    folds_and_targets_verified_against_saved_files=True))
            if target == "fan_mean":
                groups = d.groupby(["season", "week"]).actual.agg(n="size", total="sum")
                tables["normalization_checks"].append(dict(
                    scope=scope, weeks=len(groups), min_week_roster=int(groups.n.min()),
                    max_week_roster=int(groups.n.max()), distinct_roster_sizes=",".join(map(str, sorted(groups.n.unique()))),
                    maximum_absolute_week_sum_error=float(np.max(np.abs(groups.total-1))),
                    affine_map_preserving_all_week_sums_must_be_identity=bool(groups.n.nunique() > 1),
                    identity_scale_in_report_required_interval=bool(lo_sd/sd <= 1 <= hi_sd/sd)))
    for path, digest in inputs.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    manifest = dict(
        audit="metric identifiability, no fitting", inputs_sha256=inputs,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        variance_convention="population weighted variance with weights normalized to sum 1; ddof=0 for equal rows",
        rounding="conservative closed intervals centered on printed values with half-width 0.0005",
        assumptions="R2 = 1 - SSE/SST, RMSE = sqrt(MSE); common rows/units unless explicitly labeled alternative",
        artifacts={name: len(rows) for name, rows in tables.items()},
        source_outputs_modified=False, models_fitted=0,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest["artifacts"], indent=2))
    print(pd.DataFrame(tables["range_certificates"])[["scope", "target", "max_possible_rmse_at_report_r2", "any_subset_or_row_weights_excluded"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/v5_metric_identifiability")
    run(parser.parse_args().out)
