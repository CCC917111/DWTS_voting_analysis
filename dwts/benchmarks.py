"""Explore fan-share baselines and ablations with nested season-grouped validation.

Targets are frozen, whole-season posterior means, not observed audience votes.
The study was motivated by diagnostics on this same dataset. Its uncertainty
intervals describe internal validation and do not establish independent,
confirmatory or prospective prediction performance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .analysis import check_output_path, load_source
from .attribution import _designs, _forest
from .partner_identity import canonical_primary_partners

KEYS = ["season", "week", "contestant_id"]
BASELINES = ("training_mean", "uniform_share", "last_share", "history_mean_share",
             "last_share_normalized", "history_mean_share_normalized")
RF_MODELS = ("rf_tuned_raw", "rf_tuned_normalized", "rf_demographics_raw",
             "rf_demographics_normalized", "rf_pair_history_raw",
             "rf_pair_history_normalized", "rf_pair_history_demographics_raw",
             "rf_pair_history_demographics_normalized")
MODELS = BASELINES + RF_MODELS
INTERPRETATION = (
    "Exploratory internal validation on frozen retrospective latent fan shares. "
    "The normalization hypothesis was discovered using this same dataset; outer "
    "folds and bootstrap intervals are not independent confirmatory evidence. "
    "No observed-audience-vote accuracy or causal professional effect is measured."
)


@dataclass(frozen=True)
class Protocol:
    seed: int = 2627761
    trees: int = 300
    outer_folds: int = 5
    inner_folds: int = 3
    min_samples_leaf: int = 5
    history_grid: tuple[float, ...] = (0., .25, .5, .75, 1.)
    bootstrap_draws: int = 5000
    bootstrap_seed: int = 2627777

    def __post_init__(self):
        for name in ("seed", "trees", "outer_folds", "inner_folds", "min_samples_leaf",
                     "bootstrap_draws", "bootstrap_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name.endswith("folds") else 1):
                raise ValueError(f"Invalid {name}")
        if (not self.history_grid or tuple(sorted(set(self.history_grid))) != self.history_grid
                or not all(np.isfinite(w) and 0 <= w <= 1 for w in self.history_grid)):
            raise ValueError("history_grid must be sorted unique finite weights in [0,1]")


def read_protocol(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) - set(Protocol.__dataclass_fields__):
        raise ValueError("Unknown benchmark configuration fields")
    if "history_grid" in values:
        values["history_grid"] = tuple(values["history_grid"])
    return Protocol(**values)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_panel(rows):
    needed = set(KEYS + ["fan_mean", "judge_mean", "n_active", "age", "partner", "industry"])
    if needed - set(rows):
        raise ValueError(f"Missing benchmark columns: {sorted(needed-set(rows))}")
    if rows.empty or rows[KEYS].isna().any().any() or rows.duplicated(KEYS).any():
        raise ValueError("Benchmark rows need nonempty unique contestant-week keys")
    if not np.isfinite(rows[["fan_mean", "judge_mean", "n_active"]].to_numpy(float)).all():
        raise ValueError("Non-finite benchmark inputs")
    if not rows.fan_mean.between(0, 1).all():
        raise ValueError("Invalid fan share")
    for _, group in rows.groupby(["season", "week"]):
        if not group.n_active.eq(len(group)).all():
            raise ValueError("Every validation week must contain its complete active roster")
        if not np.isclose(group.fan_mean.sum(), 1., atol=1e-10, rtol=0):
            raise ValueError("Fan targets must sum to one within each week")


def normalize_weekly(rows, prediction):
    """Normalize predictions using complete validation rosters, never targets.

    Only keys/n_active are inspected, so changing held-out target values cannot
    alter the normalization. Different seasons with the same week stay separate.
    """
    p = np.asarray(prediction, dtype=float)
    if p.shape != (len(rows),) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Share predictions must be finite nonnegative vectors")
    work = rows[KEYS + ["n_active"]].reset_index(drop=True).copy()
    if work[KEYS].isna().any().any() or work.duplicated(KEYS).any():
        raise ValueError("Normalization needs unique contestant-week keys")
    count = work.groupby(["season", "week"]).season.transform("size")
    if not count.eq(work.n_active).all():
        raise ValueError("Normalization requires complete weekly rosters")
    work["prediction"] = p
    total = work.groupby(["season", "week"]).prediction.transform("sum").to_numpy()
    if (total <= 0).any():
        raise ValueError("Cannot normalize a zero-total prediction week")
    return p / total


def baseline_predictions(rows, train, test):
    """Fold-mean and own-past-share baselines; first appearances use 1/n.

    Past-share baselines use earlier rows of the same held-out season only,
    preserving the study's explicitly retrospective latent-target convention.
    """
    d = rows.reset_index(drop=True)
    ordered = d.sort_values(["season", "contestant_id", "week"])
    group = ordered.groupby(["season", "contestant_id"]).fan_mean
    last = group.shift().reindex(d.index)
    history = group.transform(lambda x: x.shift().expanding().mean()).reindex(d.index)
    neutral = 1. / d.n_active
    result = {
        "training_mean": np.full(len(test), d.iloc[train].fan_mean.mean()),
        "uniform_share": neutral.iloc[test].to_numpy(),
        "last_share": last.fillna(neutral).iloc[test].to_numpy(),
        "history_mean_share": history.fillna(neutral).iloc[test].to_numpy(),
    }
    for name in ["last_share", "history_mean_share"]:
        result[name+"_normalized"] = normalize_weekly(d.iloc[test], result[name])
    return result


def feature_columns(names, feature_set):
    names = np.asarray(names)
    history = np.array(["pq_" in name for name in names])
    if feature_set == "combined":
        return np.arange(len(names))
    if feature_set == "demographics":
        return np.flatnonzero(~history)
    if feature_set == "pair_history":
        return np.flatnonzero(history)
    raise ValueError(f"Unknown feature set: {feature_set}")


def _predict(rows, train, test, weight, feature_set, protocol, seed, cache):
    a, b, names = _designs(rows, train, test, weight, cache, fan_scale="relative")
    columns = feature_columns(names, feature_set)
    model = _forest("regression", protocol.trees, seed, protocol.min_samples_leaf)
    model.fit(a[:, columns], rows.fan_mean.to_numpy()[train])
    return model.predict(b[:, columns])


def select_weights(rows, outer_train, protocol, outer_fold, cache=None):
    """Choose raw and normalized weights on their respective inner-fold losses."""
    cache = {} if cache is None else cache
    training = rows.iloc[outer_train]
    if training.season.nunique() < protocol.inner_folds:
        raise ValueError("Not enough training seasons for declared inner folds")
    splits = list(GroupKFold(protocol.inner_folds).split(training, groups=training.season))
    records = []
    for weight in protocol.history_grid:
        for inner_fold, (itrain, itest) in enumerate(splits):
            train, test = outer_train[itrain], outer_train[itest]
            raw = _predict(rows, train, test, weight, "combined", protocol,
                           protocol.seed+1000*(outer_fold+1)+inner_fold, cache)
            normalized = normalize_weekly(rows.iloc[test], raw)
            actual = rows.fan_mean.to_numpy()[test]
            for method, prediction in [("raw", raw), ("normalized", normalized)]:
                records.append(dict(outer_fold=outer_fold, inner_fold=inner_fold,
                    method=method, history_weight=weight, validation_n=len(test),
                    squared_error_sum=float(np.square(actual-prediction).sum()),
                    train_seasons=",".join(map(str, sorted(rows.iloc[train].season.unique()))),
                    validation_seasons=",".join(map(str, sorted(rows.iloc[test].season.unique())))))
    trials = pd.DataFrame(records)
    selected = {}
    for method in ["raw", "normalized"]:
        totals = trials[trials.method.eq(method)].groupby("history_weight")[[
            "squared_error_sum", "validation_n"]].sum()
        # Sorted weights and idxmin define a deterministic smallest-weight tie break.
        selected[method] = float((totals.squared_error_sum/totals.validation_n).idxmin())
    return selected, trials


def scores(rows, prediction):
    y = rows.fan_mean.to_numpy()
    sums = pd.Series(np.asarray(prediction), index=rows.index).groupby(
        [rows.season, rows.week]).sum()
    return dict(n=len(rows), seasons=int(rows.season.nunique()), weeks=len(sums),
        r2=float(r2_score(y, prediction)), rmse=float(np.sqrt(mean_squared_error(y, prediction))),
        mae=float(mean_absolute_error(y, prediction)),
        weekly_prediction_sum_mae=float(sums.sub(1).abs().mean()))


def paired_season_bootstrap(rows, predictions, draws=5000, seed=2627777):
    """Resample whole seasons jointly for paired pooled RMSE/R2 differences.

    Models are not refitted. Intervals condition on this dataset, fitted OOF
    models and the exploratory study protocol; they do not account for latent
    posterior uncertainty, model-selection history or independent future data.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    seasons = np.sort(rows.season.unique())
    if len(seasons) < 2:
        raise ValueError("Bootstrap requires at least two seasons")
    models = list(predictions)
    for prediction in predictions.values():
        if np.asarray(prediction).shape != (len(rows),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid bootstrap prediction vector")
    counts = rows.groupby("season").size().reindex(seasons).to_numpy()
    y = rows.fan_mean.to_numpy()
    sums = rows.groupby("season").fan_mean.sum().reindex(seasons).to_numpy()
    squares = rows.assign(y2=y*y).groupby("season").y2.sum().reindex(seasons).to_numpy()
    errors = np.array([rows.assign(error=np.square(y-np.asarray(predictions[name]))).groupby(
        "season").error.sum().reindex(seasons).to_numpy() for name in models])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(seasons), size=(draws, len(seasons)))
    n = counts[sampled].sum(axis=1)
    sy = sums[sampled].sum(axis=1)
    sst = squares[sampled].sum(axis=1)-sy*sy/n
    if (sst <= 0).any():
        raise ValueError("Bootstrap R2 is undefined for a zero-variance target resample")
    sampled_errors = errors[:, sampled].sum(axis=2)
    boot = {"rmse": np.sqrt(sampled_errors/n), "r2": 1-sampled_errors/sst}
    point = {"rmse": np.sqrt(errors.sum(axis=1)/counts.sum()),
             "r2": 1-errors.sum(axis=1)/(squares.sum()-sums.sum()**2/counts.sum())}
    reference = "rf_tuned_raw"
    if reference not in predictions:
        raise ValueError("Bootstrap comparisons require rf_tuned_raw")
    comparisons = [(model, reference) for model in models if model != reference]
    if "rf_tuned_normalized" in models and "last_share" in models:
        comparisons.append(("rf_tuned_normalized", "last_share"))
    for baseline in ["last_share_normalized", "history_mean_share_normalized"]:
        if "rf_tuned_normalized" in models and baseline in models:
            comparisons.append(("rf_tuned_normalized", baseline))
    records = []
    for candidate, reference in comparisons:
        a, b = models.index(candidate), models.index(reference)
        for metric in ["rmse", "r2"]:
            delta = boot[metric][a]-boot[metric][b]
            lower, upper = np.quantile(delta, [.025, .975])
            records.append(dict(candidate=candidate, reference=reference, metric=metric,
                difference=float(point[metric][a]-point[metric][b]), lower_95=float(lower),
                upper_95=float(upper), resampling_unit="season", season_clusters=len(seasons),
                bootstrap_draws=draws, bootstrap_seed=seed, interval="paired_percentile",
                difference_definition="candidate_minus_reference",
                interpretation="exploratory_conditional_on_saved_oof_models_no_latent_uncertainty"))
    return pd.DataFrame(records)


def evaluate(rows, protocol=Protocol(), progress=None):
    """Fresh nested CV for all models, preserving every outer test row once."""
    d = rows.reset_index(drop=True).copy()
    validate_panel(d)
    if d.season.nunique() < protocol.outer_folds:
        raise ValueError("Not enough seasons for declared outer folds")
    splits = list(GroupKFold(protocol.outer_folds).split(d, groups=d.season))
    predictions = {name: np.full(len(d), np.nan) for name in MODELS}
    fold_id = np.full(len(d), -1, dtype=int)
    trials, selections, folds, fold_metrics = [], [], [], []
    cache = {}
    for outer_fold, (train, test) in enumerate(splits):
        if progress:
            progress(f"Outer fold {outer_fold+1}/{len(splits)}: fresh nested fan-share fits")
        selected, trial = select_weights(d, train, protocol, outer_fold, cache)
        trials.append(trial)
        fold_id[test] = outer_fold
        for name, values in baseline_predictions(d, train, test).items():
            predictions[name][test] = values
        fitted = {}

        def get_prediction(weight, feature_set):
            key = (weight, feature_set)
            if key not in fitted:
                fitted[key] = _predict(d, train, test, weight, feature_set, protocol,
                                      protocol.seed+outer_fold, cache)
            return fitted[key]

        specifications = [("rf_tuned", None, "combined"),
                          ("rf_demographics", 0., "demographics"),
                          ("rf_pair_history", 0., "pair_history"),
                          ("rf_pair_history_demographics", 0., "combined")]
        for prefix, fixed_weight, feature_set in specifications:
            for method in ["raw", "normalized"]:
                weight = selected[method] if fixed_weight is None else fixed_weight
                raw = get_prediction(weight, feature_set)
                name = prefix+"_"+method
                predictions[name][test] = (normalize_weekly(d.iloc[test], raw)
                                           if method == "normalized" else raw)
                selections.append(dict(model=name, outer_fold=outer_fold,
                    history_weight=weight, feature_set=feature_set,
                    selection="inner_season_GroupKFold_MSE" if fixed_weight is None else "fixed_ablation",
                    selection_prediction_scale=method,
                    train_seasons=",".join(map(str, sorted(d.iloc[train].season.unique()))),
                    test_seasons=",".join(map(str, sorted(d.iloc[test].season.unique())))))
        for name in MODELS:
            fold_metrics.append(dict(model=name, outer_fold=outer_fold,
                                     **scores(d.iloc[test], predictions[name][test])))
        folds.append(dict(outer_fold=outer_fold, train_n=len(train), test_n=len(test),
            train_seasons=",".join(map(str, sorted(d.iloc[train].season.unique()))),
            test_seasons=",".join(map(str, sorted(d.iloc[test].season.unique())))))
    if (fold_id < 0).any() or any(not np.isfinite(p).all() for p in predictions.values()):
        raise AssertionError("Every row must have exactly one finite out-of-fold prediction")
    metrics = pd.DataFrame([dict(model=name, **scores(d, p)) for name, p in predictions.items()])
    oof = pd.concat([d[KEYS].assign(model=name, fold=fold_id, actual=d.fan_mean,
                                  prediction=p) for name, p in predictions.items()], ignore_index=True)
    bootstrap = paired_season_bootstrap(d, predictions, protocol.bootstrap_draws, protocol.bootstrap_seed)
    return {"metrics": metrics, "fold_metrics": pd.DataFrame(fold_metrics),
            "weight_selection": pd.DataFrame(selections), "inner_trials": pd.concat(trials, ignore_index=True),
            "folds": pd.DataFrame(folds), "oof_predictions": oof, "paired_differences": bootstrap}


def study_protocol(source, config_path, protocol, source_config, rows):
    source = Path(source)
    code = [Path(__file__), Path(__file__).with_name("analysis.py"),
            Path(__file__).with_name("attribution.py"), Path(__file__).with_name("features.py"),
            Path(__file__).with_name("data.py"), Path(__file__).with_name("partner_identity.py")]
    source_files = ["votes.csv", "manifest.json", "run_config.json", "sampling_status.json"]
    return dict(study="fan_share_nested_benchmarks_v2", interpretation=INTERPRETATION,
        evaluation="pooled_contestant_week_out_of_fold", target="frozen_full_season_posterior_mean_fan_share",
        target_observed=False, prospective=False, independent_confirmation=False,
        effective_config=asdict(protocol), config_sha256=digest(config_path),
        source_sha256={name:digest(source/name) for name in source_files},
        code_sha256={"dwts/"+p.name:digest(p) for p in code},
        source_raw_data_sha256=json.loads((source/"manifest.json").read_text())["input_sha256"],
        partner_identity_mode=source_config.get("partner_identity_mode", "raw"),
        population=dict(seasons=sorted(map(int, rows.season.unique())), contestant_weeks=len(rows),
                        weeks=rows.groupby(["season", "week"]).ngroups),
        models=list(MODELS), baseline_cold_start="1/n_active; training_mean uses training folds only",
        baseline_extension="The same weekly normalization is applied to last-share and history-mean-share baselines; this comparison was also motivated by same-dataset diagnostics before this full replay",
        normalization="divide predictions by their season-week sum on complete validation rosters; no targets",
        inner_selection="separate pooled MSE for raw and normalized pipelines; smallest weight breaks ties",
        ablations={"demographics":"age and industry only",
                   "pair_history":"own-pair prior relative judge/fan history only, weight fixed zero",
                   "pair_history_demographics":"same own-pair history plus age and industry, weight fixed zero",
                   "tuned":"age, industry, partner relative histories; historical blend selected in inner folds"},
        inference="paired season-cluster percentile bootstrap of pooled scores; no refits; no latent uncertainty",
        packages={name:metadata.version(name) for name in ["numpy", "pandas", "scikit-learn"]},
        python=platform.python_version())


def run(source, config_path, out, data):
    source, config_path, out = Path(source), Path(config_path), Path(out)
    check_output_path(source, out)
    protocol = read_protocol(config_path)
    votes, _, _, source_config = load_source(source, data)
    identity_mode = source_config.get("partner_identity_mode", "raw")
    if identity_mode not in {"raw", "annotations_only", "verified_canonical_identity"}:
        raise ValueError("Unknown source partner_identity_mode")
    if identity_mode != "raw":
        votes = canonical_primary_partners(votes, include_verified_aliases=identity_mode == "verified_canonical_identity")
    validate_panel(votes)
    record = study_protocol(source, config_path, protocol, source_config, votes)
    out.mkdir(parents=True)
    write_json(out/"status.json", dict(completed=False, interpretation=INTERPRETATION))
    write_json(out/"protocol.json", record)
    started = datetime.now(timezone.utc).isoformat()
    results = evaluate(votes, protocol, progress=lambda message: print(message, flush=True))
    # Refuse completion if mutable input/code changed while fitting.
    final_record = study_protocol(source, config_path, protocol, source_config, votes)
    if record != final_record:
        raise RuntimeError("Source, configuration or code changed during the benchmark run")
    for name, frame in results.items():
        frame.to_csv(out/f"{name}.csv", index=False)
    write_json(out/"status.json", dict(completed=True, started_utc=started,
        completed_utc=datetime.now(timezone.utc).isoformat(), interpretation=INTERPRETATION,
        independent_confirmation=False, all_source_sampling_diagnostics_pass=True,
        protocol_sha256=digest(out/"protocol.json"),
        artifacts_sha256={p.name:digest(p) for p in sorted(out.glob("*.csv"))},
        row_level_outputs="oof_predictions.csv and inner_trials.csv remain local; publish aggregate summaries only"))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/benchmark_study.json"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data", default=Path("data/raw/2026_MCM_Problem_C_Data.csv"), type=Path)
    args = parser.parse_args()
    results = run(args.source, args.config, args.out, args.data)
    print(results["metrics"].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
