"""OLS + season-grouped RF/SHAP analysis with explicit feature provenance."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import INDUSTRIES
from .features import labels, partner_features


def pcir(importance, names):
    pro = np.array(["pq_" in n for n in names])
    numerator = float(np.asarray(importance)[pro].sum())
    denominator = float(np.asarray(importance)[~pro].sum())
    return numerator, denominator, numerator / denominator if denominator > 0 else np.nan


def ols_analysis(votes, outdir, history_weight=0.5):
    import statsmodels.api as sm
    d = partner_features(votes, history_weight=history_weight)
    rows = []
    for target in ["judge_mean", "fan_mean", "placement"]:
        if target == "placement":
            work = d.groupby(["season", "contestant_id"], as_index=False).agg(
                age=("age", "first"), pq_judge=("pq_judge", "mean"),
                industry=("industry", "first"), placement=("placement", "first"))
        else:
            work = d.copy()
        numeric = work[["age", "pq_judge"]].astype(float)
        numeric = numeric.fillna(numeric.mean()).fillna(0)
        numeric = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
        cats = pd.get_dummies(pd.Categorical(work.industry, categories=INDUSTRIES),
                              prefix="industry", dtype=float).drop(columns="industry_Acting")
        cats.index = work.index
        x = sm.add_constant(pd.concat([numeric, cats], axis=1), has_constant="add")
        y = work[target].to_numpy(float)
        y = (y - y.mean()) / (y.std() or 1)
        model = sm.OLS(y, x)
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": work.season}) if work.season.nunique() > 1 else model.fit(cov_type="HC3")
        ci = fit.conf_int()
        for name in x:
            rows.append({"target": target, "feature": name, "coefficient": fit.params[name],
                         "lower_95": ci.loc[name, 0], "upper_95": ci.loc[name, 1],
                         "pvalue": fit.pvalues[name], "r2_in_sample": fit.rsquared,
                         "n": len(work), "design_rank": np.linalg.matrix_rank(x),
                         "design_columns": x.shape[1], "covariance": fit.cov_type})
    pd.DataFrame(rows).to_csv(Path(outdir) / "ols_coefficients.csv", index=False)


def _designs(d, train, test, weight, cache, fan_scale="relative"):
    """Cache feature endpoints by split, never targets or held-out labels.

    Partner features are affine in the history weight. Endpoints avoid repeated
    history scans; preprocessing is still fitted using the training fold only.
    """
    key = (tuple(train), tuple(test), float(weight), fan_scale)
    if key in cache:
        return cache[key]
    basekey = (tuple(train), tuple(test), "endpoints", fan_scale)
    if basekey not in cache:
        training = d.iloc[train]
        cache[basekey] = [
            (partner_features(training, reference=training, history_weight=w, fan_scale=fan_scale),
             partner_features(d.iloc[test], reference=training, history_weight=w, fan_scale=fan_scale))
            for w in [0., 1.]]
    (a0, b0), (a1, b1) = cache[basekey]
    a, b = a0.copy(), b0.copy()
    cols = ["pq_judge", "pq_vote"]
    a[cols] = (1-weight)*a0[cols] + weight*a1[cols]
    b[cols] = (1-weight)*b0[cols] + weight*b1[cols]
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True)),
         ["age", "pq_judge", "pq_vote"]),
        ("industry", OneHotEncoder(categories=[INDUSTRIES], handle_unknown="ignore", sparse_output=False), ["industry"])
    ])
    result = transform.fit_transform(a), transform.transform(b), transform.get_feature_names_out()
    cache[key] = result
    return result


def _forest(kind, trees, seed, min_samples_leaf):
    common = dict(n_estimators=trees, min_samples_leaf=min_samples_leaf,
                  max_features=1.0, random_state=seed, n_jobs=1)
    return (RandomForestRegressor(**common) if kind == "regression" else
            RandomForestClassifier(**common, class_weight="balanced"))


def _probability(model, x):
    p = model.predict_proba(x)
    return p[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(x))


def select_history_weight(d, y, outer_train, kind, weights, inner_folds=3,
                          trees=300, seed=2627761, min_samples_leaf=5,
                          cache=None, fan_scale="relative"):
    """Inner season-grouped CV: outer test rows never enter feature references or loss.

    Minimize pooled squared error (MSE / Brier). Report does not publish its
    candidate grid, inner folds, or selection objective; these are explicit choices.
    """
    weights = sorted(set(float(w) for w in weights))
    if not weights or not all(np.isfinite(w) and 0 <= w <= 1 for w in weights):
        raise ValueError("history_grid must contain finite weights in [0,1]")
    cache = {} if cache is None else cache
    training = d.iloc[outer_train]
    n = min(inner_folds, training.season.nunique())
    if n < 2:
        raise ValueError("Nested history-weight selection needs two inner season groups")
    records = []
    splits = list(GroupKFold(n_splits=n).split(training, groups=training.season))
    for weight in weights:
        for fold, (itrain, itest) in enumerate(splits):
            tr, te = outer_train[itrain], outer_train[itest]
            a, b, _ = _designs(d, tr, te, weight, cache, fan_scale)
            model = _forest(kind, trees, seed+fold, min_samples_leaf)
            model.fit(a, y[tr])
            prediction = model.predict(b) if kind == "regression" else _probability(model, b)
            records.append({"inner_fold": fold, "history_weight": weight,
                            "squared_error_sum": float(np.square(y[te]-prediction).sum()),
                            "validation_n": len(te),
                            "train_seasons": ",".join(map(str, sorted(d.iloc[tr].season.unique()))),
                            "validation_seasons": ",".join(map(str, sorted(d.iloc[te].season.unique())))})
    totals = pd.DataFrame(records).groupby("history_weight")[["squared_error_sum", "validation_n"]].sum()
    score = totals.squared_error_sum / totals.validation_n
    # Deterministic tie-break: smallest weight. No reference to report metrics.
    return float(score.idxmin()), records


def rf_analysis(votes, outdir, seed=2627761, trees=300, folds=5, history_weight=0.5,
                with_shap=True, history_grid=None, inner_folds=3, min_samples_leaf=5,
                judge_target="judge_mean", fan_scale="relative", tasks_only=None):
    d = votes.reset_index(drop=True).copy()
    if d.season.nunique() < 2:
        raise ValueError("RF GroupKFold needs at least two seasons")
    if judge_target not in {"judge_mean", "judge_total"}:
        raise ValueError("judge_target must be judge_mean or judge_total")
    if trees < 1 or min_samples_leaf < 1 or folds < 2:
        raise ValueError("Invalid forest or CV configuration")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = {"A1_judge": ("regression", d[judge_target].to_numpy()),
             "A2_fan": ("regression", d.fan_mean.to_numpy())}
    for name, y in labels(d).items():
        tasks[("A3_" if name.startswith("top_half") else "A4_") + name] = ("classification", y)
    if tasks_only is not None:
        tasks = {k: tasks[k] for k in tasks_only}
    splits = list(GroupKFold(n_splits=min(folds, d.season.nunique())).split(d, groups=d.season))
    records, predictions, importance_rows, ratios, tuning, selections, fold_metrics = [], [], [], [], [], [], []
    cache = {}
    for task, (kind, y) in tasks.items():
        print(f"RF {task}: " + ("nested history-weight CV" if history_grid is not None else "fixed history weight"), flush=True)
        oof = np.full(len(d), np.nan)
        prob = np.full(len(d), np.nan)
        fold_ids = np.full(len(d), -1)
        for fold, (train, test) in enumerate(splits):
            weight = history_weight
            if history_grid is not None:
                weight, trials = select_history_weight(d, y, train, kind, history_grid,
                    inner_folds, trees, seed+1000*(fold+1), min_samples_leaf, cache, fan_scale)
                tuning.extend(dict(task=task, outer_fold=fold, **r) for r in trials)
            selections.append({"task": task, "outer_fold": fold, "history_weight": weight,
                               "selection": "inner_GroupKFold" if history_grid is not None else "fixed",
                               "criterion": "MSE" if kind == "regression" else "Brier",
                               "train_seasons": ",".join(map(str, sorted(d.iloc[train].season.unique()))),
                               "test_seasons": ",".join(map(str, sorted(d.iloc[test].season.unique())))})
            a, b, names = _designs(d, train, test, weight, cache, fan_scale)
            model = _forest(kind, trees, seed+fold, min_samples_leaf)
            model.fit(a, y[train])
            oof[test] = model.predict(b)
            fold_ids[test] = fold
            if kind == "classification":
                prob[test] = _probability(model, b)
            fm = {"task": task, "fold": fold, "n": len(test)}
            if kind == "regression":
                fm.update(r2=r2_score(y[test], oof[test]), rmse=np.sqrt(mean_squared_error(y[test], oof[test])), target_sd=np.std(y[test]))
            else:
                fm.update(f1=f1_score(y[test], oof[test], zero_division=0),
                          auc=roc_auc_score(y[test], prob[test]) if len(np.unique(y[test])) == 2 else np.nan)
            fold_metrics.append(fm)
            values = {"MDI": model.feature_importances_}
            if with_shap and kind == "regression":
                import shap
                sv = np.asarray(shap.TreeExplainer(model).shap_values(b, check_additivity=True))
                values["SHAP"] = np.abs(sv).mean(axis=0)
            for method, imp in values.items():
                numerator, denominator, ratio = pcir(imp, names)
                ratios.append({"task": task, "fold": fold, "method": method,
                               "pro_importance": numerator, "celebrity_importance": denominator,
                               "pcir": ratio})
                for name, value in zip(names, imp):
                    importance_rows.append({"task": task, "fold": fold, "method": method,
                                            "feature": name, "importance": float(value), "test_n": len(test)})
        row = {"task": task, "kind": kind, "n": len(d), "folds": len(splits),
               "validation": "Nested_GroupKFold_by_season_retrospective_latent_target" if history_grid is not None else "GroupKFold_by_season_retrospective_latent_target",
               "judge_target": judge_target, "fan_scale": fan_scale}
        if kind == "regression":
            row.update(r2=r2_score(y, oof), rmse=np.sqrt(mean_squared_error(y, oof)), target_sd=float(np.std(y)))
        else:
            row.update(f1=f1_score(y, oof, zero_division=0),
                       auc=roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan)
        records.append(row)
        for i in range(len(d)):
            predictions.append({"task": task, "fold": fold_ids[i], "season": d.season.iloc[i],
                                "week": d.week.iloc[i], "contestant_id": d.contestant_id.iloc[i],
                                "actual": y[i], "prediction": oof[i], "probability": prob[i]})
    for filename, rows in [("rf_metrics", records), ("rf_predictions", predictions),
                           ("feature_importance", importance_rows), ("pcir", ratios),
                           ("history_weight_selection", selections), ("rf_fold_metrics", fold_metrics)]:
        pd.DataFrame(rows).to_csv(out / f"{filename}.csv", index=False)
    if tuning:
        pd.DataFrame(tuning).to_csv(out / "history_weight_trials.csv", index=False)
    return pd.DataFrame(records)
