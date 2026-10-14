#!/usr/bin/env python3
"""Frozen differential nested CV for reviewed partner-identity corrections.

The default is a zero-fit plan. --execute is an explicit opt-in after review.
No report reference metric is read or used to select a weight or a branch.
Reuse requires exact design/feature-name equality, unchanged targets, matching
archived feature/model code, frozen configuration and matching package versions.
The production design, estimator, probabilities and PCIR come from attribution.py.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dwts.attribution import _designs, _forest, _probability, pcir
from dwts.data import INDUSTRIES
from dwts.features import labels
from dwts.partner_identity import canonical_primary_partners, POLICY, CANONICAL_POLICY


@dataclass(frozen=True)
class Protocol:
    seed: int = 2627761
    trees: int = 300
    folds: int = 5
    inner_folds: int = 3
    min_samples_leaf: int = 5
    weights: tuple = (0., .25, .5, .75, 1.)
    judge_target: str = "judge_mean"
    fan_scale: str = "relative"
    with_shap: bool = True


PROTOCOL = Protocol()
KEYS = ["season", "week", "contestant_id"]
ARTIFACTS = ("votes.csv", "manifest.json", "run_config.json", "execution_source.zip",
             "rf_metrics.csv", "rf_predictions.csv", "feature_importance.csv", "pcir.csv",
             "history_weight_selection.csv", "history_weight_trials.csv", "rf_fold_metrics.csv")
PROVENANCE_COLUMNS = {"partner_raw", "partner_guest_annotation", "partner_annotation_week",
                      "partner_annotation_this_week", "partner_identity_changed", "partner_id_policy"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def array_digest(value):
    a = np.asarray(value)
    header = json.dumps({"shape": a.shape, "dtype": str(a.dtype)}, sort_keys=True).encode()
    data = json.dumps(a.tolist(), ensure_ascii=False).encode() if a.dtype.kind in "OUS" else a.tobytes(order="C")
    return hashlib.sha256(header + b"\0" + data).hexdigest()


def exact_arrays(a, b):
    """Strict value, shape and dtype comparison; no numerical tolerance."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def compare_designs(baseline, canonical):
    result = {}
    for name, a, b in zip(("train", "test", "feature_names"), baseline, canonical):
        result[f"{name}_equal"] = exact_arrays(a, b)
        result[f"baseline_{name}_sha256"] = array_digest(a)
        result[f"canonical_{name}_sha256"] = array_digest(b)
        if name != "feature_names":
            result[f"{name}_changed_cells"] = int(np.count_nonzero(a != b)) if a.shape == b.shape else -1
    result["design_equal"] = all(result[f"{x}_equal"] for x in ("train", "test", "feature_names"))
    return result


def tasks_for(d, protocol=PROTOCOL):
    tasks = {"A1_judge": ("regression", d[protocol.judge_target].to_numpy()),
             "A2_fan": ("regression", d.fan_mean.to_numpy())}
    for name, y in labels(d).items():
        tasks[("A3_" if name.startswith("top_half") else "A4_") + name] = ("classification", y)
    return tasks


def seasons(d, rows):
    return ",".join(map(str, sorted(d.iloc[rows].season.unique())))


def choose_weight(trials):
    # Identical pooled loss and ascending-weight tie break to select_history_weight.
    totals = pd.DataFrame(trials).groupby("history_weight")[["squared_error_sum", "validation_n"]].sum()
    return float((totals.squared_error_sum / totals.validation_n).idxmin())


def metrics(kind, actual, prediction, probability):
    if kind == "regression":
        return dict(r2=r2_score(actual, prediction), rmse=np.sqrt(mean_squared_error(actual, prediction)),
                    target_sd=float(np.std(actual)))
    return dict(f1=f1_score(actual, prediction, zero_division=0),
                auc=roc_auc_score(actual, probability) if len(np.unique(actual)) == 2 else np.nan)


def _literal_assignment(source, variable):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Cannot verify archived {variable}")


def frozen_guards(baseline_dir, protocol=PROTOCOL):
    """Verify archived implementation/configuration/environment before any reuse."""
    baseline_dir = Path(baseline_dir)
    manifest = json.loads((baseline_dir / "manifest.json").read_text())
    config = json.loads((baseline_dir / "run_config.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Baseline analysis is not completed")
    if manifest.get("source_votes_sha256") != digest(baseline_dir / "votes.csv"):
        raise ValueError("Baseline votes do not match the completed baseline manifest")
    expected = {"rf_trees": protocol.trees, "rf_folds": protocol.folds,
                "history_grid": list(protocol.weights), "rf_inner_folds": protocol.inner_folds,
                "rf_min_samples_leaf": protocol.min_samples_leaf, "rf_judge_target": protocol.judge_target,
                "rf_fan_scale": protocol.fan_scale, "history_weight": .5}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Baseline does not use the frozen v4 forest/grid/targets configuration")
    allowed = set(expected) | {"fit", "simulation_scope"}
    if set(config) - allowed:
        raise ValueError("Unsupported baseline configuration keys")
    code = {}
    with zipfile.ZipFile(baseline_dir / "execution_source.zip") as archive:
        for relative in ("dwts/attribution.py", "dwts/features.py"):
            archived = archive.read(relative)
            current = (ROOT / relative).read_bytes()
            if current != archived:
                raise ValueError(f"Production implementation differs from baseline: {relative}")
            code[relative] = hashlib.sha256(current).hexdigest()
        archived_industries = _literal_assignment(archive.read("dwts/data.py").decode(), "INDUSTRIES")
        if INDUSTRIES != archived_industries:
            raise ValueError("Industry categories/order differ from baseline")
        code["INDUSTRIES_literal_sha256"] = json_digest(INDUSTRIES)
        archived_config = archive.read("configs/report_verification.json")
        if hashlib.sha256(archived_config).hexdigest() != manifest["config_sha256"]:
            raise ValueError("Archived configuration hash differs from baseline manifest")
        if json.loads(archived_config) != config:
            raise ValueError("Saved configuration differs from archived baseline configuration")
        cli = ast.parse(archive.read("dwts/cli.py").decode())
        analyze = next(n for n in cli.body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
        defaults = dict(zip([a.arg for a in analyze.args.args][-len(analyze.args.defaults):], analyze.args.defaults))
        if ast.literal_eval(defaults["seed"]) != protocol.seed:
            raise ValueError("Baseline analyze default seed differs")
        verification = ast.parse(archive.read("dwts/verification.py").decode())
        run = next(n for n in verification.body if isinstance(n, ast.FunctionDef) and n.name == "run_verification")
        calls = [n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "analyze"]
        if len(calls) != 1 or len(calls[0].args) != 3 or calls[0].keywords:
            raise ValueError("Cannot prove baseline verification used the frozen default seed")
    source_manifest = manifest["source_manifest"]
    if platform.python_version() != source_manifest["python"]:
        raise ValueError(f"Python version differs: expected {source_manifest['python']}, got {platform.python_version()}")
    packages = {}
    for name, expected_version in source_manifest["packages"].items():
        packages[name] = metadata.version(name)
        if packages[name] != expected_version:
            raise ValueError(f"Package version differs: {name} expected {expected_version}, got {packages[name]}")
    for required in ("numpy", "pandas", "scipy", "scikit-learn", "shap", "numba"):
        if required not in packages:
            raise ValueError(f"Baseline environment lacks recorded {required}")
    return {"protocol": asdict(protocol), "protocol_sha256": json_digest(asdict(protocol)),
            "production_code_sha256": code, "baseline_config_sha256": digest(baseline_dir / "run_config.json"),
            "baseline_artifact_sha256": {p: digest(baseline_dir / p) for p in ARTIFACTS},
            "python": platform.python_version(), "packages": packages,
            "environment_scope": "Exact recorded Python and package versions; no claim of identical machine binaries"}


def load_identity(baseline_dir, canonical_path, identity_manifest):
    # Default parser reproduces run_verification's in-memory baseline values.
    raw = pd.read_csv(Path(baseline_dir) / "votes.csv").reset_index(drop=True)
    # to_csv emits round-trip representations of that in-memory dataframe.
    canonical = pd.read_csv(canonical_path, float_precision="round_trip").reset_index(drop=True)
    provenance = json.loads(Path(identity_manifest).read_text())
    if provenance.get("source_sha256") != digest(Path(baseline_dir) / "votes.csv"):
        raise ValueError("Identity mapping was not prepared from this frozen votes file")
    if provenance.get("canonical_votes_sha256") != digest(canonical_path):
        raise ValueError("Canonical votes hash differs from identity audit manifest")
    if not set(raw).issubset(canonical) or set(canonical) - set(raw) - PROVENANCE_COLUMNS:
        raise ValueError("Canonical input has missing source columns or unreviewed extra columns")
    pd.testing.assert_frame_equal(raw.drop(columns="partner"), canonical[list(raw)].drop(columns="partner"),
                                  check_exact=True)
    if raw.duplicated(KEYS).any() or canonical.duplicated(["season", "week", "partner"]).any():
        raise ValueError("Duplicate source row keys or canonical pro-week identities")
    if not canonical.partner.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("Canonical partners must be nonempty strings")
    if not set(PROVENANCE_COLUMNS).issubset(canonical):
        raise ValueError("Missing identity provenance columns")
    if not raw.partner.equals(canonical.partner_raw):
        raise ValueError("Canonical partner_raw does not match frozen input")
    if not canonical.partner.ne(raw.partner).equals(canonical.partner_identity_changed):
        raise ValueError("Identity change flags do not match the actual partner edits")
    if provenance.get("policy") not in {POLICY, CANONICAL_POLICY}:
        raise ValueError("Identity policy is not one of the two source-reviewed branches")
    if provenance.get("module_sha256") != digest(ROOT / "dwts/partner_identity.py"):
        raise ValueError("Identity implementation differs from its audit manifest")
    reviewed = canonical_primary_partners(raw, include_verified_aliases=provenance["policy"] == CANONICAL_POLICY)
    for column in ["partner"] + sorted(PROVENANCE_COLUMNS):
        a, b = reviewed[column], canonical[column]
        if column == "partner_guest_annotation":
            a, b = a.fillna(""), b.fillna("")
        elif column == "partner_annotation_week":
            a, b = a.astype(float), b.astype(float)
        pd.testing.assert_series_equal(a, b, check_exact=True)
    # Never replace a numeric feature/target with a reserialized approximation.
    d = raw.copy()
    d["partner"] = canonical.partner
    for column in canonical:
        if column not in raw:
            d[column] = canonical[column]
    return raw, d, provenance


class Baseline:
    """Read saved results with lossless float parsing and check full coverage."""
    def __init__(self, directory, d, protocol=PROTOCOL):
        self.directory, self.protocol = Path(directory), protocol
        self.tasks = tasks_for(d, protocol)
        self.splits = list(GroupKFold(n_splits=min(protocol.folds, d.season.nunique())).split(d, groups=d.season))
        self.tables = {name: pd.read_csv(self.directory / f"{name}.csv", float_precision="round_trip") for name in (
            "history_weight_trials", "history_weight_selection", "rf_predictions", "feature_importance", "pcir",
            "rf_fold_metrics", "rf_metrics")}
        for name, frame in self.tables.items():
            if set(frame.task) != set(self.tasks):
                raise ValueError(f"Baseline {name} does not cover exactly all six tasks")
        self.trials = self._index("history_weight_trials", ["task", "outer_fold", "history_weight", "inner_fold"])
        self.selections = self._index("history_weight_selection", ["task", "outer_fold"])
        self.fold_metrics = self._index("rf_fold_metrics", ["task", "fold"])
        self._index("feature_importance", ["task", "fold", "method", "feature"])
        self._index("pcir", ["task", "fold", "method"])
        self._index("rf_predictions", ["task"] + KEYS)
        self._index("rf_metrics", ["task"])
        if len(self.selections) != len(self.tasks) * len(self.splits) or len(self.fold_metrics) != len(self.selections):
            raise ValueError("Incomplete outer fold records")
        expected_trials = 0
        self.predictions = {}
        for task, (kind, y) in self.tasks.items():
            pred = self.tables["rf_predictions"].loc[lambda x: x.task.eq(task)].reset_index(drop=True)
            pd.testing.assert_frame_equal(pred[KEYS], d[KEYS], check_exact=True)
            if not np.array_equal(pred.actual.to_numpy(), y) or not np.isfinite(pred.prediction).all():
                raise ValueError(f"Baseline prediction targets/order/values invalid: {task}")
            if kind == "classification" and not pred.probability.between(0, 1).all():
                raise ValueError("Invalid baseline classification probabilities")
            self.predictions[task] = pred
            for fold, (train, test) in enumerate(self.splits):
                if not pred.iloc[test].fold.eq(fold).all():
                    raise ValueError("Saved prediction folds differ from regenerated GroupKFold")
                saved = self.selections[(task, fold)]
                if saved["selection"] != "inner_GroupKFold" or saved["criterion"] != ("MSE" if kind == "regression" else "Brier"):
                    raise ValueError("Baseline selection protocol differs")
                if saved["train_seasons"] != seasons(d, train) or saved["test_seasons"] != seasons(d, test):
                    raise ValueError("Baseline outer split differs")
                inner = list(GroupKFold(n_splits=min(protocol.inner_folds, d.iloc[train].season.nunique())).split(
                    d.iloc[train], groups=d.iloc[train].season))
                records = []
                for weight in protocol.weights:
                    for inner_fold, (itr, ite) in enumerate(inner):
                        tr, te = train[itr], train[ite]
                        record = self.trials[(task, fold, weight, inner_fold)]
                        if (record["validation_n"] != len(te) or record["train_seasons"] != seasons(d, tr)
                                or record["validation_seasons"] != seasons(d, te)
                                or not np.isfinite(record["squared_error_sum"]) or record["squared_error_sum"] < 0):
                            raise ValueError("Baseline inner loss/split invalid")
                        records.append(record)
                expected_trials += len(records)
                if choose_weight(records) != saved["history_weight"]:
                    raise ValueError("Baseline selected weight does not minimize saved inner pooled loss")
                imp = self.subset("feature_importance", task, fold)
                ratios = self.subset("pcir", task, fold)
                expected_methods = {"MDI", "SHAP"} if kind == "regression" and protocol.with_shap else {"MDI"}
                names = ["numeric__age", "numeric__pq_judge", "numeric__pq_vote"] + [f"industry__industry_{i}" for i in INDUSTRIES]
                if set(imp.method) != expected_methods or set(ratios.method) != expected_methods or len(ratios) != len(expected_methods):
                    raise ValueError("Baseline importance/PCIR methods incomplete")
                for method in expected_methods:
                    values = imp[imp.method.eq(method)]
                    if values.feature.tolist() != names or not values.test_n.eq(len(test)).all() or not np.isfinite(values.importance).all():
                        raise ValueError("Baseline feature names/order/importance values invalid")
                    numerator, denominator, ratio = pcir(values.importance.to_numpy(), names)
                    saved_ratio = ratios[ratios.method.eq(method)].iloc[0]
                    for key, value in (("pro_importance", numerator), ("celebrity_importance", denominator), ("pcir", ratio)):
                        if not _same_number(saved_ratio[key], value):
                            raise ValueError("Baseline PCIR is inconsistent with saved importance")
                if self.fold_metrics[(task, fold)]["n"] != len(test):
                    raise ValueError("Baseline fold metric sample count differs")
                computed = metrics(kind, y[test], pred.iloc[test].prediction.to_numpy(), pred.iloc[test].probability.to_numpy())
                if any(not _same_number(self.fold_metrics[(task, fold)][key], value) for key, value in computed.items()):
                    raise ValueError("Baseline fold metrics are inconsistent with saved predictions")
            saved_metrics = self.tables["rf_metrics"].loc[lambda x: x.task.eq(task)].iloc[0]
            computed = metrics(kind, y, pred.prediction.to_numpy(), pred.probability.to_numpy())
            if any(not _same_number(saved_metrics[key], value) for key, value in computed.items()):
                raise ValueError("Baseline pooled metrics are inconsistent with saved predictions")
        if len(self.trials) != expected_trials:
            raise ValueError("Extra/missing baseline inner trials")

    def _index(self, table, columns):
        frame = self.tables[table]
        if frame.duplicated(columns).any():
            raise ValueError(f"Duplicate baseline {table} keys")
        return {tuple(row[c] for c in columns): row for row in frame.to_dict("records")}

    def subset(self, table, task, fold):
        frame = self.tables[table]
        return frame[frame.task.eq(task) & frame.fold.eq(fold)]


def _same_number(a, b):
    return a == b or (pd.isna(a) and pd.isna(b))


def build_plan(raw, canonical, baseline, protocol=PROTOCOL):
    """Evaluate every split/weight once; never instantiate or fit an estimator."""
    for task, (kind, y) in tasks_for(canonical, protocol).items():
        bkind, by = baseline.tasks[task]
        if kind != bkind or not exact_arrays(y, by):
            raise ValueError(f"Target changed: {task}")
    cache_raw, cache_new = {}, {}
    comparisons, plan, outer_candidates = {}, [], []
    for outer, (train, test) in enumerate(baseline.splits):
        training = raw.iloc[train]
        inner_splits = list(GroupKFold(n_splits=min(protocol.inner_folds, training.season.nunique())).split(
            training, groups=training.season))
        for weight in protocol.weights:
            for inner, (itr, ite) in enumerate(inner_splits):
                tr, te = train[itr], train[ite]
                a = _designs(raw, tr, te, weight, cache_raw, protocol.fan_scale)
                b = _designs(canonical, tr, te, weight, cache_new, protocol.fan_scale)
                comparison = compare_designs(a, b)
                comparisons[(outer, weight, inner)] = comparison
                for task, (kind, y) in baseline.tasks.items():
                    plan.append(dict(stage="inner", task=task, outer_fold=outer, inner_fold=inner,
                                     history_weight=weight, seed=protocol.seed + 1000 * (outer + 1) + inner,
                                     target_train_sha256=array_digest(y[tr]), target_test_sha256=array_digest(y[te]),
                                     **comparison, action="reuse_baseline" if comparison["design_equal"] else "refit"))
            a = _designs(raw, train, test, weight, cache_raw, protocol.fan_scale)
            b = _designs(canonical, train, test, weight, cache_new, protocol.fan_scale)
            comparison = compare_designs(a, b)
            comparisons[(outer, weight, None)] = comparison
            for task, (kind, y) in baseline.tasks.items():
                selected = baseline.selections[(task, outer)]["history_weight"]
                outer_candidates.append(dict(stage="outer_candidate", task=task, outer_fold=outer,
                                             history_weight=weight, baseline_selected_weight=selected,
                                             selected_weight_matches_baseline=weight == selected,
                                             reuse_eligible=weight == selected and comparison["design_equal"],
                                             seed=protocol.seed + outer, target_train_sha256=array_digest(y[train]),
                                             target_test_sha256=array_digest(y[test]), **comparison))
        print(f"Planned outer fold {outer + 1}/{len(baseline.splits)}; zero model fits", flush=True)
    plan, outer_candidates = pd.DataFrame(plan), pd.DataFrame(outer_candidates)
    bounds = []
    for (task, outer), trials in plan.groupby(["task", "outer_fold"], sort=False):
        eligible = outer_candidates.loc[lambda d: d.task.eq(task) & d.outer_fold.eq(outer)]
        unchanged = bool(trials.design_equal.all())
        weight = baseline.selections[(task, outer)]["history_weight"] if unchanged else None
        exact_reuse = bool(eligible.loc[eligible.history_weight.eq(weight), "reuse_eligible"].iloc[0]) if unchanged else None
        bounds.append(dict(task=task, outer_fold=int(outer), selection_known_without_fitting=unchanged,
                           known_selected_weight=weight, outer_refits_min=int(not exact_reuse) if unchanged else int(not eligible.reuse_eligible.any()),
                           outer_refits_max=int(not exact_reuse) if unchanged else 1))
    return {"inner": plan, "outer_candidates": outer_candidates, "outer_bounds": pd.DataFrame(bounds),
            "comparisons": comparisons, "cache_raw": cache_raw, "cache_new": cache_new}


def fold_importance(model, b, names, task, kind, fold, protocol=PROTOCOL):
    values = {"MDI": model.feature_importances_}
    if protocol.with_shap and kind == "regression":
        import shap
        sv = np.asarray(shap.TreeExplainer(model).shap_values(b, check_additivity=True))
        values["SHAP"] = np.abs(sv).mean(axis=0)
    importance, ratios = [], []
    for method, imp in values.items():
        numerator, denominator, ratio = pcir(imp, names)
        ratios.append(dict(task=task, fold=fold, method=method, pro_importance=numerator,
                           celebrity_importance=denominator, pcir=ratio))
        importance.extend(dict(task=task, fold=fold, method=method, feature=name, importance=float(value), test_n=len(b))
                          for name, value in zip(names, imp))
    return importance, ratios


def execute_plan(raw, canonical, baseline, plan, out, protocol=PROTOCOL):
    """Refit exactly changed trials and ineligible outer folds; retain all six tasks."""
    out = Path(out)
    tuning, selections, fold_rows, importance, ratios, predictions, rows, ledger = [], [], [], [], [], [], [], []
    task_targets = tasks_for(canonical, protocol)
    for task, (kind, y) in task_targets.items():
        print(f"Differential nested CV: {task}", flush=True)
        oof, probability = np.full(len(canonical), np.nan), np.full(len(canonical), np.nan)
        fold_ids = np.full(len(canonical), -1)
        for outer, (train, test) in enumerate(baseline.splits):
            training = canonical.iloc[train]
            inner_splits = list(GroupKFold(n_splits=min(protocol.inner_folds, training.season.nunique())).split(
                training, groups=training.season))
            trials = []
            for weight in protocol.weights:
                for inner, (itr, ite) in enumerate(inner_splits):
                    tr, te = train[itr], train[ite]
                    comparison = plan["comparisons"][(outer, weight, inner)]
                    record = dict(baseline.trials[(task, outer, weight, inner)])
                    seed = protocol.seed + 1000 * (outer + 1) + inner
                    action = "reuse_baseline" if comparison["design_equal"] else "refit"
                    if action == "refit":
                        a, b, _ = _designs(canonical, tr, te, weight, plan["cache_new"], protocol.fan_scale)
                        model = _forest(kind, protocol.trees, seed, protocol.min_samples_leaf)
                        model.fit(a, y[tr])
                        p = model.predict(b) if kind == "regression" else _probability(model, b)
                        record["squared_error_sum"] = float(np.square(y[te] - p).sum())
                    trials.append(record)
                    ledger.append(dict(stage="inner", task=task, outer_fold=outer, inner_fold=inner,
                                       history_weight=weight, baseline_selected_weight=baseline.selections[(task, outer)]["history_weight"],
                                       seed=seed, action=action, reason="exact_design_and_target_equality" if action == "reuse_baseline" else "design_changed",
                                       target_train_sha256=array_digest(y[tr]), target_test_sha256=array_digest(y[te]), **comparison))
            tuning.extend(trials)
            weight = choose_weight(trials)
            selection = dict(baseline.selections[(task, outer)])
            selection["history_weight"] = weight
            selections.append(selection)
            comparison = plan["comparisons"][(outer, weight, None)]
            baseline_weight = baseline.selections[(task, outer)]["history_weight"]
            reuse = weight == baseline_weight and comparison["design_equal"]
            if reuse:
                prior = baseline.predictions[task].iloc[test]
                oof[test], probability[test] = prior.prediction.to_numpy(), prior.probability.to_numpy()
                fold_rows.append(dict(baseline.fold_metrics[(task, outer)]))
                importance.extend(baseline.subset("feature_importance", task, outer).to_dict("records"))
                ratios.extend(baseline.subset("pcir", task, outer).to_dict("records"))
            else:
                a, b, names = _designs(canonical, train, test, weight, plan["cache_new"], protocol.fan_scale)
                model = _forest(kind, protocol.trees, protocol.seed + outer, protocol.min_samples_leaf)
                model.fit(a, y[train])
                oof[test] = model.predict(b)
                if kind == "classification":
                    probability[test] = _probability(model, b)
                fold_rows.append(dict(task=task, fold=outer, n=len(test), **metrics(kind, y[test], oof[test], probability[test])))
                fold_imp, fold_ratios = fold_importance(model, b, names, task, kind, outer, protocol)
                importance.extend(fold_imp)
                ratios.extend(fold_ratios)
            fold_ids[test] = outer
            ledger.append(dict(stage="outer", task=task, outer_fold=outer, inner_fold=None, history_weight=weight,
                               baseline_selected_weight=baseline_weight, selected_weight_matches_baseline=weight == baseline_weight,
                               seed=protocol.seed + outer, action="reuse_baseline" if reuse else "refit",
                               reason="exact_design_target_and_selected_weight_equality" if reuse else
                               ("selected_weight_changed" if weight != baseline_weight else "design_changed"),
                               target_train_sha256=array_digest(y[train]), target_test_sha256=array_digest(y[test]), **comparison))
            print(f"  fold {outer}: weight={weight:g}; outer {'reused' if reuse else 'refitted'}", flush=True)
        if not np.isfinite(oof).all() or np.any(fold_ids < 0):
            raise ValueError("Incomplete out-of-fold predictions")
        rows.append(dict(task=task, kind=kind, n=len(canonical), folds=len(baseline.splits),
                         validation="Nested_GroupKFold_by_season_retrospective_latent_target",
                         judge_target=protocol.judge_target, fan_scale=protocol.fan_scale,
                         **metrics(kind, y, oof, probability)))
        for i in range(len(canonical)):
            predictions.append(dict(task=task, fold=fold_ids[i], season=canonical.season.iloc[i], week=canonical.week.iloc[i],
                                    contestant_id=canonical.contestant_id.iloc[i], actual=y[i], prediction=oof[i], probability=probability[i]))
        # Progress is durable, but deliberately not a resume cache: outputs may not
        # be reused until a completed manifest and full guards prove equivalence.
        pd.DataFrame(ledger).to_csv(out / "reuse_ledger.csv", index=False)
        print("Completed task metrics: " + json.dumps(rows[-1]), flush=True)
    tables = {"rf_metrics": rows, "rf_predictions": predictions, "feature_importance": importance, "pcir": ratios,
              "history_weight_selection": selections, "rf_fold_metrics": fold_rows, "history_weight_trials": tuning}
    for name, values in tables.items():
        pd.DataFrame(values).to_csv(out / f"{name}.csv", index=False)
    return pd.DataFrame(ledger)


def run(baseline_dir, canonical_path, out, *, identity_manifest=None, execute=False):
    baseline_dir, canonical_path, out = Path(baseline_dir).resolve(), Path(canonical_path).resolve(), Path(out).resolve()
    identity_manifest = Path(identity_manifest).resolve() if identity_manifest else canonical_path.parent / "manifest.json"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Choose a new or empty output directory; prior results are never overwritten")
    source_dirs = (baseline_dir, canonical_path.parent, identity_manifest.parent)
    if any(out == source or out in source.parents or source in out.parents for source in source_dirs):
        raise ValueError("Output must be separate from frozen source and identity audit")
    guards = frozen_guards(baseline_dir)
    raw, canonical, identity = load_identity(baseline_dir, canonical_path, identity_manifest)
    baseline = Baseline(baseline_dir, raw)
    manifest = {"status": "planning", "operation": "source_backed_partner_identity_differential_nested_cv",
                "execute_requested": bool(execute), "model_fits_performed": 0, "started_utc": datetime.now(timezone.utc).isoformat(),
                "baseline_dir": str(baseline_dir), "canonical_input": str(canonical_path),
                "canonical_input_sha256": digest(canonical_path), "identity_manifest_sha256": digest(identity_manifest),
                "identity_policy": identity.get("policy"), "identity_manifest": identity,
                "runner_sha256": digest(__file__), "guards": guards,
                "target_sha256": {name: array_digest(y) for name, (_, y) in baseline.tasks.items()},
                "target_and_nonpartner_columns_exactly_unchanged": True,
                "baseline_csv_parser": "pandas default, matching archived run_verification",
                "canonical_csv_parser": "round_trip; verify all original nonpartner values, apply partner only",
                "cross_branch_fit_cache": False,
                "selection_basis": "Frozen five-weight grid, inner season GroupKFold pooled MSE/Brier; smallest weight breaks ties",
                "report_targets_used_for_selection": False,
                "limitations": ["Identity correction is independently source-backed, not recovered authors' preprocessing.",
                                "Retrospective posterior targets and pre-existing within-season information assumptions remain unchanged."]}
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / "execution_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in ("scripts/run_identity_cv.py", "tests/test_identity_cv.py", "dwts/attribution.py",
                         "dwts/features.py", "dwts/data.py", "dwts/partner_identity.py", "requirements-v4-verified.txt"):
            archive.write(ROOT / relative, relative)
    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save_manifest()
    try:
        plan = build_plan(raw, canonical, baseline)
        for name in ("inner", "outer_candidates", "outer_bounds"):
            plan[name].to_csv(out / f"plan_{name}.csv", index=False)
        summary = {"inner_trials_total": len(plan["inner"]),
                   "inner_trials_reusable": int(plan["inner"].design_equal.sum()),
                   "inner_trials_to_refit": int((~plan["inner"].design_equal).sum()),
                   "outer_folds_total": len(plan["outer_bounds"]),
                   "outer_refits_min": int(plan["outer_bounds"].outer_refits_min.sum()),
                   "outer_refits_max": int(plan["outer_bounds"].outer_refits_max.sum()),
                   "outer_selections_known_without_fitting": int(plan["outer_bounds"].selection_known_without_fitting.sum())}
        manifest["plan"] = summary
        if execute:
            if frozen_guards(baseline_dir) != guards:
                raise ValueError("Baseline inputs, configuration, code or environment changed during planning")
            if digest(canonical_path) != manifest["canonical_input_sha256"] or digest(identity_manifest) != manifest["identity_manifest_sha256"]:
                raise ValueError("Identity inputs changed during planning")
            manifest["status"] = "running"
            save_manifest()
            ledger = execute_plan(raw, canonical, baseline, plan, out)
            manifest["model_fits_performed"] = int(ledger.action.eq("refit").sum())
            manifest["reused_baseline_results"] = int(ledger.action.eq("reuse_baseline").sum())
            manifest["fits_by_stage"] = ledger[ledger.action.eq("refit")].groupby("stage").size().to_dict()
            manifest["status"] = "completed"
            canonical.to_csv(out / "votes.csv", index=False)
            (out / "run_config.json").write_bytes((baseline_dir / "run_config.json").read_bytes())
        else:
            manifest["status"] = "planned_no_models_fitted"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"}
        save_manifest()
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        # Partial work is never declared completed or made eligible for reuse.
        if (out / "reuse_ledger.csv").exists():
            partial = pd.read_csv(out / "reuse_ledger.csv")
            manifest["completed_recorded_fits"] = int(partial.action.eq("refit").sum())
        manifest["model_fits_performed"] = None if execute else 0
        save_manifest()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="outputs/report_verified")
    parser.add_argument("--canonical-votes", required=True)
    parser.add_argument("--identity-manifest", help="Defaults to manifest.json beside canonical votes")
    parser.add_argument("--out", required=True)
    parser.add_argument("--execute", action="store_true", help="Fit changed trials/folds only; omit for zero-fit plan")
    args = parser.parse_args()
    result = run(args.baseline, args.canonical_votes, args.out, identity_manifest=args.identity_manifest, execute=args.execute)
    print(json.dumps({k: result[k] for k in ("status", "model_fits_performed", "plan")}, indent=2))


if __name__ == "__main__":
    main()
