"""Chronological conditional-exit and online-expert study.

Only aggregate results are suitable for publication. Individual forecasts stay
in the chosen local output directory. Historical replay is exploratory; neither
exit prediction nor a mixture regret bound validates unobserved audience votes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import __version__
from .conditional_exit import build_events, fit, nll, baseline_prob, basic_loss, predictions
from .data import prepare
from .expert_mixture import AdaptiveOnline, ForecastPacket, EXPERTS, PRIOR, select_candidate

RAW_SHA256 = "ea99caec6ea243bdb450a1971a95ba8a95701a93be7ff29f0ba3c57d72ddff52"
INTERPRETATION = (
    "Exploratory chronological replay on previously examined historical data. "
    "Current complete rosters and scores precede prediction; exits are released "
    "after prediction. Only eligible single ordinary exits update mixture weights. "
    "No independent prospective validation or observed true audience votes. "
    "Seasons 1-2 and the judge-save regime from season 28 are outside this study. "
    "Bootstrap intervals condition on fixed predictions and exclude fitting, "
    "selection and repeated historical exploration uncertainty."
)
PUBLIC_FILES = ("metrics.csv", "paired_differences.csv", "season_metrics.csv",
                "selection.json", "protocol.json", "summary.json")


def default_config():
    return dict(schema_version=1,
        library_pairs=[dict(train=list(range(3, a+1)), test=list(range(a+1, b+1)))
                       for a, b in [(5, 8), (8, 12), (12, 16), (16, 20), (20, 27)]],
        ridge_grid=[.0001, .001, .01, .1, 1.], startup_ridge=.001,
        temperature_grid=[.25, .5, .75, 1., 1.5, 2., 3., 6., 12., 24.],
        experts=list(EXPERTS), initial_weights=PRIOR.tolist(),
        share_gamma_grid=[0., .05, .2], tempered_eta_grid=[1., .5, 2.],
        tie_tolerance=1e-9, calibration_bins=5, bootstrap_draws=20000,
        bootstrap_seed=20261015, outcome_policy="label_first")


def read_config(path=None):
    cfg = default_config() if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    expected = default_config()
    if set(cfg) != set(expected) or cfg["schema_version"] != 1:
        raise ValueError("Incomplete or unknown exit-forecast configuration")
    # The published study fixes its population, ordering and candidate families.
    for key in ("library_pairs", "experts", "initial_weights", "share_gamma_grid",
                "tempered_eta_grid", "outcome_policy"):
        if cfg[key] != expected[key]:
            raise ValueError(f"Use the declared chronological study {key}")
    for key in ("ridge_grid", "temperature_grid"):
        x = cfg[key]
        if not isinstance(x, list) or not x or len(set(x)) != len(x) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                and np.isfinite(v) and v > 0 for v in x):
            raise ValueError(f"Invalid {key}")
    for key in ("startup_ridge", "tie_tolerance"):
        if not isinstance(cfg[key], (int, float)) or isinstance(cfg[key], bool) or not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    for key in ("calibration_bins", "bootstrap_draws", "bootstrap_seed"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key.endswith("seed") else 1):
            raise ValueError(f"Invalid {key}")
    return cfg


def expert_library(panel, cfg):
    events = build_events(panel)
    eligible = [e for e in events if e.eligible]
    cache, diagnostics, settings, rows = {}, [], [], []

    def fitted(seasons, ridge):
        key = tuple(seasons), float(ridge)
        if key not in cache:
            beta, check = fit([e for e in eligible if e.season in seasons], ridge, 2,
                              f"train_3_{max(seasons)}_ridge_{ridge:g}")
            cache[key] = beta
            diagnostics.append(check)
        return cache[key]

    for pair in cfg["library_pairs"]:
        train, test = pair["train"], pair["test"]
        inner = [b for b in cfg["library_pairs"] if set(b["train"]+b["test"]) <= set(train)]
        selection_losses = []
        for ridge in cfg["ridge_grid"] if inner else []:
            total, count = 0., 0
            for b in inner:
                validation = [e for e in eligible if e.season in b["test"]]
                if not validation:
                    raise ValueError("Missing eligible inner-validation events")
                total += len(validation)*nll(validation, fitted(b["train"], ridge))
                count += len(validation)
            selection_losses.append(total/count)
        # Preserve the original expert's exact-minimum ridge rule.
        ridge = cfg["ridge_grid"][int(np.argmin(selection_losses))] if inner else cfg["startup_ridge"]
        beta = fitted(train, ridge)
        training = [e for e in eligible if e.season in train]
        target = [e for e in eligible if e.season in test]
        if not target:
            raise ValueError("Missing eligible target events")
        temps = {}
        for model in ("past_judges", "current_judges"):
            losses = [np.mean([basic_loss(e, baseline_prob(e, model, t))["log_loss"]
                               for e in training]) for t in cfg["temperature_grid"]]
            temps[model] = cfg["temperature_grid"][int(np.argmin(losses))]
        settings.append(dict(train_end=max(train), train_seasons=train, test_seasons=test,
            ridge=ridge, coefficients=beta.tolist(), temperatures=temps,
            inner_block_count=len(inner), training_events=len(training)))
        for model in EXPERTS:
            r, _ = predictions(target, model, beta=beta if model == "choice_two" else None,
                temperature=temps.get(model, 0.), phase="library", fold=str(max(train)))
            rows.extend(r)
    return pd.DataFrame(rows), settings, diagnostics


def packets(frame):
    result = []
    for (fold, season, week), g in frame.groupby(["fold", "season", "week"], sort=True):
        matrices, rosters, labels = [], [], []
        for expert in EXPERTS:
            part = g[g.model.eq(expert)].sort_values("contestant_id")
            matrices.append(part.probability.to_numpy(float))
            rosters.append(tuple(part.contestant_id))
            labels.append(part.observed_exit.to_numpy(bool))
        if any(r != rosters[0] for r in rosters) or any(not np.array_equal(y, labels[0]) for y in labels):
            raise ValueError("Experts must have identical rosters and outcomes")
        if labels[0].sum() != 1:
            raise ValueError("Exactly one eligible exit is required")
        result.append((ForecastPacket("library", str(fold), int(season), int(week),
                                     rosters[0], np.stack(matrices)), int(np.flatnonzero(labels[0])[0])))
    return result


def evaluate(events, model, eta=1., gamma=0.):
    learner, previous, rows, person_rows = AdaptiveOnline(eta=eta, gamma=gamma), None, [], []
    for packet, outcome in events:
        key = packet.phase, packet.season
        if key != previous:
            learner.reset(*key)
            previous = key
        q, _ = learner.predict(packet)
        header = dict(phase=packet.phase, fold=packet.fold, season=packet.season, week=packet.week, model=model)
        y = np.zeros(len(q)); y[outcome] = 1
        rows.append(dict(**header, log_loss=float(-np.log(q[outcome])),
            brier=float(np.square(q-y).sum()), top1=int(np.argmax(q) == outcome), confidence=float(q.max())))
        person_rows.extend(dict(**header, contestant_id=identifier, probability=float(q[i]),
                                observed_exit=bool(i == outcome)) for i, identifier in enumerate(packet.ids))
        learner.release(outcome)
    return pd.DataFrame(rows), pd.DataFrame(person_rows)


def summarize(scores, bins):
    rows = []
    for (phase, model), g in scores.groupby(["phase", "model"], sort=True):
        bin_ids = np.minimum((g.confidence.to_numpy()*bins).astype(int), bins-1)
        ece = sum(len(x)/len(g)*abs(x.confidence.mean()-x.top1.mean())
                  for b in range(bins) if len(x := g.iloc[np.flatnonzero(bin_ids == b)]))
        rows.append(dict(phase=phase, model=model, n_events=len(g), n_seasons=g.season.nunique(),
            log_loss=g.log_loss.mean(), brier=g.brier.mean(), top1=g.top1.mean(), ece=ece))
    return pd.DataFrame(rows)


def paired_differences(scores, cfg):
    rows = []
    for phase, g in scores.groupby("phase", sort=True):
        comparisons = {
            "choice_two": ("past_judges", "current_judges"),
            "online_bayes": ("past_judges", "current_judges", "choice_two"),
            "selected_fixed_share": ("past_judges", "current_judges", "choice_two", "online_bayes"),
            "selected_tempered": ("past_judges", "current_judges", "choice_two", "online_bayes"),
        }
        for candidate, baselines in comparisons.items():
            left = g[g.model.eq(candidate)].set_index(["season", "week"]).sort_index()
            seasons = np.sort(left.index.get_level_values("season").unique())
            draws = np.random.default_rng(cfg["bootstrap_seed"]).integers(
                0, len(seasons), (cfg["bootstrap_draws"], len(seasons)))
            for baseline in baselines:
                right = g[g.model.eq(baseline)].set_index(["season", "week"]).sort_index()
                if not left.index.equals(right.index):
                    raise ValueError("Paired populations differ")
                for metric in ("log_loss", "brier", "top1"):
                    delta = left[metric]-right[metric]
                    sums = delta.groupby(level="season").sum().reindex(seasons).to_numpy()
                    counts = delta.groupby(level="season").size().reindex(seasons).to_numpy()
                    boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
                    low, high = np.quantile(boot, [.025, .975])
                    means = delta.groupby(level="season").mean()
                    better = means > 1e-12 if metric == "top1" else means < -1e-12
                    worse = means < -1e-12 if metric == "top1" else means > 1e-12
                    rows.append(dict(phase=phase, candidate=candidate, baseline=baseline, metric=metric,
                        difference=delta.mean(), lower=low, upper=high, n_events=len(delta),
                        n_seasons=len(seasons), draws=cfg["bootstrap_draws"],
                        improved_seasons=int(better.sum()), worse_seasons=int(worse.sum()),
                        tied_seasons=int((~better & ~worse).sum())))
    return pd.DataFrame(rows)


def run_study(panel, cfg):
    library, expert_settings, fit_checks = expert_library(panel, cfg)
    by_end = {int(end): packets(g) for end, g in library.groupby("fold", sort=True)}
    variants = [("control", 1., 0.), ("share_005", 1., .05), ("share_020", 1., .2),
                ("eta_05", .5, 0.), ("eta_2", 2., 0.)]
    validation = {name: pd.concat([evaluate(by_end[end], name, eta, gamma)[0]
                                  for end in (5, 8, 12, 16)], ignore_index=True)
                  for name, eta, gamma in variants}
    selection, all_scores, all_predictions = [], [], []
    for fold, end in enumerate((8, 12, 16, 20), 1):
        inner_ends = [int(b["train"][-1]) for b in cfg["library_pairs"]
                      if max(b["test"]) <= end]
        chosen = []
        for family, candidates in [
            ("selected_fixed_share", variants[:3]),
            ("selected_tempered", [variants[0], variants[3], variants[4]])]:
            traces = []
            for name, eta, gamma in candidates:
                part = validation[name][validation[name].fold.astype(int).isin(inner_ends)]
                traces.append(dict(eta=eta, gamma=gamma, n_events=len(part), log_loss=float(part.log_loss.mean())))
            index = select_candidate([x["log_loss"] for x in traces], cfg["tie_tolerance"])
            selection.append(dict(family=family, train_end=end, inner_train_ends=inner_ends,
                                  candidates=traces, selected=traces[index]))
            _, eta, gamma = candidates[index]
            chosen.append((family, eta, gamma))
        phase = "legacy_exploratory" if end == 20 else "nested_development"
        fold_name = "final" if end == 20 else str(fold)
        target = [(replace(p, phase=phase, fold=fold_name), y) for p, y in by_end[end]]
        for name, eta, gamma in [("online_bayes", 1., 0.), *chosen]:
            s, p = evaluate(target, name, eta, gamma)
            all_scores.append(s); all_predictions.append(p)
        expert_frame = library[library.fold.eq(str(end))].copy()
        expert_frame["phase"], expert_frame["fold"] = phase, fold_name
        all_predictions.append(expert_frame)
        for expert in EXPERTS:
            for packet, y in target:
                p = packet.probabilities[EXPERTS.index(expert)]
                observed = np.zeros(len(p)); observed[y] = 1
                all_scores.append(pd.DataFrame([dict(phase=phase, fold=fold_name, season=packet.season,
                    week=packet.week, model=expert, log_loss=-np.log(p[y]), brier=np.square(p-observed).sum(),
                    top1=int(np.argmax(p) == y), confidence=p.max())]))
    scores = pd.concat(all_scores, ignore_index=True)
    season_metrics = scores.groupby(["phase", "season", "model"], as_index=False).agg(
        n_events=("week", "size"), log_loss=("log_loss", "mean"), brier=("brier", "mean"), top1=("top1", "mean"))
    return dict(metrics=summarize(scores, cfg["calibration_bins"]), season_metrics=season_metrics,
        paired_differences=paired_differences(scores, cfg), private_predictions=pd.concat(all_predictions, ignore_index=True),
        private_event_scores=scores), dict(experts=expert_settings, online=selection), fit_checks


def synthetic_panel():
    """Invented scores and exits; no official data or inferred audience votes."""
    rows = []
    for season in range(3, 28):
        ids = [f"synthetic-{i}" for i in range(5)]
        for week in range(1, 4):
            scores = np.array([6.+i+((season+week+i) % 3)/4 for i in range(len(ids))])
            exit_index = (season+week) % min(2, len(ids))
            for i, name in enumerate(ids):
                rows.append(dict(season=season, week=week, contestant_id=name, judge_mean=scores[i],
                    eliminated=bool(i == exit_index), withdrawal=False, is_final_week=week == 3))
            ids.pop(exit_index)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--demo", action="store_true", help="Run on invented scores and exits")
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise ValueError("Choose a new output directory; existing results are never overwritten")
        if args.demo and args.data is not None or not args.demo and args.data is None:
            raise ValueError("Choose --demo or provide --data")
        cfg = read_config(args.config)
        raw_sha = None
        if args.demo:
            panel = synthetic_panel()
        else:
            raw_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
            if raw_sha != RAW_SHA256:
                raise ValueError("Official COMAP data checksum differs from the declared study")
            panel, _ = prepare(args.data, outcome_policy=cfg["outcome_policy"], include_final_week_eliminations=True)
        started = time.perf_counter()
        tables, selection, fits = run_study(panel, cfg)
        args.out.mkdir(parents=True, exist_ok=False)
        for name, table in tables.items():
            table.to_csv(args.out/f"{name}.csv", index=False)
        summary = dict(data_kind="synthetic" if args.demo else "official_COMAP_historical",
            package_version=__version__,
            project_lead="Yike Chen", interpretation=INTERPRETATION,
            raw_data_sha256=raw_sha, elapsed_seconds=time.perf_counter()-started,
            conditional_model_fits=len(fits), all_optimizer_gates_passed=True,
            source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ("exit_study.py", "conditional_exit.py", "expert_mixture.py")},
            cohorts={str(p): int(len(g[g.model.eq("online_bayes")]))
                     for p, g in tables["private_event_scores"].groupby("phase")},
            publication_allowlist=list(PUBLIC_FILES),
            result_scope="Synthetic illustration only." if args.demo else
                         "Historical findings require paired assessment and independent future validation.")
        for name, value in [("protocol", cfg), ("selection", selection), ("summary", summary)]:
            (args.out/f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print("SYNTHETIC FORECAST DEMO" if args.demo else "EXPLORATORY HISTORICAL EXIT STUDY")
        print(tables["metrics"].to_string(index=False))
        print(f"Wrote results to {args.out}; only the six allowlisted aggregate files are publishable.")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
