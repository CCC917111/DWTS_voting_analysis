"""Exploratory controlled comparison of static and dynamic latent support.

Historical scoring uses already examined data. Synthetic comparisons share
truths, scores and exits across methods; neither track observes real fan votes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.special import softmax

from .data import prepare
from .dynamic_model import DynamicPosterior
from .forward_model import exit_log_probs
from .forecasting import (aggregate_scores, digest, eligibility_summary, forecast_season,
                          weekly_scores, write_json, _ensemble)

LATENT_MODELS = ("static_bootstrap", "dynamic_bootstrap")
BASELINES = ("uniform", "judge_only", "past_survival", "past_judges")
INTERPRETATION = (
    "Exploratory extension on previously examined COMAP data, not prospective "
    "confirmation. Static and dynamic support share a normalized categorical "
    "percentage likelihood, current-score clock, initial prior grid, development "
    "candidate budget and bootstrap particle inference. The static comparator has "
    "no MH moves: it is not a rerun of the published resample-move forecast study. "
    "No historical individual fan-vote truth is observed."
)


def read_protocol(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema_version", "development_seasons", "evaluation_seasons",
        "prior_sd_grid", "temperature_grid", "dynamic_process_sd", "dynamic_drift",
        "baseline_settings", "development_particles", "particles", "particle_checks",
        "seeds", "judge_weight", "ess_fraction", "bootstrap_draws", "bootstrap_seed",
        "calibration_bins", "mc_thresholds", "synthetic", "outcome_policy",
        "update_scope", "evaluation_scope"}
    if (set(cfg) != expected or cfg["schema_version"] != 1
            or cfg["development_seasons"] != list(range(3, 21))
            or cfg["evaluation_seasons"] != list(range(21, 28))
            or cfg["outcome_policy"] != "label_first"
            or cfg["update_scope"] != "single_ordinary_exit_n_ge_3_no_observed_withdrawal"
            or cfg["evaluation_scope"] != "single_ordinary_exit_nonfinal_n_ge_3_no_observed_withdrawal"):
        raise ValueError("Use the declared chronological percentage-only protocol")
    for name in ("prior_sd_grid", "temperature_grid"):
        if (not cfg[name] or len(set(cfg[name])) != len(cfg[name])
                or not all(np.isfinite(x) and x > 0 for x in cfg[name])):
            raise ValueError(f"Invalid {name}")
    for name in ("development_particles", "particles", "bootstrap_draws", "calibration_bins"):
        if type(cfg[name]) is not int or cfg[name] < (16 if "particles" in name else 1):
            raise ValueError(f"Invalid {name}")
    if (not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"])
            or not all(type(x) is int and x >= 0 for x in cfg["seeds"])
            or not cfg["particle_checks"]
            or not all(type(x) is int and x >= 16 for x in cfg["particle_checks"])
            or max(cfg["particle_checks"]) <= cfg["particles"]
            or not np.isfinite(cfg["dynamic_process_sd"]) or cfg["dynamic_process_sd"] < 0
            or not np.isfinite(cfg["dynamic_drift"])
            or not 0 < cfg["ess_fraction"] <= 1
            or not 0 <= cfg["judge_weight"] <= 1
            or type(cfg["bootstrap_seed"]) is not int or cfg["bootstrap_seed"] < 0):
        raise ValueError("Invalid filter controls")
    # These controls are frozen from development-only selection in forecast-study-v2.
    if cfg["baseline_settings"] != dict(judge_temperature=12., history_temperature=6.,
                                        history_strength=0., past_judge_temperature=12.):
        raise ValueError("Keep the previously development-selected strong baselines fixed")
    thresholds = {"mean_absolute_probability_difference", "max_absolute_probability_difference",
                  "absolute_log_loss_difference", "repeat_log_loss_span"}
    if set(cfg["mc_thresholds"]) != thresholds or not all(
            np.isfinite(x) and x > 0 for x in cfg["mc_thresholds"].values()):
        raise ValueError("Invalid MC thresholds")
    syn = cfg["synthetic"]
    required = {"replicates_per_condition", "contestants", "exits", "prior_sd", "temperature",
        "process_sd", "drift", "shock_size", "judge_correlations", "particles",
        "higher_particles", "stability_replicates_per_condition", "seed"}
    if (set(syn) != required or not all(type(syn[x]) is int and syn[x] >= 1 for x in
            ("replicates_per_condition", "contestants", "exits", "particles", "higher_particles",
             "stability_replicates_per_condition", "seed"))
            or syn["contestants"]-syn["exits"] < 2 or syn["particles"] < 16
            or syn["higher_particles"] <= syn["particles"]
            or syn["stability_replicates_per_condition"] > syn["replicates_per_condition"]
            or not all(np.isfinite(syn[x]) and syn[x] > 0 for x in
                       ("prior_sd", "temperature", "shock_size"))
            or not np.isfinite(syn["process_sd"]) or syn["process_sd"] < 0
            or not np.isfinite(syn["drift"]) or not syn["judge_correlations"]
            or not all(np.isfinite(x) and 0 <= x < 1 for x in syn["judge_correlations"])):
        raise ValueError("Invalid synthetic protocol")
    return cfg


def forecast_latent_season(frame, selected, cfg, *, particles, seed):
    """Only current IDs/scores enter prediction; current events are read later."""
    needed = {"season", "week", "contestant_id", "judge_mean", "eliminated",
              "withdrawal", "is_final_week"}
    if (needed-set(frame) or frame.empty or frame.season.nunique() != 1
            or frame.duplicated(["week", "contestant_id"]).any()):
        raise ValueError("Need one nonempty season with unique contestant-week rows")
    known, filters, rows, diagnostics = {}, {}, [], []
    season = int(frame.season.iloc[0])
    for week, group in frame.groupby("week", sort=True):
        g = group.sort_values("contestant_id").reset_index(drop=True)
        current = g[["contestant_id", "judge_mean"]].copy()
        ids, judges = current.contestant_id.astype(str).tolist(), current.judge_mean.to_numpy(float)
        if len(ids) < 2:
            continue
        new = [x for x in ids if x not in known]
        for identifier in new:
            known[identifier] = len(known)
        indices = np.array([known[x] for x in ids], dtype=int)
        for model in LATENT_MODELS:
            if model not in filters:
                filters[model] = DynamicPosterior(len(known), **selected[model],
                    judge_weight=cfg["judge_weight"], ess_fraction=cfg["ess_fraction"],
                    particles=particles, seed=np.random.SeedSequence([seed, season]))
            elif new:
                filters[model].extend(len(new))
            filters[model].advance(indices, judges, int(week))
        predictions = {model: filters[model].predict(indices, judges) for model in LATENT_MODELS}
        # No event, final placement or next roster has entered the above filter API.
        exits = np.flatnonzero(g.eliminated.to_numpy(bool))
        update = len(g) >= 3 and len(exits) == 1 and not g.withdrawal.any()
        eligible = bool(update and not g.is_final_week.any())
        for model, prediction in predictions.items():
            for i, identifier in enumerate(ids):
                rows.append(dict(season=season, week=int(week), contestant_id=identifier,
                    model=model, probability=float(prediction[i]), observed_exit=bool(i in exits),
                    eligible=eligible, updated=bool(update)))
            if update:
                d = filters[model].observe(indices, judges, int(exits[0]))
                diagnostics.append(dict(season=season, week=int(week), model=model, **d))
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def run_population(panel, seasons, selected, cfg, particles, seed, *, baselines=True):
    rows, diagnostics = [], []
    for season in seasons:
        frame = panel[panel.season.eq(season)]
        if frame.empty:
            raise ValueError(f"Missing requested season {season}")
        r, d = forecast_latent_season(frame, selected, cfg, particles=particles, seed=seed)
        rows.append(r)
        diagnostics.append(d)
        if baselines:
            b, _ = forecast_season(frame, particles=16, seed=seed, mh_steps=0,
                                   **cfg["baseline_settings"])
            rows.append(b[b.model.isin(BASELINES)])
    return pd.concat(rows, ignore_index=True), pd.concat(diagnostics, ignore_index=True)


def choose_development(panel, cfg):
    """Identical initial-scale/temperature grids and candidate counts per model."""
    selected, records = {}, []
    for model in LATENT_MODELS:
        transition = (dict(process_sd=0., drift=0.) if model == "static_bootstrap" else
            dict(process_sd=cfg["dynamic_process_sd"], drift=cfg["dynamic_drift"]))
        for sd, temperature in itertools.product(cfg["prior_sd_grid"], cfg["temperature_grid"]):
            candidate = dict(prior_sd=sd, temperature=temperature, **transition)
            placeholders = {name: candidate for name in LATENT_MODELS}
            repeats = []
            for seed in cfg["seeds"]:
                rows, _ = run_population(panel, cfg["development_seasons"], placeholders, cfg,
                    cfg["development_particles"], seed, baselines=False)
                repeats.append(rows)
            rows = _ensemble(repeats)
            loss = float(weekly_scores(rows).query("model == @model").log_loss.mean())
            records.append(dict(model=model, **candidate, development_log_loss=loss,
                                development_repeats=len(repeats), selected=False))
            print(f"development {model} sd={sd:g} temperature={temperature:g}: {loss:.6f}", flush=True)
        options = [i for i, x in enumerate(records) if x["model"] == model]
        best = min(options, key=lambda i: records[i]["development_log_loss"])
        records[best]["selected"] = True
        selected[model] = {key: records[best][key] for key in
                           ("prior_sd", "temperature", "process_sd", "drift")}
    return selected, pd.DataFrame(records)


def paired_differences(scores, draws=5000, seed=2627857):
    """Paired season bootstrap of fixed predictions, without fitting uncertainty."""
    seasons = np.sort(scores.season.unique())
    if len(seasons) < 2:
        raise ValueError("Need at least two season clusters")
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(seasons), (draws, len(seasons)))
    records = []
    comparisons = [("dynamic_bootstrap", "static_bootstrap")]+[
        (latent, other) for latent in LATENT_MODELS for other in BASELINES
        if other in set(scores.model)]
    for left, right in comparisons:
        a = scores[scores.model.eq(left)].set_index(["season", "week"]).sort_index()
        b = scores[scores.model.eq(right)].set_index(["season", "week"]).sort_index()
        if not a.index.equals(b.index):
            raise ValueError("All comparisons require the same scored weeks")
        for metric in ("log_loss", "brier", "top1"):
            delta = a[metric]-b[metric]
            sums = delta.groupby(level="season").sum().reindex(seasons).to_numpy()
            counts = delta.groupby(level="season").size().reindex(seasons).to_numpy()
            dist = sums[sample].sum(axis=1)/counts[sample].sum(axis=1)
            lo, hi = np.quantile(dist, [.025, .975])
            records.append(dict(comparison=f"{left} minus {right}", metric=metric,
                difference=float(delta.mean()), lower=float(lo), upper=float(hi),
                bootstrap_draws=draws, n_seasons=len(seasons)))
    return pd.DataFrame(records)


def synthetic_season(condition, correlation, replicate, settings):
    """Generate one common truth/roster/exit stream, independent of fit methods."""
    if condition not in ("static", "drift", "shock"):
        raise ValueError("Unknown synthetic condition")
    condition_id = ("static", "drift", "shock").index(condition)
    rng = np.random.default_rng(np.random.SeedSequence(
        [settings["seed"], condition_id, int(round(1000*correlation)), replicate]))
    n = settings["contestants"]
    state = rng.normal(0, settings["prior_sd"], n)
    active, last_relative, result = np.arange(n), np.full(n, np.nan), []
    # Persistent judge aptitude may correlate with initial latent support; this
    # conditional-score stress test is not an exactly matched independence model.
    aptitude = correlation*state/settings["prior_sd"]+np.sqrt(1-correlation**2)*rng.normal(size=n)
    for week in range(1, settings["exits"]+1):
        judges = 7.5+2*np.tanh(aptitude[active]+rng.normal(0, .5, len(active)))
        relative = len(active)*judges/judges.sum()
        if week > 1 and condition == "drift":
            state[active] += settings["drift"]*(relative-last_relative[active])
            state[active] += rng.normal(0, settings["process_sd"], len(active))
        if week == 3 and condition == "shock":
            # Choose among the currently observed roster; no future-roster input.
            state[int(rng.choice(active))] += settings["shock_size"]
        truth = softmax(state[active])
        p = np.exp(exit_log_probs(state, active, judges, temperature=settings["temperature"]))
        outcome = int(rng.choice(len(active), p=p))
        result.append(dict(week=week, indices=active.copy(), judges=judges.copy(),
                           truth=truth.copy(), outcome=outcome))
        last_relative[active] = relative
        active = np.delete(active, outcome)
    return result


def weighted_quantile(draws, weights, probabilities=(.025, .975)):
    """Inverse weighted empirical CDF, columnwise."""
    result = []
    for values in np.asarray(draws).T:
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(weights[order])
        cumulative[-1] = 1.
        result.append([values[order[min(np.searchsorted(cumulative, np.nextafter(p, -np.inf), side="left"),
                                      len(order)-1)]] for p in probabilities])
    return np.asarray(result).T


def fit_synthetic(events, settings, *, particles, seed):
    """Same events for both methods; filtering, never future-conditioned smoothing."""
    filters = {model: DynamicPosterior(settings["contestants"],
        prior_sd=settings["prior_sd"], temperature=settings["temperature"],
        process_sd=0. if model == "static_bootstrap" else settings["process_sd"],
        drift=0. if model == "static_bootstrap" else settings["drift"],
        particles=particles, seed=seed) for model in LATENT_MODELS}
    records = []
    for event in events:
        idx, judges = event["indices"], event["judges"]
        for model, posterior in filters.items():
            posterior.advance(idx, judges, event["week"])
            prediction = posterior.predict(idx, judges)
            onehot = np.eye(len(idx))[event["outcome"]]
            prior_mean = posterior.weights @ posterior.fan_shares(idx)
            posterior.observe(idx, judges, event["outcome"])
            shares = posterior.fan_shares(idx)
            mean = posterior.weights @ shares
            lo, hi = weighted_quantile(shares, posterior.weights)
            # Pair concordance of posterior-mean and true current shares.
            pairs = np.triu_indices(len(idx), 1)
            sign = np.sign(mean[:, None]-mean[None, :])*np.sign(
                event["truth"][:, None]-event["truth"][None, :])
            records.append(dict(week=event["week"], model=model,
                log_loss=float(-np.log(prediction[event["outcome"]])),
                brier=float(np.square(prediction-onehot).sum()),
                forecast_share_mse=float(np.square(prior_mean-event["truth"]).mean()),
                filtering_share_mse=float(np.square(mean-event["truth"]).mean()),
                coverage=float(((lo <= event["truth"]) & (event["truth"] <= hi)).mean()),
                interval_width=float((hi-lo).mean()),
                rank_concordance=float(np.where(sign[pairs] > 0, 1.,
                                                 np.where(sign[pairs] == 0, .5, 0.)).mean()),
                n_contestants=len(idx), initial_ancestor_fraction=len(
                    np.unique(posterior.ancestors))/particles))
    return pd.DataFrame(records)


def run_synthetic(cfg, out):
    settings, records, checks = cfg["synthetic"], [], []
    for condition, corr in itertools.product(("static", "drift", "shock"),
                                             settings["judge_correlations"]):
        for replicate in range(settings["replicates_per_condition"]):
            events = synthetic_season(condition, corr, replicate, settings)
            seed = np.random.SeedSequence([settings["seed"]+1, replicate,
                int(corr*1000), ("static", "drift", "shock").index(condition)])
            rows = fit_synthetic(events, settings, particles=settings["particles"], seed=seed)
            rows["condition"], rows["judge_correlation"], rows["replicate"] = condition, corr, replicate
            records.append(rows)
            if replicate < settings["stability_replicates_per_condition"]:
                high = fit_synthetic(events, settings, particles=settings["higher_particles"],
                    seed=np.random.SeedSequence([settings["seed"]+1000003, replicate,
                        int(corr*1000), ("static", "drift", "shock").index(condition)]))
                for model in LATENT_MODELS:
                    a, b = rows[rows.model.eq(model)], high[high.model.eq(model)]
                    checks.append(dict(condition=condition, judge_correlation=corr,
                        replicate=replicate, model=model,
                        absolute_log_loss_difference=float(abs(a.log_loss.mean()-b.log_loss.mean())),
                        absolute_filtering_rmse_difference=float(abs(np.sqrt(a.filtering_share_mse.mean())-
                                                                     np.sqrt(b.filtering_share_mse.mean()))),
                        absolute_coverage_difference=float(abs(a.coverage.mean()-b.coverage.mean()))))
        print(f"synthetic {condition}, judge correlation={corr:g} complete", flush=True)
    rows = pd.concat(records, ignore_index=True)
    rows.to_csv(out/"private_synthetic_rows.csv", index=False)
    summaries, paired = [], []
    metrics = ("log_loss", "brier", "forecast_share_mse", "filtering_share_mse",
               "coverage", "interval_width", "rank_concordance", "initial_ancestor_fraction")
    for (condition, corr, model), group in rows.groupby(["condition", "judge_correlation", "model"]):
        record = dict(condition=condition, judge_correlation=corr, model=model,
            n_seasons=int(group.replicate.nunique()), n_weeks=len(group))
        for metric in metrics:
            name = metric.replace("_mse", "_rmse")
            record[name] = float(np.sqrt(group[metric].mean()) if metric.endswith("_mse")
                                 else group[metric].mean())
        summaries.append(record)
    for (condition, corr), group in rows.groupby(["condition", "judge_correlation"]):
        rng = np.random.default_rng(cfg["bootstrap_seed"])
        for metric in metrics[:-1]:
            # Equal simulated season lengths make season means a paired unit.
            means = group.groupby(["replicate", "model"])[metric].mean().unstack()
            delta = means.dynamic_bootstrap-means.static_bootstrap
            sample = rng.integers(0, len(delta), (cfg["bootstrap_draws"], len(delta)))
            dist = delta.to_numpy()[sample].mean(axis=1)
            lo, hi = np.quantile(dist, [.025, .975])
            paired.append(dict(condition=condition, judge_correlation=corr,
                comparison="dynamic_bootstrap minus static_bootstrap", metric=metric,
                difference=float(delta.mean()), lower=float(lo), upper=float(hi),
                n_seasons=len(delta), bootstrap_draws=cfg["bootstrap_draws"]))
    pd.DataFrame(summaries).to_csv(out/"synthetic_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(out/"synthetic_paired_differences.csv", index=False)
    pd.DataFrame(checks).to_csv(out/"synthetic_stability.csv", index=False)
    return dict(independent_seasons=settings["replicates_per_condition"]*3*len(settings["judge_correlations"]),
        filtering_intervals="95% model-conditional weighted empirical intervals after each released exit; no smoothing; coverage and other summaries weight each week equally",
        uncertainty="paired simulated-season percentile bootstrap of fixed particle estimates; no MC correction",
        score_correlation="persistent judge aptitude correlated with initial support at stated level; conditional-score stress test",
        static_truth="independent Normal initial support, no changes",
        drift_truth="same Normal initial support, declared score-change drift and Gaussian process noise",
        shock_truth="static initial support with a randomly selected active contestant receiving a positive shock at week 3",
        stability_scope="first fixed replicates in every condition, independently seeded higher-particle reruns",
        not_measured="historical fan-vote accuracy; uniform fixed-parameter coverage; accuracy of retrospective soft-constraint model")


def run_study(data, config, out):
    data, config, out = Path(data).resolve(), Path(config).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Use a fresh output directory; never overwrite a prior run")
    cfg = read_protocol(config)
    root = Path(__file__).resolve().parents[1]
    sources = [root/"dwts"/name for name in
               ("dynamic_model.py", "dynamic_study.py", "forward_model.py", "forecasting.py", "data.py")]
    frozen = dict(raw_data_sha256=digest(data), config_sha256=digest(config),
                  code_sha256={p.name: digest(p) for p in sources})
    out.mkdir(parents=True)
    started, started_clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    write_json(out/"status.json", dict(status="running", started_utc=started))
    protocol = dict(schema_version=1, interpretation=INTERPRETATION, configuration=cfg,
        integrity=frozen, python=platform.python_version(),
        packages={name: metadata.version(name) for name in ("numpy", "pandas", "scipy")},
        clock="current roster/scores -> transition -> normalized probabilities -> release exits/withdrawals -> weight update",
        transition="z_t=z_previous+drift*(n_t*judge_share_t-n_previous*judge_share_previous)+Normal(0,process_sd*sqrt(elapsed_weeks))",
        first_appearance="independent Normal prior only; no first-week transition noise, no future roster preallocation",
        transition_on_skipped_events="current scores advance state even when current released outcome is not modeled",
        inference="bootstrap particle filter, likelihood importance weights, ESS systematic resampling; no MH or extra jitter",
        static_comparator="zero-noise zero-drift bootstrap filter; no MH; finite approximation differs from published resample-move study",
        selection="same prior SD x likelihood temperature grid, candidate count and seed ensemble for both models; minimum development mean log loss of ensemble probabilities, input-order ties",
        dynamic_transition_selection="process noise and score drift fixed in configuration before fitting, not optimized on development or evaluation",
        baseline_source="development-selected settings from results/forecast_study/protocol.json at repository revision 21e6cb8",
        primary_metric="paired difference of eligible-week categorical log loss; lower is better",
        intervals="whole-season percentile bootstrap of fixed predictions; six evaluation clusters, no fitting or MC uncertainty",
        calibration="fixed equal-width top-label confidence bins; descriptive",
        limitations=["Previously examined historical data: no independent confirmation.",
            "The selected dynamic transition is one prespecified alternative, not a comprehensive dynamic-model search.",
            "Filtering states can have ancestral degeneracy; process noise is not a general rejuvenation correction.",
            "Identical first-week Normal scales do not force later prior predictive variances to match.",
            "Historical labels assume current-score availability and post-week outcome release, not externally timestamped events.",
            "No true fan votes, judge-save model, full-season causal intervention or academic priority established."],
        inspiration=dict(url="https://github.com/aidanhaya/DWTS-Fan-Vote-Estimator/blob/24942a3a77847009bce1502b4b3652da09f800e1/dynamic_model.py",
                         role="dynamic-support/score-change idea; independently implemented normalized likelihood and bootstrap filter, no copied code"))
    write_json(out/"protocol.json", protocol)
    panel, _ = prepare(data, outcome_policy="label_first", include_final_week_eliminations=True)
    eligibility_summary(panel, cfg).to_csv(out/"eligibility_summary.csv", index=False)
    selected, trials = choose_development(panel, cfg)
    trials.to_csv(out/"model_selection.csv", index=False)
    repeats, diagnostics = [], []
    for repeat, seed in enumerate(cfg["seeds"]):
        r, d = run_population(panel, cfg["evaluation_seasons"], selected, cfg, cfg["particles"], seed)
        d["particles"], d["repeat"] = cfg["particles"], repeat
        repeats.append(r)
        diagnostics.append(d)
        print(f"evaluation {cfg['particles']} particles repeat {repeat+1} complete", flush=True)
    ensemble = _ensemble(repeats)
    ensemble.to_csv(out/"private_predictions.csv", index=False)
    scores = weekly_scores(ensemble)
    metrics, calibration = aggregate_scores(scores, cfg["calibration_bins"])
    metrics.to_csv(out/"metrics.csv", index=False)
    calibration.to_csv(out/"calibration.csv", index=False)
    season_metrics = []
    for season, group in scores.groupby("season"):
        m, _ = aggregate_scores(group, cfg["calibration_bins"])
        m["season"] = int(season)
        season_metrics.append(m)
    pd.concat(season_metrics, ignore_index=True).to_csv(out/"season_metrics.csv", index=False)
    paired_differences(scores, cfg["bootstrap_draws"], cfg["bootstrap_seed"]).to_csv(
        out/"paired_differences.csv", index=False)
    checks, stability, assessment = {}, [], {}
    keys = ["season", "week", "contestant_id", "model"]
    for count in [cfg["particles"], *cfg["particle_checks"]]:
        count_rows = []
        for repeat, seed in enumerate(cfg["seeds"]):
            if count == cfg["particles"]:
                r = repeats[repeat]
            else:
                r, d = run_population(panel, cfg["evaluation_seasons"], selected, cfg,
                    count, seed+1000003)
                d["particles"], d["repeat"] = count, repeat
                diagnostics.append(d)
            count_rows.append(r)
            for model in LATENT_MODELS:
                reference = ensemble[ensemble.model.eq(model) & ensemble.eligible].set_index(keys)
                current = r[r.model.eq(model) & r.eligible].set_index(keys)
                diff = current.probability-reference.probability
                s = weekly_scores(r).query("model == @model")
                stability.append(dict(model=model, particles=count, repeat=repeat,
                    log_loss=float(s.log_loss.mean()), brier=float(s.brier.mean()),
                    mean_absolute_probability_difference=float(diff.abs().mean()),
                    max_absolute_probability_difference=float(diff.abs().max())))
        checks[count] = _ensemble(count_rows)
        print(f"particle sensitivity count={count} complete", flush=True)
    pd.DataFrame(stability).to_csv(out/"particle_stability.csv", index=False)
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    diagnostics.to_csv(out/"private_filter_diagnostics.csv", index=False)
    high = checks[max(cfg["particle_checks"])]
    for model in LATENT_MODELS:
        a = ensemble[ensemble.model.eq(model) & ensemble.eligible].set_index(keys).probability
        b = high[high.model.eq(model) & high.eligible].set_index(keys).probability
        high_loss = float(weekly_scores(high).query("model == @model").log_loss.mean())
        main_loss = float(scores.query("model == @model").log_loss.mean())
        repeat_loss = [float(weekly_scores(r).query("model == @model").log_loss.mean()) for r in repeats]
        observed = dict(mean_absolute_probability_difference=float((a-b).abs().mean()),
            max_absolute_probability_difference=float((a-b).abs().max()),
            absolute_log_loss_difference=abs(high_loss-main_loss),
            repeat_log_loss_span=max(repeat_loss)-min(repeat_loss))
        passed = {key: value <= cfg["mc_thresholds"][key] for key, value in observed.items()}
        assessment[model] = dict(observed=observed, passed=passed, all_checks_pass=all(passed.values()))
    write_json(out/"mc_assessment.json", dict(thresholds=cfg["mc_thresholds"], models=assessment,
        interpretation="Independent-seed/persistent-particle sensitivity, not proof of convergence, latent accuracy or model validity."))
    protocol["selected_models"] = selected
    protocol["evaluation_population"] = dict(weeks=int(scores.query("model == 'static_bootstrap'").shape[0]),
        represented_seasons=sorted(map(int, scores.season.unique())),
        seasons_without_eligible_weeks=sorted(set(cfg["evaluation_seasons"])-set(scores.season.unique())))
    protocol["filter_diagnostics"] = {model: dict(minimum_ess_fraction=float(g.ess_fraction.min()),
        median_ess_fraction=float(g.ess_fraction.median()), resampling_fraction=float(g.resampled.mean()),
        minimum_initial_ancestor_fraction=float(g.initial_ancestor_fraction.min()))
        for model, g in diagnostics.groupby("model")}
    protocol["synthetic"] = run_synthetic(cfg, out)
    after = dict(raw_data_sha256=digest(data), config_sha256=digest(config),
                 code_sha256={p.name: digest(p) for p in sources})
    if frozen != after:
        raise RuntimeError("Data, protocol or implementation changed during the run")
    protocol["runtime_seconds"] = time.monotonic()-started_clock
    write_json(out/"protocol.json", protocol)
    write_json(out/"status.json", dict(status="complete", started_utc=started,
        completed_utc=datetime.now(timezone.utc).isoformat(), runtime_seconds=protocol["runtime_seconds"],
        integrity=frozen, artifacts_sha256={p.name: digest(p) for p in sorted(out.iterdir())
                                           if p.name != "status.json"}))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--config", default="configs/dynamic_study.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(run_study(args.data, args.config, args.out).to_string(index=False))


if __name__ == "__main__":
    main()
