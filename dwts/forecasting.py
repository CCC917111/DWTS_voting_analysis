"""Retrospective temporal backtest of a separate normalized forward exit model.

Predictions use the current roster/scores and previously released single exits.
They never use whole-season posterior fan means, placement or result-label
features. The COMAP panel is retrospective: event labels and withdrawals are
assumed released at each week's end, and schedule-conditioned eligibility is
determined after predicting. This is not a pristine prospective holdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.special import softmax

from .data import prepare
from .forward_model import SequentialPosterior

MODELS = ("forward_bayes", "uniform", "judge_only", "past_survival", "past_judges")
KEYS = ["season", "week", "contestant_id", "model"]
INTERPRETATION = (
    "Retrospective temporal backtest on previously examined COMAP data, not "
    "independent confirmatory or prospective evidence. Development seasons 3-20 "
    "select a fixed model; seasons 21-27 evaluate chronological predictions. "
    "Seasons 1-2 use a different rule and 28-34 have an unmodeled judge-save "
    "mechanism, so neither enters this percentage-only backtest. The categorical "
    "forward model is distinct from retrospective soft-constraint fan inference. "
    "Weekly scores, active rosters, single-exit schedule and reconstructed "
    "post-week event-release times are assumed known at their declared clock."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_protocol(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "development_seasons", "evaluation_seasons",
                "prior_sd_grid", "temperature_grid", "judge_temperature_grid",
                "history_strength_grid", "development_particles", "particles",
                "particle_checks", "seeds", "judge_weight", "mh_steps",
                "proposal_scale", "ess_fraction", "bootstrap_draws", "bootstrap_seed",
                "calibration_bins", "mc_thresholds", "outcome_policy", "update_scope", "evaluation_scope"}
    if set(cfg) != required or cfg["schema_version"] != 1:
        raise ValueError("Incomplete or unknown forecast protocol")
    if (cfg["development_seasons"] != list(range(3, 21))
            or cfg["evaluation_seasons"] != list(range(21, 28))
            or cfg["outcome_policy"] != "label_first"
            or cfg["update_scope"] != "single_ordinary_exit_n_ge_3_no_observed_withdrawal"
            or cfg["evaluation_scope"] != "single_ordinary_exit_nonfinal_n_ge_3_no_observed_withdrawal"):
        raise ValueError("Use the declared chronological percentage-only population")
    for name in ("prior_sd_grid", "temperature_grid", "judge_temperature_grid"):
        if (not cfg[name] or len(set(cfg[name])) != len(cfg[name])
                or not all(np.isfinite(x) and x > 0 for x in cfg[name])):
            raise ValueError(f"Invalid {name}")
    if (not cfg["history_strength_grid"] or
            not all(np.isfinite(x) and x >= 0 for x in cfg["history_strength_grid"])):
        raise ValueError("Invalid history strengths")
    for name in ("development_particles", "particles", "mh_steps", "bootstrap_draws",
                 "bootstrap_seed", "calibration_bins"):
        if type(cfg[name]) is not int or cfg[name] < (16 if "particles" in name else 1):
            raise ValueError(f"Invalid {name}")
    if (not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"])
            or not all(type(s) is int and s >= 0 for s in cfg["seeds"])
            or not cfg["particle_checks"]
            or not all(type(p) is int and p >= 16 for p in cfg["particle_checks"])
            or max(cfg["particle_checks"]) <= cfg["particles"]
            or not np.isfinite(cfg["judge_weight"]) or not 0 <= cfg["judge_weight"] <= 1
            or not np.isfinite(cfg["proposal_scale"]) or cfg["proposal_scale"] <= 0
            or not 0 < cfg["ess_fraction"] <= 1):
        raise ValueError("Invalid SMC controls")
    if (set(cfg["mc_thresholds"]) != {"mean_absolute_probability_difference",
            "max_absolute_probability_difference", "absolute_log_loss_difference", "repeat_log_loss_span"}
            or not all(np.isfinite(x) and x > 0 for x in cfg["mc_thresholds"].values())):
        raise ValueError("Invalid Monte Carlo adequacy thresholds")
    return cfg


def _judge_hazard(scores, temperature):
    j = np.asarray(scores, float)
    if (j.ndim != 1 or len(j) < 2 or not np.isfinite(j).all()
            or (j < 0).any() or j.sum() <= 0):
        raise ValueError("Invalid current judges' scores")
    return softmax(-temperature*len(j)*j/j.sum())


def forecast_season(frame, *, prior_sd=1., temperature=3., particles=2048,
                    seed=2627761, judge_temperature=3., history_temperature=3.,
                    history_strength=1., past_judge_temperature=3., judge_weight=.5, mh_steps=4,
                    proposal_scale=.35, ess_fraction=.7):
    """Predict, release current events, then update; no future roster preload.

    Only the two current-roster columns are exposed to prediction. Released
    ``eliminated`` and ``withdrawal`` flags are read afterwards. ``is_final_week``
    affects retrospective scoring only, and never changes model updates.
    """
    needed = {"season", "week", "contestant_id", "judge_mean", "eliminated",
              "withdrawal", "is_final_week"}
    if (needed-set(frame) or frame.empty or frame.season.nunique() != 1
            or frame.duplicated(["week", "contestant_id"]).any()):
        raise ValueError("Need one nonempty season with unique contestant-week rows")
    known, support, score_history, posterior, records, diagnostics = {}, {}, {}, None, [], []
    season = int(frame.season.iloc[0])
    for week, group in frame.groupby("week", sort=True):
        g = group.sort_values("contestant_id").reset_index(drop=True)
        # This adapter deliberately drops every outcome and future metadata field.
        current = g[["contestant_id", "judge_mean"]].copy()
        ids = current.contestant_id.astype(str).tolist()
        scores = current.judge_mean.to_numpy(float)
        if len(ids) < 2:
            continue
        new = [identifier for identifier in ids if identifier not in known]
        for identifier in new:
            known[identifier] = len(known)
            support[identifier] = 0.
            score_history[identifier] = []
        if posterior is None:
            posterior = SequentialPosterior(len(known), prior_sd=prior_sd,
                temperature=temperature, particles=particles,
                seed=np.random.SeedSequence([seed, season]), judge_weight=judge_weight,
                mh_steps=mh_steps, proposal_scale=proposal_scale, ess_fraction=ess_fraction)
        elif new:
            posterior.extend(len(new))
        indices = np.array([known[x] for x in ids], dtype=int)
        history_judge = _judge_hazard(scores, history_temperature)
        relative_judges = len(g)*scores/scores.sum()
        prior_judges = np.array([np.mean(score_history[x]) if score_history[x]
                                else relative_judges[i] for i, x in enumerate(ids)])
        probabilities = {
            "forward_bayes": posterior.predict(indices, scores),
            "uniform": np.full(len(g), 1/len(g)),
            "judge_only": _judge_hazard(scores, judge_temperature),
            "past_survival": softmax(np.log(history_judge)-history_strength*
                                      np.array([support[x] for x in ids])),
            "past_judges": softmax(-past_judge_temperature*prior_judges),
        }
        # The observation is released only after every model made its prediction.
        exits = np.flatnonzero(g.eliminated.to_numpy(bool))
        update = len(g) >= 3 and len(exits) == 1 and not g.withdrawal.any()
        eligible = bool(update and not g.is_final_week.any())
        for model, prediction in probabilities.items():
            if (not np.isfinite(prediction).all() or (prediction <= 0).any()
                    or not np.isclose(prediction.sum(), 1, atol=1e-12, rtol=0)):
                raise RuntimeError("Invalid normalized categorical prediction")
            for i, identifier in enumerate(ids):
                records.append(dict(season=season, week=int(week), contestant_id=identifier,
                    model=model, probability=float(prediction[i]),
                    observed_exit=bool(i in exits), eligible=eligible, updated=bool(update)))
        if update:
            d = posterior.observe(indices, scores, int(exits[0]))
            diagnostics.append(dict(season=season, week=int(week), **d))
            for i, identifier in enumerate(ids):
                if i != exits[0]:
                    support[identifier] += float(history_judge[i])
        # Known scores remain available even when the released event schedule
        # is outside the single-exit categorical observation model.
        for i, identifier in enumerate(ids):
            score_history[identifier].append(float(relative_judges[i]))
    return pd.DataFrame(records), pd.DataFrame(diagnostics)


def _settings(cfg):
    return {name: cfg[name] for name in ("judge_weight", "mh_steps", "proposal_scale", "ess_fraction")}


def run_population(panel, seasons, cfg, selected, particles, seed):
    rows, diagnostics = [], []
    for season in seasons:
        current = panel[panel.season.eq(season)]
        if current.empty:
            raise ValueError(f"Missing requested season {season}")
        r, d = forecast_season(current, **selected, **_settings(cfg), particles=particles, seed=seed)
        rows.append(r)
        diagnostics.append(d)
    nonempty = [d for d in diagnostics if not d.empty]
    return pd.concat(rows, ignore_index=True), (pd.concat(nonempty, ignore_index=True)
                                               if nonempty else pd.DataFrame())


def eligibility_summary(panel, cfg):
    """Publish aggregate schedule coverage without any contestant-level records."""
    records = []
    for phase, seasons in [("development", cfg["development_seasons"]),
                            ("evaluation", cfg["evaluation_seasons"])]:
        for season in seasons:
            weeks = list(panel[panel.season.eq(season)].groupby("week"))
            eligible = updates = multi = withdrawal = final = 0
            for _, g in weeks:
                final += int(g.is_final_week.any())
                withdrawal += int(g.withdrawal.any())
                multi += int(g.eliminated.sum() > 1)
                update = len(g) >= 3 and g.eliminated.sum() == 1 and not g.withdrawal.any()
                updates += int(update)
                eligible += int(update and not g.is_final_week.any())
            records.append(dict(phase=phase, season=season, observed_roster_weeks=len(weeks),
                eligible_evaluation_weeks=eligible, modeled_update_weeks=updates,
                multiple_ordinary_exit_weeks=multi, withdrawal_weeks=withdrawal,
                final_weeks=final))
    return pd.DataFrame(records)


def weekly_scores(predictions):
    records = []
    for (season, week, model), group in predictions[predictions.eligible].groupby(
            ["season", "week", "model"], sort=True):
        p = group.probability.to_numpy(float)
        y = group.observed_exit.to_numpy(bool)
        if y.sum() != 1 or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
            raise ValueError("Scoring needs one observed exit and a normalized complete roster")
        # Lexical ID order is deterministic for ties, including uniform baseline.
        correct = int(y[np.argmax(p)])
        records.append(dict(season=int(season), week=int(week), model=model,
                            log_loss=float(-np.log(p[y][0])),
                            brier=float(np.square(p-y).sum()), top1=correct,
                            confidence=float(p.max())))
    return pd.DataFrame(records)


def aggregate_scores(scores, bins=5, phase="evaluation"):
    tables, calibration = [], []
    for model, group in scores.groupby("model", sort=True):
        bin_id = np.minimum((group.confidence.to_numpy()*bins).astype(int), bins-1)
        ece = 0.
        for b in range(bins):
            x = group[bin_id == b]
            confidence = float(x.confidence.mean()) if len(x) else None
            accuracy = float(x.top1.mean()) if len(x) else None
            if len(x):
                ece += len(x)/len(group)*abs(confidence-accuracy)
            calibration.append(dict(model=model, bin=b, lower=b/bins, upper=(b+1)/bins,
                n_weeks=len(x), mean_confidence=confidence, observed_accuracy=accuracy))
        tables.append(dict(phase=phase, model=model, n_weeks=len(group),
            n_seasons=int(group.season.nunique()), log_loss=float(group.log_loss.mean()),
            brier=float(group.brier.mean()), top1=float(group.top1.mean()), ece=ece))
    return pd.DataFrame(tables), pd.DataFrame(calibration)


def paired_bootstrap(scores, draws=5000, seed=2627857):
    seasons = np.sort(scores.season.unique())
    if len(seasons) < 2:
        raise ValueError("Paired season bootstrap requires at least two seasons")
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(seasons), (draws, len(seasons)))
    results = []
    bayes = scores[scores.model.eq("forward_bayes")].set_index(["season", "week"])
    for other in ("uniform", "judge_only", "past_survival", "past_judges"):
        base = scores[scores.model.eq(other)].set_index(["season", "week"])
        if not bayes.index.equals(base.index):
            raise ValueError("All baselines must score the same weeks")
        for metric in ("log_loss", "brier", "top1"):
            delta = bayes[metric]-base[metric]
            sums = delta.groupby(level="season").sum().reindex(seasons).to_numpy()
            counts = delta.groupby(level="season").size().reindex(seasons).to_numpy()
            distribution = sums[sample].sum(axis=1)/counts[sample].sum(axis=1)
            lower, upper = np.quantile(distribution, [.025, .975])
            results.append(dict(comparison=f"forward_bayes minus {other}", metric=metric,
                difference=float(delta.mean()), lower=float(lower), upper=float(upper),
                bootstrap_draws=draws, n_seasons=len(seasons)))
    return pd.DataFrame(results)


def choose_development(panel, cfg):
    """Tune each nontrivial model on its declared development-only candidate grid."""
    trials = []
    placeholder = dict(judge_temperature=1., history_temperature=1., history_strength=0.,
                       past_judge_temperature=1.)
    for sd, temp in itertools.product(cfg["prior_sd_grid"], cfg["temperature_grid"]):
        rows, _ = run_population(panel, cfg["development_seasons"], cfg,
            dict(prior_sd=sd, temperature=temp, **placeholder),
            cfg["development_particles"], cfg["seeds"][0])
        scores = weekly_scores(rows)
        loss = scores.loc[scores.model.eq("forward_bayes"), "log_loss"].mean()
        trials.append(dict(model="forward_bayes", prior_sd=sd, temperature=temp,
                           history_strength=None, development_log_loss=float(loss)))
        print(f"development Bayesian sd={sd:g} temperature={temp:g} log_loss={loss:.6f}", flush=True)
    # Baselines need no SMC and use the same prediction/release implementation.
    for model, candidates in [
        ("judge_only", [(t, 0.) for t in cfg["judge_temperature_grid"]]),
        ("past_judges", [(t, 0.) for t in cfg["judge_temperature_grid"]]),
        ("past_survival", list(itertools.product(cfg["temperature_grid"],
                                               cfg["history_strength_grid"])))]:
        for temp, strength in candidates:
            rows, _ = run_population(panel, cfg["development_seasons"], cfg,
                dict(prior_sd=1., temperature=1., judge_temperature=temp,
                     history_temperature=temp, history_strength=strength,
                     past_judge_temperature=temp),
                16, cfg["seeds"][0])
            loss = weekly_scores(rows).query("model == @model").log_loss.mean()
            trials.append(dict(model=model, prior_sd=None, temperature=temp,
                history_strength=strength, development_log_loss=float(loss)))
    trials = pd.DataFrame(trials)
    trials["selected"] = False
    selected = {}
    for model in ("forward_bayes", "judge_only", "past_survival", "past_judges"):
        # Candidate input order supplies declared deterministic tie breaking.
        idx = trials[trials.model.eq(model)].development_log_loss.idxmin()
        trials.loc[idx, "selected"] = True
        row = trials.loc[idx]
        if model == "forward_bayes":
            selected.update(prior_sd=float(row.prior_sd), temperature=float(row.temperature))
        elif model == "judge_only":
            selected["judge_temperature"] = float(row.temperature)
        elif model == "past_judges":
            selected["past_judge_temperature"] = float(row.temperature)
        else:
            selected.update(history_temperature=float(row.temperature),
                            history_strength=float(row.history_strength))
    return selected, trials


def _ensemble(repeats):
    combined = pd.concat(repeats, ignore_index=True)
    fixed = ["season", "week", "contestant_id", "model", "observed_exit", "eligible", "updated"]
    return combined.groupby(fixed, sort=True, as_index=False).probability.mean()


def run_study(data, config, out):
    data, config, out = Path(data).resolve(), Path(config).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Use a fresh output directory; completed or partial runs are never overwritten")
    cfg = read_protocol(config)
    root = Path(__file__).resolve().parents[1]
    code_files = [root/"dwts"/name for name in ("forward_model.py", "forecasting.py", "data.py")]
    frozen = dict(raw_data_sha256=digest(data), config_sha256=digest(config),
                  code_sha256={p.name: digest(p) for p in code_files})
    out.mkdir(parents=True)
    write_json(out/"status.json", dict(status="running", started_utc=datetime.now(timezone.utc).isoformat()))
    protocol = dict(schema_version=1, interpretation=INTERPRETATION, configuration=cfg,
        integrity=frozen, python=platform.python_version(),
        packages={p: metadata.version(p) for p in ("numpy", "pandas", "scipy")},
        likelihood="normalized categorical softmax(-temperature*n*weighted_score_share)",
        latent_prior="independent static season contestant Normal(0, prior_sd)",
        inference="SMC exact likelihood weights; ESS systematic resampling; full-past symmetric RW MH",
        clock="current roster/scores -> probabilities -> release exit/withdrawal -> update; final flag scores only",
        assumptions=["Current weekly roster and judges' scores are available before predicting the exit.",
            "Retrospective label-first exits and reconstructed withdrawals are assumed released at week end.",
            "Single ordinary exit with no observed withdrawal and at least three contestants updates inference.",
            "Final-week metadata never enters inference; it filters reported evaluation only.",
            "Future contestant identifiers are not preloaded; newly observed entrants get independent priors."],
        primary_metric="mean categorical log loss over eligible weeks",
        brier_definition="sum over active contestants of (predicted exit probability - one-hot exit)^2",
        top1_tie_break="lexically smallest contestant identifier",
        calibration=f"{cfg['calibration_bins']} fixed equal-width top-label confidence bins; descriptive, small sample",
        intervals="paired whole-season percentile bootstrap of fixed predictions; no refit or independent-data uncertainty",
        history_baseline="cumulative past judge-hazard exit probability for actual survivors; same released-event update gate",
        past_judge_baseline="own mean past n*judge_share, current relative score for cold starts; softmax negative temperature times score",
        mc_check_seed_offset=1000003,
        not_measured=["actual audience vote accuracy", "late-era judge-save forecasts", "independent prospective validation"])
    write_json(out/"protocol.json", protocol)
    panel, _ = prepare(data, outcome_policy="label_first", include_final_week_eliminations=True)
    coverage = eligibility_summary(panel, cfg)
    coverage.to_csv(out/"eligibility_summary.csv", index=False)
    selected, trials = choose_development(panel, cfg)
    trials.to_csv(out/"model_selection.csv", index=False)
    write_json(out/"selected_model.json", selected)
    repeats, diag = [], []
    for repeat, seed in enumerate(cfg["seeds"]):
        rows, diagnostics = run_population(panel, cfg["evaluation_seasons"], cfg, selected,
                                            cfg["particles"], seed)
        repeats.append(rows)
        diagnostics["particles"], diagnostics["repeat"] = cfg["particles"], repeat
        diag.append(diagnostics)
        print(f"evaluation {cfg['particles']} particles repeat {repeat+1}/{len(cfg['seeds'])}", flush=True)
    ensemble = _ensemble(repeats)
    # These individual prediction records are explicitly private, excluded from packaging.
    ensemble.to_csv(out/"private_predictions.csv", index=False)
    scores = weekly_scores(ensemble)
    metrics, calibration = aggregate_scores(scores, cfg["calibration_bins"])
    metrics.to_csv(out/"metrics.csv", index=False)
    calibration.to_csv(out/"calibration.csv", index=False)
    season_records = []
    for season, group in scores.groupby("season", sort=True):
        m, _ = aggregate_scores(group, cfg["calibration_bins"])
        m["season"] = int(season)
        season_records.append(m)
    pd.concat(season_records, ignore_index=True).to_csv(out/"season_metrics.csv", index=False)
    paired_bootstrap(scores, cfg["bootstrap_draws"], cfg["bootstrap_seed"]).to_csv(
        out/"paired_differences.csv", index=False)
    reference = ensemble[ensemble.model.eq("forward_bayes") & ensemble.eligible].set_index(KEYS)
    stability, check_ensembles = [], {}
    for count in [cfg["particles"], *cfg["particle_checks"]]:
        check_repeats = []
        for repeat, seed in enumerate(cfg["seeds"]):
            if count == cfg["particles"]:
                rows = repeats[repeat]
            else:
                rows, diagnostics = run_population(panel, cfg["evaluation_seasons"], cfg,
                                                      selected, count, seed+1000003)
                diagnostics["particles"], diagnostics["repeat"] = count, repeat
                diag.append(diagnostics)
            check_repeats.append(rows)
            current = rows[rows.model.eq("forward_bayes") & rows.eligible].set_index(KEYS)
            difference = current.probability-reference.probability
            m = weekly_scores(rows).query("model == 'forward_bayes'")
            stability.append(dict(particles=count, repeat=repeat, log_loss=float(m.log_loss.mean()),
                brier=float(m.brier.mean()), top1=float(m.top1.mean()),
                mean_absolute_probability_difference=float(difference.abs().mean()),
                max_absolute_probability_difference=float(difference.abs().max()),
                reference=f"{cfg['particles']}-particle {len(cfg['seeds'])}-repeat probability ensemble"))
            print(f"particle check {count}, repeat {repeat+1}", flush=True)
        check_ensembles[count] = _ensemble(check_repeats)
    pd.DataFrame(stability).to_csv(out/"particle_stability.csv", index=False)
    diagnostics = pd.concat(diag, ignore_index=True)
    diagnostics.to_csv(out/"smc_diagnostics.csv", index=False)
    larger = max(cfg["particle_checks"])
    high = check_ensembles[larger]
    high_probability = high[high.model.eq("forward_bayes") & high.eligible].set_index(KEYS).probability
    delta_probability = high_probability-reference.probability
    main_loss = float(scores[scores.model.eq("forward_bayes")].log_loss.mean())
    high_loss = float(weekly_scores(high).query("model == 'forward_bayes'").log_loss.mean())
    repeat_loss = [weekly_scores(r).query("model == 'forward_bayes'").log_loss.mean() for r in repeats]
    observed_mc = dict(mean_absolute_probability_difference=float(delta_probability.abs().mean()),
        max_absolute_probability_difference=float(delta_probability.abs().max()),
        absolute_log_loss_difference=abs(high_loss-main_loss),
        repeat_log_loss_span=float(max(repeat_loss)-min(repeat_loss)))
    passed = {name: value <= cfg["mc_thresholds"][name] for name, value in observed_mc.items()}
    mc_assessment = dict(interpretation="Approximation sensitivity check, not proof of posterior convergence or model validity.",
        main_particles=cfg["particles"], higher_particles=larger, independent_check_seed_offset=1000003,
        repeats=len(cfg["seeds"]), scope="eligible evaluation-week categorical probabilities",
        thresholds=cfg["mc_thresholds"], observed=observed_mc, passed=passed,
        all_checks_pass=all(passed.values()))
    write_json(out/"mc_assessment.json", mc_assessment)
    protocol["smc_aggregate_diagnostics"] = dict(observed_updates=len(diagnostics),
        minimum_pre_resampling_ess_fraction=float(diagnostics.ess_fraction.min()),
        median_pre_resampling_ess_fraction=float(diagnostics.ess_fraction.median()),
        resampling_fraction=float(diagnostics.resampled.mean()),
        mh_acceptance=float(diagnostics.mh_accepted.sum()/max(1, diagnostics.mh_attempts.sum())),
        approximation_checks_pass=all(passed.values()))
    after = dict(raw_data_sha256=digest(data), config_sha256=digest(config),
                 code_sha256={p.name: digest(p) for p in code_files})
    if after != frozen:
        raise RuntimeError("Data, configuration or implementation changed during the study")
    protocol["selected_model"] = selected
    protocol["evaluation_population"] = dict(seasons=cfg["evaluation_seasons"],
        represented_seasons=sorted(map(int, scores.season.unique())),
        seasons_without_eligible_weeks=sorted(set(cfg["evaluation_seasons"])-set(map(int, scores.season.unique()))),
        weeks=int(scores[scores.model.eq("forward_bayes")].shape[0]),
        update_event_assumption="post-week released ordinary exits and reconstructed withdrawal labels")
    write_json(out/"protocol.json", protocol)
    artifacts = {p.name: digest(p) for p in sorted(out.iterdir()) if p.name != "status.json"}
    write_json(out/"status.json", dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                                     integrity=frozen, artifacts_sha256=artifacts))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--config", default="configs/forecast_study.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(run_study(args.data, args.config, args.out).to_string(index=False))


if __name__ == "__main__":
    main()
