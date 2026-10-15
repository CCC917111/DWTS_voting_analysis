import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import multiprocessing
import logging

import numpy as np
import pandas as pd

from .data import data_summary, prepare
from .model import FitConfig, fit_season


def dump_json(path, obj):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def _fit_job(frame, cfg, outdir):
    """Process-level season parallelism; no shared random stream or model state."""
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "fit.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import pymc  # initialize handlers before binding this job's file
        logger = logging.getLogger("pymc")
        old_handlers, old_propagate, old_level = logger.handlers[:], logger.propagate, logger.level
        handler = logging.StreamHandler(log)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            return fit_season(frame, cfg, outdir)
        finally:
            logger.handlers = old_handlers
            logger.propagate = old_propagate
            logger.setLevel(old_level)
            handler.close()


def code_digest():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run(args):
    config = json.loads(Path(args.config).read_text())
    compare_reference = bool(getattr(args, "compare_reference", False))
    allowed = {"fit", "seasons", "simulation_scope", "rf_trees", "rf_folds", "history_weight", "data", "season_workers", "history_grid", "rf_inner_folds", "rf_min_samples_leaf", "rf_judge_target", "rf_fan_scale", "partner_identity_mode"}
    if set(config) - allowed:
        raise ValueError(f"Unknown config keys: {set(config) - allowed}")
    if config.get('partner_identity_mode', 'raw') not in {'raw', 'annotations_only', 'verified_canonical_identity'}:
        raise ValueError('Unknown partner_identity_mode')
    cfg = FitConfig(**config.get("fit", {}))
    cfg.validate()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty; choose a new --out to preserve prior runs")
    out.mkdir(parents=True, exist_ok=True)
    panel, audit = prepare(args.data, **config.get("data", {}))
    seasons = args.seasons or config.get("seasons") or sorted(panel.season.unique())
    if not seasons or set(seasons) - set(panel.season.unique()):
        raise ValueError("Requested seasons unavailable")
    selected = panel[panel.season.isin(seasons)].copy()
    run_config = {**config, "fit": asdict(cfg), "seasons": list(map(int, seasons))}
    input_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    source_hash = code_digest()
    if args.resume:
        old_manifest = json.loads((out / "manifest.json").read_text())
        old_config = json.loads((out / "run_config.json").read_text())
        if (old_config != run_config or old_manifest["input_sha256"] != input_hash
                or old_manifest.get("code_sha256") != source_hash
                or bool(old_manifest.get("reference_comparison_requested", False)) != compare_reference):
            raise ValueError("Resume refused: config, input data, code, or reference-comparison mode changed; use a new output directory")
    dump_json(out / "data_summary_all_seasons.json", data_summary(panel))
    audit.to_csv(out / "preprocessing_audit.csv", index=False)
    panel.to_csv(out / "panel_all_seasons.csv", index=False)
    dump_json(out / "run_config.json", run_config)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "input_sha256": input_hash, "code_sha256": source_hash,
                "study": {"team": "2627761", "problem": "2026 MCM Problem C", "topic": "DWTS voting systems"},
                "reference_comparison_requested": compare_reference, "status": "running",
                "packages": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "scipy", "pymc", "pytensor", "arviz", "scikit-learn", "shap", "statsmodels", "numba"]}}
    dump_json(out / "manifest.json", manifest)
    results, statuses = [], []
    pending = []
    for season, g in selected.groupby("season"):
        dest = out / f"season_{season:02d}"
        if args.resume and (dest / "fit_status.json").exists():
            results.append(pd.read_csv(dest / "votes.csv"))
            statuses.append(json.loads((dest / "fit_status.json").read_text()))
        else:
            pending.append((season, g, dest))
    workers = int(config.get("season_workers", 1))
    if workers < 1 or (workers > 1 and cfg.cores > 1):
        raise ValueError("Use positive season_workers and cores=1 when parallelizing seasons")
    def record(result, status):
        results.append(result)
        statuses.append(status)
        statuses.sort(key=lambda s: s["season"])
        dump_json(out / "sampling_status.json", statuses)
        print(f"Season {status['season']} finished ({len(statuses)}/{len(seasons)}); max R-hat={status['max_rhat']:.3f}", flush=True)
    if workers == 1:
        for season, g, dest in pending:
            print(f"Fitting season {season}: {len(g)} contestant-weeks", flush=True)
            record(*_fit_job(g, cfg, dest))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = {pool.submit(_fit_job, g, cfg, dest): season for season, g, dest in pending}
            for future in as_completed(jobs):
                record(*future.result())
    votes = pd.concat(results, ignore_index=True).sort_values(["season", "week", "contestant_id"]).reset_index(drop=True)
    votes.to_csv(out / "votes.csv", index=False)
    analyze(votes, out, config, cfg.seed, args.skip_attribution)
    from .plots import make_study_figures
    make_study_figures(out, statuses)
    if compare_reference:
        from .reconcile import write_comparison
        write_comparison(votes, out, statuses)
    manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                    all_sampling_diagnostics_pass=all(s["sampling_diagnostics_pass"] for s in statuses),
                    attribution_ran=not args.skip_attribution and len(seasons) >= 2)
    dump_json(out / "manifest.json", manifest)
    print(f"Completed: {out}. Check sampling_status.json before interpreting estimates.", flush=True)


def analyze(votes, out, config, seed=2627761, skip_attribution=False):
    from .evaluation import evaluate, weight_grid
    from .plots import make_plots
    identity_mode = config.get('partner_identity_mode', 'raw')
    if identity_mode not in {'raw', 'annotations_only', 'verified_canonical_identity'}:
        raise ValueError('Unknown partner_identity_mode')
    include_withdrawals = config.get("fit", {}).get("withdrawal_policy", "skip_week") == "include_scored"
    tables = evaluate(votes, include_withdrawal_weeks=include_withdrawals)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    detail, grid = weight_grid(votes, config.get("simulation_scope", "single_elimination"), include_withdrawals)
    detail.to_csv(out / "weight_grid_weekly.csv", index=False)
    grid.to_csv(out / "weight_grid_summary.csv", index=False)
    make_plots(votes, tables["influence_balance"], grid, out / "figures")
    if not skip_attribution and votes.season.nunique() >= 2:
        from .attribution import ols_analysis, rf_analysis
        attribution_votes = votes
        if identity_mode != 'raw':
            from .partner_identity import canonical_primary_partners
            attribution_votes = canonical_primary_partners(
                votes, include_verified_aliases=identity_mode == 'verified_canonical_identity')
            attribution_votes.to_csv(out / 'attribution_input.csv', index=False)
            provenance_columns = ['season', 'week', 'contestant_id', 'partner', 'partner_raw',
                                  'partner_guest_annotation', 'partner_annotation_week',
                                  'partner_annotation_this_week', 'partner_identity_changed', 'partner_id_policy']
            attribution_votes[provenance_columns].to_csv(out / 'partner_identity_provenance.csv', index=False)
        print("Running OLS and season-grouped Random Forest / SHAP", flush=True)
        kw = {"history_weight": config.get("history_weight", 0.5)}
        ols_analysis(attribution_votes, out, **kw)
        rf_analysis(attribution_votes, out, seed, trees=config.get("rf_trees", 300),
                    folds=config.get("rf_folds", 5), history_grid=config.get("history_grid"),
                    inner_folds=config.get("rf_inner_folds", 3), min_samples_leaf=config.get("rf_min_samples_leaf", 5),
                    judge_target=config.get("rf_judge_target", "judge_mean"), fan_scale=config.get("rf_fan_scale", "relative"), **kw)


def main():
    parser = argparse.ArgumentParser(description="Analyze DWTS fan support and voting rules")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Inspect all official data without MCMC")
    p.add_argument("--data", default="data/raw/2026_MCM_Problem_C_Data.csv")
    p.add_argument("--out", default="outputs/prepared")
    p = sub.add_parser("run", help="Run preprocessing, MCMC, simulations, OLS and RF/SHAP")
    p.add_argument("--data", default="data/raw/2026_MCM_Problem_C_Data.csv")
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--out", default="outputs/smoke")
    p.add_argument("--seasons", type=int, nargs="+")
    p.add_argument("--skip-attribution", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse completed seasons only if code, input and config match")
    p.add_argument("--compare-reference", action="store_true", help="Also export comparisons with the original report's values")
    p = sub.add_parser("verify", help="Recompute analyses from saved votes; exit 2 if report numbers do not match")
    p.add_argument("--source", default="outputs/report_final")
    p.add_argument("--config", default="configs/report_verification.json")
    p.add_argument("--out", default="outputs/report_verified")
    args = parser.parse_args()
    if args.command == "verify":
        from .verification import run_verification
        raise SystemExit(run_verification(args))
    if args.command == "run":
        run(args)
    else:
        p, audit = prepare(args.data)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        p.to_csv(out / "panel.csv", index=False)
        audit.to_csv(out / "audit.csv", index=False)
        summary = data_summary(p)
        dump_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=2, default=lambda x: x.item()))

