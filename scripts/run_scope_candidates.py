"""Run two predeclared, one-factor Bayesian scope sensitivities; no attribution fits.

Unchanged seasons reuse verified archived summaries, whose raw chains live in the
previous posterior packages. No report target is used to choose a protocol/setting.
"""
import argparse
import ast
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dwts.cli import _fit_job, dump_json
from dwts.data import data_summary, prepare
from dwts.model import FitConfig

KEYS = ["season", "week", "contestant_id"]
MODEL_INPUTS = KEYS + ["judge_mean", "judge_z", "judge_share", "eliminated", "withdrawal", "rule"]
PROTOCOLS = {
    "candidate_zero_first": {
        "data": {"outcome_policy": "zero_first", "include_final_week_eliminations": False,
                 "exclude_withdrawn_contestants": False},
        "expected_changed_seasons": [18],
        "interpretation": "Literal first-zero convention sensitivity; all other baseline scope held fixed.",
    },
    "candidate_exclude_withdrawn": {
        "data": {"outcome_policy": "label_first", "include_final_week_eliminations": False,
                 "exclude_withdrawn_contestants": True},
        "expected_changed_seasons": [3, 7, 9, 16, 18, 21, 28, 29, 31],
        "interpretation": "Withdrawn-roster sensitivity motivated by EDA evidence, not an identified author Bayesian scope.",
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ordered(frame):
    return frame.sort_values(KEYS).reset_index(drop=True)


def inputs_equal(a, b):
    return ordered(a)[MODEL_INPUTS].equals(ordered(b)[MODEL_INPUTS])


def input_digest(frame):
    # Stable textual serialization of canonical, full-precision model inputs.
    return hashlib.sha256(ordered(frame)[MODEL_INPUTS].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def assert_inputs_match(a, b, context, exact=True):
    try:
        pd.testing.assert_frame_equal(ordered(a)[MODEL_INPUTS], ordered(b)[MODEL_INPUTS],
                                      check_exact=exact, rtol=0, atol=1e-14)
    except AssertionError as exc:
        raise ValueError(f"Model-input consistency failed: {context}") from exc


def posterior_rule_dependency(source):
    """Fingerprint the sole rules dependency imported by both posterior engines.

    The engine source files themselves must remain byte-identical. They import
    only rules.ranks, whose full AST and np/rankdata bindings are guarded here.
    Simulation-only tie ordering is not part of the posterior target.
    """
    tree = ast.parse(source)
    rank_functions = [node for node in tree.body
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'ranks']
    if len(rank_functions) != 1:
        raise ValueError('Expected exactly one posterior ranks function')
    relevant = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue  # Module documentation has no numerical effect.
        if (isinstance(node, ast.ImportFrom) and node.module == 'fractions' and node.level == 0
                and len(node.names) == 1 and node.names[0].name == 'Fraction' and node.names[0].asname is None):
            continue  # The sole reviewed new import, used only by simulation ordering.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != 'ranks':
            # Definitions execute decorators/defaults/annotations at import time.
            # Guard those while allowing changes inside simulation-only bodies.
            signature = copy.copy(node)
            signature.body = []
            relevant.append(signature)
        else:
            # Includes nested top-level control flow, rebinding, imports, classes,
            # and all of ranks. Unreviewed executable changes therefore fail closed.
            relevant.append(node)
    return tuple(ast.dump(node, include_attributes=False) for node in relevant)


def verify_posterior_source(archive, root):
    for name in ['dwts/model.py', 'dwts/fast_mh.py']:
        if archive.read(name) != (root / name).read_bytes():
            raise ValueError(f'Frozen posterior implementation changed: {name}')
    archived = posterior_rule_dependency(archive.read('dwts/rules.py'))
    current = posterior_rule_dependency((root / 'dwts/rules.py').read_bytes())
    if archived != current:
        raise ValueError('Frozen posterior implementation changed: dwts/rules.py ranks dependency')


def preflight(data, base):
    manifest = json.loads((base / "manifest.json").read_text())
    config = json.loads((base / "run_config.json").read_text())
    if manifest.get("status") != "completed" or not manifest.get("all_sampling_diagnostics_pass"):
        raise ValueError("Baseline must be completed and diagnostically passed")
    if sha(data) != manifest["input_sha256"]:
        raise ValueError("Raw input hash differs from frozen baseline")
    fit = FitConfig(**config["fit"]); fit.validate()
    if asdict(fit) != config["fit"]:
        raise ValueError("Baseline FitConfig must specify all current settings")
    with ZipFile(base / "execution_source.zip") as archive:
        verify_posterior_source(archive, ROOT)
    # Exercise the actual output backend before any expensive sampling starts.
    import h5py
    import xarray as xr
    with tempfile.TemporaryDirectory(prefix="dwts-netcdf-preflight-") as temporary:
        path = Path(temporary) / "test.nc"
        xr.Dataset({"value": ("observation", np.array([0.0]))}).to_netcdf(path, engine="h5netcdf")
        with h5py.File(path) as saved:
            if saved["value"][0] != 0.0:
                raise ValueError("NetCDF serialization preflight did not round trip")
    canonical, _ = prepare(data, **config["data"])
    saved_panel = pd.read_csv(base / "panel_all_seasons.csv", float_precision="round_trip")
    assert_inputs_match(canonical, saved_panel, "baseline prepared panel", exact=True)
    votes = pd.read_csv(base / "votes.csv", float_precision="round_trip")
    # The historical assembly rewrote CSVs after parsing with pandas' default
    # parser. Permit only serialization-level differences in this combined file;
    # original per-season fit inputs below must agree exactly.
    assert_inputs_match(canonical, votes, "baseline aggregate votes", exact=False)
    posterior_cols = ["fan_mean", "fan_lower", "fan_upper", "relative_uncertainty"]
    for season, frame in canonical.groupby("season"):
        directory = base / f"season_{season:02d}"
        status = json.loads((directory / "fit_status.json").read_text())
        if status["config"] != config["fit"] or not status["sampling_diagnostics_pass"]:
            raise ValueError(f"Baseline season {season} fit settings/status disagree")
        original = pd.read_csv(directory / "votes.csv", float_precision="round_trip")
        assert_inputs_match(frame, original, f"baseline season {season}", exact=True)
        pd.testing.assert_frame_equal(ordered(original)[posterior_cols],
                                      ordered(votes[votes.season == season])[posterior_cols],
                                      check_exact=False, rtol=0, atol=1e-13)
    return manifest, config, canonical, votes


def fit_candidate(frame, fit_dict, destination):
    """Diagnostic-only escalation changes storage duration, never the target."""
    cfg = FitConfig(**fit_dict)
    attempts = []
    for attempt, multiplier in enumerate([1, 2, 4], start=1):
        current = replace(cfg, draws=cfg.draws * multiplier)
        if attempt > 1:
            old = destination / f"attempt_{attempt - 1:02d}"
            old.mkdir()
            for path in list(destination.iterdir()):
                if path.is_file():
                    shutil.move(str(path), old / path.name)
        started = time.perf_counter()
        result, status = _fit_job(frame, current, destination)
        attempts.append({"attempt": attempt, "draw_multiplier": multiplier,
                         "runtime_seconds": time.perf_counter() - started,
                         "config": asdict(current), "status": status,
                         "posterior_sha256": sha(destination / "posterior.nc")})
        if status["sampling_diagnostics_pass"]:
            break
    dump_json(destination / "candidate_attempts.json", attempts)
    return result, status, attempts


def snapshot(out):
    paths = sorted((ROOT / "dwts").glob("*.py")) + [Path(__file__), ROOT / "tests/test_scope_candidates.py", ROOT / "pyproject.toml"]
    with ZipFile(out / "execution_source.zip", "w", ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(ROOT)))
    return sha(out / "execution_source.zip")


def run(data, base, output_root, workers=4):
    started = time.perf_counter()
    destinations = {name: output_root / name for name in PROTOCOLS}
    for out in destinations.values():
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Preserve existing results; output is not empty: {out}")
    baseline_manifest, config, canonical, baseline_votes = preflight(data, base)
    contexts, jobs = {}, []
    prior_origins = pd.read_csv(base / "season_provenance.csv").set_index("season")
    for name, spec in PROTOCOLS.items():
        out = destinations[name]; out.mkdir(parents=True, exist_ok=True)
        panel, audit = prepare(data, **spec["data"])
        changed = [s for s in range(1, 35) if not inputs_equal(canonical[canonical.season == s], panel[panel.season == s])]
        if changed != spec["expected_changed_seasons"]:
            raise ValueError(f"Unexpected affected seasons for {name}: {changed}")
        run_config = {**config, "data": spec["data"], "seasons": list(range(1, 35))}
        protocol = {"protocol_id": name, "declared_utc": datetime.now(timezone.utc).isoformat(),
                    **spec, "fit": config["fit"], "baseline_directory": str(base.resolve()),
                    "baseline_manifest_sha256": sha(base / "manifest.json"),
                    "baseline_votes_sha256": sha(base / "votes.csv"), "raw_sha256": sha(data),
                    "reuse_rule": "Exact canonical per-season model-input equality; identical full FitConfig; verified raw and source hashes.",
                    "diagnostic_escalation": "Only failed fits: repeat with 2x then 4x saved draws, same seed/tune/target. Preserve all attempts; no target-score selection.",
                    "downstream_analysis": "Not run by this script. Parent owns RF/OLS and report comparisons.",
                    "legacy_aggregate_csv_tolerance": "Absolute 1e-14 for combined input CSV, 1e-13 for combined posterior summaries; original per-season model inputs exact.",
                    "raw_chains_for_reused_seasons": "Absent from supplied main bundle; preserve archived posterior hashes and refer to prior posterior ZIPs."}
        # Both complete protocols are written before any model fitting starts.
        dump_json(out / "protocol.json", protocol)
        dump_json(out / "run_config.json", run_config)
        panel.to_csv(out / "panel_all_seasons.csv", index=False)
        audit.to_csv(out / "preprocessing_audit.csv", index=False)
        dump_json(out / "data_summary_all_seasons.json", data_summary(panel))
        source_hash = snapshot(out)
        manifest = {"status": "prepared", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "input_sha256": sha(data), "protocol_sha256": sha(out / "protocol.json"),
                    "source_snapshot_sha256": source_hash, "baseline_manifest": baseline_manifest,
                    "python": platform.python_version(),
                    "packages": {package: importlib.metadata.version(package) for package in
                                 ["numpy", "pandas", "scipy", "pymc", "pytensor", "arviz", "numba", "h5netcdf", "h5py", "xarray"]},
                    "changed_seasons": changed, "reused_seasons": [s for s in range(1, 35) if s not in changed],
                    "attribution_ran": False, "generation": "fixed_target_scope_sensitivity"}
        dump_json(out / "manifest.json", manifest)
        results, statuses, provenance = {}, {}, []
        for season, frame in panel.groupby("season"):
            dest = out / f"season_{season:02d}"
            digest = input_digest(frame)
            if season in changed:
                jobs.append((name, int(season), frame, dest))
                continue
            source = base / dest.name
            shutil.copytree(source, dest)
            records = []
            for file in sorted(source.rglob("*")):
                if file.is_file():
                    copy = dest / file.relative_to(source)
                    source_sha = sha(file)
                    if source_sha != sha(copy):
                        raise ValueError(f"Copied file mismatch: {copy}")
                    records.append({"source": str(file.resolve()), "destination": str(copy.resolve()), "sha256": source_sha})
            # Preserve exactly the frozen combined posterior summary values while
            # assigning the canonical full-precision input columns for this scope.
            result = ordered(baseline_votes[baseline_votes.season == season]).copy()
            fresh = ordered(frame)
            for col in fresh:
                result[col] = fresh[col].to_numpy()
            results[int(season)] = result
            statuses[int(season)] = json.loads((source / "fit_status.json").read_text())
            provenance.append({"season": int(season), "kind": "reused", "model_input_sha256": digest,
                               "baseline_model_input_sha256": input_digest(canonical[canonical.season == season]),
                               "source_directory": str(source.resolve()), "files": records,
                               "archived_posterior_sha256": prior_origins.loc[season, "posterior_sha256"],
                               "original_source_directory": prior_origins.loc[season, "source_directory"],
                               "posterior_present_locally": (source / "posterior.nc").exists(),
                               "aggregate_posterior_summary_origin": str((base / "votes.csv").resolve())})
        contexts[name] = dict(out=out, manifest=manifest, results=results, statuses=statuses, provenance=provenance, panel=panel)
        dump_json(out / "season_provenance.json", provenance)
    for context in contexts.values():
        context["manifest"]["status"] = "running"
        dump_json(context["out"] / "manifest.json", context["manifest"])
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(fit_candidate, frame, config["fit"], dest): (name, season, frame, dest)
                   for name, season, frame, dest in jobs}
        for future in as_completed(futures):
            name, season, frame, dest = futures[future]
            result, status, attempts = future.result()
            context = contexts[name]
            context["results"][season] = result; context["statuses"][season] = status
            context["provenance"].append({"season": season, "kind": "refit", "model_input_sha256": input_digest(frame),
                "baseline_model_input_sha256": input_digest(canonical[canonical.season == season]),
                "source_directory": str(dest.resolve()), "posterior_sha256": sha(dest / "posterior.nc"),
                "posterior_present_locally": True, "attempts": attempts,
                "files": [{"path": str(p.resolve()), "sha256": sha(p)} for p in sorted(dest.rglob("*")) if p.is_file()]})
            dump_json(context["out"] / "season_provenance.json", sorted(context["provenance"], key=lambda x: x["season"]))
            dump_json(context["out"] / "sampling_status.json", [context["statuses"][s] for s in sorted(context["statuses"])])
            print(f"{name} S{season}: pass={status['sampling_diagnostics_pass']}, max R-hat={status['max_rhat']:.6f}, attempts={len(attempts)}", flush=True)
    for name, context in contexts.items():
        votes = ordered(pd.concat([context["results"][s] for s in range(1, 35)], ignore_index=True))
        assert_inputs_match(context["panel"], votes, f"assembled {name}", exact=True)
        votes.to_csv(context["out"] / "votes.csv", index=False)
        statuses = [context["statuses"][s] for s in range(1, 35)]
        dump_json(context["out"] / "sampling_status.json", statuses)
        manifest = context["manifest"]
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds_total_protocol_batch=time.perf_counter() - started,
                        all_sampling_diagnostics_pass=all(s["sampling_diagnostics_pass"] for s in statuses),
                        votes_sha256=sha(context["out"] / "votes.csv"),
                        locally_available_raw_chain_seasons=manifest["changed_seasons"])
        dump_json(context["out"] / "manifest.json", manifest)
        print(f"Completed {name}: {sum(s['sampling_diagnostics_pass'] for s in statuses)}/34 diagnostic passes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--base", type=Path, default=ROOT / "outputs/report_final")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    run(args.data, args.base, args.output_root, args.workers)


if __name__ == "__main__":
    main()
