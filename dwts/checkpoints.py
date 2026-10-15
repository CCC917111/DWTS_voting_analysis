"""Integrity-checked season checkpoints for interrupted study runs.

A checkpoint is committed after all fit artifacts have been written. It guards
against accidental changes and inconsistent cached inputs, not malicious edits
to both an artifact and its checksum record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
CHECKPOINT_NAME = "checkpoint.json"
ARTIFACTS = (
    "votes.csv", "fit_status.json", "diagnostics.csv", "posterior.nc",
    "posterior_eliminations.csv",
)
KEYS = ["season", "week", "contestant_id"]


def _record(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Checkpoint artifact missing or not a regular file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_fit(directory, expected, fit_config):
    votes = pd.read_csv(directory / "votes.csv", float_precision="round_trip")
    status = json.loads((directory / "fit_status.json").read_text())
    if not isinstance(status, dict):
        raise ValueError("Checkpoint fit status must be a JSON object")
    required = set(expected.columns) | set(KEYS) | {"fan_mean", "fan_lower", "fan_upper"}
    if (required - set(votes) or votes.empty or votes[KEYS].isna().any().any()
            or votes.duplicated(KEYS).any()):
        raise ValueError("Checkpoint votes must have complete, unique input row keys and probability columns")
    seasons = expected.season.unique()
    if len(seasons) != 1 or type(status.get("season")) is not int or status["season"] != int(seasons[0]):
        raise ValueError("Checkpoint status season disagrees with the requested season")
    if status.get("config") != fit_config:
        raise ValueError("Checkpoint fit configuration disagrees with the requested configuration")
    for column in expected.select_dtypes(include="bool").columns:
        if not pd.api.types.is_bool_dtype(votes[column]):
            raise ValueError(f"Checkpoint input {column} must contain booleans")
    try:
        pd.testing.assert_frame_equal(
            votes.sort_values(KEYS)[expected.columns].reset_index(drop=True),
            expected.sort_values(KEYS).reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
    except AssertionError as error:
        raise ValueError("Checkpoint rows disagree with the requested preprocessed season") from error
    probabilities = votes[["fan_mean", "fan_lower", "fan_upper"]].to_numpy(float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or (votes.fan_lower > votes.fan_upper).any()
            or not np.allclose(votes.groupby(["season", "week"]).fan_mean.sum(), 1,
                               atol=1e-10, rtol=0)):
        raise ValueError("Checkpoint fan probabilities must be finite, bounded and normalized by week")
    diagnostics = pd.read_csv(directory / "diagnostics.csv", float_precision="round_trip")
    columns = ["r_hat", "ess_bulk", "ess_tail"]
    if (diagnostics.empty or set(columns + ["parameter"]) - set(diagnostics)
            or diagnostics.parameter.isna().any() or diagnostics.parameter.duplicated().any()):
        raise ValueError("Checkpoint diagnostics are incomplete")
    values = diagnostics[columns].to_numpy(float)
    passed = bool(np.isfinite(values).all() and (diagnostics.r_hat <= 1.01).all()
                  and (diagnostics.ess_bulk >= 400).all() and (diagnostics.ess_tail >= 400).all())
    if type(status.get("sampling_diagnostics_pass")) is not bool or status["sampling_diagnostics_pass"] != passed:
        raise ValueError("Checkpoint diagnostic pass flag disagrees with diagnostic values")
    summaries = {"max_rhat": diagnostics.r_hat.max(),
                 "min_ess_bulk": diagnostics.ess_bulk.min(),
                 "min_ess_tail": diagnostics.ess_tail.min()}
    for name, value in summaries.items():
        if name not in status or not np.isclose(status[name], value, rtol=0, atol=0, equal_nan=True):
            raise ValueError(f"Checkpoint diagnostic summary disagrees with diagnostics: {name}")
    return votes, status


def seal_checkpoint(directory, expected, fit_config):
    """Validate a finished fit, then atomically mark all its artifacts reusable."""
    directory = Path(directory)
    records = {name: _record(directory / name) for name in ARTIFACTS}
    votes, status = _read_fit(directory, expected, fit_config)
    metadata = {"schema_version": SCHEMA_VERSION, "season": status["season"],
                "fit_config": fit_config, "files": records}
    target = directory / CHECKPOINT_NAME
    temporary = directory / f".{CHECKPOINT_NAME}.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return votes, status


def load_checkpoint(directory, expected, fit_config):
    """Read an intact checkpoint without modifying any saved fit or run record."""
    directory = Path(directory)
    metadata = json.loads((directory / CHECKPOINT_NAME).read_text())
    seasons = expected.season.unique()
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("files"), dict)
            or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != SCHEMA_VERSION or len(seasons) != 1
            or type(metadata.get("season")) is not int or metadata["season"] != int(seasons[0])
            or metadata.get("fit_config") != fit_config or set(metadata.get("files", {})) != set(ARTIFACTS)):
        raise ValueError("Checkpoint schema, season, fit configuration or artifact inventory disagrees")
    for name in ARTIFACTS:
        if _record(directory / name) != metadata["files"][name]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return _read_fit(directory, expected, fit_config)
