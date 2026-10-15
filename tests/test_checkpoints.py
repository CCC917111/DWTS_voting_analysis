"""A resumed fit must preserve the completed fit and its input identity."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from dwts.checkpoints import load_checkpoint, seal_checkpoint


ARTIFACTS = (
    "votes.csv", "fit_status.json", "diagnostics.csv", "posterior.nc",
    "posterior_eliminations.csv",
)


@pytest.fixture
def checkpoint(tmp_path):
    expected = pd.DataFrame({
        "season": [3, 3, 3, 3],
        "week": [1, 1, 2, 2],
        "contestant_id": ["A", "B", "A", "B"],
        "judge_share": [0.2727272727272727, 0.7272727272727273, 0.4, 0.6],
        "judge_z": [-0.12345678901234566, 0.12345678901234566, -1.0, 1.0],
        "eliminated": [False, False, True, False],
    })
    votes = expected.assign(
        fan_mean=[0.4, 0.6, 0.3, 0.7],
        fan_lower=[0.2, 0.4, 0.1, 0.5],
        fan_upper=[0.6, 0.8, 0.5, 0.9],
    )
    config = {"sampler": "fast_mh", "draws": 8, "tune": 4,
              "chains": 4, "seed": 1729}
    diagnostics = pd.DataFrame({
        "parameter": ["alpha[0]", "sigma"], "r_hat": [1.005, 1.001],
        "ess_bulk": [800.0, 900.0], "ess_tail": [600.0, 700.0],
    })
    status = {
        "season": 3, "sampling_diagnostics_pass": True,
        "max_rhat": 1.005, "min_ess_bulk": 800.0,
        "min_ess_tail": 600.0, "config": config,
    }
    directory = tmp_path / "season_03"
    directory.mkdir()
    votes.to_csv(directory / "votes.csv", index=False)
    diagnostics.to_csv(directory / "diagnostics.csv", index=False)
    (directory / "fit_status.json").write_text(json.dumps(status), encoding="utf-8")
    # Unit fixtures test byte integrity; the CLI integration writes real NetCDF.
    (directory / "posterior.nc").write_bytes(b"synthetic posterior bytes\x00\x01")
    pd.DataFrame({"season": [3], "week": [2], "probability": [0.8]}).to_csv(
        directory / "posterior_eliminations.csv", index=False)
    return directory, expected, config


def _rewrite_status(directory, **changes):
    path = directory / "fit_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    status.update(changes)
    path.write_text(json.dumps(status), encoding="utf-8")


def test_seal_and_load_preserve_input_floats_and_artifact_identity(checkpoint):
    directory, expected, config = checkpoint
    sealed_votes, sealed_status = seal_checkpoint(directory, expected, config)
    votes, status = load_checkpoint(directory, expected, config)

    pd.testing.assert_frame_equal(votes[expected.columns], expected, check_exact=True)
    pd.testing.assert_frame_equal(votes, sealed_votes, check_exact=True)
    assert status == sealed_status
    assert status["sampling_diagnostics_pass"] is True
    metadata = json.loads((directory / "checkpoint.json").read_text())
    assert metadata["schema_version"] == 1
    assert metadata["season"] == 3
    assert metadata["fit_config"] == config
    assert set(metadata["files"]) == set(ARTIFACTS)
    for name, record in metadata["files"].items():
        contents = (directory / name).read_bytes()
        assert record == {"bytes": len(contents),
                          "sha256": hashlib.sha256(contents).hexdigest()}


@pytest.mark.parametrize("name", ARTIFACTS)
def test_changed_artifact_is_rejected_without_rewriting_it(checkpoint, name):
    directory, expected, config = checkpoint
    seal_checkpoint(directory, expected, config)
    path = directory / name
    path.write_bytes(path.read_bytes() + b"\n")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError):
        load_checkpoint(directory, expected, config)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("name", ARTIFACTS)
def test_missing_artifact_cannot_be_reused(checkpoint, name):
    directory, expected, config = checkpoint
    seal_checkpoint(directory, expected, config)
    (directory / name).unlink()
    with pytest.raises(ValueError):
        load_checkpoint(directory, expected, config)


@pytest.mark.parametrize("corruption", [
    "wrong_season", "duplicate_row", "missing_row", "input_changed",
    "non_normalized", "nan_mean", "out_of_bounds", "reversed_interval",
])
def test_sealing_rejects_invalid_votes_before_publishing_metadata(checkpoint, corruption):
    directory, expected, config = checkpoint
    path = directory / "votes.csv"
    votes = pd.read_csv(path, float_precision="round_trip")
    if corruption == "wrong_season":
        votes["season"] = 4
    elif corruption == "duplicate_row":
        votes = pd.concat([votes, votes.iloc[[0]]], ignore_index=True)
    elif corruption == "missing_row":
        votes = votes.iloc[:-1]
    elif corruption == "input_changed":
        votes.loc[0, "judge_z"] += 0.01
    elif corruption == "non_normalized":
        votes.loc[0, "fan_mean"] = 0.5
    elif corruption == "nan_mean":
        votes.loc[0, "fan_mean"] = np.nan
    elif corruption == "out_of_bounds":
        votes.loc[0, "fan_lower"] = -0.01
    else:
        votes.loc[0, ["fan_lower", "fan_upper"]] = [0.9, 0.1]
    votes.to_csv(path, index=False)

    with pytest.raises(ValueError):
        seal_checkpoint(directory, expected, config)
    assert not (directory / "checkpoint.json").exists()


@pytest.mark.parametrize("changes", [
    {"season": 4}, {"season": "3"}, {"sampling_diagnostics_pass": False},
    {"sampling_diagnostics_pass": 1}, {"max_rhat": 1.0},
    {"min_ess_bulk": 1000.0}, {"min_ess_tail": 1000.0},
    {"config": {"sampler": "fast_mh", "seed": 999}},
])
def test_sealing_rejects_status_that_disagrees_with_fit(checkpoint, changes):
    directory, expected, config = checkpoint
    _rewrite_status(directory, **changes)
    with pytest.raises(ValueError):
        seal_checkpoint(directory, expected, config)
    assert not (directory / "checkpoint.json").exists()


def test_diagnostic_gate_is_derived_instead_of_trusting_pass_flag(checkpoint):
    directory, expected, config = checkpoint
    path = directory / "diagnostics.csv"
    diagnostics = pd.read_csv(path)
    diagnostics.loc[0, "r_hat"] = 1.1
    diagnostics.to_csv(path, index=False)
    _rewrite_status(directory, max_rhat=1.1)
    with pytest.raises(ValueError):
        seal_checkpoint(directory, expected, config)

    _rewrite_status(directory, sampling_diagnostics_pass=False)
    seal_checkpoint(directory, expected, config)
    _, status = load_checkpoint(directory, expected, config)
    assert status["sampling_diagnostics_pass"] is False


def test_honest_undefined_diagnostics_remain_provisional_on_resume(checkpoint):
    directory, expected, config = checkpoint
    path = directory / "diagnostics.csv"
    diagnostics = pd.read_csv(path)
    diagnostics[["r_hat", "ess_bulk", "ess_tail"]] = np.nan
    diagnostics.to_csv(path, index=False)
    _rewrite_status(directory, sampling_diagnostics_pass=False,
                    max_rhat=float("nan"), min_ess_bulk=float("nan"),
                    min_ess_tail=float("nan"))

    seal_checkpoint(directory, expected, config)
    votes, status = load_checkpoint(directory, expected, config)
    pd.testing.assert_frame_equal(votes[expected.columns], expected, check_exact=True)
    assert status["sampling_diagnostics_pass"] is False
    assert all(np.isnan(status[name]) for name in (
        "max_rhat", "min_ess_bulk", "min_ess_tail"))


@pytest.mark.parametrize("corruption", ["absent", "null", "duplicate"])
def test_diagnostic_parameters_must_be_identifiable(checkpoint, corruption):
    directory, expected, config = checkpoint
    path = directory / "diagnostics.csv"
    diagnostics = pd.read_csv(path)
    if corruption == "absent":
        diagnostics = diagnostics.drop(columns="parameter")
    elif corruption == "null":
        diagnostics.loc[0, "parameter"] = None
    else:
        diagnostics["parameter"] = "alpha[0]"
    diagnostics.to_csv(path, index=False)
    with pytest.raises(ValueError):
        seal_checkpoint(directory, expected, config)


@pytest.mark.parametrize("schema_version", [True, "1", 1.0, 2])
def test_checkpoint_schema_version_has_strict_integer_type(checkpoint, schema_version):
    directory, expected, config = checkpoint
    seal_checkpoint(directory, expected, config)
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text())
    metadata["schema_version"] = schema_version
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        load_checkpoint(directory, expected, config)


@pytest.mark.parametrize("change", ["config", "input"])
def test_checkpoint_must_belong_to_requested_fit(checkpoint, change):
    directory, expected, config = checkpoint
    seal_checkpoint(directory, expected, config)
    if change == "config":
        config = {**config, "seed": config["seed"] + 1}
    else:
        expected = expected.copy()
        expected.loc[0, "judge_z"] += 0.01
    with pytest.raises(ValueError):
        load_checkpoint(directory, expected, config)
