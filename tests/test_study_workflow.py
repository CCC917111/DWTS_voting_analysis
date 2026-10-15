"""A study run is independent of optional comparisons to a reference report."""

import hashlib
import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest

from dwts import cli


@pytest.fixture
def study_run(monkeypatch, tmp_path):
    """Keep the run orchestration real while replacing expensive model fitting."""
    import dwts.plots as plots
    import dwts.reconcile as reconcile

    data = tmp_path / "synthetic.csv"
    data.write_text("synthetic input for the orchestration test\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "seasons": [3],
        "fit": {"sampler": "fast_mh", "draws": 4, "tune": 0, "chains": 2},
    }), encoding="utf-8")
    panel = pd.DataFrame({
        "season": [3, 3], "week": [1, 1], "contestant_id": ["B", "A"],
        "judge_share": [0.4, 0.6], "judge_z": [-1.0, 1.0],
    })
    votes = panel.assign(fan_mean=[0.45, 0.55], fan_lower=[0.3, 0.4], fan_upper=[0.6, 0.7])
    events = []
    state = {"diagnostics_pass": True}
    monkeypatch.setattr(cli, "prepare", lambda *a, **k: (
        panel.copy(), pd.DataFrame({"check": ["synthetic fixture"]})))
    monkeypatch.setattr(cli, "data_summary", lambda frame: {
        "seasons": int(frame.season.nunique()), "contestant_weeks": len(frame)})
    monkeypatch.setattr(cli, "code_digest", lambda: "synthetic-orchestration-source")

    def fit(frame, cfg, out):
        pd.testing.assert_frame_equal(frame, panel)
        assert cfg.sampler == "fast_mh"
        events.append("fit")
        status = {
            "season": 3, "sampling_diagnostics_pass": state["diagnostics_pass"],
            "max_rhat": 1.01 if state["diagnostics_pass"] else 1.4,
            "min_ess_bulk": 500 if state["diagnostics_pass"] else 4,
            "min_ess_tail": 500 if state["diagnostics_pass"] else 3,
            "config": asdict(cfg),
        }
        out.mkdir(parents=True)
        cli.dump_json(out / "fit_status.json", status)
        votes.to_csv(out / "votes.csv", index=False)
        pd.DataFrame({"parameter": ["beta"], "r_hat": [status["max_rhat"]],
                      "ess_bulk": [status["min_ess_bulk"]],
                      "ess_tail": [status["min_ess_tail"]]}).to_csv(out / "diagnostics.csv", index=False)
        (out / "posterior.nc").write_bytes(b"synthetic orchestration placeholder")
        (out / "posterior_eliminations.csv").write_text("synthetic orchestration placeholder\n")
        return votes.copy(), status

    def analyze(actual_votes, out, config, seed, skip_attribution):
        expected = votes.sort_values("contestant_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_votes, expected)
        assert skip_attribution is True
        events.append("analysis")
        actual_votes.groupby(["season", "week"]).fan_mean.sum().rename(
            "fan_share_sum").to_csv(out / "scientific_summary.csv")

    def comparison(actual_votes, out, statuses):
        assert len(actual_votes) == 2 and statuses[0]["season"] == 3
        events.append("comparison")
        (out / "reference_comparison.csv").write_text("comparison requested\n")

    def study_figures(out, statuses):
        assert statuses[0]["season"] == 3
        events.append("study_figures")
        (out / "sampling_diagnostics.svg").write_text("<svg/>")

    def legacy_report_figures(*args, **kwargs):
        pytest.fail("The study workflow should use the neutral figure entry point")

    monkeypatch.setattr(cli, "_fit_job", fit)
    monkeypatch.setattr(cli, "analyze", analyze)
    monkeypatch.setattr(reconcile, "write_comparison", comparison)
    monkeypatch.setattr(plots, "make_study_figures", study_figures, raising=False)
    monkeypatch.setattr(plots, "make_report_figures", legacy_report_figures)
    args = SimpleNamespace(
        config=config, data=data, out=tmp_path / "results", seasons=None,
        resume=False, skip_attribution=True,
    )
    return args, state, events, votes


@pytest.mark.parametrize("explicit_default", [False, True])
def test_default_study_run_writes_results_without_reference_comparison(study_run, explicit_default):
    args, _, events, expected_votes = study_run
    if explicit_default:
        args.compare_reference = False
    cli.run(args)

    assert events == ["fit", "analysis", "study_figures"]
    saved_votes = pd.read_csv(args.out / "votes.csv")
    pd.testing.assert_frame_equal(
        saved_votes, expected_votes.sort_values("contestant_id").reset_index(drop=True))
    summary = pd.read_csv(args.out / "scientific_summary.csv")
    assert summary.fan_share_sum.tolist() == pytest.approx([1.0])
    assert not (args.out / "reference_comparison.csv").exists()
    assert (args.out / "sampling_diagnostics.svg").is_file()
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["reference_comparison_requested"] is False
    assert manifest["all_sampling_diagnostics_pass"] is True
    assert "source_report" not in manifest
    assert manifest["input_sha256"] == hashlib.sha256(args.data.read_bytes()).hexdigest()
    assert json.loads((args.out / "sampling_status.json").read_text())[0][
        "sampling_diagnostics_pass"] is True


def test_reference_comparison_is_explicit_and_runs_after_scientific_analysis(study_run):
    args, _, events, _ = study_run
    args.compare_reference = True
    cli.run(args)

    assert events[:2] == ["fit", "analysis"]
    assert sorted(events[2:]) == ["comparison", "study_figures"]
    assert (args.out / "scientific_summary.csv").is_file()
    assert (args.out / "reference_comparison.csv").is_file()
    assert (args.out / "sampling_diagnostics.svg").is_file()
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["reference_comparison_requested"] is True


def test_completed_execution_preserves_failed_sampling_diagnostics(study_run):
    args, state, events, _ = study_run
    state["diagnostics_pass"] = False
    cli.run(args)

    assert events == ["fit", "analysis", "study_figures"]
    assert (args.out / "votes.csv").is_file()
    assert (args.out / "scientific_summary.csv").is_file()
    manifest = json.loads((args.out / "manifest.json").read_text())
    statuses = json.loads((args.out / "sampling_status.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["all_sampling_diagnostics_pass"] is False
    assert statuses[0]["sampling_diagnostics_pass"] is False
    assert statuses[0]["max_rhat"] == 1.4
    assert manifest["attribution_ran"] is False


def test_resume_rejects_changed_comparison_mode_without_mutating_prior_results(study_run):
    args, _, events, _ = study_run
    args.compare_reference = True
    cli.run(args)
    previous_files = {
        path.relative_to(args.out): path.read_bytes()
        for path in args.out.rglob("*") if path.is_file()
    }
    assert "comparison" in events
    events.clear()

    args.resume = True
    args.compare_reference = False
    with pytest.raises(ValueError, match="Resume refused"):
        cli.run(args)

    assert events == []
    assert {
        path.relative_to(args.out): path.read_bytes()
        for path in args.out.rglob("*") if path.is_file()
    } == previous_files
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert manifest["reference_comparison_requested"] is True
    assert (args.out / "reference_comparison.csv").is_file()


def _saved_files(directory):
    return {path.relative_to(directory): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("change", ["skip_attribution", "environment", "config", "input", "code", "checkpoint", "legacy"])
def test_resume_validation_rejects_changes_before_any_output_write(study_run, monkeypatch, change):
    args, _, events, _ = study_run
    cli.run(args)
    if change == "skip_attribution":
        args.skip_attribution = False
    elif change == "environment":
        environment = cli.environment_identity()
        environment["packages"]["numpy"] = "different-version"
        monkeypatch.setattr(cli, "environment_identity", lambda: environment)
    elif change == "config":
        config = json.loads(args.config.read_text())
        config["fit"]["draws"] = 8
        args.config.write_text(json.dumps(config))
    elif change == "input":
        args.data.write_text("changed input bytes\n")
    elif change == "code":
        monkeypatch.setattr(cli, "code_digest", lambda: "changed-code")
    elif change == "checkpoint":
        checkpoint_votes = args.out / "season_03" / "votes.csv"
        checkpoint_votes.write_text(checkpoint_votes.read_text() + "\n")
    elif change == "legacy":
        manifest = json.loads((args.out / "manifest.json").read_text())
        manifest.pop("resume_schema_version")
        (args.out / "manifest.json").write_text(json.dumps(manifest))
    before = _saved_files(args.out)
    events.clear()
    args.resume = True
    with pytest.raises(ValueError, match="Resume refused"):
        cli.run(args)
    assert events == []
    assert _saved_files(args.out) == before


def test_all_cached_resume_rebuilds_diagnostics_and_preserves_original_identity(study_run):
    args, _, events, _ = study_run
    cli.run(args)
    original_manifest = json.loads((args.out / "manifest.json").read_text())
    original_votes = (args.out / "votes.csv").read_bytes()
    (args.out / "sampling_status.json").unlink()
    events.clear()
    args.resume = True
    cli.run(args)
    assert events == ["analysis", "study_figures"]
    assert (args.out / "votes.csv").read_bytes() == original_votes
    statuses = json.loads((args.out / "sampling_status.json").read_text())
    assert [status["season"] for status in statuses] == [3]
    manifest = json.loads((args.out / "manifest.json").read_text())
    for key in ["started_utc", "python", "python_implementation", "packages", "run_options"]:
        assert manifest[key] == original_manifest[key]
    assert manifest["resume_count"] == 1 and manifest["status"] == "completed"


def test_real_synthetic_cli_recovers_fit_saved_before_parent_records_completion(tmp_path, monkeypatch):
    """Real tiny MCMC, NetCDF, analysis and plotting; this tests execution, not convergence."""
    data = tmp_path / "synthetic.csv"
    pd.DataFrame([
        dict(celebrity_name=name, ballroom_partner=f"P{index}", celebrity_industry="actor/actress",
             season=3, results="Eliminated Week 1" if index == 0 else f"{index} Place",
             placement=4 if index == 0 else index, celebrity_age_during_season=20+index*5,
             week1_judge1_score=score, week2_judge1_score=0 if index == 0 else score+1)
        for index, (name, score) in enumerate(zip("ABCD", [4., 6., 7., 9.]))
    ]).to_csv(data, index=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(seasons=[3], fit=dict(
        sampler="fast_mh", draws=4, tune=0, chains=2, seed=914))))
    out = tmp_path / "results"
    command = ["dwts", "run", "--data", str(data), "--config", str(config),
               "--out", str(out), "--skip-attribution"]
    original_fit = cli._checkpointed_fit_job

    class InterruptedAfterFit(RuntimeError):
        pass

    def interrupted(*args):
        original_fit(*args)
        raise InterruptedAfterFit("parent interrupted after the worker committed its checkpoint")

    monkeypatch.setattr(cli, "_checkpointed_fit_job", interrupted)
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(InterruptedAfterFit):
        cli.main()
    assert (out / "season_03" / "checkpoint.json").is_file()
    assert json.loads((out / "sampling_status.json").read_text()) == []
    checkpoint_votes = pd.read_csv(out / "season_03" / "votes.csv", float_precision="round_trip")
    before_resume = json.loads((out / "manifest.json").read_text())
    (out / "sampling_status.json").unlink()

    def no_refit(*args):
        pytest.fail("A complete, sealed checkpoint should be reused without another fit")

    monkeypatch.setattr(cli, "_checkpointed_fit_job", no_refit)
    monkeypatch.setattr(sys, "argv", command + ["--resume"])
    cli.main()
    saved_votes = pd.read_csv(out / "votes.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(saved_votes, checkpoint_votes, check_exact=True)
    statuses = json.loads((out / "sampling_status.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(statuses) == 1 and statuses[0]["season"] == 3
    assert statuses[0]["sampling_diagnostics_pass"] is False
    assert manifest["all_sampling_diagnostics_pass"] is False
    assert manifest["status"] == "completed"
    assert manifest["started_utc"] == before_resume["started_utc"]
    assert manifest["packages"] == before_resume["packages"]
    assert (out / "figures" / "sampling_diagnostics.png").is_file()
    assert (out / "accuracy_summary.csv").is_file()


@pytest.mark.parametrize("flag, expected", [([], False), (["--compare-reference"], True)])
def test_run_command_parses_reference_comparison_as_opt_in(monkeypatch, flag, expected):
    received = []
    monkeypatch.setattr(cli, "run", lambda args: received.append(args))
    monkeypatch.setattr(sys, "argv", ["dwts", "run", *flag])
    cli.main()
    assert len(received) == 1
    assert received[0].compare_reference is expected
