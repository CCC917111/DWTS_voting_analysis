"""A study run is independent of optional comparisons to a reference report."""

import hashlib
import json
import sys
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
    votes = panel.assign(fan_mean=[0.45, 0.55])
    events = []
    state = {"diagnostics_pass": True}
    monkeypatch.setattr(cli, "prepare", lambda *a, **k: (
        panel.copy(), pd.DataFrame({"check": ["synthetic fixture"]})))
    monkeypatch.setattr(cli, "data_summary", lambda frame: {
        "seasons": int(frame.season.nunique()), "contestant_weeks": len(frame)})

    def fit(frame, cfg, out):
        pd.testing.assert_frame_equal(frame, panel)
        assert cfg.sampler == "fast_mh"
        events.append("fit")
        status = {
            "season": 3, "sampling_diagnostics_pass": state["diagnostics_pass"],
            "max_rhat": 1.01 if state["diagnostics_pass"] else 1.4,
            "min_ess_bulk": 500 if state["diagnostics_pass"] else 4,
            "min_ess_tail": 500 if state["diagnostics_pass"] else 3,
        }
        out.mkdir(parents=True)
        cli.dump_json(out / "fit_status.json", status)
        votes.to_csv(out / "votes.csv", index=False)
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


@pytest.mark.parametrize("flag, expected", [([], False), (["--compare-reference"], True)])
def test_run_command_parses_reference_comparison_as_opt_in(monkeypatch, flag, expected):
    received = []
    monkeypatch.setattr(cli, "run", lambda args: received.append(args))
    monkeypatch.setattr(sys, "argv", ["dwts", "run", *flag])
    cli.main()
    assert len(received) == 1
    assert received[0].compare_reference is expected
