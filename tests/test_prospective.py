"""Separated-clock, restart and publication-boundary checks for the draft runner."""
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from dwts import prospective as p
from dwts.forecasting import MODELS, forecast_season


def protocol():
    cfg = p.default_protocol()
    cfg["model"].update(particles=64, seeds=[71], mh_steps=2, ess_fraction=.99)
    cfg["baselines"]["history_strength"] = .8
    return cfg


def metadata(*, final=False, scheduled=True):
    return dict(scheduled_single_exit=scheduled, is_final=final, known_withdrawal=False,
                scores_complete=True, scores_available_at="2000-01-01T20:00:00Z",
                earliest_outcome_at="2000-01-01T21:00:00Z", source_sha256="0"*64)


def predict(path, sequence=1, ids=None, judges=None, **kwargs):
    return p.predict_event(path, event_id=f"event-{sequence}", sequence=sequence,
                           roster=ids or ["A", "B", "C", "D"],
                           scores=judges or [8., 7., 6., 5.], event_metadata=metadata(**kwargs))


def release(path, sequence=1, exits=None, withdrawals=None):
    return p.release_event(path, event_id=f"event-{sequence}", exits=exits or ["D"],
                           withdrawals=withdrawals or [], disclosed_at="2000-01-01T21:00:00Z",
                           source_sha256="1"*64)


def fixture_receipt(header, *, late=False):
    return dict(adapter="fixture_unverified", payload_sha256=header["payload_sha256"],
                received_at="2000-01-01T21:00:00Z" if late else "2000-01-01T20:30:00Z")


@pytest.fixture
def state(tmp_path):
    target = tmp_path/"private"
    p.initialize(target, season=7, protocol=protocol())
    return target


def test_separate_clock_matches_existing_batch_with_new_entrant_and_unmodeled_event(state):
    events = [(["D", "B", "A", "C"], [5., 7., 8., 6.], ["D"], False),
              (["A", "B", "C"], [8., 5., 6.], [], False),
              (["E", "A", "C", "B"], [5., 7., 6., 9.], ["C"], False),
              (["A", "B", "E"], [6., 8., 9.], ["B"], True)]
    rows, actual = [], []
    for week, (ids, judges, exits, final) in enumerate(events, 1):
        predict(state, week, ids, judges, final=final)
        frozen = p._load(state)["pending"]["payload"]
        actual.append(frozen)
        p.release_event(state, event_id=f"event-{week}", exits=exits, withdrawals=[],
                        disclosed_at="2000-01-01T21:00:00Z", source_sha256="1"*64)
        rows.extend(dict(season=7, week=week, contestant_id=x, judge_mean=j,
                         eliminated=x in exits, withdrawal=False, is_final_week=final)
                    for x, j in zip(ids, judges))
    cfg = protocol()
    settings = {k: v for k, v in cfg["model"].items() if k not in {"seeds", "rule"}}
    expected, _ = forecast_season(pd.DataFrame(rows), **settings, seed=71, **cfg["baselines"])
    for week, frozen in enumerate(actual, 1):
        for model in MODELS:
            observed = expected[expected.week.eq(week) & expected.model.eq(model)]
            assert observed.contestant_id.tolist() == frozen["roster"]
            np.testing.assert_array_equal(observed.probability.to_numpy(),
                                          frozen["probabilities"][model])
    saved = p._load(state)
    assert saved["known"] == {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    assert len(saved["posteriors"][0]["history"]) == 3  # final still updates
    assert p.export_aggregate(state)["coverage"]["scorable_exploratory_events"] == 2


def test_prediction_interface_has_no_outcome_and_forbids_hidden_metadata(state):
    assert not {"outcome", "exits", "withdrawals", "placement"} & set(inspect.signature(p.predict_event).parameters)
    with pytest.raises(TypeError):
        p.predict_event(state, event_id="event-1", sequence=1, roster=["A", "B", "C"],
                        scores=[1., 2., 3.], event_metadata=metadata(), outcome="A")
    bad = metadata()
    bad["eliminated"] = "A"
    with pytest.raises(ValueError, match="outcomes are forbidden"):
        p.predict_event(state, event_id="event-1", sequence=1, roster=["A", "B", "C"],
                        scores=[1., 2., 3.], event_metadata=bad)


def test_process_reload_retains_rng_and_exact_inference_state(state, tmp_path):
    header = predict(state)
    p.attach_receipt(state, fixture_receipt(header))
    clone = tmp_path/"clone"
    shutil.copytree(state, clone)
    request = tmp_path/"release.json"
    request.write_text(json.dumps(dict(event_id="event-1", exits=["D"], withdrawals=[],
        disclosed_at="2000-01-01T21:00:00Z", source_sha256="1"*64)))
    subprocess.run([sys.executable, "-m", "dwts.prospective", "release", "--state", str(clone),
                    "--input", str(request)], check=True, capture_output=True, text=True)
    release(state)
    predict(state, 2, ["A", "B", "C", "E"], [8., 6., 7., 5.])
    predict(clone, 2, ["A", "B", "C", "E"], [8., 6., 7., 5.])
    left, right = p._load(state), p._load(clone)
    assert left["posteriors"] == right["posteriors"]
    assert left["pending"]["payload"]["probabilities"] == right["pending"]["payload"]["probabilities"]


@pytest.mark.parametrize("late", [False, True])
def test_fixture_receipts_never_count_as_confirmatory(state, late):
    header = predict(state)
    p.attach_receipt(state, fixture_receipt(header, late=late))
    result = release(state)
    assert result["receipt_status"] == ("late_fixture_unverified" if late else "fixture_unverified")
    assert result["confirmatory"] is False
    exported = p.export_aggregate(state)
    assert exported["coverage"]["confirmatory_events"] == 0
    assert all(m["events"] == 1 for m in exported["metrics"])


def test_missing_receipt_retained_and_same_denominator_for_all_models(state):
    predict(state)
    assert release(state)["receipt_status"] == "missing"
    p.record_unpredicted_event(state, event_id="event-2", sequence=2, status="missed_deadline")
    predict(state, 3, ["A", "B", "C"], [6., 7., 8.])
    p.release_event(state, event_id="event-3", exits=["C"], withdrawals=["B"],
                    disclosed_at="2000-01-01T21:00:00Z", source_sha256="1"*64)
    exported = p.export_aggregate(state)
    assert exported["coverage"]["registered_events"] == 3
    assert exported["coverage"]["scorable_exploratory_events"] == 1
    assert exported["coverage"]["predeadline_eligible_events"] == 2
    assert exported["coverage"]["unknown_predeadline_eligibility_events"] == 1
    assert exported["coverage"]["undefined_reason_counts"] == {"withdrawal": 1}
    assert {row["events"] for row in exported["metrics"]} == {1}
    assert len(p._load(state)["posteriors"][0]["history"]) == 1


def test_duplicate_prediction_receipt_and_release_are_refused(state):
    header = predict(state)
    with pytest.raises(ValueError, match="pending"):
        predict(state)
    p.attach_receipt(state, fixture_receipt(header))
    with pytest.raises(ValueError, match="duplicate receipt"):
        p.attach_receipt(state, fixture_receipt(header))
    release(state)
    with pytest.raises(ValueError, match="duplicate update"):
        release(state)
    with pytest.raises(ValueError, match="duplicate prediction"):
        predict(state, 1)
    assert len(p._load(state)["posteriors"][0]["history"]) == 1


def test_wrong_hash_and_fake_verified_receipt_refused_without_mutation(state):
    header = predict(state)
    before = (state/"private_state.json").read_bytes()
    bad = fixture_receipt(header)
    bad["payload_sha256"] = "f"*64
    with pytest.raises(ValueError, match="not bound"):
        p.attach_receipt(state, bad)
    bad = fixture_receipt(header)
    bad["verified"] = True
    with pytest.raises(ValueError, match="unverified fixture"):
        p.attach_receipt(state, bad)
    assert (state/"private_state.json").read_bytes() == before


def test_tampering_is_detected_before_any_update(state):
    predict(state)
    path = state/"private_state.json"
    text = path.read_text()
    path.write_text(text.replace('"event-1"', '"event-X"', 1))
    with pytest.raises(ValueError, match="integrity"):
        release(state)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.])
def test_unsafe_nonfinite_inputs_leave_snapshot_unchanged(state, score):
    before = (state/"private_state.json").read_bytes()
    with pytest.raises(ValueError):
        predict(state, judges=[8., 7., score, 5.])
    assert (state/"private_state.json").read_bytes() == before


def test_active_mode_and_rule_mismatch_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="Activation blocked"):
        p.initialize(tmp_path/"live", season=35, mode="active")
    for field, value in [("judge_save", True), ("carryover", True), ("rule", "rank")]:
        cfg = protocol()
        cfg["required_rule"][field] = value
        with pytest.raises(ValueError, match="rule"):
            p.initialize(tmp_path/field, season=7, protocol=cfg)


def test_crash_before_atomic_replace_can_retry_without_duplicate_observe(state, monkeypatch):
    predict(state)
    original = p.os.replace
    before = (state/"private_state.json").read_bytes()
    def fail(*args):
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(p.os, "replace", fail)
    with pytest.raises(OSError, match="interrupted"):
        release(state)
    assert (state/"private_state.json").read_bytes() == before
    monkeypatch.setattr(p.os, "replace", original)
    release(state)
    assert len(p._load(state)["posteriors"][0]["history"]) == 1
    assert not list(state.glob(".snapshot-*"))


def test_crash_after_atomic_replace_cannot_repeat_observe(state, monkeypatch):
    predict(state)
    original = p._atomic_write
    def fail_after_write(*args):
        original(*args)
        raise OSError("simulated lost acknowledgement")
    monkeypatch.setattr(p, "_atomic_write", fail_after_write)
    with pytest.raises(OSError, match="acknowledgement"):
        release(state)
    monkeypatch.setattr(p, "_atomic_write", original)
    with pytest.raises(ValueError, match="duplicate update"):
        release(state)
    assert len(p._load(state)["posteriors"][0]["history"]) == 1


def test_writer_lock_prevents_concurrent_transactions(state):
    with (state/".writer.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="writer lock"):
            predict(state)


def test_oversized_new_snapshot_is_refused_before_replacing_usable_state(state, monkeypatch):
    before = (state/"private_state.json").read_bytes()
    monkeypatch.setattr(p, "MAX_SNAPSHOT_BYTES", len(before)+1)
    with pytest.raises(ValueError, match="previous state retained"):
        predict(state)
    assert (state/"private_state.json").read_bytes() == before
    assert p._load(state)["pending"] is None


def test_export_is_aggregate_allowlist_and_pending_event_counts(state):
    predict(state)
    result = p.export_aggregate(state)
    assert result["coverage"]["registered_events"] == 1
    assert result["coverage"]["pending_events"] == 1
    assert set(result) == {"schema_version", "study_status", "interpretation", "protocol_sha256",
                           "coverage", "metrics", "primary_paired_log_loss_forward_minus_past_judges"}
    serialized = json.dumps(result)
    for forbidden in ('"roster"', '"scores"', '"particles"', '"probabilities"', '"nonce_hex"',
                      '"event_id"', '"payload"', '"support"', '"rng"'):
        assert forbidden not in serialized
    assert os.stat(state/"private_state.json").st_mode & 0o077 == 0


def test_unknown_event_sequence_and_external_roster_result_refused(state):
    with pytest.raises(ValueError, match="consecutive"):
        predict(state, sequence=2)
    predict(state)
    with pytest.raises(ValueError, match="outside"):
        release(state, exits=["unknown"])
    assert len(p._load(state)["posteriors"][0]["history"]) == 0


def test_unknown_schedule_never_promoted_to_predeadline_eligible_after_release(state):
    predict(state, scheduled=None)
    result = p.export_aggregate(state)
    assert result["coverage"]["predeadline_eligible_events"] == 0
    assert result["coverage"]["unknown_predeadline_eligibility_events"] == 1
    release(state)
    result = p.export_aggregate(state)
    assert result["coverage"]["scorable_exploratory_events"] == 1
    assert result["coverage"]["predeadline_eligible_events"] == 0
    assert result["coverage"]["unknown_predeadline_eligibility_events"] == 1


def test_synthetic_demo_and_cli_export(tmp_path):
    state = tmp_path/"demo"
    result = p.demo(state)
    assert result["coverage"]["registered_events"] == 3
    assert result["coverage"]["scorable_exploratory_events"] == 2
    target = tmp_path/"aggregate.json"
    completed = subprocess.run([sys.executable, "-m", "dwts.prospective", "export", "--state",
                               str(state), "--out", str(target)], check=True,
                              capture_output=True, text=True)
    assert json.loads(completed.stdout) == json.loads(target.read_text()) == result


def test_repository_config_matches_implemented_default():
    config = Path(__file__).parents[1]/"configs"/"prospective_protocol.json"
    assert json.loads(config.read_text()) == p.default_protocol()
