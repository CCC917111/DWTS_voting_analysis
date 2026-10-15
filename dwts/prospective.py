"""Private, resumable rehearsal of a prospective prediction/release clock.

This runner has no trusted timestamp adapter and cannot activate a confirmatory
study. Its explicit JSON snapshots contain private inputs, predictions and SMC
particles; only ``export_aggregate`` and commitment headers may be published.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata as import_metadata
import json
import os
from pathlib import Path
import platform
import re
import secrets
import tempfile

import numpy as np
from scipy.special import softmax

from .forward_model import Observation, SequentialPosterior, exit_log_probs
from .forecasting import MODELS, _judge_hazard

SCHEMA = 1
MAX_SNAPSHOT_BYTES = 128*1024*1024
INTERPRETATION = (
    "Local draft rehearsal only: fixture receipts are unverified; no independent "
    "prospective validation, true-vote validation or confirmatory events."
)
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SETTINGS = ("prior_sd", "temperature", "rule", "judge_weight", "particles",
             "mh_steps", "proposal_scale", "ess_fraction")


def _pack(value):
    """Explicit float64 hex encoding; not an RFC 8785 implementation."""
    if isinstance(value, np.ndarray):
        return _pack(value.tolist())
    if isinstance(value, np.generic):
        return _pack(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Nonfinite numeric values are forbidden")
        return {"float64_hex": value.hex()}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("JSON object keys must be strings")
        return {k: _pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(v) for v in value]
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError(f"Unsupported JSON value: {type(value).__name__}")


def _unpack(value):
    if isinstance(value, dict):
        if set(value) == {"float64_hex"}:
            if not isinstance(value["float64_hex"], str):
                raise ValueError("Invalid encoded float")
            result = float.fromhex(value["float64_hex"])
            if not np.isfinite(result):
                raise ValueError("Nonfinite encoded float")
            return result
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    return value


def canonical_bytes(value):
    return (json.dumps(_pack(value), ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")


def _hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def _time(value):
    if not isinstance(value, str):
        raise ValueError("Timestamps must be timezone-aware ISO 8601 strings")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid ISO 8601 timestamp") from error
    if stamp.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return stamp.astimezone(timezone.utc)


def _slug(value, label):
    if not isinstance(value, str) or not _SLUG.fullmatch(value):
        raise ValueError(f"{label} must be an opaque ASCII identifier, at most 80 characters")


def _sha(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("Expected a lowercase SHA-256 digest")


def runtime_identity():
    root = Path(__file__).parent
    files = ("prospective.py", "forward_model.py", "forecasting.py")
    return dict(python=platform.python_version(),
                numpy=import_metadata.version("numpy"), scipy=import_metadata.version("scipy"),
                implementation_sha256=_hash({name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                             for name in files}))


def default_protocol():
    """Return a self-contained draft; the repository config contains this value."""
    return dict(schema_version=SCHEMA, status="draft_not_registered_not_active",
        project_lead="Yike Chen", receipt_adapter="fixture_unverified",
        official_rule_evidence=None, future_seasons=None, correction_window_hours=None,
        model=dict(prior_sd=.5, temperature=6., rule="percentage", judge_weight=.5,
                   particles=2048, seeds=[2627761, 2627783, 2627819], mh_steps=4,
                   proposal_scale=.35, ess_fraction=.7),
        baselines=dict(judge_temperature=12., history_temperature=6.,
                       history_strength=0., past_judge_temperature=12.),
        required_rule=dict(rule="percentage", judge_weight=.5, judge_save=False,
                           carryover=False, score_aggregation="provided_nonnegative_weekly_scores"))


def validate_protocol(protocol):
    expected = default_protocol()
    if not isinstance(protocol, dict) or set(protocol) != set(expected):
        raise ValueError("Unknown or incomplete prospective protocol")
    if (protocol["schema_version"] != SCHEMA
            or protocol["status"] != expected["status"]
            or protocol["project_lead"] != "Yike Chen"
            or protocol["receipt_adapter"] != "fixture_unverified"
            or protocol["required_rule"] != expected["required_rule"]):
        raise ValueError("Only the declared draft percentage/no-save/no-carryover rule is supported")
    if any(protocol[x] is not None for x in ("official_rule_evidence", "future_seasons",
                                             "correction_window_hours")):
        raise ValueError("Activation fields remain unset until a trusted live adapter exists")
    model, baseline = protocol["model"], protocol["baselines"]
    if (not isinstance(model, dict) or set(model) != set(expected["model"])
            or not isinstance(baseline, dict) or set(baseline) != set(expected["baselines"])):
        raise ValueError("Unknown model or baseline settings")
    seeds = model["seeds"]
    if (not isinstance(seeds, list) or not seeds or len(seeds) > 10
            or len(set(seeds)) != len(seeds)
            or not all(type(s) is int and s >= 0 for s in seeds)
            or type(model["particles"]) is not int or not 16 <= model["particles"] <= 100000
            or model["rule"] != "percentage" or model["judge_weight"] != .5):
        raise ValueError("Invalid declared static SMC settings")
    SequentialPosterior(2, **{k: model[k] for k in _SETTINGS}, seed=0)
    if (not all(type(x) in (int, float) and np.isfinite(x) and x > 0
                for k, x in baseline.items() if k != "history_strength")
            or type(baseline["history_strength"]) not in (int, float)
            or not np.isfinite(baseline["history_strength"])
            or baseline["history_strength"] < 0):
        raise ValueError("Invalid fixed baseline settings")
    canonical_bytes(protocol)


def _posterior_state(posterior):
    return dict(settings={k: (len(posterior.particles) if k == "particles" else getattr(posterior, k))
                          for k in _SETTINGS}, particles=posterior.particles,
                weights=posterior.weights, rng=posterior.rng.bit_generator.state,
                history=[dict(indices=list(o.indices), judges=list(o.judges), outcome=o.outcome)
                         for o in posterior.history], diagnostics=posterior.diagnostics)


def _restore_posterior(state):
    if not isinstance(state, dict) or set(state) != {"settings", "particles", "weights", "rng",
                                                   "history", "diagnostics"}:
        raise ValueError("Invalid explicit SMC state")
    particles, weights = np.asarray(state["particles"], float), np.asarray(state["weights"], float)
    settings = state["settings"]
    if (not isinstance(settings, dict) or set(settings) != set(_SETTINGS)
            or particles.ndim != 2 or not 2 <= particles.shape[1] <= 1000
            or not 16 <= particles.shape[0] <= 100000
            or settings["particles"] != particles.shape[0]
            or weights.shape != (len(particles),) or not np.isfinite(particles).all()
            or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("Invalid SMC particles or weights")
    posterior = SequentialPosterior(particles.shape[1], **settings, seed=0)
    posterior.particles, posterior.weights = particles, weights
    for observation in state["history"]:
        if set(observation) != {"indices", "judges", "outcome"}:
            raise ValueError("Invalid observation fields")
        idx, judges, outcome = (observation[k] for k in ("indices", "judges", "outcome"))
        exit_log_probs(particles[0], idx, judges, rule=posterior.rule)
        if type(outcome) is not int or not 0 <= outcome < len(idx):
            raise ValueError("Invalid saved outcome")
        posterior.history.append(Observation(tuple(idx), tuple(judges), outcome))
    if (not isinstance(state["diagnostics"], list)
            or len(state["diagnostics"]) != len(posterior.history)
            or state["rng"].get("bit_generator") != "PCG64"):
        raise ValueError("Invalid diagnostics or unsupported RNG")
    posterior.rng.bit_generator.state = state["rng"]
    posterior.diagnostics = state["diagnostics"]
    return posterior


@contextmanager
def _locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory/".writer.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process holds the prospective writer lock") from error
        try:
            yield directory
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_write(path, content):
    """One durable snapshot is the transaction; abandoned temp files are ignored."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".snapshot-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save(directory, state):
    content = canonical_bytes(dict(state=state, state_sha256=_hash(state)))
    if len(content) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Snapshot exceeds the supported private state size; previous state retained")
    _atomic_write(directory/"private_state.json", content)


def _load(directory):
    path = Path(directory)/"private_state.json"
    if path.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError("Snapshot exceeds the supported private state size")
    envelope = _unpack(_read_json(path))
    if (not isinstance(envelope, dict) or set(envelope) != {"state", "state_sha256"}
            or _hash(envelope["state"]) != envelope["state_sha256"]):
        raise ValueError("Private state integrity check failed")
    state = envelope["state"]
    if state["schema_version"] != SCHEMA or state["runtime"] != runtime_identity():
        raise ValueError("State schema, implementation or numerical environment changed")
    validate_protocol(state["protocol"])
    if state["protocol_sha256"] != _hash(state["protocol"]):
        raise ValueError("Protocol integrity check failed")
    previous, seen = None, set()
    for sequence, record in enumerate(state["ledger"], 1):
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        if (record["previous_record_sha256"] != previous or record["sequence"] != sequence
                or record["event_id"] in seen or record["record_sha256"] != _hash(body)):
            raise ValueError("Private event ledger integrity check failed")
        if "payload" in record and _hash(record["payload"]) != record["payload_sha256"]:
            raise ValueError("Private prediction payload integrity check failed")
        seen.add(record["event_id"])
        previous = record["record_sha256"]
    pending = state["pending"]
    if pending is not None:
        if (pending["event_id"] in seen or pending["sequence"] != len(state["ledger"])+1
                or pending["header"]["previous_record_sha256"] != previous
                or _hash(pending["payload"]) != pending["header"]["payload_sha256"]):
            raise ValueError("Pending prediction integrity check failed")
    if len(state["known"]) != len(state["support"]) or set(state["known"]) != set(state["score_history"]):
        raise ValueError("Invalid baseline state")
    if state["posteriors"]:
        if len(state["posteriors"]) != len(state["protocol"]["model"]["seeds"]):
            raise ValueError("Missing independent SMC repeat")
        for saved in state["posteriors"]:
            posterior = _restore_posterior(saved)
            if posterior.particles.shape[1] != len(state["known"]):
                raise ValueError("SMC roster mapping mismatch")
    return state


def initialize(directory, *, season, protocol=None, mode="dry_run"):
    """Create an empty season without receiving any future roster or outcomes."""
    if mode != "dry_run":
        raise ValueError("Activation blocked: no independently verifiable receipt adapter or official live rule adapter")
    if type(season) is not int or season < 1:
        raise ValueError("season must be a positive integer")
    protocol = default_protocol() if protocol is None else protocol
    validate_protocol(protocol)
    with _locked(directory) as directory:
        if (directory/"private_state.json").exists():
            raise ValueError("State already exists; choose a new private output directory")
        state = dict(schema_version=SCHEMA, mode=mode, season=season, runtime=runtime_identity(),
                     protocol=protocol, protocol_sha256=_hash(protocol), known={}, support={},
                     score_history={}, posteriors=[], pending=None, ledger=[])
        _save(directory, state)
    return dict(status="dry_run_initialized", protocol_sha256=state["protocol_sha256"],
                interpretation=INTERPRETATION)


def _new_event(state, event_id, sequence):
    _slug(event_id, "event_id")
    if state["pending"] is not None:
        raise ValueError("Release the pending prediction before registering another event")
    if event_id in {r["event_id"] for r in state["ledger"]}:
        raise ValueError("Event already recorded; duplicate prediction/update refused")
    if type(sequence) is not int or sequence != len(state["ledger"])+1:
        raise ValueError("Event sequence must be the next consecutive integer")


def _validate_inputs(roster, scores, event_metadata):
    if (not isinstance(roster, list) or not 2 <= len(roster) <= 1000
            or len(set(roster)) != len(roster)):
        raise ValueError("Need a unique current roster with at least two opaque IDs")
    for identifier in roster:
        _slug(identifier, "contestant ID")
    j = np.asarray(scores, float)
    _judge_hazard(j, 1.)
    if len(j) != len(roster):
        raise ValueError("Roster and score lengths differ")
    fields = {"scheduled_single_exit", "is_final", "known_withdrawal", "scores_complete",
              "scores_available_at", "earliest_outcome_at", "source_sha256"}
    if not isinstance(event_metadata, dict) or set(event_metadata) != fields:
        raise ValueError("Unknown or missing predeadline event metadata; outcomes are forbidden")
    if (event_metadata["scheduled_single_exit"] is not None
            and type(event_metadata["scheduled_single_exit"]) is not bool
            or any(type(event_metadata[k]) is not bool for k in
                   ("is_final", "known_withdrawal", "scores_complete"))):
        raise ValueError("Invalid predeadline eligibility flags")
    if _time(event_metadata["scores_available_at"]) >= _time(event_metadata["earliest_outcome_at"]):
        raise ValueError("Complete scores must precede the earliest declared result disclosure")
    _sha(event_metadata["source_sha256"])
    # Stable lexical tie order agrees with the historical batch adapter.
    order = np.argsort(roster)
    return [roster[i] for i in order], j[order]


def predict_event(directory, *, event_id, sequence, roster, scores, event_metadata):
    """Freeze a private prediction. This interface cannot receive an outcome."""
    ids, judges = _validate_inputs(roster, scores, event_metadata)
    with _locked(directory) as directory:
        state = _load(directory)
        _new_event(state, event_id, sequence)
        before = _hash(state)
        new = [identifier for identifier in ids if identifier not in state["known"]]
        for identifier in new:
            state["known"][identifier] = len(state["known"])
            state["support"][identifier] = 0.
            state["score_history"][identifier] = []
        model = state["protocol"]["model"]
        if not state["posteriors"]:
            posteriors = [SequentialPosterior(len(state["known"]),
                **{k: model[k] for k in _SETTINGS}, seed=np.random.SeedSequence([seed, state["season"]]))
                for seed in model["seeds"]]
        else:
            posteriors = [_restore_posterior(saved) for saved in state["posteriors"]]
            for posterior in posteriors:
                posterior.extend(len(new))
        idx = [state["known"][x] for x in ids]
        b = state["protocol"]["baselines"]
        judge_hazard = _judge_hazard(judges, b["history_temperature"])
        relative = len(ids)*judges/judges.sum()
        previous_judges = [np.mean(state["score_history"][x]) if state["score_history"][x]
                           else relative[i] for i, x in enumerate(ids)]
        repeats = [p.predict(idx, judges) for p in posteriors]
        predictions = dict(forward_bayes=np.mean(repeats, axis=0),
            uniform=np.full(len(ids), 1/len(ids)),
            judge_only=_judge_hazard(judges, b["judge_temperature"]),
            past_survival=softmax(np.log(judge_hazard)-b["history_strength"]*
                                  np.array([state["support"][x] for x in ids])),
            past_judges=softmax(-b["past_judge_temperature"]*np.array(previous_judges)))
        for prediction in predictions.values():
            if (not np.isfinite(prediction).all() or (prediction <= 0).any()
                    or not np.isclose(prediction.sum(), 1., rtol=0, atol=1e-12)):
                raise ValueError("Invalid predictive probability vector")
        state["posteriors"] = [_posterior_state(p) for p in posteriors]
        known_ineligible = (len(ids) < 3 or event_metadata["is_final"]
                            or event_metadata["known_withdrawal"]
                            or not event_metadata["scores_complete"]
                            or event_metadata["scheduled_single_exit"] is False)
        eligible = (False if known_ineligible else event_metadata["scheduled_single_exit"])
        payload = dict(format_version=SCHEMA, protocol_sha256=state["protocol_sha256"],
            runtime=state["runtime"], season=state["season"], event_id=event_id, sequence=sequence,
            before_state_sha256=before, prediction_state_sha256=_hash(state),
            nonce_hex=secrets.token_hex(32), roster=ids, scores=judges,
            event_metadata=event_metadata, predeadline_eligible=eligible,
            probabilities=predictions, forward_repeats=repeats)
        header = dict(format_version=SCHEMA, protocol_sha256=state["protocol_sha256"],
            event_id=event_id, previous_record_sha256=(state["ledger"][-1]["record_sha256"]
                                                       if state["ledger"] else None),
            payload_sha256=_hash(payload))
        state["pending"] = dict(event_id=event_id, sequence=sequence, header=header,
                                payload=payload, receipt=None)
        _save(directory, state)
    return header


def attach_receipt(directory, receipt):
    """Bind an explicitly unverified fixture receipt; never establishes timeliness."""
    if (not isinstance(receipt, dict)
            or set(receipt) != {"adapter", "payload_sha256", "received_at"}
            or receipt["adapter"] != "fixture_unverified"):
        raise ValueError("Only an explicitly unverified fixture receipt is supported")
    _sha(receipt["payload_sha256"])
    _time(receipt["received_at"])
    with _locked(directory) as directory:
        state = _load(directory)
        pending = state["pending"]
        if pending is None or pending["receipt"] is not None:
            raise ValueError("No unreceipted pending prediction; duplicate receipt refused")
        if receipt["payload_sha256"] != pending["header"]["payload_sha256"]:
            raise ValueError("Receipt is not bound to the pending payload hash")
        pending["receipt"] = receipt
        _save(directory, state)
    return dict(status="fixture_receipt_attached", independently_verified=False,
                confirmatory_eligible=False)


def _append_record(state, record):
    record["previous_record_sha256"] = (state["ledger"][-1]["record_sha256"]
                                         if state["ledger"] else None)
    record["record_sha256"] = _hash(record)
    state["ledger"].append(record)


def release_event(directory, *, event_id, exits, withdrawals, disclosed_at, source_sha256):
    """Score frozen probabilities, then assimilate a released event exactly once."""
    _sha(source_sha256)
    disclosure = _time(disclosed_at)
    if (not isinstance(exits, list) or not isinstance(withdrawals, list)
            or len(set(exits)) != len(exits) or len(set(withdrawals)) != len(withdrawals)
            or set(exits) & set(withdrawals)):
        raise ValueError("Released exit/withdrawal IDs must be unique and disjoint")
    with _locked(directory) as directory:
        state = _load(directory)
        pending = state["pending"]
        if pending is None or pending["event_id"] != event_id:
            raise ValueError("No matching pending prediction; duplicate update refused")
        payload = pending["payload"]
        ids, judges = payload["roster"], np.asarray(payload["scores"], float)
        if not set(exits+withdrawals).issubset(ids):
            raise ValueError("Released result includes an ID outside the frozen roster")
        if disclosure < _time(payload["event_metadata"]["scores_available_at"]):
            raise ValueError("Outcome disclosure predates the declared score availability")
        actual_deadline = min(disclosure, _time(payload["event_metadata"]["earliest_outcome_at"]))
        receipt = pending["receipt"]
        receipt_status = ("missing" if receipt is None else
                          "late_fixture_unverified" if _time(receipt["received_at"]) >= actual_deadline
                          else "fixture_unverified")
        update = len(ids) >= 3 and len(exits) == 1 and not withdrawals
        scorable = bool(update and not payload["event_metadata"]["is_final"])
        scores = {}
        if scorable:
            observed = ids.index(exits[0])
            truth = np.eye(len(ids))[observed]
            for model, prediction in payload["probabilities"].items():
                p = np.asarray(prediction, float)
                scores[model] = dict(log_loss=float(-np.log(p[observed])),
                    brier=float(np.square(p-truth).sum()), top1=float(np.argmax(p) == observed))
        if update:
            indices = [state["known"][x] for x in ids]
            posteriors = [_restore_posterior(saved) for saved in state["posteriors"]]
            for posterior in posteriors:
                posterior.observe(indices, judges, ids.index(exits[0]))
            state["posteriors"] = [_posterior_state(p) for p in posteriors]
            hazard = _judge_hazard(judges, state["protocol"]["baselines"]["history_temperature"])
            for i, identifier in enumerate(ids):
                if identifier != exits[0]:
                    state["support"][identifier] += float(hazard[i])
        relative = len(ids)*judges/judges.sum()
        for i, identifier in enumerate(ids):
            state["score_history"][identifier].append(float(relative[i]))
        reason = ("final" if payload["event_metadata"]["is_final"] else
                  "withdrawal" if withdrawals else "zero_exit" if not exits else
                  "multiple_exits" if len(exits) > 1 else "roster_below_three" if len(ids) < 3 else None)
        record = dict(event_id=event_id, sequence=pending["sequence"], kind="released_prediction",
            payload=payload, payload_sha256=pending["header"]["payload_sha256"], receipt=receipt,
            receipt_status=receipt_status, confirmatory=False,
            predeadline_eligible=payload["predeadline_eligible"], scorable=scorable,
            status=("scored_exploratory" if scorable else "undefined_categorical_score"),
            undefined_reason=reason, updated=bool(update), scores=scores,
            released=dict(exits=exits, withdrawals=withdrawals, disclosed_at=disclosed_at,
                          source_sha256=source_sha256))
        _append_record(state, record)
        state["pending"] = None
        _save(directory, state)
    return dict(status=record["status"], updated=bool(update), receipt_status=receipt_status,
                confirmatory=False, record_sha256=record["record_sha256"])


def record_unpredicted_event(directory, *, event_id, sequence, status):
    """Retain missed/ineligible events in the coverage denominator without outcomes."""
    if status not in {"missed_deadline", "predeclared_ineligible", "input_or_numerical_failure"}:
        raise ValueError("Unknown unpredicted event status")
    with _locked(directory) as directory:
        state = _load(directory)
        _new_event(state, event_id, sequence)
        record = dict(event_id=event_id, sequence=sequence, kind="unpredicted_event", status=status,
                      predeadline_eligible=(False if status == "predeclared_ineligible" else None),
                      scorable=False, confirmatory=False, scores={})
        _append_record(state, record)
        _save(directory, state)
    return dict(status=status, record_sha256=record["record_sha256"])


def export_aggregate(directory):
    """Construct an allowlisted aggregate from scratch; never copy private records."""
    with _locked(directory) as directory:
        state = _load(directory)
    ledger = state["ledger"]
    scorable = [r for r in ledger if r["scorable"]]
    metrics = []
    for model in MODELS:
        rows = [r["scores"][model] for r in scorable]
        metrics.append(dict(model=model, events=len(rows),
            **{key: float(np.mean([r[key] for r in rows])) if rows else None
               for key in ("log_loss", "brier", "top1")}))
    paired = [r["scores"]["forward_bayes"]["log_loss"]-r["scores"]["past_judges"]["log_loss"]
              for r in scorable]
    return dict(schema_version=SCHEMA, study_status="draft_dry_run_not_confirmatory",
        interpretation=INTERPRETATION, protocol_sha256=state["protocol_sha256"],
        coverage=dict(registered_events=len(ledger)+int(state["pending"] is not None),
            released_or_recorded_events=len(ledger), pending_events=int(state["pending"] is not None),
            predeadline_eligible_events=sum(r["predeadline_eligible"] is True for r in ledger)+
                int(state["pending"] is not None
                    and state["pending"]["payload"]["predeadline_eligible"] is True),
            unknown_predeadline_eligibility_events=sum(r["predeadline_eligible"] is None for r in ledger)+
                int(state["pending"] is not None
                    and state["pending"]["payload"]["predeadline_eligible"] is None),
            scorable_exploratory_events=len(scorable), confirmatory_events=0,
            status_counts=dict(Counter(r["status"] for r in ledger)),
            receipt_counts=dict(Counter(r["receipt_status"] for r in ledger if "receipt_status" in r)),
            undefined_reason_counts=dict(Counter(r["undefined_reason"] for r in ledger
                                                 if r.get("undefined_reason") is not None))),
        metrics=metrics, primary_paired_log_loss_forward_minus_past_judges=(float(np.mean(paired))
                                                                         if paired else None))


def demo(directory):
    """Synthetic rehearsal only; no real contestants, future data or external I/O."""
    protocol = default_protocol()
    protocol["model"].update(particles=64, seeds=[17, 31], mh_steps=2)
    initialize(directory, season=1, protocol=protocol)
    events = [("fixture-1", ["A", "B", "C", "D"], [8., 7., 6., 5.], ["D"]),
              ("fixture-2", ["A", "B", "C"], [7., 9., 6.], ["C"])]
    for sequence, (event_id, roster, judges, exits) in enumerate(events, 1):
        metadata = dict(scheduled_single_exit=True, is_final=False, known_withdrawal=False,
            scores_complete=True, scores_available_at=f"2000-01-0{sequence}T20:00:00Z",
            earliest_outcome_at=f"2000-01-0{sequence}T21:00:00Z", source_sha256="0"*64)
        header = predict_event(directory, event_id=event_id, sequence=sequence, roster=roster,
                               scores=judges, event_metadata=metadata)
        attach_receipt(directory, dict(adapter="fixture_unverified",
            payload_sha256=header["payload_sha256"], received_at=f"2000-01-0{sequence}T20:30:00Z"))
        release_event(directory, event_id=event_id, exits=exits, withdrawals=[],
                      disclosed_at=f"2000-01-0{sequence}T21:00:00Z", source_sha256="1"*64)
    record_unpredicted_event(directory, event_id="fixture-3", sequence=3, status="missed_deadline")
    return export_aggregate(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "predict", "commit", "release", "record", "export", "demo"):
        command = sub.add_parser(name)
        command.add_argument("--state", required=True, help="Private, ignored output directory")
        if name == "init":
            command.add_argument("--season", type=int, required=True)
            command.add_argument("--protocol")
            command.add_argument("--mode", choices=("dry_run", "active"), default="dry_run")
        elif name in {"predict", "commit", "release", "record"}:
            command.add_argument("--input", required=True, help="Private JSON request file")
        elif name == "export":
            command.add_argument("--out", help="Optional aggregate-only JSON output")
    args = parser.parse_args(argv)
    if args.command == "init":
        result = initialize(args.state, season=args.season, mode=args.mode,
                            protocol=_read_json(args.protocol) if args.protocol else None)
    elif args.command == "demo":
        result = demo(args.state)
    elif args.command == "export":
        result = export_aggregate(args.state)
        if args.out:
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())
    else:
        request = _read_json(args.input)
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object")
        function = dict(predict=predict_event, commit=attach_receipt,
                        release=release_event, record=record_unpredicted_event)[args.command]
        result = (function(args.state, request) if args.command == "commit"
                  else function(args.state, **request))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
