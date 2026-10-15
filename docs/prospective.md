# Prospective workflow rehearsal

Project lead: **Yike Chen**. This tool implements the prediction/release clock
for a **local draft rehearsal**. It has not registered a study, submitted a
prediction to a timestamp service, verified a future season's rules, or produced
independent prospective evidence. Its only receipt adapter is an explicitly
unverified fixture; every result remains exploratory.

## Run the synthetic rehearsal

From the repository root, after installing the package:

The persistent runner requires Linux or macOS (POSIX `fcntl` locking). Use WSL
on Windows; the generic package demo does not have this locking requirement.

```bash
python -m dwts.prospective demo --state outputs/prospective-demo
python -m dwts.prospective export --state outputs/prospective-demo \
  --out outputs/prospective-demo-aggregate.json
```

The example uses fictitious IDs and dates, two scored synthetic events and one
missed event. Use a new output directory for a new rehearsal. The generated
`private_state.json` contains private inputs, predictions, posterior particles,
past observations and RNG state. Keep its directory under the ignored `outputs/`
tree. Do not add it, request files, individual forecasts or source evidence to
Git. The aggregate exporter constructs a restricted output from scratch.

## Separate prediction and release

Initialize a season with no roster, outcomes or future entrants:

```bash
python -m dwts.prospective init --state outputs/prospective-rehearsal \
  --season 7 --protocol configs/prospective_protocol.json
```

The default frozen candidate uses three independent 2,048-particle static SMC
runs, prior SD 0.5, exit temperature 6, and a 50/50 percentage merit rule. The
prediction is their equal average. Four fixed baselines accompany it: uniform,
current judges (temperature 12), past survival (temperature 6, history strength
0), and past judges (temperature 12, the primary comparator). Numerical defaults
come from the existing historical study; that selection does not become a new
independent holdout. Changes to a draft configuration start a new private state.

Save a private prediction request, for example under `outputs/`:

```json
{
  "event_id": "rehearsal-week-1",
  "sequence": 1,
  "roster": ["A", "B", "C"],
  "scores": [8.0, 7.0, 6.0],
  "event_metadata": {
    "scheduled_single_exit": true,
    "is_final": false,
    "known_withdrawal": false,
    "scores_complete": true,
    "scores_available_at": "2000-01-01T20:00:00Z",
    "earliest_outcome_at": "2000-01-01T21:00:00Z",
    "source_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
  }
}
```

```bash
python -m dwts.prospective predict --state outputs/prospective-rehearsal \
  --input outputs/predict-request.json
```

`predict` has no outcome argument and rejects unknown event metadata, including
exit labels. It receives only the current roster and current scores, adds newly
seen IDs to independent priors, and freezes all five probability vectors. IDs
are sorted lexically, also defining the top-1 tie convention. Past-judge history
uses only previously released events; cold starts use current relative scores.
The printed commitment header includes no individual forecasts or scores. Use
opaque public episode identifiers for `event_id`, not a person's name.

The private payload has a random 32-byte nonce. Its custom canonical encoding is
sorted compact ASCII JSON with one trailing LF, finite floats encoded as
`{"float64_hex": "0x..."}` objects, and SHA-256 over those exact bytes. It is not
claimed to implement RFC 8785. The payload binds the inputs and their declared
source digest, environment, protocol, state digests and complete predictions.
An authorized private reviewer can re-encode it with `canonical_bytes(payload)`.

An optional fixture receipt can be attached once with `commit`:

```json
{
  "adapter": "fixture_unverified",
  "payload_sha256": "REPLACE_WITH_THE_64_CHARACTER_HEADER_DIGEST",
  "received_at": "2000-01-01T20:30:00Z"
}
```

```bash
python -m dwts.prospective commit --state outputs/prospective-rehearsal \
  --input outputs/fixture-receipt.json
```

This checks hash binding and timestamp syntax. The supplied time is an assertion
in a local file, **not an authenticated server receipt**. Even an apparently
early fixture receipt never makes an event confirmatory. A missing or late
receipt stays in the coverage report; it does not remove an unfavorable score.

Only after prediction, a separate release request can provide the result:

```json
{
  "event_id": "rehearsal-week-1",
  "exits": ["C"],
  "withdrawals": [],
  "disclosed_at": "2000-01-01T21:00:00Z",
  "source_sha256": "1111111111111111111111111111111111111111111111111111111111111111"
}
```

```bash
python -m dwts.prospective release --state outputs/prospective-rehearsal \
  --input outputs/release-request.json
```

The runner scores the frozen probability vectors before observing the result.
It updates SMC only for at least three active contestants, exactly one ordinary
exit and no withdrawal. Finals follow the same update gate but do not enter the
reported forecast scores, matching the historical adapter. Zero exits, multiple
exits and withdrawals remain recorded with undefined-score reasons. Current
relative scores enter the past-judges baseline after any released event.

## Coverage, restart and privacy

Every manually registered event has a consecutive sequence number. Exactly one
prediction may be pending. Duplicate predictions, receipts and outcome updates
are rejected. An event without a prediction can be recorded with `record` and a
private request containing `event_id`, `sequence` and one of `missed_deadline`,
`predeclared_ineligible`, or `input_or_numerical_failure`. Such events remain in
the denominator. Missed/failed events have unknown predeadline eligibility,
reported separately; they are not silently classified as ineligible. A schedule
that was unknown when predicting also remains unknown after observing the exit.
This manual ledger cannot prove that every official episode
was registered; reconciliation to an independently archived calendar is still
required before a real study.

A nonblocking Linux writer lock prevents concurrent mutations. Each operation
writes one complete explicit JSON snapshot, fsyncs it, atomically replaces the
old snapshot and fsyncs the directory. The snapshot contains the event ledger
and the post-update particles, weights, complete likelihood history, settings
and PCG64 state. A crash before replacement leaves the old state usable; a crash
after replacement leaves the update committed and a retry is rejected. This
prevents double observation. Abandoned temporary files never become state.
Snapshots are limited to 128 MiB; a transaction exceeding that limit is refused
before replacement so the previous state remains usable.
The snapshot and ledger have consistency hashes. These detect corruption; a
local operator can rewrite a file and recompute its hashes, so they are not an
independent append-only registry or proof of a pre-result prediction. No pickle
or executable deserialization is used. Resume rejects changes to the recorded
Python/NumPy/SciPy versions or runner/model code.

`export` reports coverage and common-event log loss, multiclass Brier and top-1
accuracy for all five models. Its paired log-loss difference compares forward
Bayes with past judges on the same scored rehearsal events; negative favors
forward Bayes. It includes all scored exploratory events, including those whose
predeadline schedule was unknown. It reports the predeadline-eligible count
separately and makes **no confirmatory inference**. There are no confidence
intervals or superiority tests. The export contains no event IDs, names,
rosters, raw scores, individual probabilities, nonces, RNG state or particles.
An aggregate cannot support public row-by-row recomputation of private forecasts.

## What remains before real prospective validation

`--mode active` fails closed. Editing a local JSON receipt or filling in a new
season number cannot activate the runner. The draft configuration deliberately
leaves future seasons, official evidence and correction windows unset.

A live version still needs verified official rules and score aggregation,
save/tie/carryover and exit-schedule handling; independently authenticated
server receipts bound to the exact payload hash before the earliest public
disclosure; an independently archived complete calendar and unique append-only
commitment registry; a private-review arrangement; fixed correction and
stopping rules; and frozen implementation/environment identities. Changes in
the show's rules require a separately frozen compatible model, not a silent
change to this percentage-only candidate. The local input source hashes and
declared times do not themselves verify the source or rule.

The current rehearsal also does not implement the protocol draft's independent
high-particle checks or descriptive calibration table. Numerical adequacy must
be added and frozen before live activation; an ensemble alone is not a passed
Monte Carlo check. No scheduler, third-party posting or real future forecast is
created by these commands.
