# Validation record

Validated locally on 2026-10-14, Python 3.12.14 on Linux, using the recorded v5 dependency environment and writable temporary caches.

## Checks performed

- Default self-contained suite in a clean extracted ZIP: **132 passed; 14 explicitly deselected**. It excludes tests needing the separately obtained official CSV or historical evidence.
- Full suite with the pinned v5 evidence imported: **146 passed** (134 preserved v5 tests plus 12 repository-layer tests).
- Original v5 scientific modules and 134-test source files: integrity checked against hashes from the input ZIP; only the package version declaration changed among existing modules.
- Synthetic demo: module execution, CSV/JSON/SVG outputs, exact expected losers and overwrite refusal verified. SVG rendered and visually inspected.
- Package: wheel built; imported successfully from an isolated installation directory outside the checkout; both `dwts-demo` and `dwts --help` executed.
- Data loader: checksum validation checked against the actual pinned CSV, including its UTF-8 byte-order mark; changed data rejected. No fresh upstream download was needed or tested during packaging.
- Historical importer: exact v5 archive accepted; 1,803 files / 161,966,674 bytes restored as ignored local evidence; incorrect archives rejected before mutation. No report PDF or raw posterior chain imported. Checkout source files were not replaced.
- Official-data preparation and saved-evidence full-report expansion: executed successfully without model refitting. The latter still reports that complete numerical reproduction is false.
- Frozen differential-CV planning: completed all five outer folds with zero model fits; 450 inner trials assessed, 90 reusable and 360 requiring refits. This was a plan only, not a new CV execution.
- Release allowlist: tests confirm exclusion of raw data, reports, ordinary outputs, caches and credential filenames.

## Scope and limits

The default suite includes small synthetic sampler checks, deterministic fake-estimator differential-CV controls and direct density/ranking comparisons. It does not retrain the full 34-season research model. Historical evidence checks validate saved artifacts and their internal consistency, not the authors' unknown implementation.

The wheel installation test reused already-available verified dependencies. A completely fresh network dependency installation, Windows/macOS execution, and hosted GitHub Actions have not been tested. The workflow is supplied for the eventual authorized repository; no passing CI badge is claimed.

The first sandbox test attempt failed because PyTensor tried to create a cache under a read-only home directory. The documented writable-cache environment variables resolved it; no statistical code was altered for this issue.

The historical full-report expansion emitted pandas deprecation warnings about future object-dtype downcasting. The pinned run completed; future dependency upgrades need retesting.

## Saved evidence

- [Default suite log](validation/default_tests.txt)
- [Full evidence suite log](validation/full_tests.txt)
- [Historical v5 suite log](validation/v5_archived_tests.txt), retained for provenance only

Software tests, diagnostic gates, matching printed values and complete scientific replication are separate claims. Full source-report numerical replication remains **not established**.
