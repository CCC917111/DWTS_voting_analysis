# Reproduction guide

Run checkout scripts from the repository root. Use new output directories for experiments. No command below publishes code or contacts another person.

## 1. Environment

The recorded environment is Python 3.12.14 on Linux. `requirements-v4-verified.txt` pins direct relevant packages; it is not a complete transitive lockfile. Other versions/architectures may produce different numeric results and are not certified for bitwise reuse. A C++ compiler can be required by PyTensor.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v4-verified.txt
python -m pip install --no-deps -e ".[test,audit]"
```

If your home/cache directory is read-only, set writable cache locations before importing the libraries (Linux/macOS):

```bash
mkdir -p /tmp/dwts-cache
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/dwts-cache/matplotlib
export XDG_CACHE_HOME=/tmp/dwts-cache
export NUMBA_CACHE_DIR=/tmp/dwts-cache/numba
export PYTENSOR_FLAGS=base_compiledir=/tmp/dwts-cache/pytensor
```

The initial local test attempt exposed this sandbox cache limitation; moving caches to a writable temporary directory resolved it. No statistical code was changed for that fix.

## 2. Self-contained demo and tests

```bash
python -m dwts.demo --out outputs/my-demo
python -m pytest -q
```

The demo is synthetic. Default tests deselect `official_data` and `archive` cases. Original v5 test files remain byte-identical; marks are applied by exact test name in `tests/conftest.py`.

## 3. Official data and smoke execution

Read [SOURCES.md](../data/SOURCES.md) before obtaining data. The downloader verifies the pinned SHA-256 and refuses an existing file.

```bash
python scripts/download_data.py
python -m dwts prepare --out outputs/prepared
python -m pytest -q -m "not archive"
python -m dwts run --config configs/smoke.json --out outputs/smoke --skip-attribution
```

`prepare` does not fit models. It currently uses the parser's legacy defaults; full v5 experiment semantics are selected through the run configuration. The smoke run fits seasons 1, 3 and 28 with small chains; it may fail diagnostic gates and is not research evidence. Check each sampling status, not just process exit status. Avoid reusing a prior `prepare` output directory because that legacy command can replace its own output files.

## 4. Fresh full v5-style reconstruction

```bash
python -m dwts run --config configs/report_reconstruction_v5.json --out outputs/my-full-v5
```

This performs all 34 season fits plus downstream analyses. It can be computationally expensive; the package does not promise a runtime. The configuration uses four season workers, four chains and a full nested RF analysis. Reduce concurrency only deliberately, retain the changed configuration and do not claim bitwise historical equivalence without checking it.

Inspect `manifest.json`, `sampling_status.json`, `run_config.json`, `votes.csv`, `rf_metrics.csv`, and the generated comparison files. Full execution and diagnostic success are distinct from matching the report. To rerun downstream analysis with the explicit numerical gate:

```bash
python -m dwts verify --source outputs/my-full-v5 --config configs/report_verification_v5.json --out outputs/my-verification
```

For `verify`, exit code 2 means report numbers were not all verified; it is an expected scientific discrepancy status, not a program crash. Do not suppress that status and call it a pass.

## 5. Optional historical v5 evidence

If you already have the authorized `DWTS_Full_Report_Reproduction_v5.zip`, the importer can restore data/results in a clean checkout. It accepts only the pinned archive hash, refuses overwrites, and does not copy the report PDF or raw posterior chains and never replaces the checkout implementation. Historical code snapshots inside `outputs/` are retained because exact-reuse checks require them.

```bash
python scripts/import_v5_evidence.py /path/to/DWTS_Full_Report_Reproduction_v5.zip
python -m pytest -q -m ""
python scripts/audit_metric_identifiability.py --out outputs/my-metric-audit
python -m dwts.full_report --source outputs/v5_final --out outputs/my-full-report
```

Use this route instead of downloading the CSV first; the importer refuses an existing data file as well as existing evidence targets. It imports about 162 MB of historical results, all ignored by Git. It does not retrieve or grant access to the archive. Public users without that archive can run the synthetic tests and a new official-data experiment, but cannot rerun the historical-evidence tests as written.

### Exact differential CV

After importing the archive, a zero-fit plan is available:

```bash
python scripts/run_identity_cv.py --canonical-votes outputs/v5_partner_identity/verified_canonical_identity/votes_canonical_primary.csv --out outputs/my-identity-plan
```

Adding `--execute` refits changed designs and reuses only strictly verified unchanged results. The archived Python/package versions and feature/model code must match. No report metric participates in selecting the history weight. This command is an advanced historical audit; do not use it as the first installation check.

## 6. Package verification and hygiene

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

The wheel contains the `dwts` implementation and demo. Repository audit scripts, configurations and snapshots are used from the checkout, not treated as installed package data. See [VALIDATION.md](VALIDATION.md) for the exact checks performed on this package. Keep evidence imports and experiment outputs out of commits.
