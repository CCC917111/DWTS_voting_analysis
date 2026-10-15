# Running the study

Use the repository root and a new output directory for every experiment.

## Environment and demonstration

Python 3.12.14 on Linux is the recorded environment. The requirements file pins
relevant direct dependencies; it is not a complete transitive lockfile. A C++
compiler may be needed by PyTensor.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v4-verified.txt
python -m pip install --no-deps -e ".[test,audit]"
python -m dwts.demo --out outputs/my-demo
python -m pytest -q
```

The demonstration is synthetic. Default tests are self-contained; cases requiring
official data or historical experiment files are explicitly deselected.

For cloud environments with read-only home/cache directories, set writable caches
before importing the scientific libraries:

```bash
mkdir -p /tmp/dwts-cache
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/dwts-cache/matplotlib
export XDG_CACHE_HOME=/tmp/dwts-cache
export NUMBA_CACHE_DIR=/tmp/dwts-cache/numba
export PYTENSOR_FLAGS=base_compiledir=/tmp/dwts-cache/pytensor
```

## Data and execution check

Read [data sources](../data/SOURCES.md), then obtain the official dataset. The
downloader checks its SHA-256 and refuses to overwrite an existing file.

```bash
python scripts/download_data.py
python -m dwts prepare --out outputs/prepared
python -m pytest -q -m "not archive"
python -m dwts run --config configs/smoke.json --out outputs/my-smoke --skip-attribution
```

`prepare` inspects data without model fitting. The short smoke run fits seasons
1, 3 and 28 with deliberately small chains and may fail sampling diagnostics.
Completion confirms execution; research interpretation also requires diagnostic
adequacy. The full experiment's preprocessing settings come from its configuration.

## Full research workflow

```bash
python -m dwts run --config configs/study.json --out outputs/my-study
```

This fits 34 seasons and runs voting-rule simulations, OLS, nested season-grouped
random forests and SHAP. It records input/code hashes and the actual configuration.
Four season workers and four chains per season can be expensive; reduce concurrency
only deliberately and preserve the changed configuration.

The default run produces fitted estimates, scientific tables and figures.
`manifest.json` records execution completion separately from
`all_sampling_diagnostics_pass`. Inspect `sampling_status.json` and each season's
unrounded R-hat/ESS values before interpreting the estimates. Resume accepts only
identical code, input data, configuration and reference-comparison mode.

## Analyze fitted estimates

After the fitted seasons pass sampling diagnostics:

```bash
python -m dwts.analysis --source outputs/my-study --out outputs/my-analysis
```

The command validates the official CSV, declared fitted-season set, preprocessing,
normalized posterior means and sampling diagnostics. It exports rule comparisons,
EDA, OLS with prior-week judge shares, correlation and influence summaries, and
separate controversy-week and controversy-event rates. Saved RF metrics are
validated and copied when available; the command does not refit the Bayesian model
or RF.

Undefined multi-exit save cases are unscored, with supported denominators retained.
A single fitted season can be analyzed, but the OLS component is marked unavailable
because its season-cluster inference requires at least two seasons. Successful default analysis
returns exit **0**; `status.json` records execution and component availability.

## Optional comparison with the original report

The study's current results are evaluated on their declared definitions. Comparing
numbers with the original team report is a separate, optional operation:

```bash
python -m dwts.analysis --source outputs/my-study --out outputs/my-reference-check --compare-reference
```

Reference outputs are written under `reference_comparison/`. Unresolved numerical
comparisons return exit **2**, while the completed analysis and its scientific
results remain recorded. A local `--report-pdf references/2627761.pdf` may be used
with this option to verify the PDF checksum; it is never copied into outputs.
The original report is not needed for the default workflow.

A full model run can also export its legacy comparison table by adding
`--compare-reference` to `dwts run`. The historical `dwts verify` command retains
its original numerical-gate semantics for saved experiments.

## Package and material handling

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

The wheel contains the implementation and demo. Configurations and repository
scripts are used from the checkout. Original reports, raw data, fitted models,
posterior chains and experiment output directories stay outside source releases.
