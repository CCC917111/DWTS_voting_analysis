# DWTS Voting Analysis

**How do voting rules shape competition outcomes when audience votes are hidden?**

A Python project for inferring latent fan support, comparing DWTS voting rules,
and studying retrospective associations between judge scores, contestant
characteristics and professional-partner history.

[中文说明](README.zh-CN.md) · [Methods](docs/METHODS.md) · [Results](docs/RESULTS.md) · [Run guide](docs/RUNNING.md)

![Synthetic voting-rule demonstration](docs/assets/demo.svg)

## Research questions

- What fan-support patterns are compatible with public scores and elimination outcomes?
- How do rank and percentage aggregation, judge saves and fan saves change weekly outcomes?
- How are judge scores and inferred fan support associated with contestant and partner features?

The workflow combines Bayesian inverse modelling, a PyMC reference model and a
Numba Metropolis–Hastings engine, explicit rule simulations, nested season-grouped
random forests, OLS and SHAP. Configurations and input hashes record each
experiment's assumptions. Exact rank-tie handling and canonical partner identities
address two demonstrated implementation issues.

## Quick start

Python 3.12 is the verified environment. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-v4-verified.txt
python -m pip install --no-deps -e ".[test,audit]"
python -m dwts.demo --out outputs/my-demo
python -m pytest -q
```

The demo uses four invented contestants and fan shares. At equal judge/fan weight,
the rank rule eliminates A and the percentage rule eliminates C. It writes a CSV,
JSON and SVG, and refuses an existing output directory.

Default tests are self-contained. Tests requiring official data or archived
experiments are explicitly deselected. GitHub Actions runs the self-contained
tests and synthetic demo.

## Recorded results

The saved v5 experiment covers 34 seasons and 2,777 contestant-weeks. Its
retrospective models use five outer season-grouped folds and inner selection of
history weights.

| Regression target | Out-of-fold R² | Out-of-fold RMSE |
|---|---:|---:|
| Judge mean score | 0.434488 | 1.100571 |
| Inferred fan share | 0.555199 | 0.063604 |

Judge RMSE is in judge-mean score units; fan RMSE is in inferred-share units.
These are saved research results; the quick-start demo does not fit these models.
[Results and reference comparisons](docs/RESULTS.md) document all six tasks,
numerical differences and interpretation limits.

## Run an experiment

Read [data sources](data/SOURCES.md), then obtain the official COMAP CSV:

```bash
python scripts/download_data.py
python -m dwts prepare --out outputs/prepared
python -m dwts run --config configs/smoke.json --out outputs/smoke --skip-attribution
```

The smoke configuration checks execution with short chains. Research estimates
require a full run and passing sampling diagnostics; see the [run guide](docs/RUNNING.md).

After a full run passes diagnostics:

```bash
python -m dwts.analysis --source outputs/my-full-v5 --out outputs/my-analysis
```

This compares voting rules under declared save mechanisms, fits OLS using prior
judge shares, and reports week and event rates with explicit denominators.
Undefined multi-exit save cases remain unscored. It reuses fitted posterior means.
The command also records reference-comparison status: exit **2** indicates
completed execution with unresolved reference comparisons; inspect `status.json`.

## Interpretation limits

- Fan shares are latent estimates; true audience vote totals are unavailable.
- Whole-season inference supports retrospective analysis. Prospective forecast performance has not been evaluated.
- SHAP and feature importance describe associations; they do not identify causal partner effects.
- Rule comparisons retain each week's observed roster. Complete alternative-season trajectories are not simulated.
- Downstream RF/OLS use posterior means and do not propagate all latent-vote uncertainty.

## Repository map

```text
dwts/                 Models, voting rules, analysis and synthetic demo
configs/              Experiment and sensitivity configurations
scripts/              Data acquisition and validation utilities
tests/                Scientific regression tests and repository checks
results/v5_snapshot/  Saved aggregate metrics and comparison data
docs/                 Methods, results, sources and running instructions
.github/              Test workflow and contribution templates
```

## Sources and contributions

Methods are informed by Team 2627761's 2026 MCM Problem C report,
*Modeling Fairness in Competitive Voting Systems: A Bayesian Approach to Infer Fan
Support and Evaluate Contest Dynamics in DWTS*. The analysis workflow was
implemented independently with AI assistance; the team's original source code
was not supplied. See [sources and contribution scope](docs/PROVENANCE.md).

The report, raw dataset and posterior chains are not bundled. No open-source
license has been granted; see [license status](LICENSE_STATUS.md).
