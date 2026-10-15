# DWTS Voting Analysis

**How do voting rules shape competition outcomes when audience votes are hidden?**

The software implementation and subsequent extensions of Team 2627761's DWTS
study for the 2026 MCM Problem C. Yike chen participated in the original team
research. The workflow infers latent fan support, compares voting rules, and
studies associations between scores, contestant features and partner history.

[中文说明](README.zh-CN.md) · [Documentation](#documentation) · [Methods](docs/METHODS.md) · [Results](docs/RESULTS.md) · [Run guide](docs/RUNNING.md)

## Documentation

| Guide | What it covers |
|---|---|
| [Methods and assumptions](docs/METHODS.md) | Data processing, Bayesian inference, voting rules and validation. |
| [Results and interpretation](docs/RESULTS.md) | Recorded experiment results, units and interpretation limits. |
| [Running the study](docs/RUNNING.md) | Installation, data, the demo and the full research workflow. |
| [Research team and sources](docs/PROVENANCE.md) | Original team research, Yike's participation and the current code edition. |

The [documentation index](docs/README.md) also links data sources and experiment configurations.

## Research questions

- What fan-support patterns are compatible with public scores and elimination outcomes?
- How do rank and percentage aggregation, judge saves and fan saves change weekly outcomes?
- How are judge scores and inferred fan support associated with contestant and partner features?

The workflow combines Bayesian inverse modelling, a PyMC reference model and a
Numba Metropolis–Hastings engine, explicit rule simulations, nested season-grouped
random forests, OLS and SHAP. Configurations and input hashes record each
experiment's assumptions. Exact rank-tie handling and canonical partner identities
address two demonstrated implementation issues.

![Synthetic voting-rule demonstration](docs/assets/demo.svg)

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

The current implementation's saved v5 experiment covers 34 seasons and 2,777 contestant-weeks. Its
retrospective models use five outer season-grouped folds and inner selection of
history weights.

| Regression target | Out-of-fold R² | Out-of-fold RMSE |
|---|---:|---:|
| Judge mean score | 0.434488 | 1.100571 |
| Inferred fan share | 0.555199 | 0.063604 |

Judge RMSE is in judge-mean score units; fan RMSE is in inferred-share units.
These are saved research results; the quick-start demo does not fit these models.
[Results](docs/RESULTS.md) document all six tasks and interpretation limits.
The current software results and the original report's values are labeled separately.

## Run an experiment

Read [data sources](data/SOURCES.md), then obtain the official COMAP CSV:

```bash
python scripts/download_data.py
python -m dwts prepare --out outputs/prepared
python -m dwts run --config configs/smoke.json --out outputs/smoke --skip-attribution
```

The smoke configuration checks execution with short chains. For the full study:

```bash
python -m dwts run --config configs/study.json --out outputs/my-study
```

Research estimates require passing sampling diagnostics; see the [run guide](docs/RUNNING.md).

After a full run passes diagnostics:

```bash
python -m dwts.analysis --source outputs/my-study --out outputs/my-analysis
```

This compares voting rules under declared save mechanisms, fits OLS using prior
judge shares, and reports week and event rates with explicit denominators.
Undefined multi-exit save cases remain unscored. It reuses fitted posterior means.
Successful standalone analysis returns exit **0**. Optional comparisons with the
original report are enabled with `--compare-reference` and have a separate status;
see the [run guide](docs/RUNNING.md).

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

## Research team and sources

Original study: Team 2627761 (2026), *Modeling Fairness in Competitive Voting
Systems: A Bayesian Approach to Infer Fan Support and Evaluate Contest Dynamics
in DWTS*. Yike chen was a member of the original research team; the study is
credited to the full team. [Software provenance](docs/PROVENANCE.md) distinguishes
the current code edition and later extensions from the competition-time work.

The report, raw dataset and posterior chains are not bundled. No open-source
license has been granted; see [license status](LICENSE_STATUS.md).
