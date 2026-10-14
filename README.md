# DWTS Voting Audit

**How much can public scores and elimination outcomes tell us about an opaque voting system?**

A Python research project for reconstructing plausible fan support, comparing voting rules, and auditing which conclusions survive explicit assumptions. Built from the independently reconstructed v5 implementation of Team 2627761's 2026 MCM Problem C report.

[中文说明](README.zh-CN.md) · [Methods](docs/METHODS.md) · [Results](docs/RESULTS.md) · [Reproduction guide](docs/REPRODUCING.md)

> **Research status:** full numerical replication of the source report has not been established. Fan shares are inferred latent quantities, not observed votes. This is assistant-assisted independent reconstruction and audit work, not the original team's source code. Contributor identities and software licensing remain under review.

![Synthetic voting-rule demonstration](docs/assets/demo.svg)

## What this project does

- **Reconstructs plausible fan shares:** Bayesian inverse modelling with elimination constraints, a PyMC reference model, and a checked Numba Metropolis–Hastings engine.
- **Compares voting rules:** rank vs. percentage aggregation, judge/fan-save variants, and exact rational rank-tie handling.
- **Audits retrospective associations:** nested season-grouped random forests, OLS, and SHAP, with prior-only feature construction and explicit limits on interpretation.
- **Keeps an evidence trail:** versioned configurations, input/code hashes, provenance-preserving partner identity corrections, and separated numerical-comparison results.

## Quick start

Python 3.12 is the verified environment. Commands below start from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-v4-verified.txt
python -m pip install --no-deps -e ".[test,audit]"
python -m dwts.demo --out outputs/my-demo
python -m pytest -q
```

The demo uses four invented contestants and fan shares. It writes a CSV, JSON, and SVG without downloading data or fitting models. At equal judge/fan weight, the rank rule eliminates A and the percentage rule eliminates C. Reusing an existing output directory is refused.

The default test suite uses synthetic fixtures and does not need the original report, dataset, or archived model outputs. Data/archive-dependent tests are explicitly deselected. See the exact local checks in [VALIDATION.md](docs/VALIDATION.md); the included GitHub Actions workflow has not yet been run on GitHub.

## Recorded research results

The following are **v5 saved results**, not models trained by the quick-start demo. They use 2,777 contestant-weeks across 34 seasons, with five outer season-grouped folds and inner selection of history weights.

| Retrospective regression | v5 R² | Report R² | v5 RMSE | Report RMSE |
|---|---:|---:|---:|---:|
| Judge mean score | 0.434488 | 0.146 | 1.100571 | 5.320 |
| Inferred fan share | 0.555199 | 0.614 | 0.063604 | 0.072 |

Judge RMSE is in judge-mean score units; fan RMSE is in inferred-share units. Author evaluation definitions and units are not fully identified, so these columns show a discrepancy rather than a fair model leaderboard. The fan R²/RMSE gap remains unresolved.

[All six tasks and limitations](docs/RESULTS.md) · [Machine-readable snapshot](results/v5_snapshot/rf_metrics.csv)

## From public inputs to an audit

1. Parse the COMAP wide-format scores and preserve preprocessing disagreements.
2. Infer season-specific fan shares under declared elimination constraints.
3. Check posterior diagnostics before interpreting estimates.
4. Compare weekly outcomes under alternative aggregation/save rules.
5. Fit retrospective attribution models with nested season-grouped validation.
6. Compare saved outputs against report references, recording mismatches and unidentified definitions.

The reconstruction includes two demonstrated v5 corrections: floating-point rank ties and fragmented professional-partner identities. Neither correction establishes the source authors' undisclosed implementation.

## Run a real-data experiment

The COMAP CSV and the source PDF are intentionally not included in this repository. Read [data/SOURCES.md](data/SOURCES.md), obtain data from the official publisher, and respect its terms.

```bash
python scripts/download_data.py
python -m dwts prepare --out outputs/prepared
python -m dwts run --config configs/smoke.json --out outputs/smoke --skip-attribution
```

The smoke configuration checks execution, not posterior adequacy or report replication. A fresh full 34-season run is expensive; it is documented separately in [REPRODUCING.md](docs/REPRODUCING.md). An optional pinned-archive importer supports people who already have the v5 evidence bundle.

## Repository map

```text
dwts/                 Research implementation and synthetic demo
configs/              Explicit experiment configurations
scripts/              Data acquisition and guarded audit utilities
tests/                Original v5 regression tests + repository checks
results/v5_snapshot/  Small historical metrics, not fresh-run outputs
docs/                 Methods, evidence, limitations, release checklist
.github/              Synthetic-test workflow and contribution templates
```

## What the evidence does not show

- True fan-vote accuracy: real fan totals are unavailable.
- Prospective forecasts: fan targets use whole-season retrospective smoothing.
- Causal professional-dancer effects: SHAP and feature importance are associational.
- Complete alternative seasons: counterfactuals retain each week's observed roster.
- Full downstream uncertainty: RF/OLS use posterior means rather than all posterior draws.
- Complete numerical replication: the author's row-level outputs, evaluation definitions and several rule details are missing.

## Attribution and release status

Source report: Team 2627761 (2026), *Modeling Fairness in Competitive Voting Systems: A Bayesian Approach to Infer Fan Support and Evaluate Contest Dynamics in DWTS*.

This repository preserves that attribution and explicitly discloses assistant-assisted reconstruction. It does not assert the current maintainer authored the report or grant a license to COMAP materials. See [PROVENANCE.md](docs/PROVENANCE.md), [LICENSE_STATUS.md](LICENSE_STATUS.md), and the [publication checklist](docs/PUBLICATION_CHECKLIST.md).

The repository is published for review without an open-source license grant. No individual report authorship, DOI, passing CI badge, or institutional affiliation is asserted without evidence.
