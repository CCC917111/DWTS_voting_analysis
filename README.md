# DWTS Voting Analysis

**How do voting rules shape competition outcomes when audience votes are hidden?**

The software implementation and subsequent extensions of Team 2627761's DWTS
study for the 2026 MCM Problem C. Yike Chen participated in the original team
research. The workflow infers latent fan support, compares voting rules, and
studies associations between scores, contestant features and partner history.

**Project lead: Yike Chen.** The original study is credited to Team 2627761.

[Documentation](#documentation) · [Methods](docs/METHODS.md) · [Study results](docs/STUDY_RESULTS.md) · [Validation](docs/VALIDATION_RESULTS.md) · [Run guide](docs/RUNNING.md)

## Documentation

| Guide | What it covers |
|---|---|
| [What the outcomes identify](docs/identification.md) | Feasible fan-share ranges and counterfactual exit ambiguity under an additional deterministic rule. |
| [Static and dynamic support](docs/dynamic-comparison.md) | A controlled comparison with a common normalized exit likelihood and strong baselines. |
| [Preparing prospective predictions](docs/prospective.md) | Persistent local prediction/release workflow, commitments and activation limits. |
| [Prediction, recovery and policy validation](docs/VALIDATION_RESULTS.md) | Temporal exit backtesting, known-truth simulation and multi-objective policy robustness. |
| [Methods and assumptions](docs/METHODS.md) | Data processing, Bayesian inference, voting rules and validation. |
| [Study results and figures](docs/STUDY_RESULTS.md) | Baselines, feature ablations, posterior policy comparisons and sensitivity. |
| [Earlier results and report comparisons](docs/RESULTS.md) | Preserved v5 metrics, numerical differences and interpretation limits. |
| [Running the study](docs/RUNNING.md) | Installation, data, the demo and the full research workflow. |
| [Research team and sources](docs/PROVENANCE.md) | Original team research, project leadership and the current code edition. |
| [Method background and related work](docs/RELATED_WORK.md) | Established methods, related DWTS software and the scope of the contribution. |

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

## Research findings

The extended study covers 34 seasons and 2,777 contestant-weeks. Fourteen
fan-share models share five outer season folds; RF history weights are selected
in three inner season folds. Every model predicts the same inferred fan-share
target. Weekly normalization constrains predicted shares to sum to one.

| Fan-share model | Out-of-fold R² | Out-of-fold RMSE |
|---|---:|---:|
| Unconstrained tuned RF | 0.555265 | 0.063599 |
| Weekly-normalized tuned RF | 0.723234 | 0.050172 |
| Weekly-normalized own-history mean | 0.757838 | 0.046930 |

The simple normalized history baseline has the best point score among the
declared models. Both RF pipelines select history weight zero in every outer
fold, so these results do not establish an incremental cross-season partner
effect. This is exploratory internal validation of whole-season latent targets;
the normalization hypothesis arose from this dataset. It does not measure future
true-vote prediction. RMSE is in share units.

![Fan-share baselines and feature ablations](docs/assets/study-benchmarks.svg)

The policy study propagates 2,048 joint posterior draws through 18 policies on
291 eligible weekly rosters. Judge saves do not universally reduce the declared
controversial-week fraction. A separate nine-season analytic prior-scale study
shows invariant rank-policy results but sensitivity in percentage-policy
outcomes. [Study results](docs/STUDY_RESULTS.md) provide three figures, paired
comparisons, assumptions and aggregate data. [Earlier results](docs/RESULTS.md)
preserve the six-task v5 snapshot and original-report comparisons.

## Prediction, recovery and policy validation

Three additional studies test the limits of the inference and policy results:

- **Temporal backtesting:** a separate normalized exit model selects settings on
  seasons 3–20, then uses current judges' scores to predict each eligible exit
  in seasons 21–27. The
  33 eligible weeks represent six seasons; season 26 has no qualifying
  single-exit weeks. Bayesian log loss is 1.948 versus 2.154 for uniform
  probabilities and 1.621 for the best point-scoring past-judges baseline.
  The data support no demonstrated Bayesian advantage over stronger baselines.
- **Known-truth recovery:** 192 independently generated seasons support 384
  fits and 64 larger-particle checks. Matched-model intervals cover about 95–96%
  of generated contestant-week shares, but narrower priors reduce coverage to
  about 64–67%. Independent coverage probes remain imprecise, and some point
  rankings are sensitive to finite-particle error.
- **Multiple policy objectives:** 18 policies are compared on six declared
  objectives across separate matched cohorts. Seven of 153 policy pairs show
  baseline mean dominance; none retains dominance throughout either of the
  declared sensitivity grids. Tradeoffs and assumptions remain visible.

[Validation results and three figures](docs/VALIDATION_RESULTS.md) provide the
protocols, uncertainty intervals and aggregate data. These historical data were
previously examined. The chronological replay is a retrospective backtest;
independent prospective confirmation remains untested.

## What can elimination outcomes identify?

An additional deterministic percentage-rule study measures what the observed
constraints permit, without imposing a prior over fan shares. In 164 selected
single-exit weeks, the median feasible interval spans **99.63 percentage points
for survivors** and **13.60 for eliminated contestants**. These are coordinate
bounds under declared extra assumptions, rather than posterior intervals or
measured true votes.

The same joint constraints often permit several counterfactual exits. Changing
the judge weight from 50% to 25% leaves multiple attainable exits in 155 of 164
weeks; changing it to 75% does so in 87 weeks. These are ambiguity counts, not
probabilities of changing the result. Both studies retain the observed roster
and exclude saves, finals, withdrawals and multiple exits.

![Feasible share widths and counterfactual exit ambiguity](docs/assets/identification.svg)

[Identification methods and results](docs/identification.md) explain the
assumptions, constructive checks and aggregate tables. A separate
[static/dynamic comparison](docs/dynamic-comparison.md) tests how latent-state
dynamics affect forecasts. The [local prospective workflow](docs/prospective.md)
separates prediction from outcome release and preserves state across processes;
it is preparation for future evidence, not completed independent validation.

The controlled comparison uses three development repeats per candidate for both
models. On the same 33 historical weeks, static bootstrap log loss is 1.951 and
dynamic bootstrap log loss is 1.960, versus 1.621 for past judges. The paired
dynamic-minus-static difference is 0.0096 (six-season interval
−0.0101 to 0.0255). This tested dynamic specification does not demonstrate an
improvement. Both latent comparators use bootstrap filters without MH; the
earlier resample–move result remains a separate experiment.

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
- Whole-season fan inference supports retrospective analysis. A separate model has chronological backtests; independent prospective performance remains unevaluated.
- SHAP and feature importance describe associations; they do not identify causal partner effects.
- Rule comparisons retain each week's observed roster. Complete alternative-season trajectories are not simulated.
- RF/OLS use posterior means. The separate policy study propagates joint posterior draws; its intervals remain conditional on model assumptions.

## Repository map

```text
dwts/                 Models, voting rules, analysis and synthetic demo
configs/              Experiment and sensitivity configurations
scripts/              Data acquisition and validation utilities
tests/                Scientific regression tests and repository checks
results/              Aggregate benchmarks, policy studies and preserved v5 metrics
docs/                 Methods, results, sources and running instructions
.github/              Test workflow and contribution templates
```

## Research team and sources

Original study: Team 2627761 (2026), *Modeling Fairness in Competitive Voting
Systems: A Bayesian Approach to Infer Fan Support and Evaluate Contest Dynamics
in DWTS*. Yike Chen was a member of the original research team; the study is
credited to the full team. [Software provenance](docs/PROVENANCE.md) distinguishes
the current code edition and later extensions from the competition-time work.

The report, raw dataset and posterior chains are not bundled. No open-source
license has been granted; see [license status](LICENSE_STATUS.md).
