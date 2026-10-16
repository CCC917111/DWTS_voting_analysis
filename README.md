# DWTS Voting Analysis

**Studying hidden audience support, observable exits and voting-rule tradeoffs.**

DWTS Voting Analysis combines Bayesian fan-support inference, chronological
elimination forecasts and voting-rule simulations. The project develops the
2026 MCM Problem C study by **Team 2627761** and its subsequent software and
research extensions. **Project lead: Yike Chen**, a member of the original team.

[Documentation](docs/README.md) · [Forecasting study](docs/exit-forecasting.md) · [Methods](docs/METHODS.md) · [Run guide](docs/RUNNING.md) · [Research provenance](docs/PROVENANCE.md)

## Explore the project

| Research question | Approach | Evidence and limits |
|---|---|---|
| Can public information predict the next elimination? | Conditional choice models, strong score baselines and sequential expert mixtures. | [Chronological forecasting](docs/exit-forecasting.md): 86 outer-validation weeks and a previously examined 33-week diagnostic period. |
| What can eliminations reveal about hidden fan support? | Bayesian inference and a separate deterministic constraint analysis. | [Identification](docs/identification.md): feasible share ranges remain wide under the declared constraints. |
| How do voting rules change outcomes? | Rank/percentage aggregation, save mechanisms and joint posterior policy comparisons. | [Study results](docs/STUDY_RESULTS.md): uncertainty and competing policy objectives are retained. |
| Does a more elaborate model help? | Static/dynamic support comparisons, feature ablations and known-truth simulation. | [Validation](docs/VALIDATION_RESULTS.md) and [support dynamics](docs/dynamic-comparison.md): added complexity does not consistently improve results. |

The research separates predictions of **observed eliminations** from estimates
of **unobserved audience votes**. The first can be scored against outcomes; the
second remains conditional on assumptions.

## Run a demonstration

Python 3.12 is the verified environment. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v4-verified.txt
python -m pip install --no-deps -e ".[test,audit]"
python -m dwts.demo --out outputs/voting-demo
python -m dwts.exit_study --demo --out outputs/forecast-demo
python -m pytest -q
```

Both demos use invented data and need no report or official dataset. The voting
demo writes CSV, JSON and SVG outputs; the forecasting demo exercises the
chronological study workflow. Use a new output directory for each run. Default
tests are self-contained; data-dependent and archived-experiment tests are
explicitly deselected. See the [run guide](docs/RUNNING.md) for Windows,
dependencies and full experiments.

## Chronological elimination forecasts

The latest study compares simple, inspectable probability forecasts on the same
eligible events. Each prediction uses the current roster and judges' scores,
then freezes before the current outcome is released. Model selection uses only
earlier seasons. Online weights update from released eligible exits and reset
at each season boundary.

| Method | Outer validation: 86 weeks | Historical diagnostic: 33 weeks |
|---|---:|---:|
| Uniform probabilities | 2.0604 | 2.1539 |
| Past judges' scores | 1.6988 | 1.6213 |
| Current judges' scores | 1.5414 | 1.7546 |
| Two-feature conditional choice | **1.5040** | 1.6397 |
| Fixed online expert mixture | 1.5511 | **1.5737** |
| Selected fixed-share mixture | 1.5557 | 1.5809 |
| Selected tempered mixture | 1.5488 | 1.5806 |

Mean log loss; lower is better. Compare methods within each column. The
two-feature model has the best outer-validation point score, while the fixed
online mixture has the best point score in the historical diagnostic. Additional
weight recovery and learning-rate selection do not establish a consistent gain.
No single method is established as the best future predictor.

![Chronological elimination forecast comparison](docs/assets/exit-forecasting.svg)

All these historical data had been examined before the later experiments.
Chronological nesting controls the information available to each fit and
selection; it does not make repeated exploration an independent validation.
The diagnostic contains only six eligible seasons, and its online-mixture gain
over past judges is concentrated in season 27. [Methods, paired intervals and
aggregate results](docs/exit-forecasting.md) make these qualifications explicit.

To run the study with the separately obtained official data:

```bash
python scripts/download_data.py
python -m dwts.exit_study --data data/raw/2026_MCM_Problem_C_Data.csv --config configs/exit_forecast_study.json --out outputs/exit-forecast-study
```

## Fan support and voting rules

The broader study covers **34 seasons and 2,777 contestant-weeks**. It includes
a PyMC reference model, a Numba Metropolis–Hastings engine, explicit rule
simulations, nested season-grouped random forests, OLS and SHAP. Input hashes,
experiment configurations and diagnostic checks make assumptions inspectable.

On whole-season inferred fan-share targets, a weekly-normalized history baseline
has a better point score than the declared RF pipelines: out-of-fold R² is
0.7578 versus 0.7232 for the normalized RF. This is an internal comparison of
latent targets, rather than evidence of true-vote prediction. Both RF pipelines
select zero partner-history weight in every outer fold.

The policy study propagates 2,048 joint posterior draws through 18 policies on
291 eligible rosters. Judge saves do not universally reduce the declared
controversial-week fraction. A separate six-objective analysis finds that none
of seven baseline mean-dominance relations persists throughout either declared
sensitivity grid. These are conditional comparisons on observed rosters, rather
than simulated complete alternative seasons.

An additional deterministic percentage-rule analysis finds median feasible
fan-share widths of **99.63 percentage points for survivors** and **13.60 for
eliminated contestants** across 164 selected weeks. These coordinate bounds
illustrate how weakly an elimination alone identifies audience support under
the declared extra assumptions.

[Study results and figures](docs/STUDY_RESULTS.md) · [Policy and recovery validation](docs/VALIDATION_RESULTS.md) · [Identification](docs/identification.md)

## Documentation

| Guide | Contents |
|---|---|
| [Documentation index](docs/README.md) | A starting point for all project guides. |
| [Chronological forecasting](docs/exit-forecasting.md) | Features, information clock, nested selection, online aggregation and results. |
| [Methods and assumptions](docs/METHODS.md) | Data preparation, latent support, voting rules and retrospective analysis. |
| [Study results](docs/STUDY_RESULTS.md) | Fan-share baselines, ablations, posterior policy comparisons and sensitivity. |
| [Prediction, recovery and policy validation](docs/VALIDATION_RESULTS.md) | Earlier latent forecasts, known-truth simulation and multi-objective policy robustness. |
| [Identification](docs/identification.md) | Feasible shares and counterfactual exit ambiguity. |
| [Static and dynamic support](docs/dynamic-comparison.md) | A controlled comparison under a common normalized exit likelihood. |
| [Prospective workflow](docs/prospective.md) | Persistent local prediction/release preparation and activation requirements. |
| [Running the study](docs/RUNNING.md) | Installation, demonstrations, official data and experiment commands. |
| [Research provenance](docs/PROVENANCE.md) | Full-team credit, project leadership and the current software edition. |
| [Related work](docs/RELATED_WORK.md) | Standard method sources and related DWTS software. |
| [Earlier results](docs/RESULTS.md) | Preserved software metrics and numerical differences from the original report. |

## Scope and research materials

True audience vote totals are unavailable. Inferred shares, associations and
policy comparisons depend on their stated models; SHAP and feature importance
do not identify causal effects. Independent prospective forecasting has not yet
been completed. The [local prospective utility](docs/prospective.md) is a draft
workflow for preparing that evidence.

The current implementation was prepared after the original report with
AI-assisted development. Competition-time source code was unavailable when
this edition was prepared. The original study and report are credited to Team
2627761; subsequent implementations and extensions are described in the
[provenance guide](docs/PROVENANCE.md).

The original report, raw dataset, individual prediction records and posterior
chains are not distributed. Published results contain allowlisted aggregates.
See [data sources](data/SOURCES.md), [contributing](CONTRIBUTING.md) and
[license status](LICENSE_STATUS.md).

```text
dwts/       Models, rule analysis, forecasting and demonstrations
configs/    Explicit experiment definitions
tests/      Numerical, workflow and scientific regression checks
results/    Aggregate study results
docs/       Methods, figures, sources and running instructions
scripts/    Data acquisition, figure rendering and packaging utilities
```
