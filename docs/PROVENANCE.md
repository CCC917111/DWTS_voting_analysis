# Research team and software provenance

## Original research

Team 2627761 developed the DWTS study for the 2026 MCM Problem C:
*Modeling Fairness in Competitive Voting Systems: A Bayesian Approach to Infer Fan
Support and Evaluate Contest Dynamics in DWTS*.

Yike Chen participated in the original team research and leads the current DWTS
Voting Analysis project. The report and its findings are credited to the full
team. This repository presents the study's current software implementation and
subsequent extensions.

## Current software edition

The present implementation was prepared after the report with AI-assisted
development. Competition-time source code was not available when this edition
was prepared. Its baseline implementation hashes are recorded in
[source_inventory.json](source_inventory.json); intentional workflow updates and
later additions are recorded separately.

This edition includes Bayesian inference, voting-rule analysis, retrospective
models, exact rank-tie handling, partner-identity normalization, diagnostic
checks, tests and a synthetic demonstration.

The extended studies add equally constrained simple baselines, feature ablations,
paired season-bootstrap comparisons, joint posterior policy uncertainty and an
analytic rank-only prior-scale sensitivity. Their protocols and aggregate
results are linked from [STUDY_RESULTS.md](STUDY_RESULTS.md). These are subsequent
extensions of the project; they are not presented as competition-time outputs.

Further validation adds a separate normalized categorical forward model,
chronological elimination backtests, replicated known-truth recovery and
multi-objective policy robustness. [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md)
records these experiments and their limitations. They do not establish
independent prospective validation or transfer the original sampler's
diagnostics to the new finite-particle inference engine.

The subsequent identification study quantifies feasible shares and policy exits
under an additional deterministic rule. The controlled support study compares
static and dynamic bootstrap filters, and the prospective utility prepares a
local prediction/release workflow. These are later extensions; they do not
establish measured true votes or completed independent prospective validation.
Selected method sources and related software are listed in
[RELATED_WORK.md](RELATED_WORK.md).

The subsequent [conditional-exit and online-expert study](exit-forecasting.md)
adds score-history features, nested chronological selection and standard expert
aggregation. It includes unsuccessful adaptive extensions alongside the simpler
controls. These are later software studies on previously examined data, rather
than original competition-time results or independent future validation.

Saved results under `results/v5_snapshot/` belong to this software edition. They
are distinct from the original report's values; [RESULTS.md](RESULTS.md) explains
the numerical differences and evaluation limits. A successful software run does
not establish numerical equivalence to every reported result.

## Research materials

- Original report PDF SHA-256: `c731e44d92c84d6b667ce76b3a4a73085b555c5c12db7fd2afadbdfb5787999a`.
- Data: COMAP's 2026 Problem C dataset; publisher link and checksum in [SOURCES.md](../data/SOURCES.md).

The source report, extracted report text, raw COMAP CSV, per-person posterior
outputs, fitted estimators and sampling chains are not distributed here. Software,
report and data rights are separate; see [LICENSE_STATUS.md](../LICENSE_STATUS.md).
