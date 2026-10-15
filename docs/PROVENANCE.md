# Research team and software provenance

## Original research

Team 2627761 developed the DWTS study for the 2026 MCM Problem C:
*Modeling Fairness in Competitive Voting Systems: A Bayesian Approach to Infer Fan
Support and Evaluate Contest Dynamics in DWTS*.

Yike chen participated in the original team research. The report and its findings
are credited to the full team. This repository presents the study's current
software implementation and subsequent extensions.

## Current software edition

The present implementation was prepared after the report with AI-assisted
development. Competition-time source code was not available when this edition
was prepared. Its baseline implementation hashes are recorded in
[source_inventory.json](source_inventory.json); intentional workflow updates and
later additions are recorded separately.

This edition includes Bayesian inference, voting-rule analysis, retrospective
models, exact rank-tie handling, partner-identity normalization, diagnostic
checks, tests and a synthetic demonstration.

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
