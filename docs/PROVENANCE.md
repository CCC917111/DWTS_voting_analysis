# Provenance and contribution boundaries

## Source materials

- Report: Team 2627761, 2026 MCM Problem C, *Modeling Fairness in Competitive Voting Systems: A Bayesian Approach to Infer Fan Support and Evaluate Contest Dynamics in DWTS*.
- Report PDF SHA-256: `c731e44d92c84d6b667ce76b3a4a73085b555c5c12db7fd2afadbdfb5787999a`.
- Original team source code was not supplied to the reconstruction effort.
- Dataset: COMAP's 2026 Problem C data. Details and pinned checksum: [SOURCES.md](../data/SOURCES.md).
- Immediate software/evidence source: `DWTS_Full_Report_Reproduction_v5.zip`, SHA-256 `5452dc3e638bddccbba11a0e3a2686972eaa8becaaf5d4b83a953c5fe8669988`.

## Layers of work

| Layer | Attribution / scope |
|---|---|
| Original report and its claims | Team 2627761; preserved as the source report attribution |
| Public dataset/problem | COMAP; no ownership or license transfer implied |
| Reconstructed implementation and audits | Independent, assistant-assisted reconstruction from the supplied report and available data |
| Repository presentation | New synthetic demo, documentation, test tiers, workflow and publication checks |
| Maintainer / participant identity | Awaiting confirmation; no individual report authorship inferred |

The current account owner's relationship to Team 2627761 is not established by this package. GitHub account ownership alone does not establish authorship, permission, or academic contribution. Before using first-person descriptions in applications, presentations or a profile, record what the person actually designed, implemented, validated and can explain.

## Integrity and changes

[Source inventory](source_inventory.json) records hashes of the copied v5 implementation, configs, tests and retained audit scripts before packaging. Numerical/scientific v5 modules and all original test files are unchanged. The package version was aligned to `0.5.1`; a new synthetic demo was added; metadata and test selection were changed. See [CHANGELOG.md](../CHANGELOG.md).

Small metric CSVs in `results/v5_snapshot/` are byte-preserved saved v5 outputs. The small JSON manifest records their hashes and context. They are not new fitting results.

The v5 archive is unchanged and remains the separate, detailed historical evidence bundle. This repository does not redistribute the source PDF, extracted report text, original raw CSV, person-level posterior tables, fitted estimators, full historical output tree, or raw MCMC chains.

## Citation guidance

Cite the original report separately from this reconstruction. A suitable report reference is:

Team 2627761. 2026. *Modeling Fairness in Competitive Voting Systems: A Bayesian Approach to Infer Fan Support and Evaluate Contest Dynamics in DWTS*. MCM Problem C report.

A repository citation should be added only after its real maintainer, approved authors, release URL and license are confirmed. No DOI, affiliation, individual author list or publication URL is fabricated here.
