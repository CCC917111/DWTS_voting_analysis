# Before a GitHub release

Repository destination and public visibility were explicitly approved on 2026-10-14. Licensing and individual contribution attribution remain separate decisions.

## Required decisions

- [ ] Confirm the user's relationship to Team 2627761 and exactly which contributions belong to them.
- [x] Authorized destination: `CCC917111/dwts-voting-audit`, public visibility.
- [ ] Confirm code provenance and select a software license with the authorized holder(s).
- [ ] Review source-report, dataset and aggregate-output redistribution permissions.
- [ ] Finalize honest contributor names, citation metadata and any requested AI-assistance disclosure.

## Technical checks

- [ ] Run the documented commands in a fresh environment; preserve the actual outputs.
- [ ] Run the default tests and clearly state excluded data/archive checks.
- [ ] Verify demo inputs are synthetic and historical metric snapshots are labeled.
- [ ] Recheck scientific module hashes against the source inventory.
- [ ] Keep `data/raw/`, `references/`, `outputs/`, virtual environments, private paths, credentials and raw chains out of tracked files.
- [ ] Review `git diff --cached` and the full file list before the first commit/push.
- [ ] After publication is authorized, verify the actual remote commit and CI before claiming hosted checks passed.

## Suggested repository metadata, subject to confirmation

- Name: `dwts-voting-audit` (confirmed)
- Description: `Auditable retrospective reconstruction of DWTS fan support and voting rules, with explicit assumptions and reproducibility checks.`
- Topics: `bayesian-inference`, `reproducible-research`, `voting-systems`, `pymc`, `model-audit`

Avoid “official,” “original source,” “fully reproduced,” “true vote prediction,” or
unqualified “causal dancer influence.” No homepage or CI badge is included until a real URL exists.
