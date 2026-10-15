# Contributing

First read the [license status](LICENSE_STATUS.md) and [provenance](docs/PROVENANCE.md).
This is a public review repository; contribution and reuse permissions remain to be confirmed.

For a bug report, give the command, input/config hash, Python/package versions, actual
output and expected behavior. Prefer a minimal synthetic example. Never attach tokens,
private materials or raw posterior chains to an issue.

For changes:

1. Distinguish software bugs from new statistical assumptions.
2. Add a regression test before changing numerical behavior.
3. Preserve original strings and provenance when normalizing data.
4. Never choose a change because it happens to approach a printed report number.
5. Run `python -m pytest -q`; list any additional data/archive checks separately.
6. Describe changed outputs, assumptions and relevant limits in the relevant documentation or pull request.

Tests with exact-reuse guards intentionally reject even tiny changes. Do not loosen
those checks to make old experiments appear reusable. Create a new run directory instead.
