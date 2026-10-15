# Sources and contribution scope

The methods are based on Team 2627761's 2026 MCM Problem C report,
*Modeling Fairness in Competitive Voting Systems: A Bayesian Approach to Infer Fan
Support and Evaluate Contest Dynamics in DWTS*.

- Report PDF SHA-256: `c731e44d92c84d6b667ce76b3a4a73085b555c5c12db7fd2afadbdfb5787999a`.
- Data: COMAP's 2026 Problem C dataset; publisher link and checksum in [SOURCES.md](../data/SOURCES.md).
- Initial implementation bundle: independently implemented v5 package,
  `DWTS_Full_Report_Reproduction_v5.zip`, SHA-256
  `5452dc3e638bddccbba11a0e3a2686972eaa8becaaf5d4b83a953c5fe8669988`.

The analysis workflow was implemented independently with AI assistance, using
methods described in the cited report. The original team's source code was not
supplied. The original report and its claims remain attributed
to Team 2627761; repository maintenance does not imply authorship of that report.

The software includes inference, voting-rule comparisons and
retrospective attribution models. The synthetic demo, analysis entry point,
repository documentation and supporting tests are subsequent software contributions.
[Baseline hashes](source_inventory.json) distinguish the preserved v5 modules/tests
from later additions. Small files in `results/v5_snapshot/` are preserved historical
aggregate results, not fresh fitting outputs.

The repository does not distribute the source report, extracted report text, raw
COMAP CSV, per-person posterior outputs, fitted estimators or sampling chains.
Software, report and data rights are separate; see [LICENSE_STATUS.md](../LICENSE_STATUS.md).
[RESULTS.md](RESULTS.md) records differences from the report's reference values.
Evaluation definitions are incompletely specified, so numerical equivalence has
not been established and the values are not a like-for-like model leaderboard.
