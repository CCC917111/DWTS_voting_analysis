# Recorded v5 results and interpretation

Snapshot date: 2026-10-14. These are preserved aggregate results from the completed v5 reconstruction, whose audit work was performed on 2026-10-13. Repository packaging did not retrain these full research models.

## Six retrospective tasks

All use 2,777 contestant-week rows and five outer season-grouped folds. Values below are pooled out-of-fold scores. Targets in fan tasks are inferred shares, not observed votes.

| Task | Metric | v5 | Source report |
|---|---|---:|---:|
| Judge mean | R² | 0.434488 | 0.146 |
| Judge mean | RMSE | 1.100571 | 5.320 |
| Fan share | R² | 0.555199 | 0.614 |
| Fan share | RMSE | 0.063604 | 0.072 |
| Judge top-half | F1 | 0.684151 | 0.766 |
| Judge top-half | AUC | 0.772207 | 0.778 |
| Fan top-half | F1 | 0.745946 | 0.803 |
| Fan top-half | AUC | 0.842325 | 0.879 |
| Judge bottom-two | F1 | 0.487562 | 0.488 |
| Judge bottom-two | AUC | 0.720169 | 0.755 |
| Fan bottom-two | F1 | 0.443038 | 0.547 |
| Fan bottom-two | AUC | 0.705071 | 0.775 |

Judge RMSE uses judge-mean score units; fan RMSE uses inferred-share units. Author units, targets and aggregation are not fully identified. A smaller RMSE does not establish a superior comparable model. Average-rank labels also need not yield exactly two bottom-two positives when ties occur.

Machine-readable evidence:
- [rf_metrics.csv](../results/v5_snapshot/rf_metrics.csv): current verified-identity branch, byte-preserved from v5
- [numerical_verification.csv](../results/v5_snapshot/numerical_verification.csv): printed-reference comparisons plus separately labeled derived/assumed references
- [snapshot_manifest.json](../results/v5_snapshot/snapshot_manifest.json): provenance and integrity hashes

## What changed in v5

1. **Exact rank ties.** Floating arithmetic had bypassed the implementation's declared tie rule. Exact decimal/rational ordering changed 169 baseline week×weight ranking cells and 80 week×weight×mechanism elimination cells. These are cells, not independent weeks. Weight-.5 elimination sets, Tables 5–8 and percentage results did not change.
2. **Partner identity.** Six guest annotations were separated and two reviewed aliases merged, preserving original strings. The canonical branch changed 73 rows. Both regression tasks again selected history weight zero in every outer fold, so their predictions remained unchanged; several classification scores changed.
3. **Metric identifiability.** Under stated pooled definitions, the current judge-mean range cannot explain the printed R²/RMSE pair just by selecting or nonnegatively weighting existing rows. This conditional result does not prove the report is wrong. For fans, a predeclared hierarchical weighting provides a necessary-condition counterexample to any blanket impossibility claim, but the saved predictions still do not match both printed metrics.
4. **OLS unit checks.** Four predefined score quantities did not recover the complete Figure 5 coefficient-and-interval pattern. No closest candidate was relabeled as the author's method.

Detailed per-row evidence remains in the separate v5 archive. This lean repository does not reproduce that entire evidence tree or the source report.

## Numerical comparison status

Full report reproduction: **not established**.

The legacy comparison ledger contains 54 rows: 48 printed-reference comparisons, two derived arithmetic comparisons and four unreported-denominator assumptions. Six of the 48 printed-reference comparisons are numerically compatible under declared tolerances. Multiple rows can concern the same source claim; this is not an independent-test count or a whole-report success rate.

The historical 297-row full-report inventory is an organizational coverage inventory, not 297 independent statistical tests. It is not bundled into this repository and is not used as a quality badge.

## Claims to avoid

- “Predicts true fan votes” or “100% prediction accuracy”
- “Professional dancers causally explain 61.4% of fan support”
- “The original authors' implementation has been recovered”
- “All weighting choices are mathematically impossible”
- “Fully reproduced” because software tests or a diagnostic threshold passed

The defensible conclusion is narrower: the reconstructed software runs, exposes assumptions, verifies important internal properties, corrects demonstrated implementation issues, and documents unresolved disagreements with a source report.
