# Results and interpretation

Snapshot date: 2026-10-14. These are saved aggregate results from the v5 analysis. The quick-start demo does not fit these research models.

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

## Implementation checks

Exact decimal/rational ordering preserves declared rank ties. At judge/fan weight
0.5, this correction left elimination sets unchanged; other weight settings can
produce different results. Partner normalization separates six guest annotations
and merges two reviewed aliases while preserving original strings. The canonical
branch changed 73 rows. Both regression tasks selected history weight zero in
every outer fold; several classification scores changed with the identity policy.

The corresponding methods and assumptions are documented in [METHODS.md](METHODS.md).

## Reference comparison limits

Values from the cited report are shown as references. Its evaluation rows, metric
weights, units and several model settings are incompletely specified; numerical
equivalence has not been established. These columns do not form a like-for-like
model leaderboard, and passing software tests does not resolve the differences.

Under the stated pooled definitions, the current judge-mean range cannot explain
the reference R²/RMSE pair simply by selecting or nonnegatively weighting existing
rows. This conditional result does not establish an error in the report. The fan
metrics also remain different. Four predefined OLS score quantities did not match
the complete reference coefficient-and-interval pattern.

True audience vote totals are unavailable. Fan targets are whole-season latent
estimates, so these metrics describe retrospective associations. They do not
measure prospective prediction of real votes or establish causal partner effects.
