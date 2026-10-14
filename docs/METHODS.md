# Methods and declared assumptions

This describes the reconstructed implementation, not recovered author code. The original report's claims remain attributed to Team 2627761.

## Data and preprocessing

`dwts/data.py` converts the COMAP wide-format score table into contestant-season-week rows. It distinguishes missing future weeks, all-zero post-exit weeks, explicit elimination labels and withdrawals. Score means include available judges; weekly standardization uses population SD. Returning celebrities retain separate season identifiers.

The v5 reconstruction configuration uses explicit elimination labels first, omits final-week elimination constraints, and retains scored competitors in mixed withdrawal/exit weeks without treating a withdrawal itself as a voting exit. Alternative configurations are sensitivity analyses, not recovered author definitions. The audited baseline has 421 contestant-season entries, 408 names and 2,777 active contestant-weeks across 34 seasons.

## Latent fan support

Within each season, the noncentered model forms logits from a contestant-season intercept, standardized judge scores and Gaussian latent noise; softmax gives within-week fan shares. Priors are alpha ~ Normal(0, 2²), beta ~ Normal(0, 1), sigma ~ HalfCauchy(1), and epsilon ~ Normal(0, 1). A shared weekly intercept cancels in softmax and is integrated out.

Elimination information enters as a soft penalty on violations of a declared ordering. The multi-exit extension separates the eliminated set from survivors without inventing an internal order. The optional bottom-two likelihood uses membership rather than assuming observed judges' save decisions are fully available.

PyMC supplies the reference target. A Numba MH implementation uses local recomputation, independently tested against that target including the log-sigma Jacobian. This checks the implementation's target density; it does not prove equivalence to the report authors' unpublished sampler. Rank likelihoods compare logits directly to avoid softmax underflow creating false ties.

The full configuration stores 4,000 draws per chain after 6,000 tuning sweeps, using 4 chains and 20 post-tuning sweeps per saved draw. Diagnostics use unrounded rank-normalized R-hat <= 1.01 and bulk/tail ESS >= 400. Passing the gate is a screening condition, not proof that every mode was explored. The smoke configuration is deliberately too small to justify inference.

## Rule comparisons and ties

`dwts/rules.py` implements rank and percentage aggregation plus explicit save variants. Ties use fewer fan votes, then lower judge score, then lexical contestant ID. Exact rational weighted-rank keys preserve genuine decimal-grid ties; no tolerance merges nearby distinct weights. The author tiebreak remains unknown.

Counterfactuals operate on each week's observed roster. They do not generate a complete alternative season or invent future performances. Bottom-three extensions and controversy denominators are explicit assumptions; they must remain visible when comparing results.

## Professional-partner identity

`dwts/partner_identity.py` separates six season-34 guest annotations from the regular professional identity and optionally merges two reviewed aliases. Raw strings, guest/week information and the policy label remain available. The verified-identity stage changes 73 rows and reduces 60 raw string keys to 52 identity keys. It does not allocate week-9 scores to guests or alter posterior targets.

The implementation records direct source links for the reviewed mappings. The normalization is independently supported; it is not evidence that the report used the same preprocessing.

## Attribution and validation

Random forests use age, one-hot industry and lagged partner-quality features. The recorded v5 protocol uses 300 trees, minimum leaf size 5, five outer season-grouped folds, three inner season-grouped folds and history weights {0, .25, .5, .75, 1}. Inner validation selects weights without reading report target metrics. Pooled out-of-fold metrics are separate from means of fold metrics.

Training histories exclude held-out seasons. Earlier weeks within a held-out season are allowed for sequential features. However, the fan-share targets were inferred with whole-season information; lagged features and grouped splits do not turn this retrospective analysis into prospective forecasting.

OLS and RF retain their documented target/feature conventions. SHAP magnitudes quantify model dependence on features, not direction or causal influence. At selected history weight zero, a partner feature contains the current couple's past performance and does not identify a professional dancer's standalone causal effect. Downstream models use posterior means and do not fully propagate latent-vote uncertainty.

## Exact differential reuse

`scripts/run_identity_cv.py` compares target values, design matrices, feature names, settings, environment and archived model/feature code before reusing results. Changed inputs or selected weights force refits. A zero-fit plan is the default; execution is explicit. Tests exercise tiny-value changes, incorrect saved metrics, altered targets and output protection.

## Open identification problems

Missing author material includes row-level posterior summaries, evaluation-row lists, complete OOF predictions and folds, metric weights, cold-start initialization, tie handling and some save-rule definitions. More parameter search is not a substitute for those materials.
