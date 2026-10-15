# Methods and declared assumptions

This describes the current software implementation of the team's DWTS study. Research authorship and the current code edition are recorded in [PROVENANCE.md](PROVENANCE.md).

## Data and preprocessing

`dwts/data.py` converts the COMAP wide-format score table into contestant-season-week rows. It distinguishes missing future weeks, all-zero post-exit weeks, explicit elimination labels and withdrawals. Score means include available judges; weekly standardization uses population SD. Returning celebrities retain separate season identifiers.

The full v5 configuration uses explicit elimination labels first, omits final-week elimination constraints, and retains scored competitors in mixed withdrawal/exit weeks without treating a withdrawal itself as a voting exit. Alternative configurations are sensitivity analyses. The baseline has 421 contestant-season entries, 408 names and 2,777 active contestant-weeks across 34 seasons.

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

The `dwts.analysis` entry point fits OLS using the current pair's prior-week judge
shares, with explicit first-week and placement-unit alternatives. It compares late
seasons using judge save and leaves undefined multi-exit save cases unscored.
Controversial-week fractions, events per week and events per contestant row have
separate denominators. Both influence-gap aggregation orders are retained, with
sample populations and assumptions recorded in each output directory.

## Extended benchmark and uncertainty studies

`dwts.benchmarks` evaluates simple share baselines and RF feature ablations on the
same five outer season folds. Both RF and history baselines are compared before
and after dividing predictions by their weekly sum. This uses the complete
observed roster and predicted values, without consulting validation targets.
Raw and normalized RF pipelines select their history weights independently in
the inner folds. Both predict the same posterior mean fan share;
`rf_fan_scale="relative"` changes historical features, not the target. The
normalization hypothesis was motivated by examination of this dataset, so this
is exploratory internal validation. A paired bootstrap resamples whole seasons
from fixed out-of-fold predictions, without refitting or propagating latent-vote
uncertainty.

`dwts.uncertainty` propagates stored joint seasonal posterior draws through
18 policies: two rule families, three judge weights and three save mechanisms.
The scope retains nonfinal observed rosters with at least three contestants and
excludes withdrawal weeks. Every policy removes one contestant per week,
including observed no-exit or multi-exit weeks. Controversy means eliminating a
judge-top-two/fan-bottom-half contestant (Type A) or a judge-bottom-two/fan-top-half
contestant (Type B). Its denominator is the number of eligible weeks. Policies
are paired within the same draw, weeks retain their joint seasonal draw, and
independently fitted seasons use independent seeded draw permutations.

Fan ranks and fan save/tie-break order use latent logits, preserving genuine
ordering even when numeric softmax underflows. Percentage aggregate merits use
numeric shares. Credible intervals are conditional on the fitted model and
observed elimination labels; they do not establish true-vote accuracy or a
universal policy ranking. Matched alternative likelihood/penalty fits retain
identical observed rosters. Their means and separate credible intervals describe
sensitivity; they have no naturally paired cross-model posterior difference.
See [the extended results](STUDY_RESULTS.md) for the fixed protocols and outputs.

`dwts.prior_sensitivity` tests a specific joint prior-scale family on seasons
1–2 and 28–34. Scaling alpha, beta and sigma by a positive constant c sends the
latent logits to c times their baseline values and leaves every rank-likelihood
term unchanged. The prior-density change cancels the transformation Jacobian,
so this is an exact posterior pushforward for priors with alpha SD 2c, beta SD c
and sigma Half-Cauchy scale c, while epsilon remains standard Normal. The study
uses c in {0.5, 1, 2}. It requires no new MCMC and preserves rank-policy outcomes
draw by draw, but percentage-policy outcomes may change as share magnitudes
change. This identity does not hold for percentage-likelihood seasons or
arbitrary prior changes. Numerical summaries retain the baseline Monte Carlo
approximation; transformed-function ESS was not recomputed.

## Exact differential reuse

`scripts/run_identity_cv.py` compares target values, design matrices, feature names, settings, environment and archived model/feature code before reusing results. Changed inputs or selected weights force refits. A zero-fit plan is the default; execution is explicit. Tests exercise tiny-value changes, incorrect saved metrics, altered targets and output protection.

## Limits when comparing with the original report

The competition-time source and complete row-level outputs are unavailable in this code edition. Evaluation-row lists, OOF predictions and folds, metric weights, cold-start initialization and some tie/save conventions are incompletely recorded. The current implementation declares its own settings; optional historical comparisons retain these limitations.

## Execution and reference checks

Default model runs produce study estimates, tables and figures without comparing
them to printed target values. Standalone analysis validates source diagnostics
and returns success when its scientific calculations complete. Optional original-
report comparisons retain their own status and do not define analysis success.
