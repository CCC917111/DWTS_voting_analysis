# Method background and related work

This page positions the implemented methods. It is a selected comparison, not a
systematic review or a claim of research priority. The project's contribution is
the DWTS application, its explicit assumptions and the evidence from controlled
inference, forecasting and rule-comparison studies.

## Choice and ranking models

Luce's choice formulation and the Plackett–Luce ranking model are established
background for normalized choice probabilities. The current forward model uses
a softmax link over exit merits. Those merits depend on the current roster,
judge shares and fan shares or ranks. Its odds therefore need not satisfy the
fixed-worth independence property of a classical Plackett–Luce model. It is a
normalized categorical exit model, rather than an implementation of a complete
fixed-worth ranking distribution.

- Luce, R. D. (1959). *Individual Choice Behavior: A Theoretical Analysis*.
- Plackett, R. L. (1975). *The Analysis of Permutations*.
  [DOI: 10.2307/2346567](https://doi.org/10.2307/2346567).
- Turner, H. L., van Etten, J., Firth, D., and Kosmidis, I. (2020).
  *Modelling Rankings in R: The PlackettLuce Package*.
  [DOI: 10.1007/s00180-020-00959-3](https://doi.org/10.1007/s00180-020-00959-3).
  The [author-maintained overview](https://github.com/hturner/PlackettLuce/blob/411b1ef67e906ecad0b820dd54504712307cc14f/vignettes/Overview.Rmd)
  supplies the relevant choice and ranking formulas.

## Sequential inference and probability forecasts

Sequential importance weighting, ESS resampling and Metropolis moves are
standard methods. The static forward engine uses a complete-history target for
its resample–move step. A dynamic bootstrap filter instead propagates its state
through a specified transition; applying the static-state target to a dynamic
current state would describe a different model.

The [particles author tutorial](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/docs/source/notebooks/SMC_samplers_tutorial.ipynb)
and [implementation](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/particles/smc_samplers.py)
provide IBIS and full-history posterior background. They accompany Chopin and
Papaspiliopoulos's *An Introduction to Sequential Monte Carlo*.

Log loss and Brier scores are established proper scoring rules. They evaluate
the predicted observable exit, rather than the accuracy of unobserved fan
shares. Calibration, approximation error and strong baselines provide separate
checks.

- Gneiting, T., and Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
  Prediction, and Estimation*.
  [DOI: 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
- Gneiting, T., and Katzfuss, M. (2014). *Probabilistic Forecasting*.
  [DOI: 10.1146/annurev-statistics-062713-085831](https://doi.org/10.1146/annurev-statistics-062713-085831).
- The [scoringRules author manuscript](https://github.com/FK83/scoringRules/blob/4d2f12f2097243a8447db037745395f5d5cf43da/vignettes/article.Rnw)
  explains comparative probability-forecast evaluation. Its bibliography verifies
  the two references above; this project does not claim an exhaustive reading of
  every original source.

## Online expert aggregation

The [chronological exit study](exit-forecasting.md) uses established aggregation
methods. Its fixed mixture averages four probability forecasts, then updates
their weights using the observed exit likelihood. The cumulative log-loss
comparison follows the standard mixture argument. Resetting every season pays
the prior-weight constant separately for each season.

Fixed-share restores weight to experts through a transition after outcome
release. It is related to the standard tracking-experts construction; this
implementation shares toward the declared nonuniform initial prior. Tempered
updates change the likelihood exponent. They do not generally preserve the
exact Bayesian identity or its original guarantee.

- Cesa-Bianchi, N., and Lugosi, G. (2006). *Prediction, Learning, and Games*.
  Cambridge University Press.
  [DOI: 10.1017/CBO9780511546921](https://doi.org/10.1017/CBO9780511546921).
- Herbster, M., and Warmuth, M. K. (1998). *Tracking the Best Expert*.
  *Machine Learning*, 32, 151–178.
  [DOI: 10.1023/A:1007424614876](https://doi.org/10.1023/A:1007424614876).

The project contributes the declared DWTS information clock, chronological
comparisons and application-specific evidence. It does not claim a new expert
aggregation algorithm or infer audience-vote accuracy from elimination scores.

## Related DWTS software

Two public student projects illustrate substantial application overlap:

| Project at the inspected commit | Related implementation |
|---|---|
| [DWTS-Fan-Vote-Estimator](https://github.com/aidanhaya/DWTS-Fan-Vote-Estimator/tree/24942a3a77847009bce1502b4b3652da09f800e1) | Dynamic support logits, score-change drift and particle-based vote inference. |
| [MCM-2026-DWTS-Voting-Analysis](https://github.com/Jamie0504/MCM-2026-DWTS-Voting-Analysis/tree/aa6fff93f5022589e8557b2afc35c91c85892bd3) | Dirichlet constraint inference, rank/percentage mechanisms and rule uncertainty. |

These are related software, not peer-reviewed publications. Their retrospective
consistency metrics are not interchangeable with this project's chronological
log loss. No head-to-head performance advantage or competition-time priority is
established by this comparison. Dynamic support and Bayesian vote inference
are not claimed as new general methods here.

The controlled comparison holds the observation model and information clock
fixed to examine when a dynamic state helps. The identification analysis asks
what the observable constraints alone imply under an additional deterministic
rule. Neither study recovers independently measured true votes. See
[identification](identification.md), [dynamic support](dynamic-comparison.md) and
[prospective preparation](prospective.md) for the specific study definitions.
