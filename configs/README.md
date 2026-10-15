# Experiment configurations

- `study.json`: main full-study configuration, with all 34 seasons, four chains per season, full nested RF validation and verified partner identities.
- `benchmark_study.json`: same-fold fan-share baselines, raw/normalized RF pipelines, feature ablations and a paired season bootstrap; this is an exploratory internal comparison.
- `uncertainty_study.json`: 18 posterior policy comparisons on fixed eligible weekly rosters, plus matched likelihood and penalty sensitivities.
- `prior_sensitivity_study.json`: exact analytic posterior transport for the nine rank-likelihood seasons under joint prior-scale factors 0.5, 1 and 2; no new MCMC fits or percentage-likelihood transport.
- `study_late_bottom_two.json`: seasons 28–34 with the alternative bottom-two likelihood; all other fit and preprocessing settings match `study.json`.
- `study_season3_penalty_0p1.json` and `study_season3_penalty_10.json`: season 3 with the declared elimination-penalty strength changed; all other fit settings match `study.json`.
- `smoke.json`: short three-season execution check; inadequate for research inference.
- `bottom_two_sensitivity.json`: explicitly different late-season likelihood sensitivity.
- Other configurations retain earlier experiments and pilot settings. Their filenames identify historical runs; `study.json` has the same modeling settings as the preserved v5 full-study configuration.
- `report_verification_v5.json` is used only by the optional original-report comparison command.

The sampler's legacy `likelihood="report"` value selects the declared elimination-order penalty. It does not require a PDF or numerical agreement with the report.

Changing a configuration changes the experiment. Keep the actual configuration with
its output and use a new output directory. Analysis choices must follow the stated
research question rather than a target metric.
