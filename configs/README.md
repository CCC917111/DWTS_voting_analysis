# Experiment configurations

- `study.json`: main full-study configuration, with all 34 seasons, four chains per season, full nested RF validation and verified partner identities.
- `smoke.json`: short three-season execution check; inadequate for research inference.
- `bottom_two_sensitivity.json`: explicitly different late-season likelihood sensitivity.
- Other configurations retain earlier experiments and pilot settings. Their filenames identify historical runs; `study.json` has the same modeling settings as the preserved v5 full-study configuration.
- `report_verification_v5.json` is used only by the optional original-report comparison command.

The sampler's legacy `likelihood="report"` value selects the declared elimination-order penalty. It does not require a PDF or numerical agreement with the report.

Changing a configuration changes the experiment. Keep the actual configuration with
its output and use a new output directory. Analysis choices must follow the stated
research question rather than a target metric.
