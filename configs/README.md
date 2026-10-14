# Configuration map

- `smoke.json`: short three-season execution check; inadequate for research inference.
- `report_reconstruction_v5.json`: preferred full fresh run with verified partner identities.
- `report_verification_v5.json`: preferred downstream reanalysis of frozen posterior summaries.
- `report_final.json`, `report_full.json`, `report_reconstruction.json`, `report_verification.json`: preserved historical settings; do not silently treat them as v5 identity-corrected settings.
- Other files preserve earlier pilots or explicitly different likelihood/data sensitivity settings.

Changing a configuration changes the experiment. Keep the actual config with its output,
use a new output directory and do not select assumptions to approach a printed report metric.
