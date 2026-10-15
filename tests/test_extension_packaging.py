"""Publication boundaries for new research outputs and prospective state."""
import json
from pathlib import Path
import zipfile

from scripts.package_repository import package, selected_files


def test_source_release_keeps_individual_states_and_witnesses_private(tmp_path):
    public = {
        "results/identification_study/width_summary.csv",
        "results/identification_study/policy_ambiguity.csv",
        "results/dynamic_study/metrics.csv",
        "results/dynamic_study/synthetic_metrics.csv",
    }
    private = {
        "results/identification_study/private_bounds.csv",
        "results/identification_study/private_weekly.csv",
        "results/identification_study/private_policy_candidates.csv",
        "results/identification_study/witnesses.json",
        "results/dynamic_study/private_predictions.csv",
        "results/dynamic_study/private_synthetic_rows.csv",
        "results/dynamic_study/private_filter_diagnostics.csv",
        "results/prospective/state.json",
        "results/prospective/prediction_payload.json",
        "outputs/prospective/state.json",
        "docs/misplaced-state/private_state.json",
        "dwts/misplaced-state/.snapshot-interrupted.json",
        "references/report.pdf",
        "data/raw/2026_MCM_Problem_C_Data.csv",
    }
    for rel in public | private:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "9.8.7"\n')
    selected = {p.relative_to(tmp_path).as_posix() for p in selected_files(tmp_path)}
    assert public <= selected
    assert not private & selected
    archive = tmp_path / "source.zip"
    package(archive, root=tmp_path)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("dwts-voting-analysis/RELEASE_MANIFEST.json"))
        assert manifest["version"] == "9.8.7"
        members = {n.removeprefix("dwts-voting-analysis/") for n in z.namelist()}
        assert public <= members
        assert not private & members
