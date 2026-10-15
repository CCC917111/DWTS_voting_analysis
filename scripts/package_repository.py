"""Build a source ZIP from an explicit allowlist, excluding private research inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {'dwts', 'configs', 'scripts', 'tests', 'docs', 'examples', '.github', 'results'}
TOP_FILES = {'README.md', 'LICENSE_STATUS.md', 'CONTRIBUTING.md',
             'pyproject.toml', 'requirements-v4-verified.txt', '.gitignore'}
EXCLUDED_PARTS = {'__pycache__', '.pytest_cache', '.git', '.venv', 'build', 'dist'}
PRIVATE_STATE_FILES = {'private_state.json', '.writer.lock'}
ALLOWED_SUFFIXES = {'.py', '.md', '.toml', '.txt', '.json', '.csv', '.svg', '.yml', '.yaml'}
PUBLIC_RESULT_FILES = {
    'v5_snapshot': {'rf_metrics.csv', 'numerical_verification.csv', 'snapshot_manifest.json'},
    'study_benchmarks': {'metrics.csv', 'fold_metrics.csv', 'paired_differences.csv',
                         'weight_selection.csv', 'folds.csv', 'protocol.json', 'status.json'},
    'study_uncertainty': {'policy_summary.csv', 'policy_contrasts.csv', 'sensitivity_summary.csv',
                          'protocol.json', 'source_integrity.json', 'status.json'},
    'study_prior_sensitivity': {'policy_summary.csv', 'policy_contrasts.csv',
                                'prior_scale_sensitivity.csv', 'protocol.json',
                                'source_integrity.json', 'status.json'},
    'recovery_study': {'recovery_summary.csv', 'coverage_probe.csv', 'inference_checks.csv',
                       'approximation_checks.csv', 'protocol.json', 'status.json'},
    'forecast_study': {'metrics.csv', 'season_metrics.csv', 'paired_differences.csv',
                       'calibration.csv', 'particle_stability.csv', 'model_selection.csv',
                       'eligibility_summary.csv', 'protocol.json', 'status.json',
                       'mc_assessment.json'},
    'policy_robustness': {'policy_metrics.csv', 'policy_contrasts.csv', 'pareto_stability.csv',
                          'worst_case_metrics.csv', 'protocol.json', 'source_integrity.json',
                          'status.json'},
    'identification_study': {'protocol.json', 'width_summary.csv', 'eligibility_summary.csv',
                             'policy_ambiguity.csv', 'policy_candidate_counts.csv',
                             'numerical_checks.json', 'status.json'},
    'dynamic_study': {'metrics.csv', 'season_metrics.csv', 'paired_differences.csv',
                       'calibration.csv', 'model_selection.csv', 'eligibility_summary.csv',
                       'particle_stability.csv', 'mc_assessment.json', 'synthetic_metrics.csv',
                       'synthetic_paired_differences.csv', 'synthetic_stability.csv',
                       'protocol.json', 'status.json'},
}

def selected_files(root=ROOT):
    for p in sorted(Path(root).rglob('*')):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root)
        if set(rel.parts) & EXCLUDED_PARTS:
            continue
        if rel.name in PRIVATE_STATE_FILES or rel.name.startswith('.snapshot-'):
            continue
        if rel.as_posix() in TOP_FILES or rel.as_posix() == 'data/SOURCES.md':
            yield p
        elif rel.parts[0] in DIRECTORIES and p.suffix in ALLOWED_SUFFIXES:
            if rel.parts[0] == 'results':
                if (len(rel.parts) != 3 or
                        rel.name not in PUBLIC_RESULT_FILES.get(rel.parts[1], set())):
                    continue
            yield p

def package(out, root=ROOT):
    out, root = Path(out), Path(root)
    if out.exists():
        raise ValueError('Choose a new archive path; never overwrite a release')
    files = list(selected_files(root))
    records = []
    for p in files:
        data = p.read_bytes()
        if len(data) > 2_000_000:
            raise ValueError(f'Unexpected large repository file: {p.relative_to(root)}')
        records.append({'path': p.relative_to(root).as_posix(), 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
    version = tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']
    manifest = {'package': 'DWTS Voting Analysis', 'version': version,
                'study': 'Team 2627761, 2026 MCM Problem C',
                'software_edition': 'current_implementation_and_extensions',
                'release_kind': 'source_archive',
                'files': records}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, 'dwts-voting-analysis/'+p.relative_to(root).as_posix())
        z.writestr('dwts-voting-analysis/RELEASE_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'files': len(files)+1, 'bytes': out.stat().st_size,
                      'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    package(parser.parse_args().out)
