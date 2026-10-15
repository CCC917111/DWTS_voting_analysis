"""Checks for the repository layer, separate from historical scientific tests."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import pytest
from dwts import __version__
from dwts.demo import example_rows, write_demo
from scripts.download_data import validate
from scripts.import_v5_evidence import import_evidence
ROOT = Path(__file__).resolve().parents[1]


def test_package_version_is_consistent():
    assert __version__ == tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']


def test_synthetic_rule_example_exposes_mechanism_differences():
    got = {(r['rule'], r['mechanism']): r['eliminated'] for r in example_rows()}
    assert got == {('rank','none'):'A', ('rank','judge_save'):'B', ('rank','fan_save'):'A',
                   ('percentage','none'):'C', ('percentage','judge_save'):'C', ('percentage','fan_save'):'B'}


def test_demo_outputs_are_explicitly_synthetic(tmp_path):
    out = tmp_path / 'demo'
    data = write_demo(out)
    assert data['data_kind'] == 'synthetic' and data['fitted_models'] == 0
    assert sum(c['fan_share'] for c in data['contestants']) == pytest.approx(1)
    assert sorted(p.name for p in out.iterdir()) == ['comparison.svg', 'demo.json', 'rule_comparison.csv']
    assert json.loads((out / 'demo.json').read_text()) == data


def test_demo_does_not_overwrite(tmp_path):
    with pytest.raises(ValueError, match='never overwrites'):
        write_demo(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_demo_command_runs_from_module(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'dwts.demo', '--out', str(tmp_path/'demo')],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'SYNTHETIC DEMO' in result.stdout


def test_changed_dataset_is_rejected_before_write():
    with pytest.raises(ValueError, match='checksum'):
        validate(b'celebrity_name,other\nChanged,data\n')


def test_wrong_evidence_archive_is_rejected_without_mutation(tmp_path):
    archive = tmp_path/'wrong.zip'; archive.write_bytes(b'not the pinned archive')
    root = tmp_path/'repo'; root.mkdir()
    with pytest.raises(ValueError, match='pinned v5'):
        import_evidence(archive, root)
    assert list(root.iterdir()) == []


def test_scientific_source_hashes_match_documented_versions():
    inventory = json.loads((ROOT/'docs/source_inventory.json').read_text())
    count = 0
    for rel, expected in inventory['original_files_sha256'].items():
        if rel.startswith('dwts/') and rel != 'dwts/__init__.py':
            change = inventory.get('modified_files', {}).get(rel)
            if change:
                assert change['reason'] and change['baseline_sha256'] == expected, rel
                expected = change['current_sha256']
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
            count += 1
    assert count >= 15


def test_original_regression_tests_stay_identical():
    inventory = json.loads((ROOT/'docs/source_inventory.json').read_text())
    for rel, expected in inventory['original_files_sha256'].items():
        if rel.startswith('tests/'):
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel


def test_dataset_validation_accepts_utf8_bom(monkeypatch):
    from scripts import download_data
    data = b'\xef\xbb\xbfcelebrity_name,other\nSynthetic,only\n'
    monkeypatch.setattr(download_data, 'EXPECTED_SHA256', hashlib.sha256(data).hexdigest())
    download_data.validate(data)


def test_preserved_snapshot_is_internally_consistent():
    import pandas as pd
    folder = ROOT/'results/v5_snapshot'
    manifest = json.loads((folder/'snapshot_manifest.json').read_text())
    assert manifest['full_numerical_replication'] is False
    for filename, expected in manifest['files_sha256'].items():
        assert hashlib.sha256((folder/filename).read_bytes()).hexdigest() == expected
    metrics = pd.read_csv(folder/'rf_metrics.csv').set_index('task')
    ledger = pd.read_csv(folder/'numerical_verification.csv').set_index('metric')
    assert len(metrics) == 6 and metrics.n.eq(2777).all()
    for task, row in metrics.iterrows():
        for metric in ('r2','rmse','f1','auc'):
            if pd.notna(row[metric]):
                assert row[metric] == pytest.approx(ledger.loc[f'table9_{task}_{metric}', 'reconstruction_value'], abs=1e-12)
    printed = ledger[ledger.reference_class.eq('printed_report_value')]
    assert len(printed) == 48 and int(printed.numerical_match.sum()) == 6


def test_release_allowlist_excludes_private_and_generated_materials(tmp_path):
    from scripts.package_repository import selected_files
    paths = ['README.md','data/SOURCES.md','data/raw/data.csv','references/report.pdf',
             'outputs/model.csv','.env','dwts/demo.py','dwts/__pycache__/demo.pyc',
             'results/v5_snapshot/rf_metrics.csv','results/private.csv','docs/note.md']
    for rel in paths:
        p = tmp_path/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixture')
    got = {p.relative_to(tmp_path).as_posix() for p in selected_files(tmp_path)}
    assert got == {'README.md','data/SOURCES.md','dwts/demo.py',
                   'results/v5_snapshot/rf_metrics.csv','docs/note.md'}
