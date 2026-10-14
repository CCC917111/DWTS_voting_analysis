"""Build a review ZIP from an explicit allowlist, excluding all research inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {'dwts', 'configs', 'scripts', 'tests', 'docs', 'examples', '.github', 'results'}
TOP_FILES = {'README.md', 'README.zh-CN.md', 'LICENSE_STATUS.md', 'CONTRIBUTING.md',
             'CHANGELOG.md', 'pyproject.toml', 'requirements-v4-verified.txt', '.gitignore'}
EXCLUDED_PARTS = {'__pycache__', '.pytest_cache', '.git', '.venv', 'build', 'dist'}
ALLOWED_SUFFIXES = {'.py', '.md', '.toml', '.txt', '.json', '.csv', '.svg', '.yml', '.yaml'}

def selected_files(root=ROOT):
    for p in sorted(Path(root).rglob('*')):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root)
        if set(rel.parts) & EXCLUDED_PARTS:
            continue
        if rel.as_posix() in TOP_FILES or rel.as_posix() == 'data/SOURCES.md':
            yield p
        elif rel.parts[0] in DIRECTORIES and p.suffix in ALLOWED_SUFFIXES:
            if rel.parts[0] == 'results' and rel.parts[:2] != ('results', 'v5_snapshot'):
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
    manifest = {'package': 'DWTS Voting Audit', 'version': '0.5.1',
                'full_numerical_replication': False, 'remote_publication_performed': False,
                'files': records}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, 'dwts-voting-audit/'+p.relative_to(root).as_posix())
        z.writestr('dwts-voting-audit/RELEASE_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'files': len(files)+1, 'bytes': out.stat().st_size,
                      'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    package(parser.parse_args().out)
