"""Import evidence from the exact local v5 archive, without replacing checkout code.

This is optional. Obtain the archive separately from its authorized provider.
It is not downloaded, published or bundled by this repository.
Historical code snapshots inside outputs are retained for exact-reuse guards;
no source report PDF or raw posterior chain is copied.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

EXPECTED_SHA256 = "5452dc3e638bddccbba11a0e3a2686972eaa8becaaf5d4b83a953c5fe8669988"
ROOT = Path(__file__).resolve().parents[1]

def import_evidence(archive, root=ROOT):
    archive, root = Path(archive), Path(root).resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Not the pinned v5 archive; no files were written")
    with zipfile.ZipFile(archive) as z:
        selected = []
        for info in z.infolist():
            parts = PurePosixPath(info.filename).parts
            if not parts or parts[0] != "dwts-reproduction-v5" or ".." in parts:
                raise ValueError("Unsafe or unexpected archive member")
            if info.is_dir():
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Symlink members are not supported")
            rel = Path(*parts[1:])
            if rel.parts[0] not in {"outputs", "data"}:
                continue
            if rel.parts[0] == "data" and rel.as_posix() != "data/raw/2026_MCM_Problem_C_Data.csv":
                continue
            if rel.suffix == ".nc":
                continue
            dest = root / rel
            if not dest.resolve().is_relative_to(root):
                raise ValueError("Destination escapes the checkout")
            if dest.exists():
                raise ValueError(f"Refusing to overwrite {rel}; use a clean checkout")
            selected.append((info, rel, dest))
        if sum(i.file_size for i, _, _ in selected) > 250_000_000:
            raise ValueError("Unexpected archive size")
        for info, _, dest in selected:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as f:
                f.write(z.read(info))
    receipt = {"archive_sha256": digest, "imported_files": len(selected),
               "imported_bytes": sum(i.file_size for i, _, _ in selected),
               "code_overwritten": False, "report_pdf_copied": False,
               "raw_posterior_chains_copied": False}
    print(json.dumps(receipt, indent=2))
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    import_evidence(args.archive)

if __name__ == "__main__":
    main()
