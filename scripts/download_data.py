"""Download the pinned COMAP CSV explicitly; verify bytes before writing."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = "https://www.contest.comap.com/undergraduate/contests/mcm/contests/2026/problems/2026_MCM_Problem_C_Data.csv"
EXPECTED_SHA256 = "ea99caec6ea243bdb450a1971a95ba8a95701a93be7ff29f0ba3c57d72ddff52"

def validate(data):
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Source checksum differs from the audited dataset; no file was written")
    if not data.decode("utf-8-sig").startswith("celebrity_name,"):
        raise ValueError("Unexpected CSV header")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/2026_MCM_Problem_C_Data.csv"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    validate(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as f:
        f.write(data)
    print(f"Saved {len(data)} bytes to {args.out}; SHA-256 {EXPECTED_SHA256}")

if __name__ == "__main__":
    main()
