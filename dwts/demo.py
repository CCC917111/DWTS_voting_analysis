"""A deterministic synthetic example. No real votes, data downloads or fitting."""
from __future__ import annotations
import argparse
import csv
import html
import json
from pathlib import Path
from .rules import eliminate

CONTESTANTS = ("A", "B", "C", "D")
JUDGES = (9.0, 8.0, 7.0, 6.0)
FANS = (0.10, 0.11, 0.12, 0.67)

def example_rows():
    rows = []
    for rule in ("rank", "percentage"):
        for mechanism in ("none", "judge_save", "fan_save"):
            ix = eliminate(JUDGES, FANS, CONTESTANTS, rule=rule,
                           judge_weight=0.5, mechanism=mechanism)
            rows.append({"rule": rule, "mechanism": mechanism,
                         "judge_weight": 0.5,
                         "eliminated": CONTESTANTS[int(ix[0])]})
    return rows

def write_demo(out):
    out = Path(out)
    if out.exists():
        raise ValueError("Choose a new output directory; demo never overwrites results")
    rows = example_rows()
    out.mkdir(parents=True, exist_ok=False)
    with (out / "rule_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    data = {"data_kind": "synthetic", "fitted_models": 0,
            "warning": "Illustration only. These are invented fan shares, not inferred DWTS results.",
            "contestants": [{"id": c, "judge_mean": j, "fan_share": p}
                            for c, j, p in zip(CONTESTANTS, JUDGES, FANS)],
            "results": rows}
    (out / "demo.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    cards = []
    for i, row in enumerate(rows):
        x, y = 28 + (i // 3)*340, 104 + (i % 3)*87
        label = row["mechanism"].replace("_", " ").title()
        cards.append(f'<rect x="{x}" y="{y}" width="324" height="71" rx="12" fill="#172c3a"/>'
                     f'<text x="{x+16}" y="{y+27}" fill="#a5c6d4" font-size="14">{html.escape(label)}</text>'
                     f'<text x="{x+16}" y="{y+53}" fill="#ffffff" font-size="20">Eliminate {row["eliminated"]}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">'
           '<rect width="720" height="420" rx="18" fill="#0b1b27"/>'
           '<g font-family="sans-serif"><text x="28" y="36" fill="#ffffff" font-size="23">Same roster. Different rules.</text>'
           '<text x="28" y="61" fill="#b2c7d1" font-size="13">Synthetic demonstration • equal judge/fan weight • no model fitting</text>'
           '<text x="28" y="92" fill="#5ed3b2" font-size="17">Rank aggregation</text>'
           '<text x="368" y="92" fill="#f8bc77" font-size="17">Percentage aggregation</text>'
           + ''.join(cards) +
           '<text x="28" y="395" fill="#b2c7d1" font-size="12">Invented inputs: judges 9, 8, 7, 6 | fan shares .10, .11, .12, .67</text></g></svg>')
    (out / "comparison.svg").write_text(svg, encoding="utf-8")
    return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    try:
        data = write_demo(args.out)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print("SYNTHETIC DEMO: no observed or inferred DWTS fan votes; no models fitted.")
    for row in data["results"]:
        print(f"{row['rule']:10} {row['mechanism']:11} -> eliminate {row['eliminated']}")
    print(f"Wrote CSV, JSON and SVG to {args.out}")

if __name__ == "__main__":
    main()
