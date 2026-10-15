"""Render the study figures from public aggregate results, without fitting models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"judge": "#31688e", "fan": "#c06b38", "normalized": "#27816b"}
MECHANISMS = {"none": ("No save", "#31688e"),
              "judge_save": ("Judge save", "#c06b38"),
              "fan_save": ("Fan save", "#27816b")}


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "medium", "svg.fonttype": "none",
        "svg.hashsalt": "dwts-study-results-v1",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def read_table(path, columns):
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{path}: missing aggregate columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: aggregate table is empty")
    return frame


def save_figure(figure, name, out, preview_dir=None):
    out.mkdir(parents=True, exist_ok=True)
    destination = out / name
    figure.savefig(destination, format="svg", metadata={"Date": None})
    # Keep generated SVGs deterministic and free of trailing path whitespace.
    destination.write_text("\n".join(line.rstrip() for line in
                                   destination.read_text(encoding="utf-8").splitlines()) + "\n",
                           encoding="utf-8")
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_dir / name.replace(".svg", ".png"), dpi=140)
    plt.close(figure)


def check_completed(directory):
    status = json.loads((directory / "status.json").read_text())
    if not (status.get("completed") is True or status.get("execution_completed") is True):
        raise ValueError(f"{directory}: the study has not completed")


def one_row(frame, **conditions):
    selected = frame
    for column, value in conditions.items():
        selected = selected[np.isclose(selected[column], value) if isinstance(value, float)
                            else selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one aggregate row for {conditions}; got {len(selected)}")
    return selected.iloc[0]


def interval(ax, point, lower, upper, y, color, label=None, marker="o"):
    values = np.asarray([point, lower, upper], dtype=float)
    if not np.isfinite(values).all() or lower > upper:
        raise ValueError("Invalid plotted estimate or interval")
    ax.hlines(y, lower, upper, color=color, linewidth=1.6, alpha=.85)
    ax.plot(point, y, marker=marker, markersize=5, color=color, linestyle="none", label=label)


def benchmark_figure(directory, out, preview_dir):
    metrics = read_table(directory / "metrics.csv", ["model", "r2", "rmse", "n", "seasons", "weeks"])
    differences = read_table(directory / "paired_differences.csv",
        ["candidate", "reference", "metric", "difference", "lower_95", "upper_95", "bootstrap_draws"])
    models = [
        ("uniform_share", "Uniform share: 1 / roster size", "#777777"),
        ("last_share", "Previous-week inferred share", "#777777"),
        ("last_share_normalized", "Previous-week share, normalized", "#777777"),
        ("history_mean_share", "Own prior-mean inferred share", "#777777"),
        ("history_mean_share_normalized", "Own prior-mean share, normalized", "#777777"),
        ("rf_demographics_normalized", "RF: age + industry, normalized", COLORS["normalized"]),
        ("rf_pair_history_normalized", "RF: own-pair history, normalized", COLORS["normalized"]),
        ("rf_pair_history_demographics_normalized", "RF: own-pair history + attributes, normalized", COLORS["normalized"]),
        ("rf_tuned_raw", "RF: tuned history blend, unconstrained", COLORS["fan"]),
        ("rf_tuned_normalized", "RF: tuned history blend, normalized", COLORS["judge"]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 7.1), sharey=True,
                               gridspec_kw={"width_ratios": [1, 1.1]})
    y = np.arange(len(models))
    for index, (model, _, color) in enumerate(models):
        row = one_row(metrics, model=model)
        axes[0].plot(row.r2, index, "o", color=color, markersize=5)
        if model == "rf_tuned_raw":
            axes[1].plot(0, index, "o", color=color, markersize=5)
        else:
            delta = one_row(differences, candidate=model, reference="rf_tuned_raw", metric="r2")
            interval(axes[1], delta.difference, delta.lower_95, delta.upper_95, index, color)
    for ax in axes:
        ax.axvline(0, color="#aaaaaa", linestyle="--", linewidth=.8, zorder=0)
        ax.grid(axis="x", color="#e8e8e8", linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set(yticks=y, yticklabels=[label for _, label, _ in models],
                xlabel="Pooled out-of-fold R²", title="Performance on the same inferred-share target")
    axes[0].invert_yaxis()
    axes[1].set(xlabel="R² difference from unconstrained tuned RF",
                title="Paired 95% season-bootstrap intervals")
    reference = one_row(metrics, model="rf_tuned_raw")
    draws = int(differences.bootstrap_draws.iloc[0])
    figure.suptitle("Fan-share baselines and feature ablations", fontsize=14, y=.98)
    figure.text(.5, .025,
        f"{int(reference.seasons)} seasons · {int(reference.n):,} contestant-weeks · shared five outer season folds\n"
        f"{draws:,} paired season resamples of fixed out-of-fold predictions; no model refits or latent-vote uncertainty",
        ha="center", va="bottom", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.32, right=.985, top=.84, bottom=.18, wspace=.16)
    save_figure(figure, "study-benchmarks.svg", out, preview_dir)


def overall_policies(directory):
    frame = read_table(directory / "policy_summary.csv", ["source", "level", "group", "rule",
        "judge_weight", "mechanism", "metric", "mean", "lower", "upper", "weeks", "seasons", "posterior_draws"])
    return frame[frame.level.eq("overall") & frame.group.eq("all") &
                 frame.metric.eq("controversial_week_fraction")]


def uncertainty_figure(directory, out, preview_dir):
    frame = overall_policies(directory)
    frame = frame[frame.source.eq("baseline")]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), sharey=True)
    for ax, rule in zip(axes, ["rank", "percentage"]):
        for mechanism, (label, color) in MECHANISMS.items():
            rows = frame[frame.rule.eq(rule) & frame.mechanism.eq(mechanism)].sort_values("judge_weight")
            if len(rows) != 3:
                raise ValueError(f"Expected the three declared weights for {rule}/{mechanism}")
            ax.plot(rows.judge_weight, 100 * rows["mean"], color=color, marker="o", label=label)
            ax.fill_between(rows.judge_weight, 100 * rows.lower, 100 * rows.upper, color=color, alpha=.14)
        ax.set(title="Rank aggregation" if rule == "rank" else "Percentage aggregation",
               xlabel="Judge weight", xticks=[.25, .5, .75], xlim=(.21, .79))
        ax.grid(axis="y", color="#e8e8e8", linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Controversial weeks (%)")
    axes[0].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=9)
    record = frame.iloc[0]
    figure.suptitle("Voting-policy outcomes under posterior uncertainty", fontsize=14, y=.98)
    figure.text(.5, .025,
        f"{int(record.weeks)} fixed observed nonfinal rosters; at least three contestants; withdrawal weeks excluded\n"
        f"One exit per policy/week · {int(record.posterior_draws):,} joint posterior draws · pointwise 95% credible intervals",
        ha="center", va="bottom", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.08, right=.985, bottom=.23, top=.84, wspace=.13)
    save_figure(figure, "study-rule-uncertainty.svg", out, preview_dir)


def sensitivity_figure(directory, out, preview_dir):
    frame = overall_policies(directory)
    frame = frame[np.isclose(frame.judge_weight, .5)]
    policies = [(rule, mechanism) for rule in ["rank", "percentage"] for mechanism in MECHANISMS]
    specifications = [
        ("Late-season likelihood",
         [("baseline_matched_to_late_bottom_two_likelihood", "Lowest-aggregate constraint", "#555555", "o"),
          ("late_bottom_two_likelihood", "Bottom-two membership constraint", "#9467bd", "s")]),
        ("Season 3 penalty strength",
         [("season3_penalty_0p1", "λ = 0.1", "#27816b", "v"),
          ("baseline_matched_to_season3_penalty_0p1", "λ = 1", "#555555", "o"),
          ("season3_penalty_10", "λ = 10", "#c06b38", "s")]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True)
    labels = [f"{'Rank' if rule == 'rank' else 'Percentage'} · {MECHANISMS[mechanism][0].lower()}"
              for rule, mechanism in policies]
    for ax, (title, sources) in zip(axes, specifications):
        offsets = np.linspace(-.17, .17, len(sources))
        for source_index, (source, label, color, marker) in enumerate(sources):
            for index, (rule, mechanism) in enumerate(policies):
                row = one_row(frame, source=source, rule=rule, mechanism=mechanism)
                interval(ax, 100*row["mean"], 100*row.lower, 100*row.upper,
                         index + offsets[source_index], color,
                         label=label if index == 0 else None, marker=marker)
        sample = one_row(frame, source=sources[0][0], rule="rank", mechanism="none")
        season_label = "season" if int(sample.seasons) == 1 else "seasons"
        ax.set(title=f"{title}\n{int(sample.seasons)} {season_label}, {int(sample.weeks)} matched rosters",
               xlabel="Controversial weeks (%)")
        ax.grid(axis="x", color="#e8e8e8", linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False, fontsize=8.5)
    axes[0].set(yticks=np.arange(len(policies)), yticklabels=labels)
    axes[0].invert_yaxis()
    figure.suptitle("Sensitivity to the fitted elimination constraint", fontsize=14, y=.98)
    figure.text(.5, .015,
        "Judge weight 0.5 · points: conditional posterior means · bars: each model's 95% credible interval\n"
        "Each panel holds observed rosters fixed. Independently fitted models have no paired-difference interval.",
        ha="center", va="bottom", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.20, right=.985, bottom=.29, top=.80, wspace=.13)
    save_figure(figure, "study-sensitivity.svg", out, preview_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "results/study_benchmarks")
    parser.add_argument("--uncertainty", type=Path, default=ROOT / "results/study_uncertainty")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--preview-dir", type=Path, help="Optional PNG previews outside the source release")
    args = parser.parse_args()
    configure_style()
    for directory in [args.benchmarks, args.uncertainty]:
        check_completed(directory)
    benchmark_figure(args.benchmarks, args.out, args.preview_dir)
    uncertainty_figure(args.uncertainty, args.out, args.preview_dir)
    sensitivity_figure(args.uncertainty, args.out, args.preview_dir)
    print(f"Rendered three study figures from aggregate tables into {args.out}")


if __name__ == "__main__":
    main()
