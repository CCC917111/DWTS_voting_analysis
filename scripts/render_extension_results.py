"""Render identification and support-comparison figures from public aggregates."""
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
BLUE, ORANGE = "#31688e", "#c06b38"


def completed(directory):
    status = json.loads((directory / "status.json").read_text())
    if status.get("status") not in {"complete", "completed"}:
        raise ValueError(f"Study incomplete: {directory}")


def save(figure, name, out, preview):
    out.mkdir(parents=True, exist_ok=True)
    destination = out / name
    figure.savefig(destination, metadata={"Date": None})
    destination.write_text("\n".join(line.rstrip() for line in
        destination.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview / name.replace(".svg", ".png"), dpi=150)
    plt.close(figure)


def identification(directory, out, preview):
    completed(directory)
    widths = pd.read_csv(directory / "width_summary.csv").set_index("population")
    ambiguity = pd.read_csv(directory / "policy_ambiguity.csv").sort_values("counterfactual_judge_weight")
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    values = 100 * widths.loc[["survivor", "observed_exit"], "median"].to_numpy()
    axes[0].barh([1, 0], values, color=[BLUE, ORANGE], height=.48)
    for y, value in zip([1, 0], values):
        axes[0].text(min(value + 1.5, 97), y, f"{value:.2f}", va="center",
                     ha="right" if value > 95 else "left",
                     color="white" if value > 95 else "#333333")
    axes[0].set(yticks=[1, 0], yticklabels=["Survivors (n=1,179)", "Observed exits (n=164)"],
                xlim=(0, 102), xlabel="Median feasible width (percentage points)",
                title="Shares remain weakly constrained")
    values = 100 * ambiguity.ambiguous_fraction.to_numpy()
    axes[1].bar(np.arange(len(values)), values, color=[BLUE, "#999999", ORANGE], width=.55)
    for x, value, weeks in zip(range(len(values)), values, ambiguity.ambiguous_weeks):
        axes[1].text(x, value + 2, f"{value:.1f}%\n({int(weeks)}/164)", ha="center", va="bottom")
    axes[1].set(xticks=np.arange(len(values)),
                xticklabels=[f"{100*w:g}%" for w in ambiguity.counterfactual_judge_weight],
                ylim=(0, 114), ylabel="Weeks with multiple attainable exits (%)",
                xlabel="Counterfactual judge weight", title="Changing weights often leaves several exits possible")
    for ax in axes:
        ax.grid(axis="x" if ax is axes[0] else "y", color="#eeeeee")
        ax.set_axisbelow(True)
    figure.suptitle("What a single elimination leaves undetermined", fontsize=14, y=.98)
    figure.text(.5, .025, "164 selected historical weeks · observed judge weight 50% · fixed rosters, no save\n"
        "Additional deterministic-rule assumptions. Feasible ranges and ambiguity counts are not posterior intervals or probabilities.",
        ha="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.18, right=.985, top=.83, bottom=.23, wspace=.34)
    save(figure, "identification.svg", out, preview)


def dynamics(directory, out, preview):
    completed(directory)
    metrics = pd.read_csv(directory / "metrics.csv").sort_values("log_loss", ascending=False)
    synthetic = pd.read_csv(directory / "synthetic_metrics.csv")
    paired = pd.read_csv(directory / "synthetic_paired_differences.csv")
    names = {"static_bootstrap": "Static bootstrap", "dynamic_bootstrap": "Dynamic bootstrap",
             "uniform": "Uniform", "judge_only": "Current judges",
             "past_survival": "Past survival + judges", "past_judges": "Past judges"}
    figure, axes = plt.subplots(1, 3, figsize=(13.3, 5.2), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    colors = [BLUE if x == "static_bootstrap" else ORANGE if x == "dynamic_bootstrap" else "#999999"
              for x in metrics.model]
    axes[0].scatter(metrics.log_loss, np.arange(len(metrics)), c=colors, s=36)
    axes[0].set(yticks=np.arange(len(metrics)), yticklabels=[names[x] for x in metrics.model],
                xlabel="Log loss (lower is better)", title="Historical forecasts: 33 weeks")
    conditions = [(c, corr) for c in ("static", "drift", "shock")
                  for corr in sorted(synthetic.judge_correlation.unique())]
    for model, color in [("static_bootstrap", BLUE), ("dynamic_bootstrap", ORANGE)]:
        table = synthetic[synthetic.model.eq(model)].set_index(["condition", "judge_correlation"])
        ordered = table.loc[conditions]
        axes[2].plot(range(len(conditions)), ordered.coverage, "o-", color=color,
                     markersize=4, linewidth=1, label=names[model])
    differences = paired[paired.metric.eq("log_loss")].set_index(
        ["condition", "judge_correlation"]).loc[conditions]
    axes[1].vlines(range(len(conditions)), differences.lower, differences.upper,
                   color=ORANGE, linewidth=1.3)
    axes[1].scatter(range(len(conditions)), differences.difference, color=ORANGE, s=22)
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=.8)
    labels = [f"{c.title()}\nr={corr:g}" for c, corr in conditions]
    for ax in axes[1:]:
        ax.set(xticks=range(len(conditions)), xticklabels=labels, xlabel="Synthetic truth / judge correlation")
        ax.tick_params(axis="x", labelsize=8)
    axes[1].set(title="Synthetic log loss: dynamic minus static",
                ylabel="Paired difference (negative favors dynamic)")
    axes[2].legend(frameon=False, fontsize=8, loc="lower left")
    axes[2].axhline(.95, color="#777777", linestyle="--", linewidth=.8)
    axes[2].set(title="Filtering intervals after each exit", ylabel="Coverage of nominal 95% share intervals", ylim=(0, 1.02))
    for i, ax in enumerate(axes):
        ax.grid(axis="x" if i == 0 else "y", color="#eeeeee")
        ax.set_axisbelow(True)
    figure.suptitle("Static and dynamic support under one exit likelihood", fontsize=14, y=.98)
    figure.text(.5, .025, "Historical data were previously examined; both latent comparators use bootstrap filters without MH.\n"
        "144 shared synthetic seasons, 24 per condition. r=0.6 is a conditional-score stress test; middle intervals are paired season bootstrap, excluding MC error.",
        ha="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.14, right=.985, top=.83, bottom=.24, wspace=.37)
    save(figure, "support-dynamics.svg", out, preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identification", type=Path, default=ROOT/"results/identification_study")
    parser.add_argument("--dynamic", type=Path, default=ROOT/"results/dynamic_study")
    parser.add_argument("--out", type=Path, default=ROOT/"docs/assets")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "svg.fonttype": "none", "svg.hashsalt": "dwts-research-extensions-v1",
        "axes.spines.top": False, "axes.spines.right": False})
    identification(args.identification, args.out, args.preview_dir)
    dynamics(args.dynamic, args.out, args.preview_dir)


if __name__ == "__main__":
    main()
