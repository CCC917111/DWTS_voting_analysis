"""Render forward-validation figures from public aggregate tables only."""
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
RULES = {"rank": ("Rank", "#31688e"),
         "percentage": ("Percentage", "#c06b38")}
MECHANISMS = {"none": ("No save", "o"),
              "judge_save": ("Judge save", "s"),
              "fan_save": ("Fan save", "^")}
SCENARIOS = [("matched", "Matched"), ("narrow_prior", "Narrow prior"),
             ("wide_prior", "Wide prior"), ("gradual_drift", "Support drift")]
FORECAST_NAMES = {"forward_bayes": "Bayesian forward model", "uniform": "Uniform exit probability",
                  "judge_only": "Current judges only", "past_survival": "Past survival + judges",
                  "past_judges": "Past judge scores"}


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "medium", "svg.fonttype": "none",
        "svg.hashsalt": "dwts-validation-results-v1",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def read_table(path, columns):
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{path}: missing aggregate columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: empty aggregate table")
    return frame


def check_completed(directory):
    status = json.loads((directory / "status.json").read_text())
    if not (status.get("completed") is True or status.get("execution_completed") is True
            or status.get("status") in {"completed", "complete"}):
        raise ValueError(f"{directory}: the study has not completed")


def one_row(frame, **conditions):
    selected = frame
    for column, value in conditions.items():
        mask = np.isclose(selected[column], value) if isinstance(value, float) else selected[column].eq(value)
        selected = selected[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one aggregate row for {conditions}, got {len(selected)}")
    return selected.iloc[0]


def interval(ax, estimate, lower, upper, x, color, marker="o", label=None):
    values = np.asarray([estimate, lower, upper], dtype=float)
    if not np.isfinite(values).all() or lower > upper:
        raise ValueError("Invalid plotted aggregate estimate or interval")
    ax.vlines(x, lower, upper, color=color, linewidth=1.4, alpha=.85)
    ax.plot(x, estimate, marker=marker, markersize=4.5,
            color=color, linestyle="none", label=label)


def save_figure(figure, name, out, preview_dir=None):
    out.mkdir(parents=True, exist_ok=True)
    destination = out / name
    figure.savefig(destination, format="svg", metadata={"Date": None})
    destination.write_text("\n".join(line.rstrip() for line in
        destination.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_dir / name.replace(".svg", ".png"), dpi=150)
    plt.close(figure)


def forecast_figure(directory, out, preview_dir):
    metrics = read_table(directory / "metrics.csv", [
        "phase", "model", "n_weeks", "n_seasons", "log_loss", "brier", "top1", "ece"])
    primary = metrics[metrics.phase.isin(["primary", "evaluation"])]
    if primary.empty:
        raise ValueError("Forecast metrics must include the declared primary evaluation phase")
    primary = primary.sort_values("log_loss", ascending=False).reset_index(drop=True)
    figure, axes = plt.subplots(1, 3, figsize=(13.1, 5.8),
                               gridspec_kw={"width_ratios": [1, 1, 1.1]})
    names = [FORECAST_NAMES.get(str(model), str(model).replace("_", " ")) for model in primary.model]
    y = np.arange(len(primary))
    colors = ["#31688e" if "forward" in str(model).lower() else "#898989"
              for model in primary.model]
    for ax, metric, title in zip(axes[:2], ["log_loss", "brier"],
                                 ["Categorical log loss", "Multiclass Brier score"]):
        ax.scatter(primary[metric], y, c=colors, s=32, zorder=3)
        ax.set(yticks=y, yticklabels=names if ax is axes[0] else [],
               xlabel=f"{title} (lower is better)", title=title)
        ax.grid(axis="x", color="#e8e8e8", linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_ylim(-.6, len(primary)-.4)
    calibration = read_table(directory / "calibration.csv", [
        "model", "bin", "n_weeks", "mean_confidence", "observed_accuracy"])
    ax = axes[2]
    palette = plt.cm.viridis(np.linspace(.15, .85, len(primary)))
    for color, (_, row) in zip(palette, primary.iterrows()):
        rows = calibration[calibration.model.eq(row.model) & calibration.n_weeks.gt(0)].sort_values("mean_confidence")
        if rows.empty:
            raise ValueError(f"Missing forecast calibration aggregates for {row.model}")
        ax.plot(rows.mean_confidence, rows.observed_accuracy, marker="o", markersize=3.5,
                color=color, linewidth=1, label=FORECAST_NAMES.get(str(row.model), str(row.model).replace("_", " ")))
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=.8)
    ax.set(xlabel="Confidence in the most likely exit", ylabel="Observed top-choice accuracy",
           title="Calibration (descriptive)", xlim=(0, 1), ylim=(0, 1))
    ax.grid(color="#e8e8e8", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    record = primary.iloc[0]
    figure.suptitle("Chronological forecasting on seasons 21–27", fontsize=14, y=.98)
    figure.text(.5, .025,
        f"{int(record.n_weeks)} eligible single-exit weeks in {int(record.n_seasons)} seasons · development uses seasons 3–20\n"
        "Current judge scores are known; the current exit is unseen. This is a retrospective backtest on previously examined data.",
        ha="center", va="bottom", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.22, right=.985, top=.85, bottom=.20, wspace=.30)
    save_figure(figure, "validation-forecast.svg", out, preview_dir)


def recovery_figure(directory, out, preview_dir):
    frame = read_table(directory / "recovery_summary.csv", [
        "rule", "scenario", "method", "metric", "estimate", "ci_lower", "ci_upper", "n_seasons"])
    specifications = [("share_rmse", "Share RMSE", "lower"),
                      ("point_rank_concordance", "Point-ranking concordance", "higher"),
                      ("coverage_95", "Coverage of nominal 95% intervals", "nominal")]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True)
    for row_index, (rule, (rule_label, color)) in enumerate(RULES.items()):
        for column_index, (metric, title, _) in enumerate(specifications):
            ax = axes[row_index, column_index]
            for method, offset, marker, alpha, label in [
                ("prior_only", -.09, "s", .5, "Prior only"),
                ("posterior", .09, "o", 1, "After observed exits")]:
                method_color = "#949494" if method == "prior_only" else color
                for index, (scenario, _) in enumerate(SCENARIOS):
                    item = one_row(frame, rule=rule, scenario=scenario, method=method, metric=metric)
                    interval(ax, item.estimate, item.ci_lower, item.ci_upper,
                             index + offset, method_color, marker,
                             label if index == 0 else None)
            if metric == "coverage_95":
                ax.axhline(.95, color="#666666", linestyle="--", linewidth=.9)
                ax.set_ylim(0, 1.03)
            elif metric == "point_rank_concordance":
                ax.axhline(.5, color="#aaaaaa", linestyle="--", linewidth=.8)
                ax.set_ylim(0, 1.03)
            else:
                ax.set_ylim(bottom=0)
            ax.grid(axis="y", color="#e8e8e8", linewidth=.7)
            ax.set_axisbelow(True)
            ax.set(xticks=np.arange(len(SCENARIOS)),
                   xticklabels=[label for _, label in SCENARIOS], xlim=(-.4, 3.4))
            if row_index == 0:
                ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(f"{rule_label} generating rule")
            if row_index == 1:
                ax.tick_params(axis="x", labelrotation=25)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    n = int(frame.n_seasons.min())
    figure.suptitle("Known-truth recovery and model misspecification", fontsize=14, y=.985)
    figure.text(.5, .02,
        f"{n} independently simulated seasons per rule/scenario · 95% season-bootstrap intervals; SMC approximation error excluded\n"
        "Within-season posterior fan-share intervals target nominal 95% coverage; recovery uses whole-season outcomes",
        ha="center", va="bottom", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.07, right=.985, top=.88, bottom=.19, hspace=.20, wspace=.25)
    save_figure(figure, "validation-recovery.svg", out, preview_dir)


def policy_figure(directory, out, preview_dir):
    frame = read_table(directory / "policy_metrics.csv", [
        "cohort", "scenario", "rule", "judge_weight", "mechanism", "metric", "mean", "lower", "upper"])
    frame = frame[np.isclose(frame.judge_weight, .5)]
    x_metric = "eliminated_judge_rank_percentile"
    y_metric = "eliminated_fan_rank_percentile"
    coordinate_floor = max(0., np.floor(10 * frame[frame.metric.isin([x_metric, y_metric])].lower.min()) / 10)
    figure, axes = plt.subplots(1, 3, figsize=(13.1, 5.2), sharex=True, sharey=True)
    cohorts = [("baseline_full34", "All 34 seasons"),
               ("late7_likelihood_prior", "Late seven seasons\nLikelihood × prior co-scale"),
               ("season3_penalty", "Season 3\nPenalty sensitivity")]
    for ax, (cohort, title) in zip(axes, cohorts):
        selected = frame[frame.cohort.eq(cohort)]
        if selected.empty:
            raise ValueError(f"Missing policy cohort {cohort}")
        scenarios = list(selected.scenario.unique())
        for rule, (rule_label, color) in RULES.items():
            for mechanism, (mechanism_label, marker) in MECHANISMS.items():
                xs, ys = [], []
                for scenario in scenarios:
                    base = dict(cohort=cohort, scenario=scenario, rule=rule, mechanism=mechanism)
                    x = one_row(frame, metric=x_metric, **base)
                    y = one_row(frame, metric=y_metric, **base)
                    xs.append(float(x["mean"]))
                    ys.append(float(y["mean"]))
                    # Marginal model-conditional intervals are not joint uncertainty contours.
                    ax.plot([x.lower, x.upper], [y["mean"], y["mean"]], color=color, alpha=.13, linewidth=.7)
                    ax.plot([x["mean"], x["mean"]], [y.lower, y.upper], color=color, alpha=.13, linewidth=.7)
                ax.scatter(xs, ys, c=color, marker=marker, s=40, alpha=.85,
                           edgecolors="white", linewidths=.4,
                           label=f"{rule_label} · {mechanism_label.lower()}")
        ax.set(title=title, xlabel="Eliminated contestant's judge-rank percentile")
        ax.grid(color="#e8e8e8", linewidth=.7)
        ax.set_axisbelow(True)
        ax.set(xlim=(coordinate_floor, 1.01), ylim=(coordinate_floor, 1.01))
    axes[0].set_ylabel("Eliminated contestant's fan-rank percentile")
    axes[1].legend(frameon=False, ncol=3, fontsize=8.1,
                   loc="upper center", bbox_to_anchor=(.5, -.20))
    figure.suptitle("Judges and fans can favor different weekly exits", fontsize=14, y=.985)
    figure.text(.5, .025,
        "Judge weight 0.5 shown · each point is one model's conditional posterior mean; faint bars are marginal 95% credible intervals\n"
        "Higher percentiles remove a weaker-ranked contestant. All 18 policies and six objectives enter the separate Pareto tables.",
        ha="center", va="bottom", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.075, right=.985, top=.79, bottom=.33, wspace=.15)
    save_figure(figure, "validation-policy-tradeoffs.svg", out, preview_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, default=ROOT / "results/forecast_study")
    parser.add_argument("--recovery", type=Path, default=ROOT / "results/recovery_study")
    parser.add_argument("--policies", type=Path, default=ROOT / "results/policy_robustness")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--preview-dir", type=Path,
                        help="Optional PNG previews outside the source release")
    args = parser.parse_args()
    configure_style()
    for directory in [args.forecast, args.recovery, args.policies]:
        check_completed(directory)
    forecast_figure(args.forecast, args.out, args.preview_dir)
    recovery_figure(args.recovery, args.out, args.preview_dir)
    policy_figure(args.policies, args.out, args.preview_dir)
    print(f"Rendered three validation figures from aggregate tables into {args.out}")


if __name__ == "__main__":
    main()
