"""Compare computed metrics to the PDF without tuning parameters to hit them."""
from pathlib import Path
import numpy as np
import pandas as pd


def write_comparison(votes, outdir, statuses):
    out = Path(outdir)
    complete = votes.season.nunique() == 34
    converged = all(s["sampling_diagnostics_pass"] for s in statuses)
    rows = []
    def add(metric, report, actual, basis, posterior=False):
        valid = complete and (not posterior or converged)
        rows.append({"metric": metric, "report_value": report, "reconstruction_value": actual,
                     "difference": actual-report, "basis": basis,
                     "full_34_seasons": complete, "sampling_diagnostics_pass": converged,
                     "comparison_status": "eligible_for_comparison" if valid else "provisional_or_partial"})
    add("unique_celebrities", 408, votes.name.nunique(), "same input count")
    acc = pd.read_csv(out / "accuracy_all_elimination_weeks.csv")
    for name, expected in [("S1-2", 11), ("S3-27", 192), ("S28-34", 56)]:
        selected = acc[acc.scenario == name]
        if len(selected):
            add(f"table5_case_count_{name}", expected, int(selected.weeks.iloc[0]), "elimination weeks including multiple exits")
    detail = pd.read_csv(out / "accuracy_weekly.csv")
    for name, rank_hits, percentage_hits, agreements, denominator in [
        ('S1-2',9,8,7,11),('S3-27',139,189,138,192),('S28-34',44,40,28,56)]:
        group=detail[detail.scenario==name]
        if len(group):
            for col,number in [('rank_correct',rank_hits),('percentage_correct',percentage_hits),('agreement',agreements)]:
                add(f'table5_{name}_{col}',number/denominator,group[col].mean(),
                    'complete exit-set comparison; report multi-exit/tie conventions unspecified',True)
    for mode in ["correct", "any_exit_hit"]:
        add(f"table5_official_accuracy_{mode}", 242/259,
            detail[f"official_rule_{mode}"].mean(),
            "exact complete exit set" if mode == "correct" else "one predicted loser is among actual exits", True)
    correlations = pd.read_csv(out / "influence_balance.csv").set_index("rule")
    for rule, fan, judge in [("rank", .719, .727), ("percentage", .942, .293)]:
        add(f"table8_{rule}_fan_spearman", fan, correlations.loc[rule, "fan_spearman"], "mean within-week correlations", True)
        add(f"table8_{rule}_judge_spearman", judge, correlations.loc[rule, "judge_spearman"], "mean within-week correlations", True)
        add(f"table8_{rule}_gap_of_mean_correlations", abs(fan-judge), correlations.loc[rule, "influence_gap"], "absolute difference of mean correlations; common valid-week denominator", True)
        printed_gap = .084 if rule == "rank" else .659
        add(f"table8_{rule}_printed_gap_vs_mean_weekly_gap", printed_gap,
            correlations.loc[rule, "mean_weekly_influence_gap"],
            "possible report aggregation: mean of within-week absolute gaps; not confirmed", True)
    grid = pd.read_csv(out / "weight_grid_summary.csv")
    for rule, mechanism, count in [("percentage", "none", 31), ("percentage", "judge_save", 25),
                                   ("rank", "none", 11), ("rank", "judge_save", 19)]:
        x = grid[(grid.rule == rule) & (grid.mechanism == mechanism) & np.isclose(grid.judge_weight, .5)]
        if len(x):
            add(f"table8_{rule}_{mechanism}_controversies", count, int(x.controversies.iloc[0]), "fixed 0.5 weight; Type A+B definition", True)
            add(f"table8_{rule}_{mechanism}_rounds", 301, int(x.weeks.iloc[0]), "301 is inferred from the rounded count/rate pairs; not explicitly stated in the report")
    rf_path = out / "rf_metrics.csv"
    if rf_path.exists():
        rf = pd.read_csv(rf_path).set_index("task")
        for task, value in [("A1_judge", .146), ("A2_fan", .614)]:
            add(f"table9_{task}_r2", value, rf.loc[task, "r2"], "OOF GroupKFold score; feature and target conventions remain assumptions", True)
        for task,value in [('A1_judge',5.320),('A2_fan',.072)]:
            add(f'table9_{task}_rmse',value,rf.loc[task,'rmse'],
                ('A1 target=' + str(rf.loc[task].get('judge_target', 'judge_mean')) + '; report raw-score units ambiguous' if task == 'A1_judge' else 'A2 target=posterior mean fan share; pooled OOF RMSE'),True)
        for task,f1,auc in [('A3_top_half_judge',.766,.778),('A3_top_half_fan',.803,.879),
                           ('A4_bottom_two_judge',.488,.755),('A4_bottom_two_fan',.547,.775)]:
            for metric,value in [('f1',f1),('auc',auc)]:
                add(f'table9_{task}_{metric}',value,rf.loc[task,metric],'pooled OOF GroupKFold metric',True)
        pcir_path=out/'pcir.csv'
        if pcir_path.exists():
            pcir=pd.read_csv(pcir_path)
            for task,method,expected in [('A1_judge','MDI',13.57),('A2_fan','MDI',36.76),
                                        ('A1_judge','SHAP',2.62),('A2_fan','SHAP',4.68)]:
                group=pcir[(pcir.task==task)&(pcir.method==method)]
                add(f'pcir_{task}_{method}',expected,group.pcir.mean(),
                    'arithmetic mean of per-fold ratios; report pooling is unspecified',True)
    pd.DataFrame(rows).to_csv(out / "report_comparison.csv", index=False)

