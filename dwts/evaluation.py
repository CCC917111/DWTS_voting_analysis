"""Week-local counterfactuals; no imagined full-season trajectories."""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, rankdata

from .rules import controversy, eliminate, merit, violation, weighted_rank_key


def safe_corr(x, y, kind="spearman"):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float((spearmanr if kind == "spearman" else kendalltau)(x, y).statistic)


def scenario(s):
    return "S1-2" if s <= 2 else ("S3-27" if s < 28 else "S28-34")


def evaluate(votes, include_withdrawal_weeks=False):
    accuracy, correlations, cases = [], [], []
    for (season, week), g in votes.groupby(["season", "week"]):
        if len(g) < 2 or (g.withdrawal.any() and not include_withdrawal_weeks):
            continue
        j, p, ids = g.judge_mean.to_numpy(), g.fan_mean.to_numpy(), g.contestant_id.to_numpy()
        actual = set(np.flatnonzero(g.eliminated.to_numpy(bool)))
        row = {"season": season, "week": week, "scenario": scenario(season), "n_exits": len(actual)}
        predictions = {}
        for rule in ["rank", "percentage"]:
            m = merit(j, p, rule)
            correlations.append({"season": season, "week": week, "rule": rule,
                                 "fan_spearman": safe_corr(p, m), "judge_spearman": safe_corr(j, m),
                                 "fan_kendall": safe_corr(p, m, "kendall"),
                                 "judge_kendall": safe_corr(j, m, "kendall")})
            if actual and len(actual) < len(g):
                pred = set(eliminate(j, p, ids, rule, count=len(actual)))
                predictions[rule] = pred
                row[f"{rule}_correct"] = pred == actual
                row[f"{rule}_any_exit_hit"] = int(eliminate(j, p, ids, rule)[0]) in actual
                if len(actual) == 1:
                    saved = set(eliminate(j, p, ids, rule, mechanism="judge_save"))
                    row[f"{rule}_judge_save_correct"] = saved == actual
            for mechanism in ["none", "judge_save", "fan_save"]:
                ix = eliminate(j, p, ids, rule, mechanism=mechanism)
                cases.append({"season": season, "week": week, "rule": rule,
                              "mechanism": mechanism, "eliminated_id": ids[ix[0]],
                              "eliminated_name": g.name.iloc[ix[0]], "observed_exit_week": bool(actual)})
        if predictions:
            row["agreement"] = predictions["rank"] == predictions["percentage"]
            row["official_rule_correct"] = row[f"{g.rule.iloc[0]}_correct"]
            row["official_rule_any_exit_hit"] = row[f"{g.rule.iloc[0]}_any_exit_hit"]
            accuracy.append(row)
    acc, corr = pd.DataFrame(accuracy), pd.DataFrame(correlations)
    if not acc.empty:
        single = acc[acc.n_exits == 1]
        summary = single.groupby("scenario").agg(
            weeks=("week", "size"), rank_correct=("rank_correct", "sum"),
            percentage_correct=("percentage_correct", "sum"), agreement_count=("agreement", "sum"),
            rank_accuracy=("rank_correct", "mean"), percentage_accuracy=("percentage_correct", "mean"),
            rank_judge_save_accuracy=("rank_judge_save_correct", "mean"),
            percentage_judge_save_accuracy=("percentage_judge_save_correct", "mean"),
            agreement_rate=("agreement", "mean")).reset_index()
    else:
        summary = pd.DataFrame()
    # Report case counts appear to count weeks including multi-exit rounds.
    # Keep both strict set accuracy and weaker one-prediction-in-exit-set accuracy.
    all_summary = acc.groupby("scenario").agg(
        weeks=("week", "size"), rank_exact_set_accuracy=("rank_correct", "mean"),
        percentage_exact_set_accuracy=("percentage_correct", "mean"),
        rank_any_exit_hit_rate=("rank_any_exit_hit", "mean"),
        percentage_any_exit_hit_rate=("percentage_any_exit_hit", "mean")
    ).reset_index() if not acc.empty else pd.DataFrame()
    balance = corr.groupby("rule").agg(
        weeks=("week", "size"), fan_spearman=("fan_spearman", "mean"),
        judge_spearman=("judge_spearman", "mean"), fan_kendall=("fan_kendall", "mean"),
        judge_kendall=("judge_kendall", "mean")).reset_index()
    for i, row in balance.iterrows():
        paired = corr[corr.rule == row.rule].dropna(subset=["fan_spearman", "judge_spearman"])
        # Compare both parties on exactly the same weeks; different missingness must
        # not silently change denominators or violate Jensen's inequality.
        balance.loc[i, "paired_spearman_weeks"] = len(paired)
        balance.loc[i, "fan_spearman"] = paired.fan_spearman.mean()
        balance.loc[i, "judge_spearman"] = paired.judge_spearman.mean()
        balance.loc[i, "mean_weekly_influence_gap"] = (paired.fan_spearman - paired.judge_spearman).abs().mean()
    balance["influence_gap"] = abs(balance.fan_spearman - balance.judge_spearman)
    return {"accuracy_weekly": acc, "accuracy_summary": summary, "accuracy_all_elimination_weeks": all_summary,
            "correlations_weekly": corr, "influence_balance": balance,
            "counterfactual_cases": pd.DataFrame(cases)}


def weight_grid(votes, scope="single_elimination", include_withdrawal_weeks=False):
    if scope not in {"single_elimination", "all_weeks", "nonfinal_weeks"}:
        raise ValueError("Unknown simulation scope")
    rows = []
    for (season, week), g in votes.groupby(["season", "week"]):
        if len(g) < 3 or (g.withdrawal.any() and not include_withdrawal_weeks):
            continue
        if scope == "single_elimination" and g.eliminated.sum() != 1:
            continue
        if scope == "nonfinal_weeks" and g.is_final_week.any():
            continue
        j, p, ids = g.judge_mean.to_numpy(), g.fan_mean.to_numpy(), g.contestant_id.to_numpy()
        for rule in ["rank", "percentage"]:
            for w in (k / 10 for k in range(11)):
                for mechanism in ["none", "judge_save", "bottom3_judge_save", "fan_save"]:
                    ix = eliminate(j, p, ids, rule, float(w), mechanism)
                    a, b = controversy(j, p, ix)
                    rows.append({"season": season, "week": week, "rule": rule,
                                 "judge_weight": round(float(w), 1), "mechanism": mechanism,
                                 "type_a": a, "type_b": b, "controversy": a + b,
                                 "eliminated_id": ids[ix[0]], "scope": scope})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = detail.groupby(["rule", "judge_weight", "mechanism", "scope"]).agg(
        weeks=("week", "size"), type_a=("type_a", "sum"), type_b=("type_b", "sum"),
        controversies=("controversy", "sum"), controversy_rate=("controversy", "mean")).reset_index()
    return detail, summary


def posterior_eliminations(frame, samples, cfg, rank_scores=None):
    rows = []
    for week, g in frame.groupby("week"):
        if len(g) < 2 or (g.withdrawal.any() and cfg.withdrawal_policy == "skip_week") or not g.eliminated.any():
            continue
        j, ids, mask = g.judge_mean.to_numpy(), g.contestant_id.to_numpy(), g.eliminated.to_numpy(bool)
        if mask.all():
            continue
        actual = set(np.flatnonzero(mask))
        ps = samples[:, g.index.to_numpy()]
        order_scores = ps if rank_scores is None else rank_scores[:, g.index.to_numpy()]
        if g.rule.iloc[0] == "rank":
            judge_ranks = rankdata(-j, method="average")
            fan_ranks = rankdata(-order_scores, axis=1, method="average")
            m = -(cfg.judge_weight * judge_ranks + (1 - cfg.judge_weight) * fan_ranks)
            ordering_key = weighted_rank_key(judge_ranks, fan_ranks, cfg.judge_weight)
        else:
            m = cfg.judge_weight * j / j.sum() + (1 - cfg.judge_weight) * ps
            ordering_key = m
        order = np.lexsort((np.broadcast_to(np.asarray(ids, str), ps.shape), np.broadcast_to(j, ps.shape),
                           order_scores, ordering_key), axis=1)
        selected = order[:, :len(actual)]
        predicted = np.zeros(ps.shape, dtype=bool)
        np.put_along_axis(predicted, selected, True, axis=1)
        hits = np.all(predicted == mask, axis=1).sum()
        probs = np.bincount(selected.ravel(), minlength=len(g))
        constraint_merit = 2 * m if g.rule.iloc[0] == "rank" else m
        if cfg.likelihood == "bottom_two" and g.season.iloc[0] >= 28 and len(actual) == 1:
            boundary = np.sort(constraint_merit, axis=1)[:, 1]
        else:
            boundary = constraint_merit[:, ~mask].min(axis=1)
        feasible = (constraint_merit[:, mask].max(axis=1) - boundary <= 1e-10).sum()
        for i in range(len(g)):
            rows.append({"season": int(g.season.iloc[0]), "week": int(week), "contestant_id": ids[i],
                         "p_eliminated_base_rule": probs[i] / len(samples),
                         "p_exact_observed_exit_set": hits / len(samples),
                         "p_constraint_satisfied": feasible / len(samples),
                         "in_sample": True})
    return pd.DataFrame(rows)

