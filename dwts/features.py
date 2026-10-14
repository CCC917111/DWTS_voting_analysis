"""Lagged partner features. Fan inputs remain retrospective posterior estimates."""
import numpy as np
import pandas as pd


def partner_features(rows, reference=None, history_weight=0.5, fan_scale="relative"):
    """For train folds, reference excludes held-out seasons.

    A held-out season's own earlier weeks may be observed at prediction time.
    Current and future rows are never used directly. First appearances use neutral 1.
    This does not remove the full-season Bayesian smoother's indirect future information.
    """
    if not 0 <= history_weight <= 1:
        raise ValueError("history_weight must lie in [0,1]")
    if fan_scale not in {"relative", "share"}:
        raise ValueError("fan_scale must be relative or share")
    d = rows.copy()
    ref = d if reference is None else reference
    histories = {}
    for season in sorted(d.season.unique()):
        h = ref[ref.season < season].copy()
        if len(h):
            h["norm_j"] = h.judge_mean / h.groupby(["season", "week"]).judge_mean.transform("mean")
            h["norm_p"] = h.fan_mean * (h.n_active if fan_scale == "relative" else 1)
            histories[season] = h.groupby("partner")[["norm_j", "norm_p"]].mean()
        else:
            histories[season] = pd.DataFrame(columns=["norm_j", "norm_p"])
    d["pq_judge"] = 1.0
    d["pq_vote"] = 1.0
    for (season, partner), group in d.groupby(["season", "partner"]):
        historical = histories[season]
        prior = historical.loc[partner].to_numpy() if partner in historical.index else np.ones(2)
        # rows may contain multiple couples per pro/week: update history after the entire week.
        running, count = np.zeros(2), 0
        for week, g in group.sort_values("week").groupby("week"):
            neutral = np.array([1., 1. if fan_scale == "relative" else 1. / g.n_active.iloc[0]])
            within = running / count if count else neutral
            # Report Eq. (3) adds the historical blend only for returning pros.
            quality = (history_weight * prior + (1 - history_weight) * within
                       if partner in historical.index else within)
            d.loc[g.index, ["pq_judge", "pq_vote"]] = quality
            week_population = d[(d.season == season) & (d.week == week)]
            nj = g.judge_mean.to_numpy() / week_population.judge_mean.mean()
            np_ = g.fan_mean.to_numpy() * (g.n_active.to_numpy() if fan_scale == "relative" else 1)
            running += np.array([nj.sum(), np_.sum()])
            count += len(g)
    return d


def labels(d):
    result = {}
    for source, col in [("judge", "judge_mean"), ("fan", "fan_mean")]:
        rank = d.groupby(["season", "week"])[col].rank(ascending=False, method="average")
        result[f"top_half_{source}"] = (rank <= d.n_active / 2).astype(int).to_numpy()
        result[f"bottom_two_{source}"] = (rank > d.n_active - 2).astype(int).to_numpy()
    return result

