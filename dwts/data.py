"""COMAP wide scores -> an auditable contestant/season/week panel."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCORE = re.compile(r"^week(\d+)_judge(\d+)_score$")
INDUSTRIES = ["Acting", "Sports", "RealityTV", "Music", "Model", "Other"]
INDUSTRY_MAP = {
    "actor/actress": "Acting", "comedian": "Acting", "magician": "Acting",
    "producer": "Acting", "athlete": "Sports", "racing driver": "Sports",
    "sports broadcaster": "Sports", "fitness instructor": "Sports",
    "tv personality": "RealityTV", "social media personality": "RealityTV",
    "radio personality": "RealityTV", "singer/rapper": "Music",
    "musician": "Music", "model": "Model", "beauty pagent": "Model",
}


def prepare(path: str | Path, outcome_policy="zero_first", include_final_week_eliminations=True,
            exclude_withdrawn_contestants=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if outcome_policy not in {"zero_first", "label_first"}:
        raise ValueError("outcome_policy must be zero_first or label_first")
    raw = pd.read_csv(path)
    required = {"celebrity_name", "ballroom_partner", "celebrity_industry", "season",
                "results", "placement", "celebrity_age_during_season"}
    if required - set(raw):
        raise ValueError(f"Missing columns: {sorted(required - set(raw))}")
    cols: dict[int, list[str]] = {}
    for col in raw:
        match = SCORE.fullmatch(col)
        if match:
            cols.setdefault(int(match[1]), []).append(col)
    if not cols:
        raise ValueError("No weekX_judgeY_score columns found")
    if raw.duplicated(["season", "celebrity_name"]).any():
        raise ValueError("Duplicate celebrity/season rows")
    scores = raw[[c for cs in cols.values() for c in cs]].apply(pd.to_numeric, errors="raise")
    if (scores < 0).any().any() or np.isinf(scores.to_numpy()).any():
        raise ValueError("Scores must be finite nonnegative numbers or N/A")
    raw[scores.columns] = scores
    raw["season"] = pd.to_numeric(raw.season, errors="raise").astype(int)
    if not raw.season.between(1, 34).all():
        raise ValueError("This reconstruction defines rule regimes only for seasons 1–34")
    rows, audit = [], []
    for _, r in raw.iterrows():
        weekly = {w: r[cs].astype(float).dropna().to_numpy() for w, cs in cols.items()}
        active = [w for w in sorted(cols) if len(weekly[w]) and (weekly[w] > 0).any()]
        if not active:
            audit.append({"season": r.season, "name": r.celebrity_name, "issue": "no_active_week"})
            continue
        zero_after = [w for w in sorted(cols) if w > active[0] and len(weekly[w])
                      and (weekly[w] == 0).all()]
        if zero_after and any(w > zero_after[0] for w in active):
            raise ValueError(f"Return after all-zero week: {r.celebrity_name}, season {r.season}")
        result = str(r.results)
        withdrew = "withdrew" in result.lower()
        if exclude_withdrawn_contestants and withdrew:
            audit.append({"season": int(r.season), "name": r.celebrity_name,
                          "issue": "withdrawn_contestant_excluded_before_normalization"})
            continue
        match = re.search(r"Eliminated Week\s+(\d+)", result, flags=re.I)
        labeled = int(match[1]) if match else None
        inferred = zero_after[0] - 1 if zero_after else None
        preferred = (labeled if labeled is not None else inferred) if outcome_policy == "label_first" else (inferred if inferred is not None else labeled)
        elimination = None if withdrew else preferred
        if elimination is not None and elimination not in active:
            audit.append({"season": int(r.season), "name": r.celebrity_name,
                          "issue": "labeled_exit_has_no_active_score_constraint_omitted",
                          "labeled_week": labeled, "inferred_week": inferred})
            elimination = None
        if inferred != labeled and not withdrew:
            audit.append({"season": int(r.season), "name": r.celebrity_name,
                          "issue": "label_zero_disagreement" if inferred and labeled else
                                   ("placement_corrected_by_zero_transition" if inferred else "label_or_finalist"),
                          "labeled_week": labeled, "inferred_week": inferred})
        for w in active:
            values = weekly[w]
            if (values == 0).any():
                audit.append({"season": int(r.season), "name": r.celebrity_name,
                              "issue": "partial_zero_included_in_mean", "week": w})
            rows.append({
                "season": int(r.season), "week": w,
                "contestant_id": f"S{int(r.season):02d}:{r.celebrity_name}",
                "name": r.celebrity_name, "partner": r.ballroom_partner,
                "industry": INDUSTRY_MAP.get(str(r.celebrity_industry).strip().lower(), "Other"),
                "age": float(r.celebrity_age_during_season), "placement": float(r.placement),
                "nationality": r.get("celebrity_homecountry/region", "Unknown"),
                "judge_mean": float(values.mean()), "judge_total": float(values.sum()),
                "n_judges": len(values), "eliminated": w == elimination,
                "withdrawal": withdrew and w == active[-1], "result_label": result,
            })
    panel = pd.DataFrame(rows).sort_values(["season", "week", "contestant_id"]).reset_index(drop=True)
    if panel.empty:
        raise ValueError("No active contestant weeks")
    groups = panel.groupby(["season", "week"])["judge_mean"]
    sd = groups.transform(lambda x: x.std(ddof=0)).replace(0, 1)
    panel["judge_z"] = (panel.judge_mean - groups.transform("mean")) / sd
    panel["judge_share"] = panel.judge_mean / groups.transform("sum")
    panel["n_active"] = groups.transform("size")
    panel["rule"] = np.where((panel.season <= 2) | (panel.season >= 28), "rank", "percentage")
    panel["is_final_week"] = panel.week == panel.groupby("season").week.transform("max")
    if not include_final_week_eliminations:
        for _, r in panel[panel.eliminated & panel.is_final_week].iterrows():
            audit.append({"season": int(r.season), "name": r["name"],
                          "issue": "final_week_exit_omitted", "week": int(r.week)})
        panel.loc[panel.is_final_week, "eliminated"] = False
    return panel, pd.DataFrame(audit, columns=["season", "name", "issue", "labeled_week", "inferred_week", "week"])


def data_summary(panel: pd.DataFrame) -> dict:
    groups = list(panel.groupby(["season", "week"]))
    eligible = [g for _, g in groups if g.eliminated.any() and not g.withdrawal.any()]
    lowest = saves = 0
    for g in eligible:
        low = np.isclose(g.judge_mean, g.judge_mean.min())
        lowest += int(low.sum())
        saves += int((low & ~g.eliminated.to_numpy()).sum())
    return {
        "seasons": int(panel.season.nunique()), "celebrity_seasons": int(panel.contestant_id.nunique()),
        "unique_celebrities": int(panel.name.nunique()), "contestant_weeks": len(panel),
        "weeks": len(groups), "elimination_weeks_excluding_withdrawals": len(eligible),
        "elimination_weeks_including_scored_withdrawals": sum(g.eliminated.any() for _, g in groups),
        "nonfinal_weeks": sum(not g.is_final_week.any() for _, g in groups),
        "single_elimination_weeks_excluding_withdrawals": sum(g.eliminated.sum() == 1 for g in eligible),
        "multiple_elimination_weeks_excluding_withdrawals": sum(g.eliminated.sum() > 1 for g in eligible),
        "withdrawal_weeks": sum(g.withdrawal.any() for _, g in groups),
        "lowest_judge_count_in_elimination_weeks": lowest, "audience_saves": saves,
        "audience_save_rate": saves / lowest if lowest else None,
    }

