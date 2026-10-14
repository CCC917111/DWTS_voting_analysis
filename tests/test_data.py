from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dwts.data import prepare, data_summary


def test_official_data_counts_and_normalization():
    p, audit = prepare(Path(__file__).parents[1] / "data/raw/2026_MCM_Problem_C_Data.csv")
    s = data_summary(p)
    assert (s["seasons"], s["celebrity_seasons"], s["unique_celebrities"]) == (34, 421, 408)
    np.testing.assert_allclose(p.groupby(["season", "week"]).judge_share.sum(), 1)
    np.testing.assert_allclose(p.groupby(["season", "week"]).judge_z.mean(), 0, atol=1e-12)
    assert not p[p.withdrawal].eliminated.any()
    assert not p.duplicated(["season", "week", "contestant_id"]).any()


def fixture_row(name, scores, result="3rd Place"):
    return {"celebrity_name": name, "ballroom_partner": "P", "celebrity_industry": "Athlete",
            "season": 1, "results": result, "placement": 3, "celebrity_age_during_season": 25,
            **{f"week{i+1}_judge1_score": v for i, v in enumerate(scores)}}


def test_zero_transition_finalists_and_withdrawals(tmp_path):
    raw = pd.DataFrame([fixture_row("A", [8, 0, 0]), fixture_row("B", [7, 7, 0], "Withdrew"),
                        fixture_row("C", [8, 8, 9], "1st Place"), fixture_row("D", [8, 8, 8], "2nd Place")])
    f = tmp_path / "raw.csv"
    raw.to_csv(f, index=False)
    p, a = prepare(f)
    assert p.loc[p.name == "A", "eliminated"].tolist() == [True]
    assert p.loc[p.name == "B", "withdrawal"].tolist() == [False, True]
    assert not p.loc[p.name.isin(["B", "C", "D"]), "eliminated"].any()


def test_all_na_future_week_not_zero_elimination(tmp_path):
    f = tmp_path / "raw.csv"
    pd.DataFrame([fixture_row("A", [8, np.nan], "1st Place"),
                  fixture_row("B", [7, np.nan], "2nd Place")]).to_csv(f, index=False)
    p, _ = prepare(f)
    assert not p.eliminated.any()


def test_resurrection_is_flagged(tmp_path):
    f = tmp_path / "raw.csv"
    pd.DataFrame([fixture_row("A", [8, 0, 8])]).to_csv(f, index=False)
    with pytest.raises(ValueError, match="Return after"):
        prepare(f)

