import numpy as np
import pandas as pd
from dwts.features import partner_features
from dwts.attribution import pcir


def panel():
    rows = []
    for s in [1, 2, 3]:
        for w in [1, 2, 3]:
            for i in [0, 1]:
                rows.append({"season": s, "week": w, "partner": f"P{i}",
                             "judge_mean": 8 - i, "fan_mean": .7 if i == 0 else .3, "n_active": 2})
    return pd.DataFrame(rows)


def test_current_and_future_values_do_not_enter_lagged_features():
    d = panel()
    original = partner_features(d)
    changed = d.copy()
    changed.loc[(changed.season == 2) & (changed.week >= 2), "judge_mean"] = 100
    new = partner_features(changed)
    earlier = (d.season < 2) | ((d.season == 2) & (d.week <= 2))
    np.testing.assert_allclose(original.loc[earlier, ["pq_judge", "pq_vote"]],
                               new.loc[earlier, ["pq_judge", "pq_vote"]])


def test_heldout_season_excluded_from_training_history():
    d = panel()
    train = d[d.season != 2]
    a = partner_features(train, reference=train)
    modified = d.copy()
    modified.loc[modified.season == 2, "fan_mean"] = 1000
    b = partner_features(modified[modified.season != 2], reference=modified[modified.season != 2])
    np.testing.assert_allclose(a[["pq_judge", "pq_vote"]], b[["pq_judge", "pq_vote"]])


def test_cold_start_neutral_and_pcir_group_sums():
    d = panel()
    f = partner_features(d)
    np.testing.assert_allclose(f.loc[(f.season == 1) & (f.week == 1), ["pq_judge", "pq_vote"]], 1)
    assert np.isclose(pcir([.4, .2, .1, .3], ["pq_judge", "pq_vote", "age", "industry_Acting"])[2], 1.5)


def test_debut_pro_uses_within_season_history_without_shrinking_to_neutral():
    d = panel()
    f = partner_features(d, history_weight=.8)
    row = f[(f.season == 1) & (f.week == 2) & (f.partner == 'P0')].iloc[0]
    assert np.isclose(row.pq_judge, 8/7.5)
    assert np.isclose(row.pq_vote, 1.4)

