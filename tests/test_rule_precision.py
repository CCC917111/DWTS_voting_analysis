"""Rank ordering uses exact decimal weights, not float tie accidents."""
from fractions import Fraction
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from dwts.evaluation import weight_grid
from dwts.full_report import mechanisms_audit
from dwts.rules import controversy, eliminate, merit, order_worst, ranks


def example():
    return np.array([4., 3, 2, 1]), np.array([.1, .3, .4, .2]), np.array(list('ABCD'))


def test_decimal_grid_rank_tie_uses_declared_fan_tiebreak():
    j, p, ids = example()
    # A and C both have exact weighted rank 2.2 at 3/5. A has fewer fans.
    assert order_worst(j, p, ids, judge_weight=.6).tolist() == [3, 0, 2, 1]
    ix = eliminate(j, p, ids, judge_weight=.6, mechanism='fan_save')
    assert ix.tolist() == [0]
    assert controversy(j, p, ix) == (1, 0)
    assert order_worst(j, p, ids, judge_weight=Fraction(3, 5)).tolist() == [3, 0, 2, 1]


def test_nearby_weight_is_not_snapped_or_merged_into_tie():
    j, p, ids = example()
    w = np.nextafter(.6, 1.)
    assert str(float(w)) == '0.6000000000000001'
    # The rank difference is genuinely nonzero under this decimal weight.
    assert order_worst(j, p, ids, judge_weight=w).tolist() == [3, 2, 0, 1]
    assert eliminate(j, p, ids, judge_weight=w, mechanism='fan_save').tolist() == [3]


def test_all_declared_grid_weights_match_integer_oracle():
    for n in range(2, 6):
        j = np.arange(n, 0, -1, dtype=float)
        ids = np.array([str(k) for k in range(n)])
        for perm in permutations(range(1, n + 1)):
            p = np.array(perm, dtype=float)
            p /= p.sum()
            twice_j, twice_f = (2 * ranks(j)).astype(int), (2 * ranks(p)).astype(int)
            for k in range(11):
                exact = -(k * twice_j + (10 - k) * twice_f)
                expected = np.lexsort((ids, j, p, exact))
                np.testing.assert_array_equal(order_worst(j, p, ids, judge_weight=k / 10), expected)


@pytest.mark.parametrize('j', [[4, 4, 2, 1], [4, 3, 3, 1], [4, 3, 2, 2], [4, 4, 4, 1]])
def test_average_half_rank_ties_use_same_exact_oracle(j):
    j, p, ids = np.array(j, float), np.array([.1, .3, .4, .2]), np.array(list('ABCD'))
    for k in range(11):
        key = -(k * (2 * ranks(j)).astype(int) + (10 - k) * (2 * ranks(p)).astype(int))
        np.testing.assert_array_equal(order_worst(j, p, ids, judge_weight=k / 10),
                                      np.lexsort((ids, j, p, key)))


@pytest.mark.parametrize('weight', [np.nan, np.inf, -np.inf, -.1, 1.1])
def test_nonfinite_or_out_of_range_weights_still_rejected(weight):
    j, p, ids = example()
    with pytest.raises(ValueError):
        order_worst(j, p, ids, judge_weight=weight)


def test_merit_values_are_not_rounded_or_changed():
    j, p, _ = example()
    w = np.nextafter(.6, 1.)
    np.testing.assert_array_equal(merit(j, p, judge_weight=w), -(w * ranks(j) + (1 - w) * ranks(p)))


def roster():
    j, p, ids = example()
    return pd.DataFrame(dict(season=1, week=1, judge_mean=j, fan_mean=p, contestant_id=ids,
                             name=ids, eliminated=[False, False, False, True],
                             withdrawal=False, is_final_week=False))


def test_evaluation_grid_passes_intended_decimal_weights():
    weekly, _ = weight_grid(roster(), scope='nonfinal_weeks')
    row = weekly[(weekly.rule == 'rank') & (weekly.judge_weight == .6) &
                 (weekly.mechanism == 'fan_save')].iloc[0]
    assert row.eliminated_id == 'A'
    assert (row.type_a, row.type_b) == (1, 0)


def test_figure6_grid_passes_intended_decimal_weights(tmp_path):
    grid = mechanisms_audit(roster(), tmp_path)
    row = grid[(grid.scope == 'nonfinal_weeks') & (grid.judge_weight == .6) &
               (grid.mechanism == 'fan_save')].iloc[0]
    assert (row.type_a, row.type_b) == (1, 0)


def test_odd_roster_half_rank_threshold_is_not_floored():
    # With N=5, a fan midrank of 2.5 meets the printed <= N/2 predicate.
    j, p = [5, 4, 1, 3, 2], [.4, .2, .2, .15, .05]
    assert ranks(p)[2] == 2.5
    assert controversy(j, p, [2]) == (0, 1)


def test_rank_direction_and_type_sets_remain_disjoint():
    j, p = [6, 5, 4, 3, 2, 1], [.01, .09, .15, .2, .25, .3]
    a = {i for i in range(6) if controversy(j, p, [i])[0]}
    b = {i for i in range(6) if controversy(j, p, [i])[1]}
    assert a == {0, 1}
    assert b == {4, 5}
    assert a.isdisjoint(b)
