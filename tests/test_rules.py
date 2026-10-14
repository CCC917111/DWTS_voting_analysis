import numpy as np
import pytest
from scipy.special import softmax
from dwts.rules import ranks, merit, eliminate, controversy, violation


def test_comap_rank_worked_example():
    # COMAP official problem, Table 2: Rachel, Joey, John, Kelly.
    j, p = [25, 20, 21, 26], np.array([1.1, 3.7, 3.2, 2.0]) / 10
    np.testing.assert_allclose(-2 * merit(j, p), [6, 5, 5, 4])
    assert eliminate(j, p, ["Rachel", "Joey", "John", "Kelly"]).tolist() == [0]
    assert violation(j, p, [True, False, False, False]) == 0


def test_comap_percentage_worked_example():
    j, p = [29, 28, 30, 30], np.array([1.1, 3.7, 3.2, 2.0]) / 10
    assert eliminate(j, p, ["Jennie", "Marie", "Mel", "Helio"], "percentage").tolist() == [0]
    assert violation(j, p, [True, False, False, False], "percentage") == 0


def test_average_ties_and_deterministic_fan_tiebreak():
    np.testing.assert_allclose(ranks([8, 8, 4]), [1.5, 1.5, 3])
    # Rank sums are all 4. Lowest fan vote loses, independent of row order.
    j, p, ids = np.array([9, 6, 3]), np.array([.1, .3, .6]), np.array(["A", "B", "C"])
    assert ids[eliminate(j, p, ids)[0]] == "A"
    o = [2, 0, 1]
    assert ids[o][eliminate(j[o], p[o], ids[o])[0]] == "A"


@pytest.mark.parametrize("rule", ["rank", "percentage"])
def test_save_and_weight_endpoints(rule):
    j, p, ids = [9, 6, 3], [.1, .3, .6], ["A", "B", "C"]
    assert eliminate(j, p, ids, rule, 1).tolist() == [2]
    assert eliminate(j, p, ids, rule, 0).tolist() == [0]
    assert eliminate(j, p, ids, "rank", mechanism="judge_save").tolist() == [1]
    assert eliminate(j, p, ids, "rank", mechanism="fan_save").tolist() == [0]


def test_multi_exit_is_set_separation():
    assert violation([9, 7, 5, 3], [.4, .3, .2, .1], [False, False, True, True]) == 0
    assert violation([9, 7, 5, 3], [.4, .3, .2, .1], [True, False, False, True]) > 0


def test_bottom_two_constraint_is_weaker():
    j, p = [9, 6, 3], [.5, .3, .2]
    e = [False, True, False]
    assert violation(j, p, e) > 0
    assert violation(j, p, e, bottom_two=True) == 0


def test_controversy_uses_report_inequalities():
    assert controversy([9, 8, 7, 6, 5, 4], [.01, .09, .15, .2, .25, .3], [0]) == (1, 0)
    assert controversy([9, 8, 7, 6, 5, 4], [.01, .09, .15, .2, .25, .3], [5]) == (0, 1)


def test_week_effect_cancels_exactly():
    z = np.array([-.8, .3, 2.1])
    np.testing.assert_allclose(softmax(z), softmax(z + 17.4), rtol=1e-12)


def test_invalid_votes_rejected():
    with pytest.raises(ValueError):
        merit([8, 9], [.2, .3])

