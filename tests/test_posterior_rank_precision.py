"""Posterior rank decisions share exact decimal ties with scalar rule decisions."""
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from scipy.stats import rankdata

from dwts.evaluation import posterior_eliminations
from dwts.model import FitConfig
from dwts.rules import eliminate, order_worst, ranks, violation, weighted_rank_key


def roster(judges, exits, ids=None, *, season=1):
    n = len(judges)
    return pd.DataFrame(dict(season=season, week=1, judge_mean=judges,
        contestant_id=list("ABCDEF")[:n] if ids is None else ids,
        eliminated=np.isin(np.arange(n), exits), withdrawal=False, rule="rank"))


def test_decimal_weight_posterior_matches_reference_instead_of_float_accident():
    judges = np.arange(6, 0, -1, dtype=float)
    shares = np.array([1, 2, 4, 3, 5, 6], dtype=float) / 21
    frame = roster(judges, [0])
    expected = eliminate(judges, shares, frame.contestant_id, judge_weight=.4)
    assert expected.tolist() == [0]
    result = posterior_eliminations(frame, shares[None, :], FitConfig(judge_weight=.4))
    np.testing.assert_array_equal(result.p_eliminated_base_rule, [1, 0, 0, 0, 0, 0])
    assert result.p_exact_observed_exit_set.eq(1).all()


@pytest.mark.parametrize("weight,expected_hit", [(.6, 1), (np.nextafter(.6, 1), 0)])
def test_nearby_weight_is_not_snapped_to_decimal_tie(weight, expected_hit):
    judges, shares = np.array([4., 3, 2, 1]), np.array([.1, .3, .4, .2])
    frame = roster(judges, [0, 3])
    result = posterior_eliminations(frame, shares[None, :], FitConfig(judge_weight=weight))
    assert result.p_exact_observed_exit_set.eq(expected_hit).all()
    expected = eliminate(judges, shares, frame.contestant_id, judge_weight=weight, count=2)
    np.testing.assert_array_equal(result.p_eliminated_base_rule, np.isin(np.arange(4), expected))


@pytest.mark.parametrize("weight", [0., .4, .5, .6, np.nextafter(.6, 1.), 1.])
def test_vectorized_keys_and_posterior_ties_match_scalar_reference(weight):
    judges = np.array([8., 8., 5., 4.])
    ids = np.array(["D", "C", "B", "A"])
    shares = np.array([[.25, .25, .25, .25], [.1, .2, .3, .4], [.4, .3, .1, .2]])
    key = weighted_rank_key(ranks(judges), rankdata(-shares, axis=-1), weight)
    order = np.lexsort((np.broadcast_to(ids, shares.shape), np.broadcast_to(judges, shares.shape),
                       shares, key), axis=-1)
    expected = np.array([order_worst(judges, p, ids, judge_weight=weight) for p in shares])
    np.testing.assert_array_equal(order, expected)
    frame = roster(judges, [3], ids)
    result = posterior_eliminations(frame, shares, FitConfig(judge_weight=weight))
    counts = np.bincount(expected[:, 0], minlength=4) / len(shares)
    np.testing.assert_array_equal(result.p_eliminated_base_rule, counts)
    assert result.p_exact_observed_exit_set.eq((expected[:, 0] == 3).mean()).all()


@pytest.mark.parametrize("ids,exit_index", [(["D", "C", "B", "A"], 3), ([20, 3, 100, 2], 2)])
def test_lexical_id_breaks_equal_judge_and_fan_ties(ids, exit_index):
    frame = roster([8., 8., 8., 8.], [exit_index], ids)
    result = posterior_eliminations(frame, np.full((2, 4), .25), FitConfig(judge_weight=.4))
    np.testing.assert_array_equal(result.p_eliminated_base_rule, np.arange(4) == exit_index)
    assert result.p_exact_observed_exit_set.eq(1).all()


def test_extreme_logits_preserve_rank_and_tiebreak_before_share_underflow():
    judges = np.array([8., 8., 5., 4.])
    logits = np.array([[0., -1500., -2000., -1000.]])
    frame = roster(judges, [2])
    # Positive monotone surrogate shares provide the scalar rank-rule oracle.
    reference_shares = softmax(logits[0] / 1000)
    assert eliminate(judges, reference_shares, frame.contestant_id).tolist() == [2]
    shares = softmax(logits, axis=-1)
    assert (shares == 0).sum() == 3
    result = posterior_eliminations(frame, shares, FitConfig(), rank_scores=logits)
    np.testing.assert_array_equal(result.p_eliminated_base_rule, [0, 0, 1, 0])
    assert result.p_exact_observed_exit_set.eq(1).all()


def test_large_denominator_uses_python_integers_without_rounding_or_overflow():
    judges, shares = np.array([4., 3, 2, 1]), np.array([.1, .3, .4, .2])
    weight = Fraction(6 * 10**40 + 1, 10**41)
    key = weighted_rank_key(ranks(judges), ranks(shares), weight)
    assert key.dtype == object
    expected = [-(weight.numerator * int(2*j) + (weight.denominator-weight.numerator) * int(2*f))
                for j, f in zip(ranks(judges), ranks(shares))]
    assert key.tolist() == expected
    assert order_worst(judges, shares, list("ABCD"), judge_weight=weight).tolist() == [3, 2, 0, 1]


@pytest.mark.parametrize("exits,bottom_two", [([1], False), ([1, 2], False), ([1], True)])
def test_constraint_feasibility_keeps_existing_float_hinge_semantics(exits, bottom_two):
    judges = np.array([9., 8., 6., 5.])
    shares = np.array([[.1, .2, .3, .4], [.25, .25, .25, .25], [.3, .4, .1, .2]])
    frame = roster(judges, exits, season=28)
    cfg = FitConfig(judge_weight=.4, likelihood="bottom_two" if bottom_two else "report")
    result = posterior_eliminations(frame, shares, cfg)
    expected = np.mean([violation(judges, p, frame.eliminated, "rank", .4, bottom_two) <= 1e-10
                        for p in shares])
    assert result.p_constraint_satisfied.eq(expected).all()
