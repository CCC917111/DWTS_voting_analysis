"""Ranks are 1=best. All returned combined merits are higher=better."""
from fractions import Fraction

import numpy as np
from scipy.stats import rankdata


def ranks(values):
    return rankdata(-np.asarray(values, dtype=float), method="average")


def merit(judges, votes, rule="rank", judge_weight=0.5):
    j, p = np.asarray(judges, dtype=float), np.asarray(votes, dtype=float)
    if j.ndim != 1 or j.shape != p.shape or len(j) < 2:
        raise ValueError("Need aligned one-dimensional arrays with at least two contestants")
    if not np.isfinite(j).all() or not np.isfinite(p).all() or (j < 0).any() or (p < 0).any():
        raise ValueError("Scores and shares must be finite and nonnegative")
    if j.sum() <= 0 or not np.isclose(p.sum(), 1, atol=1e-6) or not 0 <= judge_weight <= 1:
        raise ValueError("Invalid shares, score total, or judge weight")
    if rule == "rank":
        return -(judge_weight * ranks(j) + (1 - judge_weight) * ranks(p))
    if rule == "percentage":
        return judge_weight * j / j.sum() + (1 - judge_weight) * p
    raise ValueError(f"Unknown rule: {rule}")


def order_worst(judges, votes, ids, rule="rank", judge_weight=0.5):
    """Ties: fewer fan votes loses, then lower judge score, then lexical ID.

    Rank ordering interprets a finite float weight by its shortest decimal
    representation, with no snapping or tolerance. For example, 0.6 means 3/5,
    while 0.6000000000000001 remains a distinct weight. Explicit Fraction
    weights retain their exact value. Average ranks are half-integers, so an
    integer key can preserve genuine ties without rounding either input.
    The numerical merits and likelihood computations are unchanged.
    """
    m = merit(judges, votes, rule, judge_weight)  # Preserve common validation.
    if rule == "rank":
        weight = (judge_weight if isinstance(judge_weight, Fraction)
                  else Fraction(str(float(judge_weight))))
        numerator, denominator = weight.numerator, weight.denominator
        twice_judge = [int(value) for value in 2 * ranks(judges)]
        twice_fan = [int(value) for value in 2 * ranks(votes)]
        # Python integers avoid overflow even for unusually precise weights.
        m = np.asarray([-(numerator * j + (denominator - numerator) * f)
                        for j, f in zip(twice_judge, twice_fan)], dtype=object)
    return np.lexsort((np.asarray(ids, str), np.asarray(judges), np.asarray(votes), m))


def eliminate(judges, votes, ids, rule="rank", judge_weight=0.5, mechanism="none", count=1):
    order = order_worst(judges, votes, ids, rule, judge_weight)
    n = len(order)
    if not 1 <= count < n:
        raise ValueError("Elimination count must be between 1 and N-1")
    if mechanism == "none":
        return order[:count]
    if count != 1:
        raise ValueError("Save simulations are defined only for single eliminations")
    if mechanism not in {"judge_save", "fan_save", "bottom3_judge_save"}:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    pool = order[:min(3 if mechanism == "bottom3_judge_save" else 2, n)]
    j, p = np.asarray(judges), np.asarray(votes)
    if mechanism == "judge_save":
        loser = min(pool, key=lambda i: (j[i], p[i], str(ids[i])))
    elif mechanism == "fan_save":
        loser = min(pool, key=lambda i: (p[i], j[i], str(ids[i])))
    else:
        # Paper says "save one of bottom three" but omits which of the two others exits.
        # Explicit extension: save highest judge score, eliminate worst aggregate among rest.
        saved = max(pool, key=lambda i: (j[i], p[i], str(ids[i])))
        loser = next(i for i in pool if i != saved)
    return np.array([loser], dtype=int)


def violation(judges, votes, eliminated, rule="rank", judge_weight=0.5, bottom_two=False):
    """Unscaled report hinge; multi-exit extension separates exits from survivors."""
    mask = np.asarray(eliminated, bool)
    if not mask.any() or mask.all():
        return 0.0
    m = merit(judges, votes, rule, judge_weight)
    if rule == "rank":
        m = 2 * m  # exact rank-sum scale at w=0.5, as in section 5.1.2
    if bottom_two and mask.sum() == 1:
        boundary = np.sort(m)[min(1, len(m) - 1)]
    else:
        boundary = m[~mask].min()
    return float(max(0.0, m[mask].max() - boundary))


def controversy(judges, votes, eliminated_indices):
    jr, fr, n = ranks(judges), ranks(votes), len(judges)
    ix = np.asarray(eliminated_indices, int)
    a = (jr[ix] <= 2) & (fr[ix] > n / 2)
    b = (jr[ix] > n - 2) & (fr[ix] <= n / 2)
    return int(a.sum()), int(b.sum())

