import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from dwts.model import FitConfig, build_model, tensor_ranks
from dwts.rules import ranks, violation


def tiny(season=1):
    j = np.array([9., 7., 5.])
    return pd.DataFrame({"season": season, "week": 1, "contestant_id": ["A", "B", "C"],
                         "judge_mean": j, "judge_z": (j-j.mean()) / j.std(),
                         "judge_share": j/j.sum(), "eliminated": [False, False, True],
                         "withdrawal": False, "rule": "rank" if season != 3 else "percentage"})


def test_tensor_tie_ranks_equal_scipy():
    import pytensor
    import pytensor.tensor as pt
    x = pt.dvector()
    fn = pytensor.function([x], tensor_ranks(x))
    for a in [[9, 9, 4], [1, 1, 1], [3, 1, 2]]:
        np.testing.assert_allclose(fn(a), ranks(a))


@pytest.mark.parametrize("season,likelihood", [(1, "report"), (3, "report"), (28, "bottom_two")])
def test_pymc_penalty_matches_independent_numpy(season, likelihood):
    cfg = FitConfig(likelihood=likelihood, parameterization="centered")
    m, d, _ = build_model(tiny(season), cfg)
    fn = m.compile_logp(vars=m.potentials)
    z = np.array([-.8, .6, 1.4])
    actual = fn({"z": z})
    expected = -100 * violation(d.judge_mean, softmax(z), d.eliminated,
                                d.rule.iloc[0], bottom_two=likelihood == "bottom_two")
    assert actual == pytest.approx(expected)


def test_no_elimination_means_no_penalty():
    d = tiny()
    d.eliminated = False
    m, _, _ = build_model(d, FitConfig())
    assert m.compile_logp(vars=m.potentials)({}) == 0

