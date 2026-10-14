"""Independent algebraic checks for the no-fit metric-identifiability audit."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_metric_identifiability.py"
spec = importlib.util.spec_from_file_location("metric_identifiability", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_rounding_interval_uses_joint_extrema():
    lo, hi = audit.implied_sd_interval(.614, .072)
    assert lo == pytest.approx(.0715 / np.sqrt(.3865))
    assert hi == pytest.approx(.0725 / np.sqrt(.3855))
    assert lo > .115 and hi < .117


def test_pooled_interval_reduces_to_identity():
    lo, hi = audit.feasible_rmse_interval([2, 3], [1., 4.], 2.7, "pooled", "pooled", .4)
    assert lo == pytest.approx(np.sqrt(.5995*2.7))
    assert hi == pytest.approx(np.sqrt(.6005*2.7))


def test_exact_bounds_contain_mse_simplex_and_attain_endpoints():
    rng = np.random.default_rng(123)
    n = np.array([10., 17., 23.])
    v = np.array([.7, 2.1, 1.2])
    pooled_v = 1.6
    q = n/n.sum(); equal = np.ones(3)/3
    r2 = .4
    for rmode in ["pooled", "fold_equal", "fold_size_weighted"]:
        c = {"pooled": q/pooled_v, "fold_equal": equal/v, "fold_size_weighted": q/v}[rmode]
        for emode in ["pooled", "fold_equal", "fold_size_weighted"]:
            lo, hi = audit.feasible_rmse_interval(n, v, pooled_v, rmode, emode, r2)
            weights = rng.dirichlet(np.ones(3), 200)
            mse = (.6 * weights) / c
            if emode == "pooled":
                value = np.sqrt(mse @ q)
                low_mse = np.zeros(3); high_mse = np.zeros(3)
                i, j = np.argmin(q/c), np.argmax(q/c)
                low_mse[i] = .5995/c[i]; high_mse[j] = .6005/c[j]
                low_value, high_value = np.sqrt(q@low_mse), np.sqrt(q@high_mse)
            else:
                b = equal if emode == "fold_equal" else q
                value = np.sqrt(mse) @ b
                low_mse = np.zeros(3); i = np.argmin(b*b/c)
                low_mse[i] = .5995/c[i]
                high_mse = .6005*(b/c)**2 / np.sum(b*b/c)
                low_value, high_value = b@np.sqrt(low_mse), b@np.sqrt(high_mse)
            assert value.min() >= lo-1e-12 and value.max() <= hi+1e-12
            assert low_value == pytest.approx(lo)
            assert high_value == pytest.approx(hi)


def test_hierarchical_weights_have_equal_season_and_within_season_week_mass():
    d = pd.DataFrame({"season": [1,1,1,2,2,2,2], "week": [1,1,2,1,1,1,1],
                      "contestant_id": list("abacdef"), "actual": np.arange(7)})
    d["weight"] = audit.natural_weights(d)["season_then_week_equal"]
    assert d.groupby("season").weight.sum().to_list() == pytest.approx([.5,.5])
    assert d[d.season == 1].groupby("week").weight.sum().to_list() == pytest.approx([.25,.25])
    assert d[d.season == 2].weight.to_list() == pytest.approx([.125]*4)


def test_no_output_in_saved_sources():
    with pytest.raises(ValueError, match="separate"):
        audit.run(audit.ROOT / "outputs/report_verified")


def test_current_audit_certificates_and_weighting_counterexample(tmp_path):
    audit.run(tmp_path / "new_audit")
    root = tmp_path / "new_audit"
    ranges = pd.read_csv(root / "range_certificates.csv")
    assert ranges[ranges.target == "judge_mean"].any_subset_or_row_weights_excluded.all()
    mixed = pd.read_csv(root / "mixed_aggregation_bounds.csv")
    assert len(mixed) == 81
    assert mixed[mixed.target.isin(["judge_mean", "fan_mean"])].report_pair_excluded.all()
    natural = pd.read_csv(root / "row_weight_protocols.csv")
    witness = natural[(natural.target == "fan_mean") & (natural.protocol == "season_then_week_equal")]
    assert len(witness) == 3
    assert not witness.report_pair_excluded_for_any_predictions.any()
    assert not witness.saved_prediction_matches_both.astype(bool).any()
