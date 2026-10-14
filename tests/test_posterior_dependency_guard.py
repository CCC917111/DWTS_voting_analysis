"""Simulation ordering changes may not weaken the posterior reuse guard."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scope_guard', ROOT/'scripts/run_scope_candidates.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

SOURCE = '''import numpy as np
from scipy.stats import rankdata
def ranks(values):
    return rankdata(-np.asarray(values, dtype=float), method="average")
def order_worst(values):
    return values
'''


def test_simulation_only_order_changes_leave_posterior_dependency_unchanged():
    changed = SOURCE.replace('return values', 'return list(reversed(values))')
    assert runner.posterior_rule_dependency(SOURCE) == runner.posterior_rule_dependency(changed)


@pytest.mark.parametrize('changed', [
    SOURCE.replace('method="average"', 'method="min"'),
    SOURCE.replace('rankdata(-np.asarray', 'rankdata(np.asarray'),
    SOURCE.replace('from scipy.stats import rankdata', 'from another_package import rankdata'),
    SOURCE + '\nrankdata = lambda x, **kwargs: x\n',
    SOURCE + '\nnp = None\n',
    SOURCE + '\nranks = lambda x: x\n',
    SOURCE + '\nif True:\n    rankdata = lambda x, **kwargs: x\n',
    SOURCE.replace('def order_worst(values):', 'def order_worst(values=print("side effect")):')
])
def test_posterior_rank_or_dependency_changes_are_detected(changed):
    assert runner.posterior_rule_dependency(SOURCE) != runner.posterior_rule_dependency(changed)


def test_missing_or_duplicate_ranks_is_refused():
    with pytest.raises(ValueError):
        runner.posterior_rule_dependency('import numpy as np')
    with pytest.raises(ValueError):
        runner.posterior_rule_dependency(SOURCE+'\ndef ranks(x): return x\n')


def test_only_reviewed_simulation_import_is_permitted():
    assert runner.posterior_rule_dependency(SOURCE) == runner.posterior_rule_dependency('from fractions import Fraction\n'+SOURCE)
    assert runner.posterior_rule_dependency(SOURCE) != runner.posterior_rule_dependency('import unknown_module\n'+SOURCE)
