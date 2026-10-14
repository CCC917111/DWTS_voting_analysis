"""Identity correction is explicit, applies to both models, and preserves targets."""
from pathlib import Path
import pandas as pd
import pytest
from types import SimpleNamespace
from dwts.cli import analyze

ROOT=Path(__file__).resolve().parents[1]


def test_explicit_identity_config_routes_both_models_and_retains_source(monkeypatch,tmp_path):
    import dwts.attribution as attribution
    import dwts.evaluation as evaluation
    import dwts.plots as plots
    votes=pd.read_csv(ROOT/'outputs/report_verified/votes.csv').query('season in [33,34] and week==1').copy()
    original=votes.copy(deep=True)
    monkeypatch.setattr(evaluation,'evaluate',lambda *a,**k:{'influence_balance':pd.DataFrame()})
    monkeypatch.setattr(evaluation,'weight_grid',lambda *a,**k:(pd.DataFrame(),pd.DataFrame()))
    monkeypatch.setattr(plots,'make_plots',lambda *a,**k:None)
    received=[]
    monkeypatch.setattr(attribution,'ols_analysis',lambda d,*a,**k:received.append(d.copy()))
    monkeypatch.setattr(attribution,'rf_analysis',lambda d,*a,**k:received.append(d.copy()))
    analyze(votes,tmp_path,{'partner_identity_mode':'verified_canonical_identity'})
    assert len(received)==2
    for frame in received:
        pd.testing.assert_frame_equal(frame[original.columns.drop('partner')],original.drop(columns='partner'),check_exact=True)
        assert not frame.partner.str.contains(r'[()/]',regex=True).any()
        assert frame.partner_raw.equals(original.partner)
        assert 'Valentin Chmerkovskiy' in set(frame.partner)
    pd.testing.assert_frame_equal(votes,original,check_exact=True)
    assert (tmp_path/'attribution_input.csv').exists()
    assert (tmp_path/'partner_identity_provenance.csv').exists()


def test_unknown_identity_mode_is_rejected_before_outputs(tmp_path):
    with pytest.raises(ValueError,match='partner_identity_mode'):
        analyze(pd.DataFrame(),tmp_path,{'partner_identity_mode':'guess'})
    assert list(tmp_path.iterdir())==[]


def test_full_reconstruction_v5_config_is_accepted_without_fitting(monkeypatch,tmp_path):
    import dwts.cli as cli
    class BeforeAnyFit(Exception):
        pass
    def stop_at_prepare(*args,**kwargs):
        raise BeforeAnyFit()
    monkeypatch.setattr(cli,'prepare',stop_at_prepare)
    args=SimpleNamespace(config=ROOT/'configs/report_reconstruction_v5.json',out=tmp_path/'new',
                         resume=False,data=ROOT/'data/raw/2026_MCM_Problem_C_Data.csv')
    with pytest.raises(BeforeAnyFit):
        cli.run(args)
