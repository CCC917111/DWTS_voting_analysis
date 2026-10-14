from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from dwts.data import prepare
from dwts.evaluation import posterior_eliminations, weight_grid
from dwts.model import FitConfig
from dwts.rules import eliminate, violation


def test_recovery_inclusion_counts_match_report_denominators():
    p, audit = prepare(Path(__file__).parents[1] / "data/raw/2026_MCM_Problem_C_Data.csv",
                       outcome_policy="label_first", include_final_week_eliminations=False)
    counts = p.groupby(["season", "week"]).eliminated.any()
    assert [int(counts[[x[0] in region for x in counts.index]].sum())
            for region in [range(1,3),range(3,28),range(28,35)]] == [11,192,56]
    assert (~p.groupby(["season", "week"]).is_final_week.any()).sum() == 301
    # Explicitly preserve the label conflict instead of inventing a week-1 voting exit.
    assert not p.loc[(p.season == 18) & (p.name == "Diana Nyad"), "eliminated"].any()
    assert "labeled_exit_has_no_active_score_constraint_omitted" in audit.issue.to_list()
    assert not p.loc[p.is_final_week, "eliminated"].any()


@pytest.mark.parametrize("rule,exits,bottom_two", [
    ("rank",[0],False),("percentage",[1],False),("rank",[1,2],False),("rank",[1],True)])
def test_vectorized_posterior_evaluation_against_scalar_oracle(rule,exits,bottom_two):
    j = np.array([9.,8.,6.,5.])
    ids = np.array(["A","B","C","D"])
    mask = np.isin(np.arange(4),exits)
    ps = np.array([[.1,.2,.3,.4],[.25,.25,.25,.25],[.3,.4,.1,.2],[.6,.1,.2,.1]])
    frame = pd.DataFrame({"season":28,"week":1,"judge_mean":j,"contestant_id":ids,
                          "eliminated":mask,"withdrawal":False,"rule":rule})
    cfg = FitConfig(likelihood="bottom_two" if bottom_two else "report")
    result = posterior_eliminations(frame,ps,cfg)
    counts=np.zeros(4); hits=feasible=0
    for p in ps:
        ix=eliminate(j,p,ids,rule,count=len(exits))
        counts[ix]+=1; hits+=set(ix)==set(exits)
        feasible+=violation(j,p,mask,rule,bottom_two=bottom_two)<=1e-10
    np.testing.assert_allclose(result.p_eliminated_base_rule,counts/len(ps))
    np.testing.assert_allclose(result.p_exact_observed_exit_set,hits/len(ps))
    np.testing.assert_allclose(result.p_constraint_satisfied,feasible/len(ps))


def test_resume_mismatch_does_not_mutate_prior_results(tmp_path):
    import json
    from dwts.cli import run
    cfg=tmp_path/'config.json';cfg.write_text('{}')
    out=tmp_path/'prior';out.mkdir()
    marker=out/'panel_all_seasons.csv';marker.write_text('keep me')
    (out/'manifest.json').write_text(json.dumps({'input_sha256':'different','code_sha256':'different'}))
    (out/'run_config.json').write_text('{}')
    args=SimpleNamespace(config=str(cfg),data=str(Path(__file__).parents[1]/'data/raw/2026_MCM_Problem_C_Data.csv'),
                         out=str(out),resume=True,seasons=[1],skip_attribution=True)
    with pytest.raises(ValueError,match='Resume refused'):
        run(args)
    assert marker.read_text()=='keep me'


def test_reused_worker_rebinds_logging_to_each_job(tmp_path,monkeypatch):
    import logging
    from dwts import cli
    def fake_fit(*args):
        logging.getLogger('pymc').info('season log is writable')
        return 'done'
    monkeypatch.setattr(cli,'fit_season',fake_fit)
    for name in ['one','two']:
        assert cli._fit_job(None,None,tmp_path/name)=='done'
        assert 'season log is writable' in (tmp_path/name/'fit.log').read_text()


def test_influence_gaps_use_common_weeks_and_respect_jensen():
    from dwts.evaluation import evaluate
    rows=[]
    for week,j,p in [(1,[9,7,5],[.1,.3,.6]),(2,[7,7,7],[.6,.3,.1]),(3,[9,7,5],[.6,.3,.1])]:
        for i in range(3):
            rows.append({'season':1,'week':week,'judge_mean':j[i],'fan_mean':p[i],
                         'contestant_id':str(i),'name':str(i),'eliminated':i==2,
                         'withdrawal':False,'rule':'rank'})
    tables=evaluate(pd.DataFrame(rows))
    for _,r in tables['influence_balance'].iterrows():
        weekly=tables['correlations_weekly']
        valid=weekly[weekly.rule==r.rule].dropna(subset=['fan_spearman','judge_spearman'])
        assert r.paired_spearman_weeks==len(valid)
        assert r.mean_weekly_influence_gap + 1e-12 >= r.influence_gap

