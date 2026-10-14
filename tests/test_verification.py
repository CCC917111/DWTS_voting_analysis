import numpy as np
import pandas as pd
from dwts.attribution import select_history_weight, _designs
from dwts.features import partner_features
from dwts.verification import compare_numbers, target_identity


def synthetic():
    rows=[]
    for season in range(1,7):
        for week in range(1,4):
            for i in range(3):
                rows.append(dict(season=season,week=week,partner=f'p{i}',contestant_id=f'{season}:{i}',
                                 age=25+i,industry='Acting',judge_mean=5+week+i,
                                 fan_mean=[.2,.3,.5][i],n_active=3))
    return pd.DataFrame(rows)


def test_nested_selection_never_uses_outer_test_targets_or_histories():
    d=synthetic(); train=np.flatnonzero(d.season<6); y=d.fan_mean.to_numpy().copy()
    weight, records=select_history_weight(d,y,train,'regression',[0,.5,1],trees=7)
    changed=d.copy(); changed.loc[changed.season==6,['judge_mean','fan_mean']]=999
    y2=y.copy(); y2[d.season==6]=-1000
    weight2, records2=select_history_weight(changed,y2,train,'regression',[0,.5,1],trees=7)
    assert weight==weight2 and records==records2
    for r in records:
        a=set(r['train_seasons'].split(','));b=set(r['validation_seasons'].split(','))
        assert not a & b and '6' not in a | b


def test_cached_affine_features_equal_direct_fold_features():
    d=synthetic();tr=np.flatnonzero(d.season!=3);te=np.flatnonzero(d.season==3)
    for w in [0,.25,.5,.75,1]:
        a,b,names=_designs(d,tr,te,w,{})
        direct=partner_features(d.iloc[te],reference=d.iloc[tr],history_weight=w)
        np.testing.assert_allclose(b[:,:3],direct[['age','pq_judge','pq_vote']].to_numpy())


def test_verification_gate_separates_convergence_and_numerical_match():
    d=pd.DataFrame({'metric':['table9_A2_fan_r2','table8_percentage_fan_spearman','table5_case_count_S1-2'],
                    'report_value':[.614,.942,11.], 'reconstruction_value':[.3898,.9424,11.],
                    'comparison_status':['eligible_for_comparison','eligible_for_comparison','provisional_or_partial']})
    out=compare_numbers(d)
    assert out.verdict.tolist()==['numeric_mismatch','numeric_match','incomplete_or_unconverged']
    assert not out.verdict.eq('numeric_match').all()


def test_pooled_r2_rmse_identity_is_checked_with_rounded_bounds(tmp_path):
    d=pd.DataFrame({'fan_mean':[.1,.2,.3,.4],'judge_mean':[5,6,7,8],'judge_total':[15,18,21,24]})
    pred=pd.DataFrame({'task':['A2_fan']*4,'fold':[0,0,1,1],'actual':d.fan_mean,'prediction':d.fan_mean})
    target_identity(d,pred,tmp_path)
    rows=pd.read_csv(tmp_path/'target_identity.csv')
    fan=rows[rows.target=='fan_mean'].iloc[0]
    assert not fan.compatible_with_report_pooled_pair
    assert np.isclose(fan.required_rmse_for_report_r2, np.std(d.fan_mean)*np.sqrt(1-.614))
