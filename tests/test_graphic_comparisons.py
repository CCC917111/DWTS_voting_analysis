import numpy as np
import pandas as pd
import pytest
from scripts.compare_graphic_targets import (
    comparison_fields,figure5_compare,figure6_compare,lookup_grid,
    protocol_paths,relative_change,NORMS,SCOPES,MECHANISMS,
)


def grid_fixture():
    rows=[]
    for scope in SCOPES:
        for mechanisms in MECHANISMS.values():
            for mechanism in mechanisms:
                for k in range(11):
                    count={'none':10,'judge_save':8,'bottom3_judge_save':12,
                           'bottom3_save_other_two_exit':18,'fan_save':6}[mechanism]
                    rows.append(dict(scope=scope,mechanism=mechanism,judge_weight=k/10,
                        weeks=100,contestant_rows=1000,type_a=count-2,type_b=2,controversies=count,
                        event_sum_per_week=count/100,any_controversy_week_rate=(count-1)/100,
                        event_sum_per_scope_contestant_row=count/1000,event_sum_per_all_source_rows=count/1200,
                        interpretation='two exits' if mechanism.endswith('two_exit') else 'one exit'))
    return pd.DataFrame(rows)


def target(panel='controversy_rate',mechanism='Original',value=5.,status='measured_partial_overlap'):
    return dict(panel=panel,mechanism=mechanism,judge_weight=.5,value_approx=value,
                absolute_graphic_tolerance=.05,status=status,note='Keep this uncertainty note',
                graphic_lower_bound=value-.05 if np.isfinite(value) else np.nan,
                graphic_upper_bound=value+.05 if np.isfinite(value) else np.nan)


def test_unavailable_graphic_never_becomes_numerical_match():
    result=comparison_fields(np.nan,np.nan,5.,'unavailable_obscured')
    assert result['comparison_status']=='graphic_unavailable'
    assert result['within_graphic_tolerance'] is None
    assert not result['exact_author_match_claim']
    assert comparison_fields(5.,.05,np.nan,'measured_graphic')['comparison_status']=='computed_undefined'


def test_graphical_band_is_inclusive_and_never_exact_author_claim():
    result=comparison_fields(5.,.05,5.05,'measured_graphic')
    assert result['within_graphic_tolerance']
    assert not result['exact_author_match_claim']
    assert comparison_fields(5.,.05,5.051,'measured_graphic')['comparison_status']=='outside_graphic_tolerance'


def test_all_original_definitions_scopes_and_normalizations_are_retained():
    result=figure6_compare(pd.DataFrame([target()]),grid_fixture(),'baseline','frozen.csv',1200)
    assert len(result)==2*3*4
    assert set(result.computed_mechanism)=={'none','judge_save'}
    assert set(result.computed_scope)==set(SCOPES)
    assert set(result.normalization)==set(NORMS)
    assert result.status.eq('measured_partial_overlap').all()
    assert result.note.eq('Keep this uncertainty note').all()
    assert result.original_definition_ambiguity.str.contains('No Save').all()
    assert set(result[result.normalization.eq('event_sum_per_all_source_rows')].normalization_denominator)=={1200}


def test_both_bottom3_rules_and_both_relative_baselines_are_retained():
    result=figure6_compare(pd.DataFrame([target('relative_cr_change','Bottom-3')]),grid_fixture(),'baseline','frozen.csv')
    assert len(result)==2*2*3*4
    assert set(result.computed_mechanism)=={'bottom3_judge_save','bottom3_save_other_two_exit'}
    assert set(result.original_definition)=={'none','judge_save'}
    row=result[(result.computed_mechanism=='bottom3_judge_save')&(result.original_definition=='none')&(result.normalization=='event_sum_per_week')].iloc[0]
    assert row.calculated==pytest.approx(20.)
    assert row.relative_sign_convention.startswith('Negative = Better')
    assert relative_change(6,10)==-40
    assert np.isnan(relative_change(6,0))


def test_missing_grid_slot_is_an_error_not_silent_imputation():
    grid=grid_fixture()
    with pytest.raises(ValueError,match='Grid coverage differs'):lookup_grid(grid.iloc[1:])


def test_figure5_compares_coefficient_and_both_ci_endpoints_separately():
    targets=pd.DataFrame([dict(outcome='Judge Score',term='Age (std)',coefficient_approx=-.06,
        ci95_low_approx=-.1,ci95_high_approx=-.02,absolute_graphic_tolerance=.01,status='measured_graphic')])
    coefficients=pd.DataFrame([dict(target='judge_mean',feature='age',coefficient=-.065,lower_95=-.2,
        upper_95=-.019,branch='literal_drop_contestant_week',covariance='conventional_OLS',
        ci_reference='Student_t',observation_unit='contestant_week',cold_start='drop',n=100,season_clusters=5)])
    result=figure5_compare(targets,coefficients,'baseline','frozen.csv')
    assert len(result)==3
    assert set(result.metric)=={'coefficient','lower_95','upper_95'}
    assert result.set_index('metric').within_graphic_tolerance.to_dict()=={'coefficient':True,'lower_95':False,'upper_95':True}
    assert not result.exact_author_match_claim.any()


def test_protocol_paths_explicitly_replace_defaults_and_reject_duplicates(tmp_path):
    cases=protocol_paths(tmp_path,[['named_case','ols','grid']])
    assert cases=={'named_case':(tmp_path/'ols',tmp_path/'grid')}
    with pytest.raises(ValueError,match='Duplicate protocol'):protocol_paths(tmp_path,[['a','x','y'],['a','q','r']])
