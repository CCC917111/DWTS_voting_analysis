#!/usr/bin/env python3
"""Compare approximate graphic coordinates with all declared saved protocols.

Reads CSVs only: no Bayesian/OLS/RF fitting, author-data recovery, branch selection,
curve interpolation, or replacement of obscured values. A compatible coordinate is
not exact numerical reproduction and is not a basis for choosing a protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CASES={
    'baseline':('report_ols_literal','full_report_baseline'),
    'zero_first':('ols_zero_first','full_report_zero_first'),
    'exclude_withdrawn':('ols_exclude_withdrawn','full_report_exclude_withdrawn'),
}
OUTCOMES={'Judge Score':'judge_mean','Fan Votes':'fan_mean','Placement (1=Best)':'placement'}
TERMS={'Partner Score (std)':'partner_score_literal','Age (std)':'age',
       **{f'Industry: {v}':f'industry_{v}' for v in ['Other','Model','Music','RealityTV','Sports']}}
MECHANISMS={'Original':['none','judge_save'],
            'Bottom-3':['bottom3_judge_save','bottom3_save_other_two_exit'],
            'Fan Save':['fan_save']}
NORMS=['event_sum_per_week','any_controversy_week_rate',
       'event_sum_per_scope_contestant_row','event_sum_per_all_source_rows']
SCOPES=['all_weeks_n_ge_3','nonfinal_weeks','elimination_weeks']
ORIGINAL_AMBIGUITY='Printed p21 prose: Original is Bottom-2 Judge Save; Figure 6 bar caption: Original (No Save). Both interpretations retained; neither identified as author intent.'
DENOMINATOR_AMBIGUITY='Report describes proportion of weeks, but plotted count/rate relation does not identify the denominator. Week/event, any-controversial-week, scope-row and all-source-row normalizations are declared alternatives, not recovered author definitions.'
BOTTOM3_AMBIGUITY='After saving one of the bottom three, the source does not fully identify which/how many remaining contestants exit. Single-exit and other-two-exit interpretations retained.'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison_fields(graphic_value,tolerance,computed_value,status):
    """Blank graphical targets can never become compatibility judgments."""
    result=dict(calculated=computed_value,absolute_error=np.nan,
                within_graphic_tolerance=None,comparison_status='graphic_unavailable',
                reference_class='approximate_graphic_coordinate',exact_author_match_claim=False)
    if status=='unavailable_obscured' or pd.isna(graphic_value):
        return result
    if pd.isna(computed_value) or not np.isfinite(float(computed_value)):
        result['comparison_status']='computed_undefined'
        return result
    error=abs(float(computed_value)-float(graphic_value))
    compatible=error<=float(tolerance)+1e-12
    result.update(absolute_error=error,within_graphic_tolerance=compatible,
                  comparison_status='within_graphic_tolerance' if compatible else 'outside_graphic_tolerance')
    return result


def figure5_compare(targets,coefficients,case,input_file):
    rows=[]
    for target in targets.to_dict('records'):
        outcome=OUTCOMES[target['outcome']];feature=TERMS[target['term']]
        models=coefficients[(coefficients.target==outcome)&(coefficients.feature==feature)]
        if models.empty:raise ValueError(f'Missing OLS comparison: {case} {outcome} {feature}')
        if models.duplicated(['branch','covariance']).any():raise ValueError('Duplicate OLS branch/covariance record')
        for model in models.to_dict('records'):
            for metric,column in [('coefficient','coefficient_approx'),('lower_95','ci95_low_approx'),('upper_95','ci95_high_approx')]:
                graphic_value=target[column];tolerance=target['absolute_graphic_tolerance']
                rows.append({**target,'source_case':case,'input_csv':input_file,
                    'model_target':outcome,'model_feature':feature,'metric':metric,
                    'graphic_value_approx':graphic_value,'graphic_lower_bound':graphic_value-tolerance,
                    'graphic_upper_bound':graphic_value+tolerance,
                    'ols_branch':model['branch'],'covariance':model['covariance'],
                    'branch_role':'declared_primary' if model['branch']=='literal_drop_contestant_week' else 'declared_sensitivity',
                    'covariance_role':'primary_conventional' if model['covariance']=='conventional_OLS' else 'season_cluster_supplement',
                    'ci_reference':model['ci_reference'],'observation_unit':model['observation_unit'],
                    'cold_start':model['cold_start'],'model_n':model['n'],'season_clusters':model['season_clusters'],
                    'ambiguity':'First-week handling, placement observation unit, covariance and standardization conventions are not fully identified; no branch selected by proximity.',
                    **comparison_fields(graphic_value,tolerance,model[metric],target['status'])})
    return pd.DataFrame(rows)


def lookup_grid(grid):
    d={}
    for row in grid.to_dict('records'):
        key=(row['scope'],row['mechanism'],round(float(row['judge_weight']),1))
        if key in d:raise ValueError(f'Duplicate grid key {key}')
        d[key]=row
    expected={(s,m,round(k/10,1)) for s in SCOPES for ms in MECHANISMS.values() for m in ms for k in range(11)}
    if set(d)!=expected:raise ValueError(f'Grid coverage differs: missing={expected-set(d)}, unexpected={set(d)-expected}')
    return d


def normalized_value(row,normalization):
    return float(row[normalization])


def relative_change(value,original):
    return 100*(value-original)/original if original else np.nan


def verify_saved_relative(grid,saved):
    """Check our all-scope extension against the existing nonfinal count ratio."""
    d=lookup_grid(grid)
    for r in saved.to_dict('records'):
        w=round(float(r['judge_weight']),1)
        original=d[('nonfinal_weeks',r['original_definition'],w)]['controversies']
        value=d[('nonfinal_weeks',r['mechanism'],w)]['controversies']
        expected=relative_change(value,original)
        if not np.isclose(expected,r['relative_change_percent'],atol=1e-9,rtol=0,equal_nan=True):
            raise ValueError(f'Saved relative-change table disagrees with grid at {r}')


def figure6_compare(targets,grid,case,input_file,all_source_rows=None):
    d=lookup_grid(grid);rows=[]
    count_cols={'type_a_count':'type_a','type_b_count':'type_b','total_count':'controversies'}
    for target in targets.to_dict('records'):
        w=round(float(target['judge_weight']),1)
        for scope in SCOPES:
            for mechanism in MECHANISMS[target['mechanism']]:
                g=d[(scope,mechanism,w)]
                norms=NORMS if target['panel'] in ['controversy_rate','relative_cr_change'] else ['unnormalized_count']
                baselines=['none','judge_save'] if target['panel']=='relative_cr_change' else ['not_applicable']
                for norm in norms:
                    for baseline in baselines:
                        orig=None
                        if target['panel']=='relative_cr_change':
                            orig=d[(scope,baseline,w)]
                            value=relative_change(normalized_value(g,norm),normalized_value(orig,norm))
                            metric=f'relative_change_of_{norm}'
                        elif target['panel']=='controversy_rate':
                            value=100*normalized_value(g,norm);metric=norm
                        else:value=float(g[count_cols[target['panel']]]);metric=count_cols[target['panel']]
                        if norm=='event_sum_per_scope_contestant_row':denominator=g['contestant_rows'];denom_unit='scope contestant rows'
                        elif norm=='event_sum_per_all_source_rows':
                            # Read declared source-row count from its saved manifest if available;
                            # never infer an author denominator from graphical targets.
                            denominator=np.nan if all_source_rows is None else all_source_rows
                            denom_unit='all source contestant rows (saved source manifest; author denominator remains unidentified)'
                        elif norm in ['event_sum_per_week','any_controversy_week_rate']:denominator=g['weeks'];denom_unit='weeks'
                        else:denominator=np.nan;denom_unit='not applicable to count'
                        rows.append({**target,'source_case':case,'input_csv':input_file,
                            'computed_scope':scope,'computed_mechanism':mechanism,'normalization':norm,
                            'computed_metric':metric,'original_definition':baseline,
                            'computed_weeks':g['weeks'],'computed_contestant_rows':g['contestant_rows'],
                            'normalization_denominator':denominator,'normalization_denominator_unit':denom_unit,
                            'computed_original_value_before_relative_change':np.nan if orig is None else orig[norm],
                            'computed_value_before_relative_change':np.nan if orig is None else g[norm],
                            'exit_interpretation':g['interpretation'],
                            'original_definition_ambiguity':ORIGINAL_AMBIGUITY,
                            'denominator_ambiguity':DENOMINATOR_AMBIGUITY,
                            'bottom3_rule_ambiguity':BOTTOM3_AMBIGUITY,
                            'relative_sign_convention':'Negative = Better, exactly as chart heading; zero Original denominator remains undefined.' if target['panel']=='relative_cr_change' else 'not applicable',
                            'protocol_selection':'Predeclared alternatives retained; no closest branch selected.',
                            **comparison_fields(target['value_approx'],target['absolute_graphic_tolerance'],value,target['status'])})
    return pd.DataFrame(rows)


def summarize(frame,columns):
    summary=frame.groupby(columns,dropna=False,sort=False).comparison_status.value_counts().unstack(fill_value=0).reset_index()
    for col in ['within_graphic_tolerance','outside_graphic_tolerance','graphic_unavailable','computed_undefined']:
        if col not in summary:summary[col]=0
    summary['interpretation']='Counts of approximate graphical compatibility only; not exact author reproduction or protocol selection.'
    return summary


def protocol_paths(root,overrides=None):
    specs=overrides or [(name,str(Path('outputs')/ols),str(Path('outputs')/grid)) for name,(ols,grid) in CASES.items()]
    cases={}
    for name,ols,grid in specs:
        if name in cases:raise ValueError(f'Duplicate protocol case name {name}')
        def resolve(value):
            p=Path(value)
            return p.resolve() if p.is_absolute() else (root/p).resolve()
        cases[name]=(resolve(ols),resolve(grid))
    return cases


def path_label(path,root):
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--targets',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--protocol',nargs=3,action='append',metavar=('CASE','OLS_DIR','FIGURE6_DIR'),help='Repeat to explicitly replace the three default source cases; relative directories resolve under --root.')
    args=parser.parse_args();root=args.root.resolve()
    targets=args.targets or root/'docs/full_audit/graphic_targets';out=args.out or root/'outputs/graphic_comparisons'
    target5=pd.read_csv(targets/'figure5_graphic_targets.csv');target6=pd.read_csv(targets/'figure6_curve_graphic_targets.csv')
    if len(target5)!=21 or len(target6)!=154:raise ValueError('Graphical target coverage unexpected')
    if target5.duplicated(['outcome','term']).any() or target6.duplicated(['panel','mechanism','judge_weight']).any():raise ValueError('Duplicate graphical target key')
    if target6.value_approx.isna().sum()!=14 or (target6.status=='measured_partial_overlap').sum()!=10:raise ValueError('Graphical uncertainty flags changed unexpectedly')
    outputs5=[];outputs6=[];inputs={}
    for filename in ['figure5_graphic_targets.csv','figure6_curve_graphic_targets.csv','axis_calibration.json','digitization_manifest.json']:
        path=targets/filename;inputs[str(path.relative_to(root)) if path.is_relative_to(root) else str(path)]=digest(path)
    cases=protocol_paths(root,args.protocol)
    for case,(ols_dir,grid_dir) in cases.items():
        ols_path=ols_dir/'ols_coefficients.csv'
        grid_path=grid_dir/'figure6_grid_protocols.csv'
        relative_path=grid_dir/'figure6_relative_change.csv'
        all_source_rows=None
        source_manifest=grid_dir/'manifest.json'
        if source_manifest.exists():
            inputs[path_label(source_manifest,root)]=digest(source_manifest)
            all_source_rows=json.loads(source_manifest.read_text()).get('rows')
        for path in [ols_path,grid_path,relative_path]:inputs[path_label(path,root)]=digest(path)
        ols=pd.read_csv(ols_path);grid=pd.read_csv(grid_path);relative=pd.read_csv(relative_path)
        verify_saved_relative(grid,relative)
        outputs5.append(figure5_compare(target5,ols,case,path_label(ols_path,root)))
        outputs6.append(figure6_compare(target6,grid,case,path_label(grid_path,root),all_source_rows=all_source_rows))
    f5=pd.concat(outputs5,ignore_index=True);f6=pd.concat(outputs6,ignore_index=True)
    out.mkdir(parents=True,exist_ok=True)
    f5.to_csv(out/'figure5_coefficient_ci_graphic_comparison.csv',index=False)
    f6.to_csv(out/'figure6_curve_graphic_comparison.csv',index=False)
    summarize(f5,['source_case','ols_branch','covariance','metric']).to_csv(out/'figure5_graphic_summary.csv',index=False)
    summarize(f6,['source_case','panel','computed_scope','computed_mechanism','normalization','original_definition']).to_csv(out/'figure6_graphic_summary.csv',index=False)
    manifest=dict(operation='comparison_of_approximate_graphic_coordinates_to_frozen_results',
        source_cases=list(cases),figure5_target_triplets=21,figure6_target_slots=154,
        figure6_measured_targets=140,figure6_unavailable_targets=14,figure6_partially_overlapped_targets=10,
        figure5_comparison_rows=len(f5),figure6_comparison_rows=len(f6),
        source_input_sha256=inputs,script_sha256=digest(__file__),
        no_model_reruns=True,no_protocol_selection=True,no_unavailable_point_imputation=True,
        exact_author_numerical_reproduction=False,full_report_reproduced=False,
        source_ambiguities=[ORIGINAL_AMBIGUITY,DENOMINATOR_AMBIGUITY,BOTTOM3_AMBIGUITY],
        comparison_status_meaning='within_graphic_tolerance means coordinate compatibility only, never exact numerical validation',
        relative_change_extension='Ratios calculated from frozen grid rates for every declared scope/normalization. Existing nonfinal count-ratio CSV cross-checked; no new simulation.',
        refresh_instruction='Rerun this script after source CSVs change; source hashes make stale comparisons detectable.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'README.txt').write_text('Graphic comparison audit\n\n'
        'These comparisons use approximate coordinates, not exact author numerical targets.\n'
        'Every branch, mechanism, scope and normalization is retained without ranking or selection.\n'
        'Figure 5 includes all saved OLS branches and both conventional/season-cluster intervals.\n'
        'Figure 6 retains partial-overlap flags, blank obscured points, both Original definitions,\n'
        'both Bottom-3 exit rules, three replay scopes and four rate normalizations.\n'
        'Relative changes are derived algebraically from saved grids; zero baselines stay undefined.\n'
        'Alternative normalizations may give identical relative changes when their common denominator cancels.\n'
        'Compatibility summary rows are duplicated protocol evaluations, not independent report claims.\n'
        'No count of compatible points establishes a reproduced curve or identifies author choices.\n'
        'See docs/full_audit/graphic_targets/GRAPHIC_DIGITIZATION_NOTES.txt for measurement limits.\n'
        'Rerun: python scripts/compare_graphic_targets.py --root /path/to/repo --out /path/to/output\n'
        'Explicit paths: repeat --protocol CASE OLS_DIR FIGURE6_DIR; this replaces defaults.\n')
    print(json.dumps({k:manifest[k] for k in ['figure5_comparison_rows','figure6_comparison_rows','no_model_reruns','exact_author_numerical_reproduction']},indent=2))

if __name__=='__main__':main()
