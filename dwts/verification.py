"""Numerical report checks, separate from execution and sampling success.

Report numbers are used only after fitting. They never enter training or selection.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd


def metric_tolerance(name):
    if name.startswith('table5_') or name == 'unique_celebrities' or name.endswith(('_rounds', '_controversies', '_count')):
        return 1e-12
    if name.startswith('pcir_'):
        return .005
    if name.endswith('gap_of_mean_correlations'):
        return .001  # derived from two independently rounded 3-decimal inputs
    if name.startswith('section4_'):
        return .00005
    return .0005  # half one unit of the report's last printed decimal


def compare_numbers(frame):
    d = frame.copy()
    d['reference_class'] = np.where(
        d.metric.str.startswith('table8_') & d.metric.str.endswith('_rounds'),
        'inferred_scope_not_reported',
        np.where(d.metric.str.endswith('gap_of_mean_correlations'),
                 'derived_from_rounded_report_values', 'printed_report_value'))
    assumed=d.reference_class.eq('inferred_scope_not_reported')
    previous=d.get('legacy_assumed_reference_value',pd.Series(np.nan,index=d.index))
    d['legacy_assumed_reference_value']=previous.combine_first(d.report_value.where(assumed)).where(assumed)
    d['compatible_with_table8_rounded_pairs']=np.where(assumed,d.reconstruction_value.isin([300,301]),False)
    d.loc[assumed,'report_value']=np.nan
    if 'difference' in d:
        d.loc[assumed,'difference']=np.nan
    d['absolute_tolerance'] = [metric_tolerance(x) for x in d.metric]
    d['numerical_match'] = (np.isfinite(d.reconstruction_value) &
                           ((d.reconstruction_value-d.report_value).abs() <= d.absolute_tolerance + 1e-12))
    d['verdict'] = np.where(d.numerical_match, 'numeric_match', 'numeric_mismatch')
    # Table 8 does not print a denominator. Both 300 and 301 are compatible
    # with all four rounded count/rate pairs; 301 is not an observed target.
    d.loc[d.reference_class == 'inferred_scope_not_reported', 'verdict'] = 'scope_assumption_not_validation'
    if 'comparison_status' in d:
        d.loc[d.comparison_status != 'eligible_for_comparison', 'verdict'] = 'incomplete_or_unconverged'
    return d


def target_identity(votes, predictions, out):
    """For pooled metrics on identical rows, MSE=(1-R2)*Var(y).

    Do NOT apply this identity to means of fold R2/RMSE. Those are emitted separately.
    """
    rows = []
    for target, r2, rmse in [('fan_mean', .614, .072), ('judge_mean', .146, 5.320), ('judge_total', .146, 5.320)]:
        variance = float(votes[target].var(ddof=0))
        # Include reported rounding uncertainty in the compatibility check.
        low = (rmse-.0005)**2/(1-(r2-.0005))
        high = (rmse+.0005)**2/(1-(r2+.0005))
        rows.append(dict(target=target, n=len(votes), target_sd=np.sqrt(variance),
                         report_r2=r2, report_rmse=rmse,
                         report_implied_target_sd=rmse/np.sqrt(1-r2),
                         required_rmse_for_report_r2=np.sqrt((1-r2)*variance),
                         compatible_with_report_pooled_pair=low <= variance <= high,
                         condition='same unweighted rows, pooled R2 and RMSE; NOT fold-averaged metrics'))
    pd.DataFrame(rows).to_csv(out/'target_identity.csv', index=False)
    from sklearn.metrics import r2_score, mean_squared_error
    fold_rows=[]
    for task, d in predictions[predictions.task.isin(['A1_judge','A2_fan'])].groupby('task'):
        values=[(len(g),r2_score(g.actual,g.prediction),mean_squared_error(g.actual,g.prediction)**.5)
                for _,g in d.groupby('fold')]
        for aggregation in ['pooled','fold_mean','fold_size_weighted_mean']:
            if aggregation=='pooled':
                a,b=r2_score(d.actual,d.prediction),mean_squared_error(d.actual,d.prediction)**.5
            else:
                weights=None if aggregation=='fold_mean' else [x[0] for x in values]
                a=np.average([x[1] for x in values],weights=weights)
                b=np.average([x[2] for x in values],weights=weights)
            fold_rows.append(dict(task=task,aggregation=aggregation,r2=a,rmse=b))
    pd.DataFrame(fold_rows).to_csv(out/'metric_aggregation_audit.csv',index=False)


def extended_comparison(votes, out):
    """Check report sections that were absent from the original 47-row comparison."""
    from scipy.stats import skew, kurtosis
    df=pd.read_csv(out/'report_comparison.csv')
    extra=[]
    def add(name, expected, actual, basis):
        extra.append(dict(metric=name,report_value=expected,reconstruction_value=actual,
                          difference=actual-expected,basis=basis,full_34_seasons=votes.season.nunique()==34,
                          sampling_diagnostics_pass=bool(df.sampling_diagnostics_pass.all()),
                          comparison_status='eligible_for_comparison' if len(df) and df.comparison_status.eq('eligible_for_comparison').all() else 'provisional_or_partial'))
    x=votes.judge_mean.to_numpy(); q1,q3=np.quantile(x,[.25,.75])
    for name,expected,actual in [('judge_cv',.1846,np.std(x,ddof=1)/np.mean(x)),
                                 ('judge_iqr_coefficient',.125,(q3-q1)/(q3+q1)),
                                 ('judge_skew',-.2648,skew(x,bias=False)),
                                 ('judge_excess_kurtosis',-.1707,kurtosis(x,bias=False))]:
        add('section4_'+name,expected,actual,'all active scored rows; sample/unbiased moments')
    grid=pd.read_csv(out/'weight_grid_summary.csv')
    for mechanism,expected in [('fan_save',75),('judge_save',83),('bottom3_judge_save',122)]:
        g=grid[(grid.rule=='rank') & (grid.mechanism==mechanism) & np.isclose(grid.judge_weight,.5)]
        add('section8_'+mechanism+'_controversies',expected,float(g.controversies.iloc[0]),
            'Section 8 w=.5; current Type A+B and nonfinal-week scope; report scope not fully specified')
    return pd.concat([df,pd.DataFrame(extra)],ignore_index=True)


def correlation_scope_audit(votes, out):
    corr=pd.read_csv(out/'correlations_weekly.csv')
    flags=votes.groupby(['season','week']).agg(has_exit=('eliminated','any'),is_final=('is_final_week','any')).reset_index()
    corr=corr.merge(flags,on=['season','week'],validate='many_to_one')
    rows=[]
    for scope, frame in [('all_weeks',corr),('elimination_weeks',corr[corr.has_exit]),('nonfinal_weeks',corr[~corr.is_final])]:
        for rule,g in frame.groupby('rule'):
            g=g.dropna(subset=['fan_spearman','judge_spearman'])
            for aggregate in ['weekly_mean','season_mean']:
                v=g[['fan_spearman','judge_spearman']] if aggregate=='weekly_mean' else g.groupby('season')[['fan_spearman','judge_spearman']].mean()
                rows.append(dict(scope=scope,rule=rule,aggregation=aggregate,weeks=len(g),
                                 fan_spearman=v.fan_spearman.mean(),judge_spearman=v.judge_spearman.mean()))
    pd.DataFrame(rows).to_csv(out/'correlation_scope_audit.csv',index=False)


def audit_outputs(source, out):
    source=Path(source)
    votes=pd.read_csv(out/'votes.csv')
    comparisons=compare_numbers(extended_comparison(votes,out))
    comparisons.to_csv(out/'numerical_verification.csv',index=False)
    target_identity(votes,pd.read_csv(out/'rf_predictions.csv'),out)
    correlation_scope_audit(votes,out)
    result={'execution_completed':True,
            'report_numerically_verified':bool(comparisons.loc[~comparisons.reference_class.eq('inferred_scope_not_reported'),'verdict'].eq('numeric_match').all()),
            'checks':len(comparisons),
            'numeric_matches':int(comparisons.verdict.eq('numeric_match').sum()),
            'numeric_mismatches':int(comparisons.verdict.eq('numeric_mismatch').sum()),
            'printed_report_checks':int(comparisons.reference_class.eq('printed_report_value').sum()),
            'printed_report_matches':int((comparisons.reference_class.eq('printed_report_value') & comparisons.verdict.eq('numeric_match')).sum()),
            'scope_assumptions_not_validation':int(comparisons.reference_class.eq('inferred_scope_not_reported').sum()),
            'full_report_coverage':False,
            'coverage_note':'Legacy 54-row ledger includes 48 comparisons to printed references, 2 derived arithmetic comparisons and 4 unreported denominator assumptions. Some source claims are compared under multiple protocols; rows are not independent tests. Comprehensive coverage is handled separately. Numeric matches alone do not establish identical protocols.',
            'baseline_source':str(source),
            'latent_targets':f'posterior means read from {source}; no posterior fitting in this analysis stage',
            'posterior_source_manifest':str(source/'manifest.json'),
            'audit_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'completed_utc':datetime.now(timezone.utc).isoformat()}
    (out/'verification_status.json').write_text(json.dumps(result,indent=2))
    return result


def run_verification(args):
    from .cli import analyze, code_digest, dump_json
    from .reconcile import write_comparison
    from .plots import make_report_figures
    source,out=Path(args.source),Path(args.out)
    config=json.loads(Path(args.config).read_text())
    # Analyze-only path must reject unsupported options rather than silently ignore them.
    allowed={'simulation_scope','rf_trees','rf_folds','history_weight','history_grid','rf_inner_folds',
             'rf_min_samples_leaf','rf_judge_target','rf_fan_scale','fit','partner_identity_mode'}
    if set(config)-allowed:
        raise ValueError('Unknown verification config keys: '+str(set(config)-allowed))
    if config.get('partner_identity_mode', 'raw') not in {'raw', 'annotations_only', 'verified_canonical_identity'}:
        raise ValueError('Unknown partner_identity_mode')
    if set(config.get('fit',{}))-{'withdrawal_policy'}:
        raise ValueError('Analysis-only verification cannot alter the Bayesian fit')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new verification output directory')
    votes=pd.read_csv(source/'votes.csv')
    statuses=json.loads((source/'sampling_status.json').read_text())
    if len(statuses)!=34 or {int(s['season']) for s in statuses}!=set(range(1,35)) or not all(s['sampling_diagnostics_pass'] for s in statuses):
        raise ValueError('Verification requires all 34 diagnosed season results')
    if set(votes.season.unique()) != set(range(1,35)) or votes.duplicated(['season','week','contestant_id']).any():
        raise ValueError('Incomplete or duplicate frozen vote rows')
    sums=votes.groupby(['season','week']).fan_mean.sum()
    if not np.allclose(sums,1,atol=1e-10,rtol=0) or not votes.fan_mean.between(0,1).all():
        raise ValueError('Invalid frozen posterior means')
    out.mkdir(parents=True,exist_ok=True)
    for name in ['votes.csv','sampling_status.json','data_summary_all_seasons.json','preprocessing_audit.csv','panel_all_seasons.csv']:
        shutil.copy2(source/name,out/name)
    config_hash=hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    manifest={'status':'running','operation':'analysis_only_with_frozen_latent_targets','code_sha256':code_digest(),
              'config_sha256':config_hash,'source_votes_sha256':hashlib.sha256((source/'votes.csv').read_bytes()).hexdigest(),
              'source_manifest':json.loads((source/'manifest.json').read_text()),
              'started_utc':datetime.now(timezone.utc).isoformat()}
    dump_json(out/'manifest.json',manifest); dump_json(out/'run_config.json',config)
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(out/'execution_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ['dwts','scripts','configs','tests']:
            for path in sorted((root/folder).glob('*')):
                if path.is_file() and path.suffix in {'.py','.json'}:
                    archive.write(path,path.relative_to(root))
    analyze(votes,out,config)
    write_comparison(votes,out,statuses)
    make_report_figures(out,statuses)
    status=audit_outputs(source,out)
    manifest.update(status='completed',completed_utc=status['completed_utc'],report_numerically_verified=status['report_numerically_verified'])
    dump_json(out/'manifest.json',manifest)
    print(json.dumps(status,indent=2),flush=True)
    return 0 if status['report_numerically_verified'] else 2
