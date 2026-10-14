"""Supplemental report coverage from computed votes, without target fitting.

Different plausible definitions remain separate named protocols. Report p-values
are not compared to invented averages of weekly p-values. Nothing here refits
the posterior or selects parameters against a report number.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau
from .rules import ranks, merit, eliminate, controversy, order_worst
from .verification import compare_numbers


def compatible_denominators():
    pairs=[(31,10.3),(25,8.3),(11,3.7),(19,6.3)]
    return [n for n in range(31,10001)
            if all(abs(100*k/n-rate)<.05+1e-12 for k,rate in pairs)]


def literal_spearman(x,y):
    n=len(x)
    a,b=rankdata(x,method='average'),rankdata(y,method='average')
    return float(1-6*np.square(a-b).sum()/(n*(n*n-1))) if n>1 else np.nan


def kendall_a(x,y):
    x,y=np.asarray(x),np.asarray(y)
    n=len(x); i,j=np.triu_indices(n,1)
    return float(np.sign(x[i]-x[j]).dot(np.sign(y[i]-y[j]))/(n*(n-1)/2)) if n>1 else np.nan


def correlation_audit(votes,out):
    rows=[]
    for (season,week),g in votes.groupby(['season','week']):
        j,p=g.judge_mean.to_numpy(),g.fan_mean.to_numpy()
        rm,pm=merit(j,p,'rank'),merit(j,p,'percentage')
        pairs={'percentage_vs_rank':(pm,rm),'percentage_vs_fan':(pm,p),
               'rank_vs_fan':(rm,p),'percentage_vs_judge':(pm,j),'rank_vs_judge':(rm,j)}
        for name,(x,y) in pairs.items():
            constant=np.ptp(x)==0 or np.ptp(y)==0
            sp=(np.nan,np.nan) if constant else spearmanr(x,y)
            kt=(np.nan,np.nan) if constant else kendalltau(x,y,variant='b')
            rows.append(dict(season=season,week=week,n=len(g),comparison=name,
                is_final=bool(g.is_final_week.any()),has_exit=bool(g.eliminated.any()),
                spearman_tie_aware=sp[0],spearman_weekly_pvalue=sp[1],
                kendall_tau_b=kt[0],kendall_weekly_pvalue=kt[1],
                spearman_printed_shortcut=literal_spearman(x,y),kendall_printed_tau_a=kendall_a(x,y)))
    weekly=pd.DataFrame(rows);weekly.to_csv(out/'table6_weekly_correlations.csv',index=False)
    summary=[]
    for scope,d in [('all_weeks',weekly),('nonfinal_weeks',weekly[~weekly.is_final]),('elimination_weeks',weekly[weekly.has_exit])]:
        for comparison,g in d.groupby('comparison'):
            for method in ['spearman_tie_aware','kendall_tau_b','spearman_printed_shortcut','kendall_printed_tau_a']:
                for aggregation in ['equal_week_mean','equal_season_mean']:
                    values=g[method].dropna() if aggregation=='equal_week_mean' else g.groupby('season')[method].mean().dropna()
                    summary.append(dict(scope=scope,comparison=comparison,method=method,aggregation=aggregation,
                                        units=len(values),coefficient=values.mean(),reported_pvalue_status='not_identifiable_test_unit_and_aggregation_missing'))
    summary=pd.DataFrame(summary);summary.to_csv(out/'table6_correlation_protocols.csv',index=False)
    expected={'percentage_vs_rank':(.774,.6602),'percentage_vs_fan':(.945,.8794),'rank_vs_fan':(.713,.5668)}
    checks=[]
    for comparison,(sp,kt) in expected.items():
        for method,target,tol in [('spearman_tie_aware',sp,.0005),('kendall_tau_b',kt,.00005)]:
            row=summary[(summary.scope=='all_weeks')&(summary.comparison==comparison)&(summary.method==method)&(summary.aggregation=='equal_week_mean')].iloc[0]
            checks.append(dict(item='Table 6',report_page=12,comparison=comparison,method=method,report_value=target,
                               calculated=row.coefficient,numerical_match=abs(row.coefficient-target)<=tol,
                               status='conditional_protocol_comparison',basis='equal mean of valid weekly tie-aware correlations; report aggregation not specified'))
    pd.DataFrame(checks).to_csv(out/'table6_report_comparison.csv',index=False)
    fig4=[]
    targets={'fan':{'rank':.713,'percentage':.945,'difference':.233},'judge':{'rank':.718,'percentage':.281,'difference':-.437}}
    for party in ['fan','judge']:
        paired=weekly[weekly.comparison.isin(['rank_vs_'+party,'percentage_vs_'+party])].pivot(index=['season','week'],columns='comparison',values='spearman_tie_aware').dropna()
        vals={'rank':weekly.loc[weekly.comparison.eq('rank_vs_'+party),'spearman_tie_aware'].mean(),
              'percentage':weekly.loc[weekly.comparison.eq('percentage_vs_'+party),'spearman_tie_aware'].mean(),
              'difference':(paired['percentage_vs_'+party]-paired['rank_vs_'+party]).mean()}
        for name,target in targets[party].items():
            n_valid=(len(paired) if name=='difference' else int(weekly.loc[
                weekly.comparison.eq(('percentage' if name=='percentage' else 'rank')+'_vs_'+party),
                'spearman_tie_aware'].notna().sum()))
            fig4.append(dict(item='Figure 4',report_page=11,party=party,metric=name,report_value=target,
                             calculated=vals[name],weeks=n_valid,numerical_match=abs(vals[name]-target)<=.0005,
                             basis='exact legend reading; individual valid-week histogram means, paired valid-week differences'))
    pd.DataFrame(fig4).to_csv(out/'figure4_legend_comparison.csv',index=False)
    return weekly


def table7_audit(votes,out):
    cases=[(2,'Jerry Rice'),(4,'Billy Ray Cyrus'),(11,'Bristol Palin'),(27,'Bobby Bones')]
    targets={2:['finalist',7,'finalist',7],4:['survives_week8',7,8,7],
             11:['finalist',8,'finalist',8],27:[8,8,'finalist','finalist']}
    settings=[('rank','none'),('rank','judge_save'),('percentage','none'),('percentage','judge_save')]
    rows=[];detail=[]
    for season,name in cases:
        sv=votes[votes.season==season];finalweek=int(sv.week.max())
        for k,(rule,mechanism) in enumerate(settings):
            eliminated=[]
            for week,g in sv.groupby('week'):
                if name not in set(g.name):continue
                ix=eliminate(g.judge_mean.to_numpy(),g.fan_mean.to_numpy(),g.contestant_id.to_numpy(),rule,mechanism=mechanism)
                hit=str(g.name.iloc[ix[0]])==name
                if hit and week<finalweek:eliminated.append(int(week))
                detail.append(dict(season=season,week=week,name=name,rule=rule,mechanism=mechanism,
                                   counterfactual_eliminated=hit,is_season_final=week==finalweek))
            target=targets[season][k];earliest=min(eliminated) if eliminated else None
            if target=='survives_week8':matched=8 not in eliminated;calculated='survives_week8' if matched else 'eliminated_week8'
            elif target=='finalist':matched=earliest is None;calculated='finalist' if matched else f'eliminated_week{earliest}'
            else:matched=earliest==target;calculated='finalist' if earliest is None else f'eliminated_week{earliest}'
            rows.append(dict(item='Table 7',report_page=13,season=season,name=name,season_final_week=finalweek,
                             rule=rule,mechanism=mechanism,report_outcome=str(target),calculated_outcome=calculated,
                             earliest_pre_final_elimination=earliest,all_pre_final_elimination_weeks=';'.join(map(str,eliminated)),
                             match=matched,basis='week-local observed rosters; earliest pre-final exit; no hypothetical season trajectory'))
    pd.DataFrame(rows).to_csv(out/'table7_outcome_comparison.csv',index=False)
    pd.DataFrame(detail).to_csv(out/'table7_weekly_detail.csv',index=False)


def table5_audit(votes,out):
    rows=[]
    for (season,week),g in votes.groupby(['season','week']):
        actual=set(np.flatnonzero(g.eliminated.to_numpy()))
        if not actual:continue
        scenario='S1-2' if season<=2 else ('S3-27' if season<=27 else 'S28-34')
        j,p,ids=g.judge_mean.to_numpy(),g.fan_mean.to_numpy(),g.contestant_id.to_numpy()
        for mechanism in ['none','judge_save']:
            supported=mechanism=='none' or len(actual)==1
            predictions={rule:set(eliminate(j,p,ids,rule,mechanism=mechanism,count=len(actual))) if supported else None for rule in ['rank','percentage']}
            rows.append(dict(season=season,week=week,scenario=scenario,mechanism=mechanism,actual_exits=len(actual),
                             supported=supported,rank_correct=predictions['rank']==actual if supported else np.nan,
                             percentage_correct=predictions['percentage']==actual if supported else np.nan,
                             agreement=predictions['rank']==predictions['percentage'] if supported else np.nan))
    weekly=pd.DataFrame(rows);weekly.to_csv(out/'table5_weekly_rule_protocols.csv',index=False)
    targets={'S1-2':(11,9,8,7),'S3-27':(192,139,189,138),'S28-34':(56,44,40,28)}
    comparisons=[]
    for (scenario,mechanism),g in weekly.groupby(['scenario','mechanism']):
        total,*counts=targets[scenario];expected='judge_save' if scenario=='S28-34' else 'none'
        supported=int(g.supported.sum());definition=mechanism==expected
        for metric,target in zip(['rank_correct','percentage_correct','agreement'],counts):
            value=int(g[metric].fillna(False).astype(bool).sum())
            comparisons.append(dict(item='Table 5',scenario=scenario,mechanism=mechanism,metric=metric,
                reported_correct=target,reported_weeks=total,calculated_correct_supported_only=value,
                total_reconstructed_weeks=len(g),supported_weeks=supported,unsupported_weeks=len(g)-supported,
                accuracy_supported_only=value/supported if supported else np.nan,
                mechanism_matches_report_heading=definition,
                numeric_match_when_comparable=table5_count_match(definition,supported,len(g),total,value,target),
                status='unsupported_multi_exit_save_rule' if supported<len(g) else ('conditional_protocol_comparison' if definition else 'different_mechanism_from_report_heading'),
                basis='complete exit set; Judge Save higher-score assumption; unspecified multi-exit save not invented'))
    pd.DataFrame(comparisons).to_csv(out/'table5_report_comparison.csv',index=False)


def table5_count_match(heading_matches,supported,reconstructed_weeks,reported_weeks,computed_correct,reported_correct):
    return bool(heading_matches and reconstructed_weeks==reported_weeks and
                supported==reconstructed_weeks and computed_correct==reported_correct)


def table8_audit(source,out):
    if not (source/'influence_balance.csv').exists():return
    balance=pd.read_csv(source/'influence_balance.csv').set_index('rule')
    grid=pd.read_csv(source/'weight_grid_summary.csv')
    targets={'percentage':(.942,.293,.659), 'rank':(.719,.727,.084)}
    counts={('percentage','none'):(31,10.3),('percentage','judge_save'):(25,8.3),
            ('rank','none'):(11,3.7),('rank','judge_save'):(19,6.3)}
    rows=[]
    for (rule,mechanism),(count,rate) in counts.items():
        b=balance.loc[rule];g=grid[(grid.rule==rule)&(grid.mechanism==mechanism)&np.isclose(grid.judge_weight,.5)].iloc[0]
        fan,judge,gap=targets[rule]
        metrics=[('fan_spearman',fan,b.fan_spearman,.0005),('judge_spearman',judge,b.judge_spearman,.0005),
                 ('influence_gap_mean_of_weekly_absolute_gaps',gap,b.mean_weekly_influence_gap,.0005),
                 ('influence_gap_absolute_difference_of_means',gap,b.influence_gap,.0005),
                 ('controversy_count',count,g.controversies,0),('controversy_rate_percent',rate,100*g.controversy_rate,.05)]
        for metric,target,value,tolerance in metrics:
            correlation_metric=metric not in {'controversy_count','controversy_rate_percent'}
            rows.append(dict(item='Table 8',rule=rule,mechanism=mechanism,metric=metric,report_value=target,
                             calculated=value,tolerance=tolerance,numerical_match=abs(value-target)<=tolerance+1e-12,
                             computed_weeks=int(b.paired_spearman_weeks if correlation_metric else g.weeks),
                             computed_scope='all_weeks_common_valid_correlations' if correlation_metric else str(g.scope),
                             reported_denominator='not stated; rounded count/rate pairs admit 300 or 301',
                             basis='correlation on common valid weeks; both gap aggregations retained; one local simulated exit'))
    pd.DataFrame(rows).to_csv(out/'table8_direct_comparison.csv',index=False)


def mechanisms_audit(votes,out):
    """All alternatives predeclared; none is selected to match printed counts."""
    rows=[]
    for (season,week),g in votes.groupby(['season','week']):
        if len(g)<3:continue
        j,p,ids=g.judge_mean.to_numpy(),g.fan_mean.to_numpy(),g.contestant_id.to_numpy()
        for w in (k / 10 for k in range(11)):
            for mechanism in ['none','judge_save','fan_save','bottom3_judge_save','bottom3_save_other_two_exit']:
                if mechanism=='bottom3_save_other_two_exit':
                    pool=order_worst(j,p,ids,'rank',float(w))[:3]
                    saved=max(pool,key=lambda i:(j[i],p[i],str(ids[i])))
                    ix=np.array([i for i in pool if i!=saved])
                else:ix=eliminate(j,p,ids,'rank',float(w),mechanism)
                a,b=controversy(j,p,ix)
                rows.append(dict(season=season,week=week,judge_weight=round(float(w),1),mechanism=mechanism,
                                 n=len(g),is_final=bool(g.is_final_week.any()),has_exit=bool(g.eliminated.any()),
                                 type_a=a,type_b=b,controversies=a+b,controversial_week=int(a+b>0),exit_count=len(ix)))
    allrows=pd.DataFrame(rows);allrows.to_csv(out/'figure6_weekly_protocols.csv',index=False)
    totals=[]
    for scope,d in [('all_weeks_n_ge_3',allrows),('nonfinal_weeks',allrows[~allrows.is_final]),('elimination_weeks',allrows[allrows.has_exit])]:
        for (w,mechanism),g in d.groupby(['judge_weight','mechanism']):
            count=int(g.controversies.sum());n=len(g);row_n=int(g.n.sum())
            totals.append(dict(scope=scope,judge_weight=w,mechanism=mechanism,weeks=n,contestant_rows=row_n,
                               type_a=int(g.type_a.sum()),type_b=int(g.type_b.sum()),controversies=count,
                               event_sum_per_week=count/n,any_controversy_week_rate=g.controversial_week.mean(),
                               event_sum_per_scope_contestant_row=count/row_n,
                               event_sum_per_all_source_rows=count/len(votes),
                               interpretation='two exits per round is an alternate interpretation' if mechanism.endswith('other_two_exit') else 'single simulated exit per round'))
    total=pd.DataFrame(totals);total.to_csv(out/'figure6_grid_protocols.csv',index=False)
    expected={'none':(23,60),'judge_save':(23,60),'bottom3_judge_save':(5,117),
              'bottom3_save_other_two_exit':(5,117),'fan_save':(67,8)}
    checks=[]
    for mechanism,(a,b) in expected.items():
        g=total[(total.scope=='nonfinal_weeks')&(total.judge_weight==.5)&(total.mechanism==mechanism)].iloc[0]
        for metric,target in [('type_a',a),('type_b',b),('controversies',a+b)]:
            checks.append(dict(item='Figure 6',report_page=21,mechanism=mechanism,metric=metric,report_value=target,
                               calculated=g[metric],numerical_match=g[metric]==target,
                               basis='both Original definitions and both bottom-three exit interpretations retained separately'))
    pd.DataFrame(checks).to_csv(out/'figure6_bar_comparison.csv',index=False)
    # Explicitly distinguish mathematical event rate from alternate plot normalization.
    rates=[]
    for count,label in [(75,'Fan Save'),(83,'Original'),(122,'Bottom-3')]:
        for denominator in [259,266,300,301,335,2738,2777]:
            rates.append(dict(label=label,report_count=count,denominator=denominator,implied_percent=100*count/denominator,
                              denominator_status='candidate; not uniquely identified from plot'))
    pd.DataFrame(rates).to_csv(out/'figure6_report_rate_denominator_candidates.csv',index=False)
    minima=[];changes=[]
    primary=total[total.scope=='nonfinal_weeks']
    for mechanism,target in [('none',.3),('judge_save',.3),('bottom3_judge_save',.1),('bottom3_save_other_two_exit',.1),('fan_save',.7)]:
        g=primary[primary.mechanism==mechanism]
        weights=g[np.isclose(g.event_sum_per_week,g.event_sum_per_week.min(),atol=1e-12,rtol=0)].judge_weight.tolist()
        minima.append(dict(mechanism=mechanism,report_legend_minimizer=target,computed_all_minimizers=';'.join(map(str,weights)),
                           report_weight_among_computed_minimizers=target in weights,
                           basis='exact weight label in Figure 6 legend; equal grid .1; count of event types per week'))
    for baseline in ['none','judge_save']:
        orig=primary[primary.mechanism==baseline].set_index('judge_weight').controversies
        for mechanism in ['bottom3_judge_save','bottom3_save_other_two_exit','fan_save']:
            values=primary[primary.mechanism==mechanism].set_index('judge_weight').controversies
            for w in values.index:
                changes.append(dict(original_definition=baseline,mechanism=mechanism,judge_weight=w,
                    relative_change_percent=100*(values[w]-orig[w])/orig[w] if orig[w] else np.nan,
                    convention='negative means better; zero Original denominator undefined'))
    pd.DataFrame(minima).to_csv(out/'figure6_minimum_comparison.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'figure6_relative_change.csv',index=False)
    return total


def attribution_audit(source,out):
    """Reconcile saved importance aggregations and all six reported drivers."""
    p=source/'feature_importance.csv'
    if not p.exists():return
    imp=pd.read_csv(p);ratios=[];drivers=[]
    for (task,method),g in imp.groupby(['task','method']):
        grouped=g.groupby('feature').importance.mean()
        weighted=(g.assign(weighted=g.importance*g.test_n).groupby('feature').weighted.sum()/g.groupby('feature').test_n.sum())
        for aggregation,values in [('equal_fold_mean_importance',grouped),('test_size_weighted_mean_importance',weighted)]:
            pro=values.index.str.contains('pq_')
            ratio=values[pro].sum()/values[~pro].sum() if values[~pro].sum()>0 else np.nan
            top=values.idxmax()
            expected='pq_vote' if 'fan' in task else 'pq_judge'
            drivers.append(dict(task=task,method=method,aggregation=aggregation,reported_driver=expected,
                                calculated_driver=top,match=top.endswith(expected),
                                basis='Table 9 top-driver method/aggregation not stated'))
            if task in ['A1_judge','A2_fan']:
                target={('A1_judge','MDI'):13.57,('A2_fan','MDI'):36.76,
                        ('A1_judge','SHAP'):2.62,('A2_fan','SHAP'):4.68}[(task,method)]
                ratios.append(dict(task=task,method=method,aggregation=aggregation,report_value=target,
                                   calculated=ratio,numerical_match=abs(ratio-target)<=.005,
                                   basis='ratio of aggregate importances; separate from mean of fold ratios'))
    if (source/'pcir.csv').exists():
        saved=pd.read_csv(source/'pcir.csv')
        for (task,method),g in saved[saved.task.isin(['A1_judge','A2_fan'])].groupby(['task','method']):
            target={('A1_judge','MDI'):13.57,('A2_fan','MDI'):36.76,
                    ('A1_judge','SHAP'):2.62,('A2_fan','SHAP'):4.68}[(task,method)]
            ratios.append(dict(task=task,method=method,aggregation='equal_mean_fold_ratios',report_value=target,
                               calculated=g.pcir.mean(),numerical_match=abs(g.pcir.mean()-target)<=.005,
                               basis='legacy reconstruction convention; author aggregation unspecified'))
    pd.DataFrame(drivers).to_csv(out/'table9_top_driver_comparison.csv',index=False)
    pd.DataFrame(ratios).to_csv(out/'pcir_aggregation_comparison.csv',index=False)


def qualitative_audit(source,grid,out):
    rows=[]
    def add(item,claim,value,status,basis):rows.append(dict(item=item,claim=claim,calculated_evidence=value,status=status,basis=basis))
    d=grid[grid.scope=='nonfinal_weeks']
    fan=d[d.mechanism=='fan_save'];minimum=fan.event_sum_per_week.min()
    minima=fan[np.isclose(fan.event_sum_per_week,minimum,atol=1e-12,rtol=0)].judge_weight.tolist()
    add('Section 8.2','Fan Save optimum lies in judge weights .6 to .8',str(minima),
        'direction_supported' if minima and all(.6<=w<=.8 for w in minima) else 'not_supported_under_current_protocol',
        'all minimizers on predeclared .1 grid; week-based Type A+B metric')
    for baseline in ['none','judge_save']:
        orig=d[d.mechanism==baseline].set_index('judge_weight').controversies
        fv=fan.set_index('judge_weight').controversies
        improvements=((orig-fv)/orig.replace(0,np.nan))*100
        add('Section 8.1','Fan Save reduces controversy by up to approximately 45 percent',
            f'baseline={baseline}; maximum={improvements.max()}; .2-.6 maximum={improvements.loc[.2:.6].max()}',
            'conditional_protocol_comparison','both conflicting Original definitions retained; undefined zero denominators omitted')
    table=source/'rf_metrics.csv'
    if table.exists():
        rf=pd.read_csv(table).set_index('task')
        if {'A1_judge','A2_fan'}<=set(rf.index):
            a,b=rf.loc['A1_judge','r2'],rf.loc['A2_fan','r2']
            add('Section 6.2.3','Fan regression is more predictable than judge regression',f'judge_r2={a}; fan_r2={b}',
                'direction_supported' if b>a else 'not_supported_under_current_protocol','retrospective grouped latent-target validation; not causal attribution')
        for task in ['A4_bottom_two_judge','A4_bottom_two_fan']:
            if task in rf.index:
                value=rf.loc[task,'auc']
                add('Section 6.2.3',task+' AUC exceeds .75',value,
                    'direction_supported' if value>.75 else 'not_supported_under_current_protocol','existing computed AUC')
    add('Summary','61.4 percent of fan variability is explained by professional dancer quality alone',
        'full-model R2 includes age and industry as well as partner features','not_identifiable_from_report_metric',
        'a full-model R2 is not a unique variance attribution or causal effect for one predictor group')
    add('Summary','Recovered vote uncertainty is calibrated',
        'posterior intervals are available; true fan-vote labels are unavailable','not_empirically_testable_with_supplied_data',
        'Bayesian conditional credible intervals do not by themselves establish frequentist coverage')
    pd.DataFrame(rows).to_csv(out/'qualitative_claim_checks.csv',index=False)


def plot_mechanism_panels(grid,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=grid[grid.scope=='nonfinal_weeks']
    mechanisms=[('judge_save','Original: bottom-2 judge save'),('bottom3_judge_save','Bottom-3: single-exit extension'),('fan_save','Bottom-2 fan save')]
    colors=['#516b84','#c98043','#549774']
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for (mechanism,label),color in zip(mechanisms,colors):
        g=d[d.mechanism==mechanism].sort_values('judge_weight')
        for ax,value,title in [(axes[0,0],100*g.event_sum_per_week,'Controversial events per week (%)'),
                               (axes[0,1],g.type_a,'Type A controversy count'),
                               (axes[0,2],g.type_b,'Type B controversy count'),
                               (axes[1,0],g.controversies,'Total controversial events')]:
            ax.plot(g.judge_weight,value,marker='o',label=label,color=color)
            ax.set(xlabel='Judge weight',ylabel=title)
    axes[0,0].legend(fontsize=7)
    labels=[];av=[];bv=[]
    for mechanism,label in mechanisms:
        g=d[(d.mechanism==mechanism)&(d.judge_weight==.5)].iloc[0]
        labels.append(label.replace(': ','\n'));av.append(g.type_a);bv.append(g.type_b)
    axes[1,1].bar(np.arange(3),av,label='Type A',color='#6486a1')
    axes[1,1].bar(np.arange(3),bv,bottom=av,label='Type B',color='#cd925a')
    axes[1,1].set_xticks(np.arange(3),labels,fontsize=8)
    totals=np.asarray(av)+np.asarray(bv)
    axes[1,1].set_ylim(0,max(1,float(totals.max()))*1.2)
    for index,total in enumerate(totals):
        axes[1,1].text(index,total+.02*max(1,float(totals.max())),str(int(total)),ha='center',fontsize=9)
    axes[1,1].set(ylabel='Events at judge weight .5');axes[1,1].legend(fontsize=8)
    original=d[d.mechanism=='judge_save'].set_index('judge_weight').controversies
    for mechanism,label,color in [('bottom3_judge_save','Bottom-3 single-exit','#c98043'),('fan_save','Fan Save','#549774')]:
        values=d[d.mechanism==mechanism].set_index('judge_weight').controversies
        change=100*(values-original)/original.replace(0,np.nan)
        axes[1,2].plot(change.index,change,marker='o',label=label,color=color)
    axes[1,2].axhline(0,color='gray',linewidth=.7)
    axes[1,2].set(xlabel='Judge weight',ylabel='Relative CR change (%)\nNegative = better');axes[1,2].legend(fontsize=7)
    fig.suptitle('Figure 6 analysis from computed votes\nWeek-based Type A+B metric; alternate definitions remain separate in CSV')
    fig.savefig(out/'figure6_computed_panels.png',dpi=170);plt.close(fig)


def plot_correlation_panels(weekly,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for row,party in enumerate(['fan','judge']):
        d=weekly[weekly.comparison.isin(['rank_vs_'+party,'percentage_vs_'+party])]
        paired=d.pivot(index=['season','week'],columns='comparison',values='spearman_tie_aware').dropna()
        diff=paired['percentage_vs_'+party]-paired['rank_vs_'+party]
        axes[row,0].hist(diff,bins=25,color='#618596',alpha=.8)
        axes[row,0].axvline(diff.mean(),color='#b55f49',linestyle='--',label=f'mean={diff.mean():.3f}')
        axes[row,0].set(xlabel='Percentage minus rank correlation',ylabel=f'{party.title()}: weeks');axes[row,0].legend(fontsize=8)
        for rule,color in [('rank','#618596'),('percentage','#c99163')]:
            values=d.loc[d.comparison.eq(rule+'_vs_'+party),'spearman_tie_aware'].dropna()
            axes[row,1].hist(values,bins=np.linspace(-1,1,21),alpha=.55,color=color,label=f'{rule}: mean={values.mean():.3f}')
        axes[row,1].set(xlabel='Within-week Spearman correlation',ylabel='Weeks');axes[row,1].legend(fontsize=8)
        axes[row,2].scatter(paired['rank_vs_'+party],paired['percentage_vs_'+party],s=13,alpha=.45,color='#618596')
        axes[row,2].plot([-1,1],[-1,1],linestyle='--',color='gray')
        axes[row,2].set(xlabel='Rank correlation',ylabel='Percentage correlation',xlim=(-1,1),ylim=(-1,1))
    fig.suptitle('Figure 4 analysis from computed votes\nHistograms restored in the middle panels; no original curve values reused')
    fig.savefig(out/'figure4_computed_panels.png',dpi=170);plt.close(fig)


def run(source,out):
    source,out=Path(source),Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    votes=pd.read_csv(source/'votes.csv')
    if votes.duplicated(['season','week','contestant_id']).any():raise ValueError('Duplicate vote rows')
    if not np.allclose(votes.groupby(['season','week']).fan_mean.sum(),1,atol=1e-10,rtol=0):raise ValueError('Invalid vote normalization')
    weekly=correlation_audit(votes,out)
    table5_audit(votes,out)
    table8_audit(source,out)
    table7_audit(votes,out)
    grid=mechanisms_audit(votes,out)
    attribution_audit(source,out)
    qualitative_audit(source,grid,out)
    plot_mechanism_panels(grid,out)
    plot_correlation_panels(weekly,out)
    legacy=source/'numerical_verification.csv'
    if legacy.exists():compare_numbers(pd.read_csv(legacy)).to_csv(out/'legacy_comparison_corrected_references.csv',index=False)
    manifest={'operation':'supplemental_full_report_coverage','source':str(source),
              'source_votes_sha256':hashlib.sha256((source/'votes.csv').read_bytes()).hexdigest(),
              'audit_module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'rows':len(votes),'seasons':int(votes.season.nunique()),'table8_compatible_denominators':compatible_denominators(),
              'no_refitting':True,'all_report_values_reproduced':False,
              'caveats':['Table 6 aggregate p-values are not identifiable from missing test units',
                         'Literal no-tie coefficient formulas are not equivalent to tie-aware estimators',
                         'Figure 6 has inconsistent Original labels and lacks the bottom-three remaining-exit rule',
                         'Alternative protocols are not selected using reported target values']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='outputs/report_verified');p.add_argument('--out',default='outputs/full_report_audit')
    a=p.parse_args();print(json.dumps(run(a.source,a.out),indent=2))
