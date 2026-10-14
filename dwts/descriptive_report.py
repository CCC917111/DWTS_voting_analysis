"""Reproducible Figure 2/3 data audits; EDA scope is not Bayesian-scope proof."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew,kurtosis
from .data import prepare

FIGURE3_LOWEST=[4,8,10,8,10,8,10,9,12,8,11,13,9,12,7,10,10,13,11,10,7,7,11,11,8,3,7,7,13,8,9,12,9,9]
FIGURE3_SAVES=[2,5,6,5,6,3,5,5,7,3,7,8,4,7,4,3,7,10,7,8,3,3,6,7,4,1,7,4,8,3,6,9,6,6]


def run(raw,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    baseline,_=prepare(raw,outcome_policy='label_first',include_final_week_eliminations=True)
    candidate=baseline[~baseline.result_label.str.contains('withdrew',case=False)].copy()
    groups=candidate.groupby(['season','week']).judge_mean
    candidate['judge_z']=(candidate.judge_mean-groups.transform('mean'))/groups.transform(lambda x:x.std(ddof=0)).replace(0,1)
    candidate['judge_share']=candidate.judge_mean/groups.transform('sum')
    candidate['n_active']=groups.transform('size')
    candidate.to_csv(out/'eda_candidate_panel.csv',index=False)
    rows=[]
    for label,d in [('all_scored_rows',baseline),('exclude_withdrawn_contestants',candidate)]:
        x=d.judge_mean.to_numpy();q1,q3=np.quantile(x,[.25,.75])
        metrics=[('figure2_original_mean',x.mean(),7.91,.005),('figure2_standardized_mean',d.judge_z.mean(),0,1e-12),
                 ('section4_cv_sample',x.std(ddof=1)/x.mean(),.1846,.00005),
                 ('section4_cv_population',x.std(ddof=0)/x.mean(),.1846,.00005),
                 ('section4_iqr_coefficient',(q3-q1)/(q3+q1),.125,.00005),
                 ('section4_skew_unbiased',skew(x,bias=False),-.2648,.00005),
                 ('section4_kurtosis_unbiased',kurtosis(x,bias=False),-.1707,.00005),
                 ('section4_kurtosis_biased',kurtosis(x,bias=True),-.1707,.00005)]
        for name,value,target,tolerance in metrics:
            rows.append(dict(scope=label,n=len(d),metric=name,report_value=target,calculated=value,
                             tolerance=tolerance,numerical_match=abs(value-target)<=tolerance,
                             basis='Moment estimators remain separate declared choices; no best-match selection'))
    pd.DataFrame(rows).to_csv(out/'figure2_section4_moment_comparison.csv',index=False)
    seasonal=[];weekly=[]
    for (season,week),g in candidate.groupby(['season','week']):
        if not g.eliminated.any():continue
        low=g.judge_mean.eq(g.judge_mean.min());saved=low&~g.eliminated
        weekly.append(dict(season=season,week=week,n=len(g),lowest=int(low.sum()),saves=int(saved.sum()),is_final=bool(g.is_final_week.any())))
    w=pd.DataFrame(weekly);w.to_csv(out/'figure3_weekly_counts.csv',index=False)
    for season,g in w.groupby('season'):
        for name,target in [('lowest',FIGURE3_LOWEST[season-1]),('saves',FIGURE3_SAVES[season-1])]:
            value=int(g[name].sum())
            seasonal.append(dict(season=season,metric=name,report_bar=target,calculated=value,match=value==target,
                                 reference='Manual reading of integer-height bars; printed p5 / PDF p6'))
    s=pd.DataFrame(seasonal);s.to_csv(out/'figure3_68_bar_comparison.csv',index=False)
    assert len(s)==68 and s.match.all()
    summary={'figure3_bar_matches':68,'figure3_bars_checked':68,'lowest':int(w.lowest.sum()),'saves':int(w.saves.sum()),
             'audience_save_rate':float(w.saves.sum()/w.lowest.sum()),'elimination_weeks':len(w),
             'source_scope':'label-first, withdrawn people excluded, explicit final exits retained, exact numeric minima',
             'boundary':'Matching EDA does not establish Bayesian or RF sample scope; exact minima are not necessarily scientifically preferable'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,col,title in [(axes[0],'judge_mean','Original weekly judge means'),(axes[1],'judge_z','Re-standardized within week')]:
        ax.hist(candidate[col],bins=25,color='#608498',alpha=.85)
        ax.axvline(candidate[col].mean(),linestyle='--',color='#bd7155',label=f'mean={candidate[col].mean():.2f}')
        ax.set(title=title,xlabel=col,ylabel='Contestant-weeks');ax.legend()
    fig.suptitle('Figure 2: candidate descriptive sample excluding withdrawn contestants')
    fig.savefig(out/'figure2_descriptive_sample.png',dpi=170);plt.close(fig)
    z=w.groupby('season')[['lowest','saves']].sum()
    fig,ax=plt.subplots(figsize=(14,4.8),constrained_layout=True)
    ax.bar(z.index,z.lowest,color='#8eb6cf',label='Lowest scorers')
    ax.bar(z.index,z.saves,color='#509b66',label='Saved by audience')
    ax.set(xlabel='Season',ylabel='Contestants',xticks=list(range(1,35)))
    second=ax.twinx();second.plot(z.index,100*z.saves/z.lowest,color='#c46152',marker='o',markersize=3,label='Save rate')
    second.set(ylabel='Audience save rate (%)',ylim=(0,105));ax.legend(loc='upper left')
    fig.suptitle('Figure 3 reproduced counts: 314 lowest scorers, 185 saves; 68/68 bars match')
    fig.savefig(out/'figure3_descriptive_counts.png',dpi=170);plt.close(fig)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',default='data/raw/2026_MCM_Problem_C_Data.csv');p.add_argument('--out',default='outputs/descriptive_report')
    a=p.parse_args();print(json.dumps(run(a.raw,a.out),indent=2))
