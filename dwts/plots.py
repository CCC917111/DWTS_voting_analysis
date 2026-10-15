from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_plots(votes, balance, grid, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].hist(votes.judge_mean, bins=25, color="#366e83", alpha=0.9)
    axes[0].set(xlabel="Weekly mean judge score", ylabel="Contestant-weeks")
    axes[1].hist(votes.judge_z, bins=25, color="#cd7955", alpha=0.9)
    axes[1].set(xlabel="Within-week z score", ylabel="Contestant-weeks")
    fig.suptitle("Observed judges' scores")
    fig.savefig(out / "judge_distribution.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    balance.set_index("rule")[["fan_spearman", "judge_spearman"]].plot.bar(ax=ax, color=["#cd7955", "#366e83"])
    ax.set(ylabel="Mean within-week Spearman correlation", xlabel="", ylim=(-1, 1))
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("Inferred fan support: rule alignment")
    fig.savefig(out / "influence_balance.png", dpi=160)
    plt.close(fig)
    if not grid.empty:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for name, g in grid[grid.rule == "rank"].groupby("mechanism"):
            ax.plot(g.judge_weight, 100 * g.controversy_rate, marker="o", label=name)
        ax.set(xlabel="Judge weight", ylabel="Controversial rounds (%)",
               title="Rank mechanisms: provisional posterior-mean simulation")
        ax.legend(fontsize=8)
        fig.savefig(out / "controversy_grid.png", dpi=160)
        plt.close(fig)


def make_study_figures(outdir, statuses):
    """Plot fitted estimates and diagnostics from the current study run."""
    out = Path(outdir)
    figs = out / 'figures'
    figs.mkdir(exist_ok=True)
    valid = all(s['sampling_diagnostics_pass'] for s in statuses)
    qualifier = 'posterior estimates' if valid else 'provisional estimates'
    v = pd.read_csv(out/'votes.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    axes[0].scatter(v.fan_mean, v.fan_upper-v.fan_lower, c=v.n_active,
                    cmap='viridis', s=9, alpha=.4)
    axes[0].set(xlabel='Posterior mean vote share', ylabel='95% posterior interval width')
    by_n = v.groupby('n_active').relative_uncertainty.agg(['median','mean'])
    by_n.plot(ax=axes[1], marker='o', color=['#366e83','#cd7955'])
    axes[1].set(xlabel='Active contestants', ylabel='Relative posterior uncertainty')
    fig.suptitle(f'Fan-vote uncertainty — {qualifier}')
    fig.savefig(figs/'fan_vote_uncertainty.png',dpi=160); plt.close(fig)

    c = pd.read_csv(out/'correlations_weekly.csv')
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    for row, party in enumerate(['fan','judge']):
        col = party+'_spearman'
        paired = c.pivot(index=['season','week'],columns='rule',values=col).dropna()
        axes[row,0].hist(paired.percentage-paired['rank'],bins=25,color='#366e83')
        axes[row,0].axvline(0,color='#999999',lw=1)
        axes[row,0].set(xlabel='Percentage minus rank correlation',ylabel=f'{party.title()}: weeks')
        for rule,color in [('rank','#366e83'),('percentage','#cd7955')]:
            values=c.loc[c.rule.eq(rule),col].dropna()
            axes[row,1].hist(values,bins=np.linspace(-1,1,21),alpha=.55,color=color,
                             label=f'{rule}: mean={values.mean():.3f}')
        axes[row,1].set(xlabel='Within-week Spearman correlation',ylabel='Weeks')
        axes[row,1].legend(fontsize=8)
        axes[row,2].scatter(paired['rank'],paired.percentage,s=10,alpha=.45,color='#366e83')
        axes[row,2].plot([-1,1],[-1,1],ls='--',c='#888888')
        axes[row,2].set(xlabel='Rank correlation',ylabel='Percentage correlation',xlim=(-1,1),ylim=(-1,1))
    fig.suptitle(f'Rule alignment — {qualifier}')
    fig.savefig(figs/'rule_correlation_comparison.png',dpi=160); plt.close(fig)

    if (out/'ols_coefficients.csv').exists():
        d = pd.read_csv(out/'ols_coefficients.csv')
        fig, axes = plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
        for ax,target in zip(axes,['judge_mean','fan_mean','placement']):
            g = d[(d.target==target)&(d.feature!='const')].copy()
            y = np.arange(len(g))
            ax.errorbar(g.coefficient,y,xerr=np.vstack([g.coefficient-g.lower_95,g.upper_95-g.coefficient]),
                        fmt='o',color='#366e83',capsize=3)
            ax.set_yticks(y,g.feature.str.replace('industry_','',regex=False))
            ax.axvline(0,c='#999999',lw=1)
            ax.set(title=target,xlabel='Standardized coefficient, 95% CI')
        fig.suptitle(f'OLS associations — {qualifier}')
        fig.savefig(figs/'ols_coefficients.png',dpi=160);plt.close(fig)
    if (out/'feature_importance.csv').exists():
        imp = pd.read_csv(out/'feature_importance.csv')
        fig, axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
        for ax,task in zip(axes,['A1_judge','A2_fan']):
            g=imp[(imp.task==task)&(imp.method=='SHAP')].copy()
            g['weighted']=g.importance*g.test_n
            agg=g.groupby('feature').agg(total=('weighted','sum'),n=('test_n','sum'))
            vals=(agg.total/agg.n).sort_values()
            ax.barh(vals.index.str.replace('numeric__','',regex=False).str.replace('industry__industry_','',regex=False),vals,color='#366e83')
            ax.set(title=task,xlabel='OOF mean absolute SHAP value')
        fig.suptitle(f'Feature attribution — {qualifier}')
        fig.savefig(figs/'shap_importance.png',dpi=160);plt.close(fig)

    s=pd.DataFrame(statuses).sort_values('season')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    axes[0].plot(s.season,s.max_rhat,'o-',color='#366e83')
    axes[0].axhline(1.01,c='#cd7955',ls='--',label='1.01 gate')
    axes[0].set(xlabel='Season',ylabel='Maximum rank-normalized R-hat');axes[0].legend()
    axes[1].plot(s.season,s.min_ess_bulk,'o-',label='Bulk',color='#366e83')
    axes[1].plot(s.season,s.min_ess_tail,'o-',label='Tail',color='#cd7955')
    axes[1].axhline(400,c='#888888',ls='--',label='400 gate')
    axes[1].set(xlabel='Season',ylabel='Minimum effective sample size',yscale='log');axes[1].legend()
    fig.suptitle('Sampling diagnostics — all seasons retained')
    fig.savefig(figs/'sampling_diagnostics.png',dpi=160);plt.close(fig)


def make_report_figures(outdir, statuses):
    """Compatibility entry point for historical saved-output verification."""
    return make_study_figures(outdir, statuses)

