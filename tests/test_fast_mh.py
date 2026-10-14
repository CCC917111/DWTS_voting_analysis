import numpy as np
import pandas as pd
import pytest
from dwts.fast_mh import pack, log_target, log_prior_component, week_energy, sample
from dwts.model import FitConfig, build_model


def frame(season=1):
    rows=[]
    for week in [1,2]:
        for i,j in enumerate([8.,8.,5.,4.]):
            rows.append(dict(season=season,week=week,contestant_id=str(i),
                             judge_mean=j,judge_z=(j-6.25)/1.7853571071357126,
                             judge_share=j/25,eliminated=i==3 or (i==2 and week==2),
                             withdrawal=False,rule='percentage' if season==3 else 'rank'))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('season,likelihood',[(1,'report'),(3,'report'),(28,'bottom_two')])
def test_full_log_density_matches_pymc_with_jacobian(season,likelihood):
    cfg=FitConfig(sampler='fast_mh',likelihood=likelihood,penalty_lambda=.7)
    d,names,_,_,args=pack(frame(season),cfg)
    model,_,_=build_model(d,cfg)
    fn=model.compile_logp()
    rng=np.random.default_rng(427)
    n=len(names)
    offsets=[]
    for _ in range(12):
        x=rng.normal(size=n+2+len(d))
        point={'alpha':x[:n],'beta':x[n],'sigma_log__':x[n+1],'epsilon':x[n+2:]}
        offsets.append(float(fn(point))-log_target(x,args))
    # The only difference is the fixed normalizing constants of the priors.
    np.testing.assert_allclose(offsets,offsets[0],rtol=0,atol=1e-10)


def test_sampler_recovers_known_unconstrained_priors():
    cfg=FitConfig(sampler='fast_mh',penalty_lambda=0,draws=3000,tune=1500,
                  sweeps_per_draw=4,chains=2,seed=194)
    idata,_=sample(frame(),cfg)
    post=idata.posterior
    assert abs(float(post.beta.mean())) < .08
    assert abs(float(post.beta.std())-1) < .08
    assert np.max(np.abs(post.alpha.mean(('chain','draw')).values)) < .15
    assert np.max(np.abs(post.alpha.std(('chain','draw')).values-2)) < .15
    # Avoid a nonexistent Half-Cauchy mean: its CDF transform must be Uniform(0,1).
    u=2/np.pi*np.arctan(post.sigma.values)
    assert abs(u.mean()-.5) < .04
    np.testing.assert_allclose(np.quantile(u,[.1,.5,.9]),[.1,.5,.9],atol=.04)
    np.testing.assert_allclose(post.fan_share.values.reshape(2,3000,2,4).sum(axis=-1),1,atol=1e-12)


@pytest.mark.parametrize('season',[1,3])
def test_cached_week_deltas_equal_full_pymc_density_changes(season):
    cfg=FitConfig(sampler='fast_mh')
    data=frame(season)
    data=data[~((data.week==2)&(data.contestant_id=='1'))].copy()
    data['judge_share']=data.judge_mean/data.groupby('week').judge_mean.transform('sum')
    d,names,_,affected,args=pack(data,cfg)
    model,_,_=build_model(d,cfg)
    fn=model.compile_logp()
    n=len(names)
    def exact(x):
        return float(fn({'alpha':x[:n],'beta':x[n],'sigma_log__':x[n+1],'epsilon':x[n+2:]}))
    rng=np.random.default_rng(558)
    for repeat in range(3):
        x=rng.normal(size=n+2+len(d))
        for k in range(len(x)):
            proposal=x.copy();proposal[k]+=rng.normal()*1.5
            local=log_prior_component(proposal[k],k,n)-log_prior_component(x[k],k,n)
            for w in np.flatnonzero(affected[k]):
                local+=week_energy(x,w,*args)-week_energy(proposal,w,*args)
            assert local==pytest.approx(exact(proposal)-exact(x),abs=1e-9)


def test_extreme_logits_preserve_real_rank_order_instead_of_zero_ties():
    from scipy.special import softmax
    from scipy.stats import rankdata
    cfg=FitConfig(sampler='fast_mh')
    d,names,_,_,args=pack(frame(),cfg)
    n=len(names)
    x=np.zeros(n+2+len(d))
    x[n+2:]=np.tile([0.,-1000.,-1500.,-2000.],2)
    assert (softmax(x[n+2:n+6])==0).sum()==3
    # True fan ranks are 1,2,3,4; the eliminated contestant has worst combined rank.
    assert week_energy(x,0,*args)==0
    model,_,_=build_model(d,cfg)
    fn=model.compile_logp(vars=model.potentials)
    assert float(fn({'alpha':x[:n],'beta':x[n],'sigma_log__':x[n+1],'epsilon':x[n+2:]}))==0
    # Give the exit fan rank 2: combined ranks are 2.5,4.5,7,6, so violation=1.
    x[n+2+1],x[n+2+2],x[n+2+3]=-1500.,-2000.,-1000.
    assert week_energy(x,0,*args)==100
    assert float(fn({'alpha':x[:n],'beta':x[n],'sigma_log__':x[n+1],'epsilon':x[n+2:]}))==-100
