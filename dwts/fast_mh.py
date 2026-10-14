"""Compiled componentwise random-walk MH for exactly the noncentered model target.

Only affected weeks are recomputed for a proposal. No gradients, changed priors,
relaxed rank penalties, or data-dependent choice of lambda are used.
"""
import numpy as np
from numba import njit


def pack(frame, cfg):
    from .rules import ranks
    d = frame.sort_values(["week", "contestant_id"]).reset_index(drop=True)
    names = sorted(d.contestant_id.unique())
    ids = np.searchsorted(np.array(names), d.contestant_id.to_numpy())
    groups = list(d.groupby("week", sort=True))
    bounds = np.array([g.index[0] for _, g in groups] + [len(d)], dtype=np.int64)
    judge_rank = np.concatenate([ranks(g.judge_mean) for _, g in groups])
    enabled, ranked, bottom_two = [], [], []
    for _, g in groups:
        e = g.eliminated.to_numpy(bool)
        enabled.append(e.any() and not e.all() and
                       (cfg.withdrawal_policy == "include_scored" or not g.withdrawal.any()))
        ranked.append(g.rule.iloc[0] == "rank")
        bottom_two.append(cfg.likelihood == "bottom_two" and int(d.season.iloc[0]) >= 28 and e.sum() == 1)
    n = len(names)
    affected = np.zeros((n + 2 + len(d), len(groups)), dtype=np.bool_)
    affected[n:n+2, :] = True
    for w, (_, g) in enumerate(groups):
        affected[ids[g.index], w] = True
        affected[n+2+g.index.to_numpy(), w] = True
    init = (d.groupby("contestant_id").judge_mean.mean().reindex(names).to_numpy()-7)/3
    args = (ids, bounds, d.judge_z.to_numpy(float), d.judge_share.to_numpy(float),
            judge_rank, d.eliminated.to_numpy(bool), np.array(enabled), np.array(ranked),
            np.array(bottom_two), n, cfg.judge_weight, 100*cfg.penalty_lambda)
    return d, names, init, affected, args


@njit(cache=True)
def log_prior_component(x, k, n):
    if k < n:
        return -x*x/8
    if k == n+1:
        # Half-Cauchy(sigma) times d sigma / d log(sigma), constants omitted.
        return -np.logaddexp(x, -x)
    return -x*x/2


@njit(cache=True)
def week_logits(state, w, ids, bounds, judge_z, n):
    start, stop = bounds[w], bounds[w+1]
    z = np.empty(stop-start)
    sigma = np.exp(state[n+1])
    for j in range(start, stop):
        z[j-start] = state[ids[j]] + state[n]*judge_z[j] + sigma*state[n+2+j]
    return z


@njit(cache=True)
def week_values(state, w, ids, bounds, judge_z, n):
    z = week_logits(state, w, ids, bounds, judge_z, n)
    p = np.exp(z-z.max())
    return p/p.sum()


@njit(cache=True)
def week_energy(state, w, ids, bounds, judge_z, judge_share, judge_rank,
                eliminated, enabled, ranked, bottom_two, n, weight, strength):
    if not enabled[w] or strength == 0:
        return 0.0
    start, stop = bounds[w], bounds[w+1]
    # Softmax is strictly monotone mathematically; rank logits to avoid zero ties
    # when very small shares underflow in floating point.
    p = (week_logits(state, w, ids, bounds, judge_z, n) if ranked[w]
         else week_values(state, w, ids, bounds, judge_z, n))
    if not np.isfinite(p).all():
        return np.inf
    merit = np.empty(stop-start)
    for i in range(len(p)):
        if ranked[w]:
            greater, equal = 0, 0
            for j in range(len(p)):
                greater += p[j] > p[i]
                equal += p[j] == p[i]
            fan_rank = 1 + greater + .5*(equal-1)
            merit[i] = -2*(weight*judge_rank[start+i] + (1-weight)*fan_rank)
        else:
            merit[i] = weight*judge_share[start+i] + (1-weight)*p[i]
    exit_max, survivor_min = -np.inf, np.inf
    for i in range(len(p)):
        if eliminated[start+i]:
            exit_max = max(exit_max, merit[i])
        else:
            survivor_min = min(survivor_min, merit[i])
    boundary = np.sort(merit)[1] if bottom_two[w] else survivor_min
    return strength*max(0.0, exit_max-boundary)


@njit(cache=True)
def log_target(state, args):
    n = args[9]
    answer = 0.0
    for k in range(len(state)):
        answer += log_prior_component(state[k], k, n)
    for w in range(len(args[1])-1):
        answer -= week_energy(state, w, *args)
    return answer


@njit(cache=True)
def _chain(initial, scales, affected, args, draws, tune, stride, adapt, seed):
    np.random.seed(seed)
    state = initial.copy()
    n, weeks = args[9], len(args[1])-1
    energies = np.array([week_energy(state, w, *args) for w in range(weeks)])
    candidates = np.empty(weeks)
    states = np.empty((draws, len(state)))
    shares = np.empty((draws, len(args[0])))
    accepted = np.empty(draws)
    lp = np.empty(draws)
    counters = np.zeros(len(state))
    interval_accepts = 0
    for sweep in range(tune+draws*stride):
        # Random scan each complete sweep; every parameter gets one proposal.
        for k in np.random.permutation(len(state)):
            old = state[k]
            proposal = old + scales[k]*np.random.normal()
            state[k] = proposal
            delta = log_prior_component(proposal, k, n)-log_prior_component(old, k, n)
            for w in range(weeks):
                if affected[k, w]:
                    candidates[w] = week_energy(state, w, *args)
                    delta += energies[w]-candidates[w]
            if np.isfinite(delta) and np.log(np.random.random()) < delta:
                for w in range(weeks):
                    if affected[k, w]:
                        energies[w] = candidates[w]
                counters[k] += 1
                if sweep >= tune:
                    interval_accepts += 1
            else:
                state[k] = old
        if adapt and sweep < tune and (sweep+1) % 100 == 0:
            for k in range(len(state)):
                rate = counters[k]/100
                factor = (0.1 if rate < .001 else .5 if rate < .05 else .9 if rate < .2
                          else 10. if rate > .95 else 2. if rate > .75 else 1.1 if rate > .5 else 1.)
                scales[k] = min(100., max(1e-5, scales[k]*factor))
            counters[:] = 0
        if sweep >= tune and (sweep-tune+1) % stride == 0:
            draw = (sweep-tune)//stride
            states[draw] = state
            for w in range(weeks):
                shares[draw, args[1][w]:args[1][w+1]] = week_values(state, w, args[0], args[1], args[2], n)
            accepted[draw] = interval_accepts/(stride*len(state))
            interval_accepts = 0
            lp[draw] = log_target(state, args)
    return states, shares, accepted, lp, scales


def sample(frame, cfg):
    import arviz as az
    if cfg.parameterization != "noncentered":
        raise ValueError("fast_mh currently supports the noncentered model only")
    d, names, init_alpha, affected, args = pack(frame, cfg)
    n, obs = len(names), len(d)
    state_draws, fan_draws, accepted, lp, final_scales = [], [], [], [], []
    for chain in range(cfg.chains):
        seed = cfg.seed + 1000*int(d.season.iloc[0]) + chain
        rng = np.random.default_rng(seed)
        initial = np.r_[init_alpha+rng.normal(0, .1, n), rng.normal(0, .1),
                        rng.normal(0, .25), rng.normal(0, .25, obs)]
        scales = np.r_[np.full(n, cfg.alpha_step), cfg.beta_step, cfg.sigma_step,
                       np.full(obs, cfg.epsilon_step)]
        x, p, a, l, s = _chain(initial, scales, affected, args, cfg.draws, cfg.tune,
                                cfg.sweeps_per_draw, cfg.adapt_proposals, seed)
        state_draws.append(x); fan_draws.append(p); accepted.append(a); lp.append(l); final_scales.append(s)
        print(f"fast_mh chain {chain+1}/{cfg.chains}: {cfg.tune+cfg.draws*cfg.sweeps_per_draw} sweeps", flush=True)
    x = np.asarray(state_draws)
    idata = az.from_dict(posterior={"alpha": x[:, :, :n], "beta": x[:, :, n],
                                    "sigma": np.exp(x[:, :, n+1]), "epsilon": x[:, :, n+2:],
                                    "fan_share": np.asarray(fan_draws)},
                         sample_stats={"accepted": np.asarray(accepted), "lp": np.asarray(lp)},
                         coords={"contestant": names, "observation": np.arange(obs)},
                         dims={"alpha": ["contestant"], "epsilon": ["observation"], "fan_share": ["observation"]})
    idata.attrs.update(sampler="compiled_componentwise_random_walk_Metropolis",
                       sweeps_per_draw=cfg.sweeps_per_draw, tuning_sweeps=cfg.tune,
                       final_proposal_scales=str(np.asarray(final_scales).tolist()))
    return idata, d
