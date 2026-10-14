"""PyMC reconstruction of report 5.1; independent fits by season are explicit."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


@dataclass
class FitConfig:
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    cores: int = 1
    seed: int = 2627761
    penalty_lambda: float = 1.0  # not disclosed by report
    judge_weight: float = 0.5
    likelihood: str = "report"  # report | bottom_two (sensitivity variant for S28+)
    alpha_step: float = 0.15
    beta_step: float = 0.08
    sigma_step: float = 0.08  # log(sigma) proposal; PyMC positive transform
    epsilon_step: float = 0.30  # not disclosed by report
    parameterization: str = "noncentered"  # same posterior as centered latent Z
    adapt_proposals: bool = True
    sampler: str = "pymc"  # pymc | fast_mh, identical noncentered target
    sweeps_per_draw: int = 1  # fast_mh only; storage spacing, not convergence evidence
    withdrawal_policy: str = "skip_week"  # skip_week | include_scored (report reconstruction variant)

    def validate(self):
        if self.draws < 4 or self.tune < 0 or self.chains < 2 or self.cores < 1:
            raise ValueError("Use >=4 draws, >=0 tuning, >=2 chains and >=1 core")
        if self.penalty_lambda < 0 or not 0 <= self.judge_weight <= 1:
            raise ValueError("Invalid lambda/weight")
        if self.likelihood not in {"report", "bottom_two"}:
            raise ValueError("likelihood must be report or bottom_two")
        if self.parameterization not in {"centered", "noncentered"}:
            raise ValueError("Unknown parameterization")
        if self.sampler not in {"pymc", "fast_mh"} or self.sweeps_per_draw < 1 or not isinstance(self.sweeps_per_draw, int):
            raise ValueError("Invalid sampler or sweeps_per_draw")
        if self.sampler == "pymc" and self.sweeps_per_draw != 1:
            raise ValueError("sweeps_per_draw is only available with fast_mh")
        if self.sampler == "fast_mh" and self.parameterization != "noncentered":
            raise ValueError("fast_mh requires noncentered parameterization")
        if self.withdrawal_policy not in {"skip_week", "include_scored"}:
            raise ValueError("Unknown withdrawal_policy")
        for v in [self.alpha_step, self.beta_step, self.sigma_step, self.epsilon_step]:
            if v <= 0:
                raise ValueError("Proposal standard deviations must be positive")


def tensor_ranks(x):
    import pytensor.tensor as pt
    # Pairwise average ranks, including ties. No differentiability assumption.
    return 1 + pt.sum(pt.gt(x[None, :], x[:, None]), axis=1) + 0.5 * (
        pt.sum(pt.eq(x[None, :], x[:, None]), axis=1) - 1)


def build_model(frame: pd.DataFrame, cfg: FitConfig):
    import pymc as pm
    import pytensor.tensor as pt
    from .rules import ranks
    cfg.validate()
    d = frame.sort_values(["week", "contestant_id"]).reset_index(drop=True)
    if d.season.nunique() != 1:
        raise ValueError("build_model expects one season")
    names = sorted(d.contestant_id.unique())
    idx = pd.Categorical(d.contestant_id, categories=names).codes
    init_alpha = (d.groupby("contestant_id").judge_mean.mean().reindex(names).to_numpy() - 7) / 3
    coords = {"contestant": names, "observation": np.arange(len(d))}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0, 2, dims="contestant", initval=init_alpha)
        beta = pm.Normal("beta", 0, 1, initval=0.0)
        sigma = pm.HalfCauchy("sigma", beta=1, initval=1.0)
        mu = alpha[idx] + beta * d.judge_z.to_numpy()
        # A common gamma_t cancels exactly from each weekly softmax. Integrate it out.
        if cfg.parameterization == "noncentered":
            epsilon = pm.Normal("epsilon", 0, 1, dims="observation")
            z = mu + sigma * epsilon
            latent = epsilon
        else:
            latent = pm.Normal("z", mu, sigma, dims="observation", initval=init_alpha[idx])
            z = latent
        probabilities, penalties = [], []
        for week, g in d.groupby("week", sort=True):
            ix = g.index.to_numpy()
            p = pm.math.softmax(z[ix])
            probabilities.append(p)
            e = g.eliminated.to_numpy(bool)
            # Withdrawals are exogenous; their weeks do not identify voting eliminations.
            if not e.any() or e.all() or (g.withdrawal.any() and cfg.withdrawal_policy == "skip_week"):
                continue
            if g.rule.iloc[0] == "rank":
                m = -2 * (cfg.judge_weight * ranks(g.judge_mean) +
                          (1 - cfg.judge_weight) * tensor_ranks(z[ix]))
            else:
                m = cfg.judge_weight * g.judge_share.to_numpy() + (1 - cfg.judge_weight) * p
            if cfg.likelihood == "bottom_two" and int(d.season.iloc[0]) >= 28 and e.sum() == 1:
                boundary = pt.sort(m)[1]
            else:
                boundary = pt.min(m[np.flatnonzero(~e)])
            penalties.append(pt.maximum(0, pt.max(m[np.flatnonzero(e)]) - boundary))
        pm.Deterministic("fan_share", pt.concatenate(probabilities), dims="observation")
        total = pt.sum(pt.stack(penalties)) if penalties else pt.as_tensor_variable(0.0)
        pm.Potential("elimination_constraint", -100 * cfg.penalty_lambda * total)
    return model, d, (alpha, beta, sigma, latent)


def fit_season(frame, cfg, outdir):
    import arviz as az
    import pymc as pm
    from .evaluation import posterior_eliminations
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.sampler == "fast_mh":
        cfg.validate()
        from .fast_mh import sample
        idata, d = sample(frame, cfg)
    else:
        idata, d = sample_pymc(frame, cfg)
    season = int(d.season.iloc[0])
    return save_result(idata, d, cfg, out)


def sample_pymc(frame, cfg):
    import pymc as pm
    model, d, variables = build_model(frame, cfg)
    season = int(d.season.iloc[0])
    with model:
        steps = [pm.Metropolis(vars=[v], S=np.full(n, scale), tune=cfg.adapt_proposals)
                 for v, n, scale in zip(variables,
                    [d.contestant_id.nunique(), 1, 1, len(d)],
                    [cfg.alpha_step, cfg.beta_step, cfg.sigma_step, cfg.epsilon_step])]
        # Distinct initial states prevent apparently identical short chains.
        rng = np.random.default_rng(cfg.seed + season)
        initial = [{variables[-1].name: rng.normal(0, 0.25, len(d))} for _ in range(cfg.chains)]
        idata = pm.sample(draws=cfg.draws, tune=cfg.tune, chains=cfg.chains,
                          cores=min(cfg.cores, cfg.chains), step=steps,
                          random_seed=cfg.seed + season, initvals=initial,
                          return_inferencedata=True, progressbar=False,
                          compute_convergence_checks=True)
    return idata, d


def save_result(idata, d, cfg, out):
    import arviz as az
    from .evaluation import posterior_eliminations
    season = int(d.season.iloc[0])
    idata.to_netcdf(out / "posterior.nc", engine="h5netcdf")
    diag = az.summary(idata, var_names=["alpha", "beta", "sigma", "fan_share"], kind="all", round_to="none")
    diag.to_csv(out / "diagnostics.csv", index_label="parameter")
    samples = idata.posterior.fan_share.values.reshape(-1, len(d))
    result = d.copy()
    result["fan_mean"] = samples.mean(axis=0)
    result["fan_lower"] = np.quantile(samples, 0.025, axis=0)
    result["fan_upper"] = np.quantile(samples, 0.975, axis=0)
    result["relative_uncertainty"] = (result.fan_upper - result.fan_lower) / result.fan_mean
    result.to_csv(out / "votes.csv", index=False)
    if "z" in idata.posterior:
        rank_scores = idata.posterior.z.values.reshape(-1, len(d))
    else:
        ids = np.searchsorted(sorted(d.contestant_id.unique()), d.contestant_id)
        a = idata.posterior.alpha.values.reshape(len(samples), -1)
        b = idata.posterior.beta.values.reshape(-1, 1)
        s = idata.posterior.sigma.values.reshape(-1, 1)
        e = idata.posterior.epsilon.values.reshape(-1, len(d))
        rank_scores = a[:, ids] + b*d.judge_z.to_numpy() + s*e
    posterior_eliminations(d, samples, cfg, rank_scores=rank_scores).to_csv(out / "posterior_eliminations.csv", index=False)
    finite = np.isfinite(diag.r_hat) & np.isfinite(diag.ess_bulk) & np.isfinite(diag.ess_tail)
    passed = bool(finite.all() and (diag.r_hat <= 1.01).all() and
                  (diag.ess_bulk >= 400).all() and (diag.ess_tail >= 400).all())
    status = {"season": season, "sampling_diagnostics_pass": passed,
              "max_rhat": float(diag.r_hat.max()), "min_ess_bulk": float(diag.ess_bulk.min()),
              "min_ess_tail": float(diag.ess_tail.min()),
              "mean_acceptance": float(idata.sample_stats.accepted.mean()),
              "config": asdict(cfg), "scope": "independent_season",
              "note": "In-sample constraint reconstruction; not held-out elimination prediction."}
    (out / "fit_status.json").write_text(json.dumps(status, indent=2))
    if not passed:
        warnings.warn(f"Season {season}: diagnostics did not pass; outputs are provisional.")
    return result, status

