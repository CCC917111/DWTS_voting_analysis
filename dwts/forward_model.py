"""Normalized, forward-only categorical exit model and sequential inference.

This model is separate from the retrospective soft-constraint model. A season
has static independent Normal support logits. Each observed single exit has a
proper categorical likelihood over the current roster. The inference engine is
finite-particle sequential Monte Carlo (SMC), not an exact posterior sampler.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata


def exit_log_probs(alpha, indices, judges, *, rule="percentage", judge_weight=.5,
                   temperature=3.):
    """Return normalized exit log probabilities for one or many support draws.

    ``alpha`` is (..., full-season contestants), and ``indices`` selects the
    current active roster. Percentage merit is n times the weighted sum of
    judge and fan shares. Rank merit is the weighted best-oriented average
    rank divided by n-1. Higher merit means lower exit probability. Temperature
    therefore has a different interpretation in the two rule families.
    """
    a = np.asarray(alpha, dtype=float)
    idx = np.asarray(indices)
    j = np.asarray(judges, dtype=float)
    if (a.ndim not in (1, 2) or idx.ndim != 1 or j.ndim != 1 or len(idx) != len(j)
            or len(j) < 2 or not np.issubdtype(idx.dtype, np.integer)
            or len(np.unique(idx)) != len(idx) or (idx < 0).any()
            or (idx >= a.shape[-1]).any() or not np.isfinite(a).all()
            or not np.isfinite(j).all() or (j < 0).any() or j.sum() <= 0
            or not np.isfinite(temperature) or temperature <= 0
            or not np.isfinite(judge_weight) or not 0 <= judge_weight <= 1):
        raise ValueError("Invalid forward-model logits, roster, scores or parameters")
    z = a[..., idx]
    n = len(j)
    if rule == "percentage":
        merit = n * (judge_weight*j/j.sum() + (1-judge_weight)*softmax(z, axis=-1))
    elif rule == "rank":
        jr = rankdata(-j, method="average")
        fr = rankdata(-z, axis=-1, method="average")
        merit = (judge_weight*(n-jr) + (1-judge_weight)*(n-fr))/(n-1)
    else:
        raise ValueError("rule must be percentage or rank")
    logits = -float(temperature)*merit
    return logits - logsumexp(logits, axis=-1, keepdims=True)


@dataclass(frozen=True)
class Observation:
    indices: tuple[int, ...]
    judges: tuple[float, ...]
    outcome: int


class SequentialPosterior:
    """Bootstrap SMC with exact importance updates and invariant MH moves.

    When ESS falls below the declared threshold, systematic resampling gives
    equal weights. Symmetric Gaussian random-walk Metropolis steps then target
    the independent Normal prior times EVERY past normalized likelihood.
    Finite particles, resampling and finite MH steps introduce Monte Carlo
    approximation; independent repeats and particle sensitivity must assess it.
    """

    def __init__(self, n_contestants, *, prior_sd=1., temperature=3.,
                 rule="percentage", judge_weight=.5, particles=1024, seed=2627761,
                 mh_steps=4, proposal_scale=.35, ess_fraction=.7):
        if (type(n_contestants) is not int or n_contestants < 2
                or type(particles) is not int or particles < 16
                or type(mh_steps) is not int or mh_steps < 0
                or not np.isfinite(prior_sd) or prior_sd <= 0
                or not np.isfinite(proposal_scale) or proposal_scale <= 0
                or not np.isfinite(ess_fraction) or not 0 < ess_fraction <= 1):
            raise ValueError("Invalid SMC settings")
        # Validate shared likelihood parameters once, including the rule.
        exit_log_probs(np.zeros(n_contestants), np.arange(n_contestants),
                       np.ones(n_contestants), rule=rule,
                       judge_weight=judge_weight, temperature=temperature)
        self.prior_sd, self.temperature = float(prior_sd), float(temperature)
        self.rule, self.judge_weight = rule, float(judge_weight)
        self.mh_steps, self.proposal_scale = mh_steps, float(proposal_scale)
        self.ess_fraction = float(ess_fraction)
        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.normal(0, prior_sd, (particles, n_contestants))
        self.weights = np.full(particles, 1/particles)
        self.history: list[Observation] = []
        self.diagnostics: list[dict] = []

    def extend(self, n_new):
        """Introduce only newly observed entrants with independent prior support."""
        if type(n_new) is not int or n_new < 0:
            raise ValueError("Number of new entrants must be a nonnegative integer")
        if n_new:
            extra = self.rng.normal(0, self.prior_sd, (len(self.particles), n_new))
            self.particles = np.concatenate((self.particles, extra), axis=1)

    def _log_probs(self, alpha, indices, judges):
        return exit_log_probs(alpha, indices, judges, rule=self.rule,
                              judge_weight=self.judge_weight,
                              temperature=self.temperature)

    def log_target(self, alpha):
        """Unnormalized full-history posterior density (Normal constants cancel)."""
        a = np.asarray(alpha, dtype=float)
        target = -.5*np.square(a/self.prior_sd).sum(axis=-1)
        for obs in self.history:
            target += self._log_probs(a, obs.indices, obs.judges)[..., obs.outcome]
        return target

    def predict(self, indices, judges):
        probabilities = np.exp(self._log_probs(self.particles, indices, judges))
        p = self.weights @ probabilities
        return p/p.sum()

    def posterior_fan_shares(self, indices):
        """Return weighted draws, not an assertion of observed audience shares."""
        return softmax(self.particles[:, np.asarray(indices, dtype=int)], axis=1)

    def observe(self, indices, judges, outcome):
        """Assimilate an outcome only AFTER its predictive distribution is scored."""
        idx = np.asarray(indices)
        lp = self._log_probs(self.particles, idx, judges)
        if type(outcome) not in (int, np.int64, np.int32) or not 0 <= outcome < len(idx):
            raise ValueError("Observed exit must index the current roster")
        log_weights = np.log(self.weights, where=self.weights > 0,
                             out=np.full_like(self.weights, -np.inf)) + lp[:, outcome]
        normalizer = float(logsumexp(log_weights))
        self.weights = np.exp(log_weights-normalizer)
        self.history.append(Observation(tuple(map(int, idx)), tuple(map(float, judges)),
                                        int(outcome)))
        ess = float(1/np.square(self.weights).sum())
        resampled = ess < self.ess_fraction*len(self.particles)
        accepted = attempts = 0
        if resampled:
            positions = (self.rng.random()+np.arange(len(self.particles)))/len(self.particles)
            cumulative = np.cumsum(self.weights)
            cumulative[-1] = 1.
            self.particles = self.particles[np.searchsorted(cumulative, positions, side="right")].copy()
            self.weights.fill(1/len(self.weights))
            current = self.log_target(self.particles)
            for _ in range(self.mh_steps):
                proposal = self.particles + self.rng.normal(
                    0, self.proposal_scale*self.prior_sd, self.particles.shape)
                candidate = self.log_target(proposal)
                accept = np.log(self.rng.random(len(current))) < candidate-current
                self.particles[accept] = proposal[accept]
                current[accept] = candidate[accept]
                accepted += int(accept.sum())
                attempts += len(accept)
        record = dict(observation=len(self.history), predictive_log_probability=normalizer,
                      ess_before_resampling=ess, ess_fraction=ess/len(self.particles),
                      resampled=bool(resampled), mh_accepted=accepted, mh_attempts=attempts,
                      mh_acceptance=accepted/attempts if attempts else None)
        self.diagnostics.append(record)
        return record


def fit_sequential(n_contestants, observations, **settings):
    """Fit synthetic or observed normalized exits with the shared SMC engine."""
    posterior = SequentialPosterior(n_contestants, **settings)
    for observation in observations:
        if isinstance(observation, Observation):
            posterior.observe(observation.indices, observation.judges, observation.outcome)
        else:
            indices, judges, outcome = observation
            posterior.observe(indices, judges, outcome)
    return posterior
