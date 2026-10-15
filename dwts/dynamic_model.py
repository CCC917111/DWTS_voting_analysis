"""Bootstrap filtering for normalized exits with time-varying support logits.

This is a separate model from retrospective soft-constraint inference. Transition
noise is part of the model, never an uncorrected post-resampling jitter. No
static full-history Metropolis target is applied to the current dynamic state.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax

from .forward_model import exit_log_probs


class DynamicPosterior:
    """Finite-particle bootstrap filter; predictions precede released outcomes.

    At a contestant's first appearance z~Normal(0, prior_sd). At subsequent
    appearances, z_t=z_previous + drift*(r_t-r_previous) + Normal(0,
    process_sd*sqrt(elapsed_weeks)), with r=n*judge_share on each observed roster.
    Only currently observed contestants advance. Missing-week evolution is
    integrated through its elapsed-time Gaussian transition on reappearance.
    ESS-triggered systematic resampling is exact for the particle approximation;
    it does not repair finite-particle or ancestral/path degeneracy.
    """

    def __init__(self, n_contestants, *, prior_sd=1., temperature=3.,
                 process_sd=.25, drift=1., judge_weight=.5, particles=2048,
                 seed=2627761, ess_fraction=.7):
        if (type(n_contestants) is not int or n_contestants < 2
                or type(particles) is not int or particles < 16
                or not np.isfinite(prior_sd) or prior_sd <= 0
                or not np.isfinite(process_sd) or process_sd < 0
                or not np.isfinite(drift)
                or not np.isfinite(ess_fraction) or not 0 < ess_fraction <= 1):
            raise ValueError("Invalid bootstrap filter settings")
        exit_log_probs(np.zeros(n_contestants), np.arange(n_contestants),
                       np.ones(n_contestants), temperature=temperature,
                       judge_weight=judge_weight)
        self.prior_sd, self.temperature = float(prior_sd), float(temperature)
        self.process_sd, self.drift = float(process_sd), float(drift)
        self.judge_weight, self.ess_fraction = float(judge_weight), float(ess_fraction)
        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.normal(0, prior_sd, (particles, n_contestants))
        self.weights = np.full(particles, 1/particles)
        self.last_relative = np.full(n_contestants, np.nan)
        self.last_week = np.full(n_contestants, np.nan)
        self.ancestors = np.arange(particles)
        self.observations = 0
        self.diagnostics = []

    def extend(self, n_new):
        """Allocate independent priors only when new entrants are observed."""
        if type(n_new) is not int or n_new < 0:
            raise ValueError("Number of new entrants must be a nonnegative integer")
        if n_new:
            self.particles = np.column_stack((self.particles, self.rng.normal(
                0, self.prior_sd, (len(self.particles), n_new))))
            self.last_relative = np.r_[self.last_relative, np.full(n_new, np.nan)]
            self.last_week = np.r_[self.last_week, np.full(n_new, np.nan)]

    def advance(self, indices, judges, week):
        """Consume current scores, never outcomes; call exactly once per roster."""
        # Shared validator covers integer unique indices and nonnegative scores.
        exit_log_probs(self.particles, indices, judges, temperature=self.temperature,
                       judge_weight=self.judge_weight)
        if type(week) not in (int, np.int64, np.int32) or week < 1:
            raise ValueError("Week must be a positive integer")
        idx, j = np.asarray(indices, dtype=int), np.asarray(judges, dtype=float)
        seen = np.isfinite(self.last_week[idx])
        if (self.last_week[idx][seen] >= week).any():
            raise ValueError("Contestant transitions must be strictly chronological")
        relative = len(idx)*j/j.sum()
        existing = idx[seen]
        if len(existing):
            delta = relative[seen]-self.last_relative[existing]
            self.particles[:, existing] += self.drift*delta
            if self.process_sd:
                scale = self.process_sd*np.sqrt(week-self.last_week[existing])
                self.particles[:, existing] += self.rng.normal(
                    0, scale, (len(self.particles), len(existing)))
        self.last_relative[idx], self.last_week[idx] = relative, week

    def predict(self, indices, judges):
        probability = self.weights @ np.exp(exit_log_probs(self.particles,
            indices, judges, temperature=self.temperature, judge_weight=self.judge_weight))
        return probability/probability.sum()

    def fan_shares(self, indices):
        """Current filtering particles; these are model shares, not observed votes."""
        return softmax(self.particles[:, np.asarray(indices, dtype=int)], axis=1)

    def observe(self, indices, judges, outcome):
        lp = exit_log_probs(self.particles, indices, judges,
                            temperature=self.temperature, judge_weight=self.judge_weight)
        if type(outcome) not in (int, np.int64, np.int32) or not 0 <= outcome < len(indices):
            raise ValueError("Observed exit must index the current roster")
        logw = np.log(self.weights, where=self.weights > 0,
                      out=np.full_like(self.weights, -np.inf))+lp[:, outcome]
        log_predictive = float(logsumexp(logw))
        self.weights = np.exp(logw-log_predictive)
        ess = float(1/np.square(self.weights).sum())
        resampled = ess < self.ess_fraction*len(self.weights)
        if resampled:
            positions = (self.rng.random()+np.arange(len(self.weights)))/len(self.weights)
            cumulative = np.cumsum(self.weights)
            cumulative[-1] = 1.
            selected = np.searchsorted(cumulative, positions, side="right")
            self.particles = self.particles[selected].copy()
            self.ancestors = self.ancestors[selected]
            self.weights.fill(1/len(self.weights))
        self.observations += 1
        record = dict(observation=self.observations, predictive_log_probability=log_predictive,
            ess_fraction=ess/len(self.weights), resampled=bool(resampled),
            initial_ancestor_fraction=len(np.unique(self.ancestors))/len(self.weights))
        self.diagnostics.append(record)
        return record
