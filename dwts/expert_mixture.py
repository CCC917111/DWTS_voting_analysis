"""Sequential mixtures of categorical exit forecasts.

The four experts retain their own training and feature-release clocks. This
module only mixes their normalized predictions: call ``predict`` before an
observed exit is available, then ``release`` to update the next week's weights.
Reset the learner at each season boundary.

With ``eta=1`` and ``gamma=0`` this is Bayesian model averaging under log loss.
Positive ``gamma`` applies fixed share toward the original prior after each
release (Herbster and Warmuth, 1998, *Tracking the Best Expert*). Tempering is
exponential weighting; these are standard methods, not new algorithms. For
``eta=1``, the cumulative loss relative to expert j over T events is at most
``-log(prior[j]) - (T-1)*log(1-gamma)``. With no sharing and ``eta<=1``, the
allowance is ``-log(prior[j])/eta``. The same claim does not hold for ``eta>1``.
These bounds concern log loss, not recovery of confidential audience votes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


EXPERTS = ("past_judges", "current_judges", "choice_two", "uniform")
PRIOR = np.array([0.5, 0.2, 0.2, 0.1])
PRIOR.setflags(write=False)


@dataclass(frozen=True)
class ForecastPacket:
    """One roster's expert probabilities, with no outcome or future fields.

    Rows of ``probabilities`` follow ``EXPERTS``; columns follow ``ids``.
    Packet arrays are copied when prediction begins, so caller mutations
    cannot change a pending forecast or its subsequent weight update.
    """

    phase: str
    fold: str
    season: int
    week: int
    ids: tuple[str, ...]
    probabilities: np.ndarray


def _positive_integer(value: object) -> bool:
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)) and value > 0)


def _finite_scalar(value: object) -> bool:
    return (np.isscalar(value) and not isinstance(value, (str, bytes, bool, np.bool_))
            and np.isrealobj(value) and bool(np.isfinite(value)))


class AdaptiveOnline:
    """Predict, freeze, and update an arithmetic mixture of four experts.

    ``eta`` controls how strongly a released exit changes the weights.
    ``gamma`` restores that fraction of the fixed prior after each update.
    Both default to the untempered, unshared Bayesian mixture. An explicit
    season reset and strictly increasing weeks prevent accidental reuse of
    future outcomes. Only the caller decides which eligible events to submit.
    """

    def __init__(self, eta: float = 1.0, gamma: float = 0.0,
                 prior: np.ndarray = PRIOR):
        try:
            prior = np.array(prior, dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("Prior must contain four probabilities") from error
        if (prior.shape != (len(EXPERTS),) or not np.isfinite(prior).all()
                or np.any(prior <= 0)
                or not np.isclose(prior.sum(), 1, rtol=0, atol=1e-14)):
            raise ValueError("Prior must be strictly positive and normalized")
        if (not _finite_scalar(eta) or eta <= 0
                or not _finite_scalar(gamma) or not 0 <= gamma < 1):
            raise ValueError("Invalid eta/gamma")
        self.eta, self.gamma = float(eta), float(gamma)
        self._log_prior = np.log(prior)
        self._log_weights = self._log_prior.copy()
        self._group: tuple[str, int] | None = None
        self._fold: str | None = None
        self._last_week: int | None = None
        self._pending: tuple[int, np.ndarray] | None = None

    @property
    def logweights(self) -> np.ndarray:
        """Return an isolated copy of the normalized log expert weights."""
        result = self._log_weights.copy()
        result.setflags(write=False)
        return result

    @property
    def log_weights(self) -> np.ndarray:
        """Readable alias for ``logweights``."""
        return self.logweights

    def reset(self, phase: str, season: int) -> None:
        """Start a season from the fixed prior after completing any release."""
        if self._pending is not None:
            raise ValueError("Release pending outcome before reset")
        if not isinstance(phase, str) or not phase.strip() or not _positive_integer(season):
            raise ValueError("Reset requires a nonempty phase and positive integer season")
        self._group = (phase, int(season))
        self._fold = None
        self._log_weights = self._log_prior.copy()
        self._last_week = None

    def predict(self, packet: ForecastPacket) -> tuple[np.ndarray, np.ndarray]:
        """Freeze a forecast and return ``(probabilities, weights_before)``.

        No label is accepted. A released outcome must follow this call before
        the next prediction. Invalid packets leave the learner unchanged.
        """
        if self._pending is not None:
            raise ValueError("Release previous outcome before prediction")
        if not isinstance(packet, ForecastPacket):
            raise ValueError("Prediction requires a ForecastPacket")
        if (not isinstance(packet.phase, str) or not packet.phase.strip()
                or not isinstance(packet.fold, str) or not packet.fold.strip()
                or not _positive_integer(packet.season) or not _positive_integer(packet.week)):
            raise ValueError("Invalid forecast phase, fold, season, or week")
        if self._group != (packet.phase, packet.season):
            raise ValueError("Explicit per-season reset required")
        if self._fold is not None and self._fold != packet.fold:
            raise ValueError("A season cannot cross evaluation folds")
        if self._last_week is not None and packet.week <= self._last_week:
            raise ValueError("Duplicate or out-of-order event")
        if (not isinstance(packet.ids, tuple) or len(packet.ids) < 3
                or any(not isinstance(identifier, str) or not identifier.strip()
                       for identifier in packet.ids)
                or len(set(packet.ids)) != len(packet.ids)):
            raise ValueError("Invalid forecast roster")
        try:
            probabilities = np.array(packet.probabilities, dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid expert probability packet") from error
        if (probabilities.shape != (len(EXPERTS), len(packet.ids))
                or not np.isfinite(probabilities).all() or np.any(probabilities <= 0)
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)):
            raise ValueError("Incomplete or invalid expert probability packet")
        log_probabilities = np.log(probabilities)
        prediction = np.exp(logsumexp(self._log_weights[:, None] + log_probabilities, axis=0))
        if (not np.isfinite(prediction).all() or np.any(prediction <= 0)
                or not np.isclose(prediction.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("Invalid mixture prediction")
        self._pending = (int(packet.week), log_probabilities)
        self._fold = packet.fold
        prediction.setflags(write=False)
        before = np.exp(self._log_weights)
        before.setflags(write=False)
        return prediction, before

    def release(self, outcome_index: int) -> np.ndarray:
        """Release the observed roster index and return next-event weights.

        The update uses log probabilities without clipping. Invalid outcome
        indices leave the frozen prediction pending so they can be corrected.
        """
        if self._pending is None:
            raise ValueError("No prediction awaiting outcome")
        week, log_probabilities = self._pending
        if (not isinstance(outcome_index, (int, np.integer))
                or isinstance(outcome_index, (bool, np.bool_))
                or not 0 <= outcome_index < log_probabilities.shape[1]):
            raise ValueError("Invalid released exit index")
        with np.errstate(over="ignore", invalid="ignore"):
            posterior = self._log_weights + self.eta * log_probabilities[:, outcome_index]
        if not np.isfinite(posterior).all():
            raise ValueError("Tempered update exceeds floating-point range")
        posterior -= logsumexp(posterior)
        if self.gamma > 0:
            next_weights = np.logaddexp(np.log1p(-self.gamma) + posterior,
                                       np.log(self.gamma) + self._log_prior)
            next_weights -= logsumexp(next_weights)
        else:
            next_weights = posterior
        self._log_weights = next_weights
        self._pending = None
        self._last_week = week
        after = np.exp(self._log_weights)
        after.setflags(write=False)
        return after


def select_candidate(losses, tolerance: float = 1e-9) -> int:
    """Return the first candidate within tolerance of the global minimum.

    Inputs are finite, lower-is-better losses computed on training-contained
    chronological validation events. The function does not choose data splits
    or permit evaluation outcomes to enter selection.
    """
    try:
        values = np.asarray(losses, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Candidate losses must be a finite nonempty vector") from error
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Candidate losses must be a finite nonempty vector")
    if not _finite_scalar(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    return int(np.flatnonzero(values <= values.min() + tolerance)[0])
