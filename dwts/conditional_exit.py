"""Regularized categorical exit forecasts from released judge-score histories.

The coefficients describe predictive associations with an observed exit. They
do not estimate real audience votes or causal effects. Features use the current
roster and scores and each contestant's previously released scores only.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


def _boolean_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or pd.isna(array).any():
        raise ValueError(f"{name} must be a one-dimensional boolean label array")
    if not all(isinstance(value, (bool, np.bool_, int, np.integer))
               and value in (0, 1) for value in array):
        raise ValueError(f"{name} must contain boolean or integer 0/1 labels")
    return array.astype(bool)


@dataclass
class Event:
    """One released roster, its score features, and subsequently read labels."""

    season: int
    week: int
    ids: tuple[str, ...]
    x: np.ndarray
    current: np.ndarray
    past: np.ndarray
    exited: np.ndarray
    update: bool
    eligible: bool

    def __post_init__(self) -> None:
        n = len(self.ids)
        if n < 2 or len(set(self.ids)) != n or any(
                not isinstance(identifier, str) or not identifier.strip()
                for identifier in self.ids):
            raise ValueError("Event requires at least two unique nonempty contestant IDs")
        self.x = np.asarray(self.x, dtype=float)
        self.current = np.asarray(self.current, dtype=float)
        self.past = np.asarray(self.past, dtype=float)
        if self.x.shape != (n, 3) or not np.isfinite(self.x).all():
            raise ValueError("Event features must have finite shape (roster size, 3)")
        for name in ("current", "past"):
            values = getattr(self, name)
            if values.shape != (n,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Event {name} scores must be finite nonnegative roster vectors")
        self.exited = _boolean_array(self.exited, "exited")
        if self.exited.shape != (n,):
            raise ValueError("Exit labels must match the current roster")
        if not isinstance(self.update, (bool, np.bool_)) or not isinstance(
                self.eligible, (bool, np.bool_)):
            raise ValueError("Event update and eligible flags must be boolean")
        if self.update and (n < 3 or self.exited.sum() != 1):
            raise ValueError("An update requires at least three contestants and exactly one exit")
        if self.eligible and not self.update:
            raise ValueError("A scoring-eligible event must also permit an update")


def current_features(current: pd.DataFrame, history: Mapping[str, Sequence[float]]):
    """Return ID-sorted centered features without reading any outcome columns.

    The three columns are current relative judge score, the contestant's mean
    previous relative score, and change from their last relative score. Relative
    score is roster size times score share. Cold starts use the current score
    for the historical mean and zero for change. Centering occurs within the
    current roster and does not alter categorical probabilities.
    """
    if not {"contestant_id", "judge_mean"}.issubset(current.columns):
        raise ValueError("Current roster requires contestant_id and judge_mean")
    if current.contestant_id.isna().any():
        raise ValueError("Contestant IDs must be nonmissing and unique")
    g = current[["contestant_id", "judge_mean"]].copy()
    g["contestant_id"] = g.contestant_id.astype(str)
    g = g.sort_values("contestant_id")
    ids = tuple(g.contestant_id)
    if len(ids) < 2 or len(set(ids)) != len(ids) or any(not item.strip() for item in ids):
        raise ValueError("Need at least two unique nonempty contestant IDs")
    scores = g.judge_mean.to_numpy(float)
    if not np.isfinite(scores).all() or np.any(scores < 0) or not np.isfinite(
            scores.sum()) or scores.sum() <= 0:
        raise ValueError("Need complete nonnegative current scores with a positive finite sum")
    relative = len(ids) * scores / scores.sum()
    if not np.isfinite(relative).all():
        raise ValueError("Current scores exceed the supported finite numerical range")
    previous = {}
    for identifier in ids:
        values = np.asarray(history.get(identifier, []), dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Released score histories must be finite nonnegative sequences")
        previous[identifier] = values
    past = np.array([np.mean(previous[key]) if len(previous[key]) else relative[i]
                     for i, key in enumerate(ids)])
    if not np.isfinite(past).all():
        raise ValueError("Released score histories exceed the supported finite numerical range")
    change = np.array([relative[i] - previous[key][-1] if len(previous[key]) else 0.
                       for i, key in enumerate(ids)])
    x = np.column_stack([relative, past, change])
    x -= x.mean(axis=0)
    return ids, x, relative, past


def build_events(panel: pd.DataFrame) -> list[Event]:
    """Build chronological events with a fresh score history for every season.

    Features are constructed before reading the current outcome flags. Scores
    from skipped, withdrawal, multi-exit, and final events still enter later
    history. Only nonfinal single-exit events with at least three contestants
    and no withdrawal are eligible for model fitting and reported evaluation.
    """
    required = {"season", "week", "contestant_id", "judge_mean", "eliminated",
                "withdrawal", "is_final_week"}
    if required - set(panel.columns):
        raise ValueError(f"Missing panel columns: {sorted(required - set(panel.columns))}")
    frame = panel.copy()
    for name in ("season", "week"):
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 1) or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain positive integers")
        frame[name] = values.astype(int)
    if frame.contestant_id.isna().any():
        raise ValueError("Contestant IDs must be nonmissing")
    frame["contestant_id"] = frame.contestant_id.astype(str)
    if frame.duplicated(["season", "week", "contestant_id"]).any():
        raise ValueError("Duplicate contestant within a season/week roster")
    events = []
    for season, sf in frame.groupby("season", sort=True):
        history = {}
        for week, raw in sf.groupby("week", sort=True):
            g = raw.sort_values("contestant_id").reset_index(drop=True)
            if len(g) < 2:
                continue
            ids, x, relative, past = current_features(g[["contestant_id", "judge_mean"]], history)
            exited = _boolean_array(g.eliminated, "eliminated")
            withdrawal = _boolean_array(g.withdrawal, "withdrawal")
            final = _boolean_array(g.is_final_week, "is_final_week")
            update = bool(len(g) >= 3 and exited.sum() == 1 and not withdrawal.any())
            eligible = bool(update and not final.any())
            events.append(Event(int(season), int(week), ids, x, relative, past,
                                exited, update, eligible))
            for i, identifier in enumerate(ids):
                history.setdefault(identifier, []).append(float(relative[i]))
    return events


def _parameters(beta, dimension: int) -> np.ndarray:
    values = np.asarray(beta, dtype=float)
    if dimension not in (2, 3) or values.shape != (dimension,) or not np.isfinite(values).all():
        raise ValueError("Coefficients must be a finite vector of dimension 2 or 3")
    return values


def _target(event: Event) -> int:
    indices = np.flatnonzero(event.exited)
    if len(indices) != 1:
        raise ValueError("Categorical scoring requires exactly one observed exit")
    return int(indices[0])


def objective(beta, events, ridge: float, dimension: int):
    """Mean event negative log likelihood plus half the ridge penalty."""
    beta = _parameters(beta, dimension)
    if not events or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Objective requires nonempty events and a finite nonnegative ridge")
    value, gradient = 0., np.zeros(dimension)
    for event in events:
        x = event.x[:, :dimension]
        logits = x @ beta
        target = _target(event)
        value += logsumexp(logits) - logits[target]
        gradient += x.T @ softmax(logits) - x[target]
    n = len(events)
    return value/n + .5*ridge*np.dot(beta, beta), gradient/n + ridge*beta


def fit(events, ridge: float, dimension: int = 2, label: str = ""):
    """Fit bounded coefficients and reject an unconverged optimizer result."""
    if not events or not all(event.eligible for event in events):
        raise ValueError("Fit requires a nonempty nonfinal scoring population")
    objective(np.zeros(dimension), events, ridge, dimension)
    result = minimize(objective, np.zeros(dimension), args=(events, ridge, dimension),
                      jac=True, method="L-BFGS-B", bounds=[(-50., 50.)]*dimension,
                      options={"maxiter": 2000, "gtol": 1e-8, "ftol": 1e-12})
    value, gradient = objective(result.x, events, ridge, dimension)
    projected = gradient.copy()
    projected[(result.x <= -50.+1e-9) & (gradient > 0)] = 0
    projected[(result.x >= 50.-1e-9) & (gradient < 0)] = 0
    diagnostic = dict(label=label, ridge=float(ridge), dimension=dimension,
        n_training_update_events=len(events),
        n_training_nonfinal_events=sum(event.eligible for event in events),
        success=bool(result.success), message=str(result.message), iterations=int(result.nit),
        objective=float(value), gradient_inf=float(np.abs(gradient).max()),
        projected_gradient_inf=float(np.abs(projected).max()), coefficients=result.x.tolist(),
        coefficient_at_bound=bool(np.any(np.isclose(np.abs(result.x), 50.))))
    if (not result.success or not np.isfinite(value) or not np.isfinite(result.x).all()
            or not np.isfinite(gradient).all()
            or diagnostic["projected_gradient_inf"] > 1e-5):
        raise RuntimeError(f"Optimizer failed gate: {diagnostic}")
    return result.x.copy(), diagnostic


def nll(events, beta) -> float:
    """Mean categorical negative log likelihood, without regularization."""
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 1:
        raise ValueError("Coefficients must be a vector")
    beta = _parameters(beta, len(beta))
    if not events:
        raise ValueError("Negative log likelihood requires nonempty events")
    return float(np.mean([logsumexp(event.x[:, :len(beta)] @ beta)
                          - (event.x[:, :len(beta)] @ beta)[_target(event)]
                          for event in events]))


def baseline_prob(event: Event, name: str, temperature: float | None):
    """Predict uniform, current-judge, or own-history judge exit probabilities."""
    if name == "uniform":
        return np.full(len(event.ids), 1/len(event.ids))
    if name not in {"past_judges", "fixed_legacy_past12", "current_judges", "judge_only"}:
        raise ValueError(f"Unknown baseline: {name}")
    if temperature is None or not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    features = event.past if name in ("past_judges", "fixed_legacy_past12") else event.current
    return softmax(-temperature*features)


def basic_loss(event: Event, probability) -> dict:
    """Score one normalized positive categorical prediction on the same roster."""
    probability = np.asarray(probability, dtype=float)
    if (probability.shape != (len(event.ids),) or not np.isfinite(probability).all()
            or np.any(probability <= 0)
            or not np.isclose(probability.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("Invalid normalized prediction matching the current roster")
    target = _target(event)
    y = event.exited
    return dict(log_loss=float(-np.log(probability[target])),
                brier=float(np.square(probability-y).sum()),
                top1=int(y[np.argmax(probability)]), confidence=float(probability.max()))


def predictions(events, model: str, beta=None, temperature=None, phase: str = "", fold=""):
    """Return per-contestant predictions and event scores for eligible events."""
    if beta is not None:
        beta = np.asarray(beta, dtype=float)
        if beta.ndim != 1:
            raise ValueError("Coefficients must be a vector")
        beta = _parameters(beta, len(beta))
    rows, scores = [], []
    for event in events:
        if not event.eligible:
            continue
        p = (softmax(event.x[:, :len(beta)] @ beta) if beta is not None
             else baseline_prob(event, model, temperature))
        scores.append(dict(phase=phase, fold=fold, season=event.season, week=event.week,
                           model=model, **basic_loss(event, p)))
        for i, identifier in enumerate(event.ids):
            rows.append(dict(phase=phase, fold=fold, season=event.season, week=event.week,
                contestant_id=identifier, model=model, probability=float(p[i]),
                observed_exit=bool(event.exited[i])))
    return rows, scores
