from __future__ import annotations

import numpy as np


def bonferroni(p_values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    return np.minimum(values * values.size, 1.0)


def holm_bonferroni(p_values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running_max = 0.0
    m = values.size
    for rank, idx in enumerate(order):
        adjusted = min((m - rank) * values[idx], 1.0)
        running_max = max(running_max, adjusted)
        adjusted_sorted[idx] = running_max
    return adjusted_sorted


def romano_wolf_stepdown(
    observed_statistics: list[float] | np.ndarray,
    bootstrap_statistics: np.ndarray,
    use_absolute: bool = True,
) -> np.ndarray:
    """Romano-Wolf step-down adjusted p-values.

    `bootstrap_statistics` must have shape `(n_resamples, n_hypotheses)` and
    should contain centered null statistics produced by the caller's resampling
    policy. This helper only applies the step-down max-statistic correction.
    """

    observed = np.asarray(observed_statistics, dtype=float)
    bootstrap = np.asarray(bootstrap_statistics, dtype=float)
    if bootstrap.ndim != 2:
        raise ValueError("bootstrap_statistics must be a 2D array.")
    if observed.ndim != 1 or bootstrap.shape[1] != observed.size:
        raise ValueError("Observed and bootstrap statistic dimensions do not match.")
    if observed.size == 0:
        return np.empty(0, dtype=float)

    obs = np.abs(observed) if use_absolute else observed.copy()
    boot = np.abs(bootstrap) if use_absolute else bootstrap
    order = np.argsort(-obs)
    adjusted = np.empty_like(obs, dtype=float)
    running_max = 0.0
    n_resamples = boot.shape[0]
    for rank, idx in enumerate(order):
        remaining = order[rank:]
        max_stats = np.max(boot[:, remaining], axis=1)
        p_value = (np.count_nonzero(max_stats >= obs[idx]) + 1.0) / (n_resamples + 1.0)
        running_max = max(running_max, min(float(p_value), 1.0))
        adjusted[idx] = running_max
    return adjusted
