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
