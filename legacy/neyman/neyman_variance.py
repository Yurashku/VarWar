"""Archived integer Neyman allocation prototype; not part of active VarWar."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LegacyNeymanEvaluation:
    score: float
    continuous_score: float
    allocation: np.ndarray
    continuous_allocation: np.ndarray


def _continuous_allocation(
    counts: np.ndarray,
    variances: np.ndarray,
    sample_size: int,
    lower: np.ndarray,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    lower = np.asarray(lower, dtype=float)
    importance = counts * np.sqrt(np.maximum(variances, 0.0))
    positive = importance > 0.0
    allocation = lower.copy()

    zero_lower = float(lower[~positive].sum())
    target = min(float(sample_size) - zero_lower, float(counts[positive].sum()))
    target = max(target, float(lower[positive].sum()))
    if positive.any():
        weights = importance[positive]
        lower_positive = lower[positive]
        counts_positive = counts[positive]
        lower_breakpoints = lower_positive / weights
        upper_breakpoints = counts_positive / weights
        breakpoints = np.concatenate([lower_breakpoints, upper_breakpoints])
        slope_changes = np.concatenate([weights, -weights])
        order = np.argsort(breakpoints, kind="stable")

        current_total = float(lower_positive.sum())
        current_slope = 0.0
        previous = 0.0
        scale = 0.0
        cursor = 0
        while cursor < order.size:
            value = float(breakpoints[order[cursor]])
            possible_total = current_total + current_slope * (value - previous)
            if current_slope > 0.0 and target <= possible_total + 1e-12:
                scale = previous + (target - current_total) / current_slope
                break
            current_total = possible_total
            while cursor < order.size and breakpoints[order[cursor]] == value:
                current_slope += float(slope_changes[order[cursor]])
                cursor += 1
            previous = value
            scale = value
        else:
            scale = float(np.max(upper_breakpoints))

        allocation[positive] = np.clip(
            weights * scale,
            lower_positive,
            counts_positive,
        )

    remaining = float(sample_size) - float(allocation.sum())
    if remaining > 1e-8:
        for index in np.flatnonzero(~positive):
            addition = min(remaining, counts[index] - allocation[index])
            allocation[index] += addition
            remaining -= addition
            if remaining <= 1e-8:
                break
    return allocation


def integer_neyman_allocation(
    counts: np.ndarray,
    variances: np.ndarray,
    sample_size: int,
    minimum: int = 2,
) -> LegacyNeymanEvaluation:
    """Return the archived exact-size allocation and its variance objective."""

    counts = np.asarray(counts, dtype=np.int64)
    variances = np.maximum(np.asarray(variances, dtype=float), 0.0)
    if minimum < 1 or np.any(counts < minimum):
        raise ValueError("Each stratum must satisfy the positive lower bound.")
    if not minimum * counts.size <= sample_size <= int(counts.sum()):
        raise ValueError("The requested sample size is infeasible.")

    lower = np.full(counts.size, minimum, dtype=np.int64)
    continuous = _continuous_allocation(counts, variances, sample_size, lower)
    allocation = np.floor(continuous + 1e-12).astype(np.int64)
    allocation = np.clip(allocation, lower, counts)
    remaining = int(sample_size - allocation.sum())
    coefficients = counts.astype(float) ** 2 * variances

    heap: list[tuple[float, int]] = []
    for index, current in enumerate(allocation):
        if current < counts[index]:
            gain = coefficients[index] / (current * (current + 1.0))
            heapq.heappush(heap, (-float(gain), index))
    while remaining > 0 and heap:
        _, index = heapq.heappop(heap)
        allocation[index] += 1
        remaining -= 1
        current = allocation[index]
        if current < counts[index]:
            gain = coefficients[index] / (current * (current + 1.0))
            heapq.heappush(heap, (-float(gain), index))
    if remaining:
        raise RuntimeError("Failed to construct the archived allocation.")

    def score(values: np.ndarray) -> float:
        terms = counts.astype(float) ** 2 * variances / values
        terms -= counts.astype(float) * variances
        return float(terms.sum() / counts.sum() ** 2)

    return LegacyNeymanEvaluation(
        score=score(allocation.astype(float)),
        continuous_score=score(continuous),
        allocation=allocation,
        continuous_allocation=continuous,
    )
