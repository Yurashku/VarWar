from __future__ import annotations

import math


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return math.nan
    return numerator / denominator


def normal_two_sided_p(statistic: float) -> float:
    if not math.isfinite(statistic):
        return math.nan
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def normal_critical(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        return math.nan
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        from statistics import NormalDist

        return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def t_two_sided_p(statistic: float, df: float) -> float:
    if not math.isfinite(statistic) or not math.isfinite(df) or df <= 0:
        return math.nan
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(2.0 * scipy_stats.t.sf(abs(statistic), df))
    except Exception:
        return normal_two_sided_p(statistic)


def t_critical(alpha: float, df: float) -> float:
    if not math.isfinite(df) or df <= 0:
        return math.nan
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        if abs(alpha - 0.05) < 1e-12:
            return 1.959963984540054
        return 1.959963984540054
