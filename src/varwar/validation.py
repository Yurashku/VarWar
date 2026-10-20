"""Monte Carlo-aware validation of statistical inference against ground truth.

The verdict rules implemented here are documented in
``docs/methods/statistical_validation.md`` (section «Валидностный экран»).
Every hard failure requires Monte Carlo evidence; a point estimate that only
crosses a threshold yields ``warning`` or ``uncertain`` instead of ``bad``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from varwar.analysis import CausalSafety, InferenceQuality, TheoryGrade


#: Ordered verdict scale used by reports and rankings.
QUALITY_ORDER: tuple[str, ...] = ("good", "warning", "uncertain", "bad", "not_applicable")
UNCERTAIN = "uncertain"

#: A null truth must be identically zero up to this tolerance in every trial.
NULL_TRUTH_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ValidationThresholds:
    alpha: float = 0.05
    #: Size is ``bad`` only when the Wilson lower bound exceeds this value.
    max_fpr: float = 0.07
    #: Point FPR below this value is flagged (warning or conservative).
    min_fpr: float = 0.03
    #: Coverage is ``bad`` only when the Wilson upper bound is below this value.
    min_coverage: float = 0.93
    #: Point coverage above this value is flagged (warning or conservative).
    max_coverage: float = 0.97
    #: Practical bias margin in units of the empirical SD of the estimator.
    max_standardized_bias: float = 0.10
    #: Normal quantile used for the Monte Carlo error of the bias.
    bias_mc_z: float = 1.96
    good_se_sd_low: float = 0.90
    good_se_sd_high: float = 1.10
    #: Benjamini-Hochberg level for p-value super-uniformity across a stage.
    superuniformity_fdr_q: float = 0.05
    # Kept for backwards compatibility of the public dataclass; not used by
    # the Monte Carlo-aware rule.
    warning_se_sd_low: float = 0.80
    warning_se_sd_high: float = 1.20


def truth_column_for_estimand(estimand_id: str) -> str:
    if estimand_id == "sample_ate":
        return "true_sample_ate"
    if estimand_id == "population_ate":
        return "true_population_ate"
    if estimand_id in {
        "trimmed_effect",
        "trimmed_mean_effect",
        "winsorized_effect",
        "winsorized_mean_effect",
        "robust_native",
        "common_effect",
    }:
        return "true_native_estimand"
    if estimand_id in {"robust_mean_ate", "sample_mean_effect"}:
        return "true_sample_ate"
    raise ValueError(f"No ground-truth mapping for estimand {estimand_id!r}.")


@lru_cache(maxsize=32)
def _normal_quantile(probability: float) -> float:
    return float(NormalDist().inv_cdf(probability))


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    z = _normal_quantile(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(values.size, dtype=float)
    running = 0.0
    total = values.size
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg_reject(
    p_values: Iterable[float],
    q: float = 0.05,
) -> np.ndarray:
    """Return BH rejections; non-finite p-values are never rejected."""

    values = np.asarray(tuple(p_values), dtype=float)
    reject = np.zeros(values.size, dtype=bool)
    finite_index = np.flatnonzero(np.isfinite(values))
    m = finite_index.size
    if m == 0:
        return reject
    order = finite_index[np.argsort(values[finite_index], kind="mergesort")]
    sorted_p = values[order]
    below = sorted_p <= q * np.arange(1, m + 1) / m
    if below.any():
        last = int(np.flatnonzero(below).max())
        reject[order[: last + 1]] = True
    return reject


def pvalue_superuniformity(
    p_values: np.ndarray,
    thresholds: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[str, float | bool]:
    finite = np.asarray(p_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "pvalue_superuniform_p": math.nan,
            "pvalue_uniform_ks_p": math.nan,
            "pvalue_superuniform_pass": False,
        }
    result: dict[str, float | bool] = {}
    levels = np.asarray(thresholds, dtype=float)
    counts = np.count_nonzero(finite[:, None] <= levels[None, :], axis=0)
    # One-sided exact binomial test P(X >= k) for every threshold at once
    # (identical to scipy.stats.binomtest(..., alternative="greater")).
    raw = np.minimum(stats.binom.sf(counts - 1, finite.size, levels), 1.0)
    for threshold, count in zip(thresholds, counts):
        result[f"pvalue_ecdf_at_{threshold:.2f}"] = int(count) / finite.size
    adjusted = _holm_adjust(raw)
    result["pvalue_superuniform_p"] = float(np.min(adjusted))
    result["pvalue_uniform_ks_p"] = float(stats.kstest(finite, "uniform").pvalue)
    # Single-row decision.  Stage-level verdicts replace it with a
    # Benjamini-Hochberg decision across all rows (see apply_stage_superuniformity).
    result["pvalue_superuniform_pass"] = bool(np.min(adjusted) >= 0.05)
    return result


def se_to_sd_interval(
    se_to_sd: float,
    n_trials: int,
    kurtosis: float = 3.0,
    z: float = 1.96,
) -> tuple[float, float]:
    """Monte Carlo CI of SE/SD from the sampling error of the empirical SD.

    ``Var(log SD) ≈ (κ − 1) / (4R)``, where ``κ`` is the (non-excess) kurtosis
    of the estimates; for a normal estimator this is ``1 / (2R)``.  The error
    of the mean SE is ignored because it is much smaller than that of the SD.
    """

    if not np.isfinite(se_to_sd) or n_trials < 2:
        return math.nan, math.nan
    kappa = kurtosis if np.isfinite(kurtosis) else 3.0
    kappa = max(kappa, 1.0 + 1e-12)
    half = z * math.sqrt((kappa - 1.0) / (4.0 * n_trials))
    return se_to_sd * math.exp(-half), se_to_sd * math.exp(half)


def summarize_trial_group(
    trials: pd.DataFrame,
    *,
    estimand_id: str,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    truth_column = truth_column_for_estimand(estimand_id)
    if truth_column not in trials:
        raise ValueError(f"Missing ground truth column: {truth_column}")
    effect = pd.to_numeric(trials["effect"], errors="coerce").to_numpy(float)
    truth = pd.to_numeric(trials[truth_column], errors="coerce").to_numpy(float)
    se = pd.to_numeric(trials["standard_error"], errors="coerce").to_numpy(float)
    p_value = pd.to_numeric(trials["p_value"], errors="coerce").to_numpy(float)
    ci_low = pd.to_numeric(trials["ci_low"], errors="coerce").to_numpy(float)
    ci_high = pd.to_numeric(trials["ci_high"], errors="coerce").to_numpy(float)
    valid = np.isfinite(effect) & np.isfinite(truth)
    errors = effect[valid] - truth[valid]
    if errors.size < 2:
        raise ValueError("At least two finite trial estimates are required.")
    empirical_sd = float(np.std(effect[valid], ddof=1))
    bias = float(np.mean(errors))
    bias_mc_se = empirical_sd / math.sqrt(errors.size)
    bias_critical = float(stats.t.ppf(0.975, errors.size - 1))
    truth_max_abs = float(np.max(np.abs(truth[valid])))
    rejection_mask = np.isfinite(p_value)
    rejection_count = int(np.count_nonzero(p_value[rejection_mask] <= alpha))
    rejection_trials = int(rejection_mask.sum())
    fpr_low, fpr_high = wilson_interval(rejection_count, rejection_trials)
    coverage_mask = valid & np.isfinite(ci_low) & np.isfinite(ci_high)
    coverage_trials = int(coverage_mask.sum())
    coverage_count = int(
        np.count_nonzero(
            (ci_low[coverage_mask] <= truth[coverage_mask])
            & (truth[coverage_mask] <= ci_high[coverage_mask])
        )
    )
    coverage_low, coverage_high = wilson_interval(coverage_count, coverage_trials)
    # Test of H0: theta = theta_true by CI inversion.  This is the size of the
    # test whenever the realised truth is not identically zero.
    truth_rejection_count = coverage_trials - coverage_count
    truth_rej_low, truth_rej_high = wilson_interval(truth_rejection_count, coverage_trials)
    finite_se = se[np.isfinite(se)]
    mean_se = float(np.mean(finite_se)) if finite_se.size else math.nan
    sd_se = float(np.std(finite_se, ddof=1)) if finite_se.size > 1 else math.nan
    native_rmse = float(math.sqrt(np.mean(errors * errors)))
    sample_truth = pd.to_numeric(
        trials.get("true_sample_ate", pd.Series(math.nan, index=trials.index)),
        errors="coerce",
    ).to_numpy(float)
    ate_mask = np.isfinite(effect) & np.isfinite(sample_truth)
    ate_errors = effect[ate_mask] - sample_truth[ate_mask]
    centred = effect[valid] - np.mean(effect[valid])
    second = float(np.mean(centred * centred))
    kurtosis = float(np.mean(centred**4) / (second * second)) if second > 0.0 else math.nan
    se_to_sd = mean_se / empirical_sd if empirical_sd > 0.0 else math.nan
    se_sd_low, se_sd_high = se_to_sd_interval(se_to_sd, int(errors.size), kurtosis)
    if {"n_control", "n_treatment"} <= set(trials.columns):
        sizes = pd.to_numeric(trials["n_control"], errors="coerce") + pd.to_numeric(
            trials["n_treatment"], errors="coerce"
        )
        df_median = float(sizes.median()) - 2.0 if sizes.notna().any() else math.nan
    else:
        df_median = math.nan
    p_diagnostics = pvalue_superuniformity(p_value[rejection_mask])
    result: dict[str, float | int | bool] = {
        "n_trials": int(errors.size),
        "mean_effect": float(np.mean(effect[valid])),
        "mean_truth": float(np.mean(truth[valid])),
        "truth_max_abs": truth_max_abs,
        "truth_is_null": bool(truth_max_abs < NULL_TRUTH_TOLERANCE),
        "bias": bias,
        "bias_mc_se": bias_mc_se,
        "bias_ci_low": bias - bias_critical * bias_mc_se,
        "bias_ci_high": bias + bias_critical * bias_mc_se,
        "empirical_sd": empirical_sd,
        "effect_kurtosis": kurtosis,
        "mean_standard_error": mean_se,
        "sd_standard_error": sd_se,
        "se_to_sd": se_to_sd,
        "se_to_sd_ci_low": se_sd_low,
        "se_to_sd_ci_high": se_sd_high,
        "mse": float(np.mean(errors * errors)),
        "rmse": native_rmse,
        "ate_displacement": float(np.mean(ate_errors)) if ate_errors.size else math.nan,
        "ate_rmse": (
            float(math.sqrt(np.mean(ate_errors * ate_errors)))
            if ate_errors.size
            else math.nan
        ),
        # Rejection of H0: theta = 0 (p <= alpha).  Under H1 this is power.
        "rejection_rate": (
            rejection_count / rejection_trials if rejection_trials else math.nan
        ),
        "rejection_count": rejection_count,
        "rejection_trials": rejection_trials,
        "rejection_ci_low": fpr_low,
        "rejection_ci_high": fpr_high,
        # Rejection of H0: theta = theta_true (CI excludes the truth).
        "truth_rejection_rate": (
            truth_rejection_count / coverage_trials if coverage_trials else math.nan
        ),
        "truth_rejection_count": truth_rejection_count,
        "truth_rejection_ci_low": truth_rej_low,
        "truth_rejection_ci_high": truth_rej_high,
        "coverage": coverage_count / coverage_trials if coverage_trials else math.nan,
        "coverage_count": coverage_count,
        "coverage_trials": coverage_trials,
        "coverage_ci_low": coverage_low,
        "coverage_ci_high": coverage_high,
        "mean_ci_length": (
            float(np.mean(ci_high[coverage_mask] - ci_low[coverage_mask]))
            if coverage_trials
            else math.nan
        ),
        "nan_pvalue_rate": 1.0 - rejection_trials / len(trials) if len(trials) else math.nan,
        "df_median": df_median,
    }
    result.update(p_diagnostics)
    return result


def _get(summary: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    value = summary.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_inference_quality(
    h0: Mapping[str, Any],
    h1: Mapping[str, Any],
    *,
    theory_grade: TheoryGrade,
    causal_safety: CausalSafety,
    thresholds: ValidationThresholds = ValidationThresholds(),
    superuniformity_rejected: bool | None = None,
    failure_rate: float = 0.0,
    nan_pvalue_rate: float = 0.0,
    inference_failure_rate: float = 0.0,
) -> dict[str, float | bool | str]:
    """Monte Carlo-aware verdict for one (H0, H1) pair of trial summaries.

    ``superuniformity_rejected`` is the stage-level Benjamini-Hochberg decision
    for this row; when omitted the single-row Holm decision is used.
    """

    reasons: list[str] = []
    warnings: list[str] = []
    # --- bias -------------------------------------------------------------
    empirical_sd = _get(h1, "empirical_sd")
    bias = _get(h1, "bias")
    n1 = _get(h1, "n_trials")
    bias_mc_se = _get(h1, "bias_mc_se")
    if not np.isfinite(bias_mc_se):
        bias_mc_se = empirical_sd / math.sqrt(n1) if np.isfinite(n1) and n1 > 0 else 0.0
    if empirical_sd > 0.0:
        standardized_bias = abs(bias) / empirical_sd
        standardized_bias_lower = (
            abs(bias) - thresholds.bias_mc_z * bias_mc_se
        ) / empirical_sd
    else:
        standardized_bias = math.inf if bias != 0.0 else 0.0
        standardized_bias_lower = standardized_bias
    margin = thresholds.max_standardized_bias
    if standardized_bias_lower > margin:
        bias_state = "fail"
        reasons.append("bias")
    elif standardized_bias > margin:
        bias_state = "uncertain"
        warnings.append("bias_uncertain")
    else:
        bias_state = "pass"
    # --- size -------------------------------------------------------------
    h0_is_null = bool(h0.get("truth_is_null", True))
    if h0_is_null:
        size_metric = "rejection_vs_zero"
        size = _get(h0, "rejection_rate")
        size_low = _get(h0, "rejection_ci_low")
        size_high = _get(h0, "rejection_ci_high")
    else:
        size_metric = "rejection_vs_truth"
        size = _get(h0, "truth_rejection_rate")
        size_low = _get(h0, "truth_rejection_ci_low")
        size_high = _get(h0, "truth_rejection_ci_high")
    se_to_sd = _get(h1, "se_to_sd")
    se_sd_ci_low = _get(h1, "se_to_sd_ci_low", se_to_sd)
    se_sd_ci_high = _get(h1, "se_to_sd_ci_high", se_to_sd)
    se_conservative = bool(np.isfinite(se_to_sd) and se_to_sd > thresholds.good_se_sd_high)
    if not np.isfinite(size):
        size_state = "not_evaluable"
        warnings.append("size_not_evaluable")
    elif size_low > thresholds.max_fpr:
        size_state = "fail"
        reasons.append("fpr")
    elif size > thresholds.max_fpr:
        size_state = "warning"
        warnings.append("fpr_high_uncertain")
    elif size < thresholds.min_fpr:
        if se_conservative:
            size_state = "conservative"
        else:
            size_state = "warning"
            warnings.append("fpr_low")
    else:
        size_state = "pass"
    # --- coverage ---------------------------------------------------------
    coverage = _get(h1, "coverage")
    coverage_low = _get(h1, "coverage_ci_low")
    coverage_high = _get(h1, "coverage_ci_high")
    if not np.isfinite(coverage):
        coverage_state = "not_evaluable"
        warnings.append("coverage_not_evaluable")
    elif coverage_high < thresholds.min_coverage:
        coverage_state = "fail"
        reasons.append("coverage")
    elif coverage < thresholds.min_coverage:
        coverage_state = "warning"
        warnings.append("coverage_low_uncertain")
    elif coverage > thresholds.max_coverage:
        if se_conservative:
            coverage_state = "conservative"
        else:
            coverage_state = "warning"
            warnings.append("coverage_high")
    else:
        coverage_state = "pass"
    # --- SE scale ---------------------------------------------------------
    if not np.isfinite(se_to_sd):
        se_sd_state = "not_evaluable"
        warnings.append("se_sd_not_evaluable")
    elif np.isfinite(se_sd_ci_high) and se_sd_ci_high < thresholds.good_se_sd_low:
        se_sd_state = "fail"
        reasons.append("se_sd")
    elif se_to_sd < thresholds.good_se_sd_low:
        se_sd_state = "warning"
        warnings.append("se_sd_low_uncertain")
    elif se_conservative:
        se_sd_state = "conservative"
    else:
        se_sd_state = "pass"
    # --- p-value calibration (only meaningful for a null truth) -----------
    if not h0_is_null:
        pvalue_state = "not_applicable"
    else:
        if superuniformity_rejected is None:
            p_super = _get(h0, "pvalue_superuniform_p")
            if not np.isfinite(p_super):
                rejected = None
            else:
                rejected = p_super < thresholds.superuniformity_fdr_q
        else:
            rejected = bool(superuniformity_rejected)
        if rejected is None:
            pvalue_state = "not_evaluable"
            warnings.append("pvalue_not_evaluable")
        elif rejected:
            pvalue_state = "fail"
            reasons.append("pvalue_calibration")
        else:
            pvalue_state = "pass"
    # --- execution --------------------------------------------------------
    failure_rate = float(failure_rate) if np.isfinite(failure_rate) else 0.0
    nan_pvalue_rate = float(nan_pvalue_rate) if np.isfinite(nan_pvalue_rate) else 0.0
    inference_failure_rate = (
        float(inference_failure_rate) if np.isfinite(inference_failure_rate) else 0.0
    )
    if failure_rate > 0.0:
        warnings.append("execution_failures")
    if inference_failure_rate > 0.0:
        warnings.append("inference_failures")
    if nan_pvalue_rate > 0.0:
        warnings.append("nan_pvalues")
    trusted_theory = theory_grade in {TheoryGrade.A, TheoryGrade.B}
    causal_ok = causal_safety in {CausalSafety.SAFE, CausalSafety.CONDITIONAL}
    if not trusted_theory:
        warnings.append("theory_grade")
    if not causal_ok:
        warnings.append("causal_safety")
    states = (bias_state, size_state, coverage_state, se_sd_state, pvalue_state)
    applicable = [state for state in states if state != "not_applicable"]
    passes = [state in {"pass", "conservative"} for state in applicable]
    score = 100.0 * sum(passes) / len(passes) if passes else math.nan
    conservative = (
        "conservative" in states
        or (se_conservative and size_state in {"pass", "conservative"})
    )
    if reasons:
        quality = InferenceQuality.BAD.value
    elif bias_state == "uncertain":
        quality = UNCERTAIN
    elif warnings:
        quality = InferenceQuality.WARNING.value
    else:
        quality = InferenceQuality.GOOD.value
    # Acceptance criteria of the confirmation stage (equivalence-style: the
    # Monte Carlo interval must lie inside the admissible zone).
    validity_confirmed = bool(
        np.isfinite(size_high)
        and size_high <= thresholds.max_fpr
        and np.isfinite(coverage_low)
        and coverage_low >= thresholds.min_coverage
        and empirical_sd > 0.0
        and (abs(bias) + thresholds.bias_mc_z * bias_mc_se) <= margin * empirical_sd
        and np.isfinite(se_sd_ci_low)
        and se_sd_ci_low >= thresholds.good_se_sd_low
        and pvalue_state in {"pass", "not_applicable"}
        and failure_rate == 0.0
        and inference_failure_rate == 0.0
    )
    fpr = _get(h0, "rejection_rate")
    return {
        "validity_confirmed": validity_confirmed,
        "h0_is_null": h0_is_null,
        "size_metric": size_metric,
        "size": size,
        "size_ci_low": size_low,
        "size_ci_high": size_high,
        "standardized_bias": standardized_bias,
        "standardized_bias_mc_lower": standardized_bias_lower,
        "bias_state": bias_state,
        "size_state": size_state,
        "coverage_state": coverage_state,
        "se_sd_state": se_sd_state,
        "pvalue_state": pvalue_state,
        "bias_pass": bias_state == "pass",
        "fpr_pass": size_state in {"pass", "conservative", "warning"},
        "fpr_uncertain": bool(
            size_state != "fail" and np.isfinite(size_high) and size_high > thresholds.max_fpr
        ),
        "coverage_pass": coverage_state in {"pass", "conservative", "warning"},
        "coverage_uncertain": bool(
            coverage_state != "fail"
            and np.isfinite(coverage_low)
            and coverage_low < thresholds.min_coverage
        ),
        "se_sd_pass": se_sd_state in {"pass", "conservative"},
        "pvalue_calibration_pass": pvalue_state in {"pass", "not_applicable"},
        "conservative": bool(conservative),
        "inference_quality_score": score,
        "inference_quality": quality,
        "verdict_reasons": ";".join(reasons + warnings),
        "fpr_excess": (fpr if h0_is_null else size) - thresholds.alpha,
        "coverage_gap": coverage - (1.0 - thresholds.alpha),
    }


def quality_rank(values: pd.Series) -> pd.Series:
    """Map verdict strings to their order (good=0 … bad=3, unknown last)."""

    order = {name: index for index, name in enumerate(QUALITY_ORDER)}
    return values.map(lambda value: order.get(str(value), len(order))).astype(int)
