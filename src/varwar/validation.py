from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from varwar.analysis import CausalSafety, InferenceQuality, TheoryGrade


@dataclass(frozen=True)
class ValidationThresholds:
    alpha: float = 0.05
    max_fpr: float = 0.07
    min_coverage: float = 0.93
    max_standardized_bias: float = 0.10
    good_se_sd_low: float = 0.90
    good_se_sd_high: float = 1.10
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


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
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
    raw = []
    result: dict[str, float | bool] = {}
    for threshold in thresholds:
        count = int(np.count_nonzero(finite <= threshold))
        p_value = float(
            stats.binomtest(
                count,
                int(finite.size),
                threshold,
                alternative="greater",
            ).pvalue
        )
        raw.append(p_value)
        result[f"pvalue_ecdf_at_{threshold:.2f}"] = count / finite.size
    adjusted = _holm_adjust(raw)
    result["pvalue_superuniform_p"] = float(np.min(adjusted))
    result["pvalue_uniform_ks_p"] = float(stats.kstest(finite, "uniform").pvalue)
    result["pvalue_superuniform_pass"] = bool(np.min(adjusted) >= 0.05)
    return result


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
    rejection_mask = np.isfinite(p_value)
    rejection_count = int(np.count_nonzero(p_value[rejection_mask] <= alpha))
    fpr_low, fpr_high = wilson_interval(rejection_count, int(rejection_mask.sum()))
    coverage_mask = valid & np.isfinite(ci_low) & np.isfinite(ci_high)
    coverage_count = int(
        np.count_nonzero(
            (ci_low[coverage_mask] <= truth[coverage_mask])
            & (truth[coverage_mask] <= ci_high[coverage_mask])
        )
    )
    coverage_low, coverage_high = wilson_interval(
        coverage_count,
        int(coverage_mask.sum()),
    )
    finite_se = se[np.isfinite(se)]
    mean_se = float(np.mean(finite_se)) if finite_se.size else math.nan
    native_rmse = float(math.sqrt(np.mean(errors * errors)))
    sample_truth = pd.to_numeric(
        trials.get("true_sample_ate", pd.Series(math.nan, index=trials.index)),
        errors="coerce",
    ).to_numpy(float)
    ate_mask = np.isfinite(effect) & np.isfinite(sample_truth)
    ate_errors = effect[ate_mask] - sample_truth[ate_mask]
    p_diagnostics = pvalue_superuniformity(p_value[rejection_mask])
    result: dict[str, float | int | bool] = {
        "n_trials": int(errors.size),
        "mean_effect": float(np.mean(effect[valid])),
        "mean_truth": float(np.mean(truth[valid])),
        "bias": bias,
        "bias_mc_se": bias_mc_se,
        "bias_ci_low": bias - bias_critical * bias_mc_se,
        "bias_ci_high": bias + bias_critical * bias_mc_se,
        "empirical_sd": empirical_sd,
        "mean_standard_error": mean_se,
        "se_to_sd": mean_se / empirical_sd if empirical_sd > 0.0 else math.nan,
        "mse": float(np.mean(errors * errors)),
        "rmse": native_rmse,
        "ate_displacement": float(np.mean(ate_errors)) if ate_errors.size else math.nan,
        "ate_rmse": (
            float(math.sqrt(np.mean(ate_errors * ate_errors)))
            if ate_errors.size
            else math.nan
        ),
        "rejection_rate": rejection_count / rejection_mask.sum(),
        "rejection_count": rejection_count,
        "rejection_trials": int(rejection_mask.sum()),
        "rejection_ci_low": fpr_low,
        "rejection_ci_high": fpr_high,
        "coverage": coverage_count / coverage_mask.sum(),
        "coverage_count": coverage_count,
        "coverage_trials": int(coverage_mask.sum()),
        "coverage_ci_low": coverage_low,
        "coverage_ci_high": coverage_high,
        "mean_ci_length": float(np.mean(ci_high[coverage_mask] - ci_low[coverage_mask])),
    }
    result.update(p_diagnostics)
    return result


def evaluate_inference_quality(
    h0: dict[str, float | int | bool],
    h1: dict[str, float | int | bool],
    *,
    theory_grade: TheoryGrade,
    causal_safety: CausalSafety,
    thresholds: ValidationThresholds = ValidationThresholds(),
) -> dict[str, float | bool | str]:
    empirical_sd = float(h1["empirical_sd"])
    standardized_bias = (
        abs(float(h1["bias"])) / empirical_sd if empirical_sd > 0.0 else math.inf
    )
    bias_pass = standardized_bias <= thresholds.max_standardized_bias
    fpr = float(h0["rejection_rate"])
    fpr_high = float(h0["rejection_ci_high"])
    fpr_low = float(h0["rejection_ci_low"])
    # A finite Monte Carlo run rarely proves that the upper confidence bound is
    # below 0.07.  It can, however, prove anti-conservatism when the lower bound
    # is already above the admissible level.  Point estimates inside the limit
    # pass provisionally; uncertainty remains visible in the warning flag.
    fpr_pass = fpr <= thresholds.max_fpr and fpr_low <= thresholds.max_fpr
    fpr_uncertain = fpr_pass and fpr_high > thresholds.max_fpr
    coverage = float(h1["coverage"])
    coverage_low = float(h1["coverage_ci_low"])
    coverage_high = float(h1["coverage_ci_high"])
    coverage_pass = coverage >= thresholds.min_coverage and coverage_high >= thresholds.min_coverage
    coverage_uncertain = coverage_pass and coverage_low < thresholds.min_coverage
    se_to_sd = float(h1["se_to_sd"])
    se_sd_pass = thresholds.good_se_sd_low <= se_to_sd <= thresholds.good_se_sd_high
    pvalue_pass = bool(h0["pvalue_superuniform_pass"])
    numeric_passes = (bias_pass, fpr_pass, coverage_pass, se_sd_pass, pvalue_pass)
    score = 100.0 * sum(numeric_passes) / len(numeric_passes)
    hard_bad = (
        fpr_low > thresholds.max_fpr
        or coverage_high < thresholds.min_coverage
        or not bias_pass
        or not pvalue_pass
    )
    trusted_theory = theory_grade in {TheoryGrade.A, TheoryGrade.B}
    causal_ok = causal_safety in {CausalSafety.SAFE, CausalSafety.CONDITIONAL}
    if hard_bad:
        quality = InferenceQuality.BAD
    elif (
        all(numeric_passes)
        and trusted_theory
        and causal_ok
        and not fpr_uncertain
        and not coverage_uncertain
    ):
        quality = InferenceQuality.GOOD
    else:
        quality = InferenceQuality.WARNING
    return {
        "standardized_bias": standardized_bias,
        "bias_pass": bias_pass,
        "fpr_pass": fpr_pass,
        "fpr_uncertain": fpr_uncertain,
        "coverage_pass": coverage_pass,
        "coverage_uncertain": coverage_uncertain,
        "se_sd_pass": se_sd_pass,
        "pvalue_calibration_pass": pvalue_pass,
        "inference_quality_score": score,
        "inference_quality": quality.value,
        "fpr_excess": fpr - thresholds.alpha,
        "coverage_gap": coverage - (1.0 - thresholds.alpha),
    }
