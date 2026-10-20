"""Robust two-arm analysis methods with inference owned by the estimator.

The classes in this module intentionally do not act as target transformers.
Trimming, winsorization, and data-adaptive outlier detection change the point
functional and must therefore be refitted inside their resampling procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .core import EstimateResult, ExperimentSpec, require_columns


ArrayEstimator = Callable[[np.ndarray, np.ndarray, np.random.Generator], tuple[float, dict[str, Any]]]


def _group_arrays(data: pd.DataFrame, spec: ExperimentSpec) -> tuple[np.ndarray, np.ndarray]:
    require_columns(data, [spec.treatment_col, spec.target_col])
    assignment = data[spec.treatment_col].to_numpy()
    target = pd.to_numeric(data[spec.target_col], errors="coerce").to_numpy(dtype=float)
    control = target[(assignment == spec.control_value) & np.isfinite(target)]
    treatment = target[(assignment == spec.treatment_value) & np.isfinite(target)]
    if control.size < 2 or treatment.size < 2:
        raise ValueError("Both groups need at least two finite target values.")
    return control, treatment


BOOTSTRAP_FAILURE_TOLERANCE = 0.05


def resampling_failure_diagnostics(n_resamples: int, valid: int) -> dict[str, Any]:
    """Share of failed resamples and the inference-failure flag.

    Dropping failed replicates silently can bias the SE and interval when the
    failures concentrate in extreme resamples. More than 5% failures marks the
    inference as failed; the point estimate is still reported.
    """

    failed = max(int(n_resamples) - int(valid), 0)
    share = failed / n_resamples if n_resamples > 0 else 0.0
    return {
        "failed_resamples": failed,
        "failed_resample_share": float(share),
        "inference_failure": bool(share > BOOTSTRAP_FAILURE_TOLERANCE),
    }


def _validate_resampling(n_resamples: int, confidence_level: float) -> None:
    if n_resamples < 99:
        raise ValueError("n_resamples must be at least 99.")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1.")


def _full_refit_bootstrap(
    control: np.ndarray,
    treatment: np.ndarray,
    estimator: ArrayEstimator,
    *,
    observed_effect: float,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
    conservative_envelope: bool = False,
) -> tuple[float, float, float, float, float, dict[str, Any]]:
    """Bootstrap both arms independently and refit the full point procedure."""

    _validate_resampling(n_resamples, confidence_level)
    rng = np.random.default_rng(random_state)
    effects = np.empty(n_resamples, dtype=float)
    valid = 0
    for _ in range(n_resamples):
        c = control[rng.integers(0, control.size, size=control.size)]
        t = treatment[rng.integers(0, treatment.size, size=treatment.size)]
        replicate_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.int64).max))
        try:
            value, _ = estimator(c, t, replicate_rng)
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(value):
            effects[valid] = value
            valid += 1
    minimum_valid = max(50, int(math.ceil(0.8 * n_resamples)))
    if valid < minimum_valid:
        raise ValueError(
            f"Only {valid} valid bootstrap replicates; at least {minimum_valid} required."
        )
    effects = effects[:valid]
    standard_error = float(effects.std(ddof=1))
    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(effects, [alpha / 2.0, 1.0 - alpha / 2.0])
    centered = effects - effects.mean()
    extreme = int(np.count_nonzero(np.abs(centered) >= abs(observed_effect)))
    p_value = float((extreme + 1.0) / (valid + 1.0))
    statistic = (
        float(observed_effect / standard_error)
        if standard_error > 0.0
        else math.copysign(math.inf, observed_effect) if observed_effect != 0.0 else 0.0
    )
    if conservative_envelope and standard_error > 0.0:
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
        normal_low = observed_effect - critical * standard_error
        normal_high = observed_effect + critical * standard_error
        ci_low = min(float(ci_low), normal_low)
        ci_high = max(float(ci_high), normal_high)
        normal_p_value = float(2.0 * stats.norm.sf(abs(statistic)))
        p_value = max(p_value, normal_p_value)
    return (
        standard_error,
        statistic,
        p_value,
        float(ci_low),
        float(ci_high),
        {
            "inference": "independent_arm_full_refit_percentile_bootstrap",
            "bootstrap_p_value": "centered_two_sided_plus_one",
            "n_resamples": n_resamples,
            "valid_resamples": valid,
            **resampling_failure_diagnostics(n_resamples, valid),
            "random_state": random_state,
            "confidence_level": confidence_level,
            "conservative_envelope": conservative_envelope,
        },
    )


def _bootstrap_result(
    name: str,
    control: np.ndarray,
    treatment: np.ndarray,
    estimator: ArrayEstimator,
    *,
    n_resamples: int,
    random_state: int,
    alpha: float,
    diagnostics: dict[str, Any],
    conservative_envelope: bool = False,
) -> EstimateResult:
    observed, point_diagnostics = estimator(
        control, treatment, np.random.default_rng(random_state)
    )
    se, statistic, p_value, ci_low, ci_high, bootstrap_diagnostics = (
        _full_refit_bootstrap(
            control,
            treatment,
            estimator,
            observed_effect=observed,
            n_resamples=n_resamples,
            confidence_level=1.0 - alpha,
            random_state=random_state,
            conservative_envelope=conservative_envelope,
        )
    )
    return EstimateResult(
        estimator=name,
        effect=observed,
        standard_error=se,
        statistic=statistic,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_control=int(control.size),
        n_treatment=int(treatment.size),
        diagnostics={**diagnostics, **point_diagnostics, **bootstrap_diagnostics},
    )


def _yuen_group(values: np.ndarray, trim_fraction: float) -> tuple[float, float, int]:
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5).")
    ordered = np.sort(np.asarray(values, dtype=float))
    n = ordered.size
    removed = int(math.floor(n * trim_fraction))
    effective_n = n - 2 * removed
    if effective_n < 2:
        raise ValueError("Trim fraction leaves fewer than two observations.")
    trimmed_mean = float(ordered[removed : n - removed].mean())
    winsorized = ordered.copy()
    if removed:
        winsorized[:removed] = ordered[removed]
        winsorized[n - removed :] = ordered[n - removed - 1]
    winsorized_variance = float(winsorized.var(ddof=2 * removed + 1))
    return trimmed_mean, winsorized_variance / effective_n, effective_n


@dataclass
class YuenTrimmedMeanEstimator:
    """Difference of symmetric trimmed means with Yuen-Welch inference.

    The estimand is the difference of population trimmed means, not the ATE.
    """

    trim_fraction: float = 0.1
    name: str = "yuen_trimmed_mean"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        mean_c, component_c, effective_c = _yuen_group(control, self.trim_fraction)
        mean_t, component_t, effective_t = _yuen_group(treatment, self.trim_fraction)
        effect = mean_t - mean_c
        se = math.sqrt(component_c + component_t)
        denominator = component_c**2 / (effective_c - 1) + component_t**2 / (
            effective_t - 1
        )
        df = (component_c + component_t) ** 2 / denominator if denominator > 0 else math.inf
        statistic = effect / se if se > 0 else math.copysign(math.inf, effect) if effect else 0.0
        p_value = float(2.0 * stats.t.sf(abs(statistic), df)) if se > 0 else float(effect == 0)
        critical = float(stats.t.ppf(1.0 - spec.alpha / 2.0, df))
        return EstimateResult(
            estimator=self.name,
            effect=effect,
            standard_error=se,
            statistic=float(statistic),
            p_value=p_value,
            ci_low=float(effect - critical * se),
            ci_high=float(effect + critical * se),
            n_control=int(control.size),
            n_treatment=int(treatment.size),
            diagnostics={
                "estimand_id": "trimmed_mean_effect",
                "ate_alignment": "not_ate",
                "inference": "yuen_welch",
                "theory_grade": "A",
                "trim_fraction": self.trim_fraction,
                "effective_n_control": effective_c,
                "effective_n_treatment": effective_t,
                "degrees_of_freedom": float(df),
            },
        )


@dataclass
class FullRefitWinsorizedMeanEstimator:
    """Difference of sample-quantile-winsorized means with full-refit bootstrap."""

    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    n_resamples: int = 999
    random_state: int = 20260805
    name: str = "winsorized_mean_full_refit"

    def _point(
        self, control: np.ndarray, treatment: np.ndarray, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]]:
        del rng
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
        c_lo, c_hi = np.quantile(control, [self.lower_quantile, self.upper_quantile])
        t_lo, t_hi = np.quantile(treatment, [self.lower_quantile, self.upper_quantile])
        c = np.clip(control, c_lo, c_hi)
        t = np.clip(treatment, t_lo, t_hi)
        return float(t.mean() - c.mean()), {
            "control_bounds": (float(c_lo), float(c_hi)),
            "treatment_bounds": (float(t_lo), float(t_hi)),
            "modified_fraction_control": float(np.mean(c != control)),
            "modified_fraction_treatment": float(np.mean(t != treatment)),
        }

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        return _bootstrap_result(
            self.name,
            control,
            treatment,
            self._point,
            n_resamples=self.n_resamples,
            random_state=self.random_state,
            alpha=spec.alpha,
            diagnostics={
                "estimand_id": "winsorized_mean_effect",
                "ate_alignment": "not_ate",
                "theory_grade": "B",
                "inference_limitation": (
                    "Percentile bootstrap targets the data-adaptive winsorized functional; "
                    "it is not an interval for the unmodified ATE."
                ),
                "lower_quantile": self.lower_quantile,
                "upper_quantile": self.upper_quantile,
            },
        )


@dataclass
class IsolationForestSensitivityEstimator:
    """Isolation-Forest trimming or winsorization with full pipeline refitting."""

    method: str = "winsorize"
    contamination: float = 0.01
    n_estimators: int = 128
    max_samples: int | float | str = "auto"
    n_resamples: int = 999
    random_state: int = 20260805
    name: str = "isolation_forest_sensitivity"

    def _point(
        self, control: np.ndarray, treatment: np.ndarray, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]]:
        if self.method not in {"trim", "winsorize"}:
            raise ValueError("method must be 'trim' or 'winsorize'.")
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5.")
        if self.n_estimators < 32:
            raise ValueError("n_estimators must be at least 32.")
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "IsolationForestSensitivityEstimator requires scikit-learn; "
                "install VarWar with the 'ml' extra."
            ) from exc

        pooled = np.concatenate((control, treatment))
        if pooled.size < 16:
            raise ValueError("Isolation Forest needs at least 16 finite observations.")
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            bootstrap=False,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(pooled.reshape(-1, 1))
        labels = model.predict(pooled.reshape(-1, 1))
        inlier_values = pooled[labels == 1]
        if inlier_values.size < 4:
            raise ValueError("Isolation Forest left too few inliers.")
        lower, upper = float(inlier_values.min()), float(inlier_values.max())
        split = control.size
        c_inlier = labels[:split] == 1
        t_inlier = labels[split:] == 1
        if self.method == "trim":
            c, t = control[c_inlier], treatment[t_inlier]
            if c.size < 2 or t.size < 2:
                raise ValueError("Isolation-Forest trimming left fewer than two values in an arm.")
        else:
            c, t = np.clip(control, lower, upper), np.clip(treatment, lower, upper)
        return float(t.mean() - c.mean()), {
            "iforest_seed": seed,
            "inlier_lower": lower,
            "inlier_upper": upper,
            "modified_fraction_control": float(np.mean(~c_inlier)),
            "modified_fraction_treatment": float(np.mean(~t_inlier)),
        }

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        return _bootstrap_result(
            f"{self.name}_{self.method}",
            control,
            treatment,
            self._point,
            n_resamples=self.n_resamples,
            random_state=self.random_state,
            alpha=spec.alpha,
            diagnostics={
                "estimand_id": f"isolation_forest_{self.method}_effect",
                "ate_alignment": "not_ate",
                "theory_grade": "C",
                "method": self.method,
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "full_refit": True,
                "inference_limitation": (
                    "Exploratory sensitivity estimand defined by a pooled outcome-adaptive "
                    "Isolation Forest; it is not a confirmatory ATE method."
                ),
            },
        )


def _median_of_means(
    values: np.ndarray, n_blocks: int, rng: np.random.Generator
) -> tuple[float, int]:
    blocks = min(max(2, n_blocks), values.size // 2)
    if blocks < 2:
        raise ValueError("Median-of-means needs at least four observations per arm.")
    shuffled = values[rng.permutation(values.size)]
    boundaries = np.linspace(0, shuffled.size, blocks + 1, dtype=int)
    means = np.fromiter(
        (shuffled[boundaries[i] : boundaries[i + 1]].mean() for i in range(blocks)),
        dtype=float,
        count=blocks,
    )
    return float(np.median(means)), blocks


@dataclass
class MedianOfMeansEstimator:
    """Median-of-means difference targeting the original arm means."""

    n_blocks: int | None = None
    n_resamples: int = 999
    random_state: int = 20260805
    name: str = "median_of_means_diff"

    def _point(
        self, control: np.ndarray, treatment: np.ndarray, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]]:
        default_blocks = max(2, int(math.ceil(math.sqrt(min(control.size, treatment.size)))))
        requested = self.n_blocks if self.n_blocks is not None else default_blocks
        c_mean, c_blocks = _median_of_means(control, requested, rng)
        t_mean, t_blocks = _median_of_means(treatment, requested, rng)
        return t_mean - c_mean, {
            "n_blocks_control": c_blocks,
            "n_blocks_treatment": t_blocks,
        }

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        return _bootstrap_result(
            self.name,
            control,
            treatment,
            self._point,
            n_resamples=self.n_resamples,
            random_state=self.random_state,
            alpha=spec.alpha,
            diagnostics={
                "estimand_id": "sample_ate",
                "ate_alignment": "consistent_under_finite_mean_conditions",
                "theory_grade": "C",
                "inference_limitation": (
                    "The point estimator has robust concentration properties, but the "
                    "full-refit conservative bootstrap envelope used here is exploratory "
                    "for the "
                    "non-smooth median-of-means functional."
                ),
            },
            conservative_envelope=True,
        )


def _catoni_psi(value: np.ndarray) -> np.ndarray:
    positive = np.log1p(value + 0.5 * value * value)
    negative = -np.log1p(-value + 0.5 * value * value)
    return np.where(value >= 0.0, positive, negative)


def _robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    if np.isfinite(mad) and mad > np.finfo(float).eps:
        return mad
    standard_deviation = float(values.std(ddof=1))
    return standard_deviation if standard_deviation > 0.0 else 1.0


def _catoni_mean(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    scale = _robust_scale(values)
    delta = max(1e-12, 1.0 - confidence_level)
    tuning = math.sqrt(2.0 * math.log(2.0 / delta) / values.size) / scale

    def equation(location: float) -> float:
        return float(_catoni_psi(tuning * (values - location)).sum())

    spread = max(scale, float(np.ptp(values)), 1.0)
    lower, upper = float(values.min() - spread), float(values.max() + spread)
    location = float(optimize.brentq(equation, lower, upper, maxiter=100))
    return location, tuning


@dataclass
class CatoniMeanEstimator:
    """Catoni M-estimate difference, an exploratory candidate for the original mean."""

    confidence_level_for_tuning: float = 0.95
    n_resamples: int = 999
    random_state: int = 20260805
    name: str = "catoni_mean_diff"

    def _point(
        self, control: np.ndarray, treatment: np.ndarray, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]]:
        del rng
        if not 0.5 < self.confidence_level_for_tuning < 1.0:
            raise ValueError("confidence_level_for_tuning must be between 0.5 and 1.")
        c_mean, c_tuning = _catoni_mean(control, self.confidence_level_for_tuning)
        t_mean, t_tuning = _catoni_mean(treatment, self.confidence_level_for_tuning)
        return t_mean - c_mean, {
            "catoni_tuning_control": c_tuning,
            "catoni_tuning_treatment": t_tuning,
        }

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        return _bootstrap_result(
            self.name,
            control,
            treatment,
            self._point,
            n_resamples=self.n_resamples,
            random_state=self.random_state,
            alpha=spec.alpha,
            diagnostics={
                "estimand_id": "sample_ate",
                "ate_alignment": "consistent_under_finite_variance_conditions",
                "theory_grade": "C",
                "confidence_level_for_tuning": self.confidence_level_for_tuning,
                "inference_limitation": (
                    "Catoni's classical deviation guarantee assumes a valid variance bound. "
                    "This implementation estimates scale from the same arm and uses full-refit "
                    "bootstrap inference, so it remains an R&D candidate, not a certified bound."
                ),
            },
            conservative_envelope=True,
        )


__all__ = [
    "CatoniMeanEstimator",
    "FullRefitWinsorizedMeanEstimator",
    "IsolationForestSensitivityEstimator",
    "MedianOfMeansEstimator",
    "YuenTrimmedMeanEstimator",
]
