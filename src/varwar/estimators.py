from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from varwar.core import EstimateResult, ExperimentSpec, require_columns
from varwar.hypex_backend import HypExBackendUnavailable, estimate_with_hypex_abtest
from varwar.stats import normal_two_sided_p, safe_divide, t_critical, t_two_sided_p


def _float_column(data: pd.DataFrame, column: str) -> np.ndarray:
    return np.asarray(data[column], dtype=float)


def _group_arrays(
    data: pd.DataFrame,
    spec: ExperimentSpec,
    target_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target = target_col or spec.target_col
    require_columns(data, [spec.treatment_col, target])
    y = _float_column(data, target)
    treatment = np.asarray(data[spec.treatment_col])
    valid = np.isfinite(y)
    control_mask = (treatment == spec.control_value) & valid
    treatment_mask = (treatment == spec.treatment_value) & valid
    return y[control_mask], y[treatment_mask]


def _mean_var_n(values: np.ndarray) -> tuple[float, float, int]:
    n = int(values.size)
    if n == 0:
        return math.nan, math.nan, 0
    mean = float(values.mean())
    var = float(values.var(ddof=1)) if n > 1 else math.nan
    return mean, var, n


def _build_result(
    name: str,
    effect: float,
    se: float,
    df: float,
    n_control: int,
    n_treatment: int,
    alpha: float,
    diagnostics: Mapping[str, Any] | None = None,
) -> EstimateResult:
    statistic = safe_divide(effect, se)
    p_value = t_two_sided_p(statistic, df)
    crit = t_critical(alpha, df)
    ci_low = effect - crit * se if math.isfinite(crit) else math.nan
    ci_high = effect + crit * se if math.isfinite(crit) else math.nan
    return EstimateResult(
        estimator=name,
        effect=effect,
        standard_error=se,
        statistic=statistic,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_control=n_control,
        n_treatment=n_treatment,
        diagnostics=diagnostics or {},
    )


@dataclass
class DifferenceInMeansEstimator:
    """Closed-form A/B estimator optimized for the common two-arm case."""

    name: str = "welch_mean_diff"
    variance: str = "welch"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        mean_c, var_c, n_c = _mean_var_n(control)
        mean_t, var_t, n_t = _mean_var_n(treatment)
        if n_c < 2 or n_t < 2:
            raise ValueError("Both groups need at least two finite observations.")

        effect = mean_t - mean_c
        if self.variance == "pooled":
            pooled = ((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)
            se = math.sqrt(pooled * (1.0 / n_c + 1.0 / n_t))
            df = float(n_c + n_t - 2)
        elif self.variance == "welch":
            se2_c = var_c / n_c
            se2_t = var_t / n_t
            se = math.sqrt(se2_c + se2_t)
            denom = (se2_c**2 / (n_c - 1)) + (se2_t**2 / (n_t - 1))
            df = ((se2_c + se2_t) ** 2) / denom if denom > 0 else math.nan
        else:
            raise ValueError(f"Unknown variance policy: {self.variance}")

        return _build_result(
            self.name,
            effect,
            se,
            df,
            n_c,
            n_t,
            spec.alpha,
            {"variance": self.variance},
        )


@dataclass
class RegressionAdjustmentEstimator:
    """OLS adjustment for randomized experiments with numeric covariates.

    Uses a heteroskedasticity-robust HC1 standard error for the treatment
    coefficient. This keeps the implementation closed-form and memory bounded.
    """

    name: str = "regression_adjustment"
    include_pre_target: bool = True

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        features = list(spec.pre_treatment_features())
        if self.include_pre_target and spec.pre_target_col and spec.pre_target_col not in features:
            features.append(spec.pre_target_col)
        require_columns(data, [spec.treatment_col, spec.target_col, *features])

        y = _float_column(data, spec.target_col)
        treatment = (np.asarray(data[spec.treatment_col]) == spec.treatment_value).astype(float)
        columns = [np.ones(len(data), dtype=float), treatment]
        columns.extend(_float_column(data, col) for col in features)
        x = np.column_stack(columns)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        x = x[valid]
        y = y[valid]
        if x.shape[0] <= x.shape[1]:
            raise ValueError("Regression adjustment needs more rows than coefficients.")

        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ beta
        xtx_inv = np.linalg.pinv(x.T @ x)
        meat = x.T @ ((residuals**2)[:, None] * x)
        n, p = x.shape
        hc1 = n / max(n - p, 1)
        cov = hc1 * xtx_inv @ meat @ xtx_inv
        effect = float(beta[1])
        se = math.sqrt(max(float(cov[1, 1]), 0.0))
        n_control = int(((np.asarray(data[spec.treatment_col]) == spec.control_value) & valid).sum())
        n_treatment = int(((np.asarray(data[spec.treatment_col]) == spec.treatment_value) & valid).sum())
        return _build_result(
            self.name,
            effect,
            se,
            float(n - p),
            n_control,
            n_treatment,
            spec.alpha,
            {"features": ",".join(features), "se": "HC1"},
        )


class HypExABTestEstimator(DifferenceInMeansEstimator):
    """Race slot for the HypEx A/B baseline.

    The public HypEx API is not available in this workspace. This adapter uses
    the same estimand and Welch inference natively, and can delegate to an
    injected backend when the internal HypEx callable is available.
    """

    def __init__(
        self,
        backend: Callable[[pd.DataFrame, ExperimentSpec], EstimateResult] | None = None,
        backend_path: str | None = None,
        prefer_pypi: bool = True,
        name: str = "hypex_ab",
    ) -> None:
        super().__init__(name=name, variance="welch")
        self.backend = backend or _load_backend(backend_path)
        self.prefer_pypi = prefer_pypi

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        if self.backend is not None:
            result = self.backend(data, spec)
            return EstimateResult(
                estimator=self.name,
                effect=result.effect,
                standard_error=result.standard_error,
                statistic=result.statistic,
                p_value=result.p_value,
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                n_control=result.n_control,
                n_treatment=result.n_treatment,
                diagnostics={**result.diagnostics, "backend": "external_hypex"},
            )
        if self.prefer_pypi:
            try:
                result = estimate_with_hypex_abtest(data, spec)
                return EstimateResult(
                    estimator=self.name,
                    effect=result.effect,
                    standard_error=result.standard_error,
                    statistic=result.statistic,
                    p_value=result.p_value,
                    ci_low=result.ci_low,
                    ci_high=result.ci_high,
                    n_control=result.n_control,
                    n_treatment=result.n_treatment,
                    diagnostics=result.diagnostics,
                )
            except HypExBackendUnavailable as exc:
                fallback_reason = str(exc)
        else:
            fallback_reason = "pypi backend disabled"
        result = super().estimate(data, spec)
        return EstimateResult(
            estimator=self.name,
            effect=result.effect,
            standard_error=result.standard_error,
            statistic=result.statistic,
            p_value=result.p_value,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            n_control=result.n_control,
            n_treatment=result.n_treatment,
            diagnostics={
                **result.diagnostics,
                "backend": "native_compatible",
                "hypex_fallback_reason": fallback_reason,
            },
        )


@dataclass
class HypExCupacEstimator:
    name: str = "hypex_cupac"
    cupac_features: tuple[str, ...] | None = None

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        features = self.cupac_features or spec.pre_treatment_features()
        try:
            result = estimate_with_hypex_abtest(
                data,
                spec,
                cupac_features={spec.target_col: list(features)},
            )
            return EstimateResult(
                estimator=self.name,
                effect=result.effect,
                standard_error=result.standard_error,
                statistic=result.statistic,
                p_value=result.p_value,
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                n_control=result.n_control,
                n_treatment=result.n_treatment,
                diagnostics={
                    **result.diagnostics,
                    "reference_only": True,
                    "cupac_features": ",".join(features),
                },
            )
        except HypExBackendUnavailable as exc:
            from varwar.core import WrappedEstimator
            from varwar.transforms import LinearCupacTransformer

            local = WrappedEstimator(
                LinearCupacTransformer(feature_cols=features, name="cupac_hypex_reference"),
                HypExABTestEstimator(prefer_pypi=False),
                name=self.name,
            ).estimate(data, spec)
            return EstimateResult(
                estimator=self.name,
                effect=local.effect,
                standard_error=local.standard_error,
                statistic=local.statistic,
                p_value=local.p_value,
                ci_low=local.ci_low,
                ci_high=local.ci_high,
                n_control=local.n_control,
                n_treatment=local.n_treatment,
                diagnostics={
                    **local.diagnostics,
                    "backend": "native_reference_fallback",
                    "hypex_fallback_reason": str(exc),
                },
            )


def _load_backend(
    backend_path: str | None,
) -> Callable[[pd.DataFrame, ExperimentSpec], EstimateResult] | None:
    if backend_path is None:
        import os

        backend_path = os.getenv("VARWAR_HYPEX_BACKEND")
    if not backend_path:
        return None
    module_name, separator, attr_name = backend_path.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(
            "HypEx backend path must use 'module:function' format, "
            "for example 'my_project.hypex_backend:estimate'."
        )
    module = import_module(module_name)
    backend = getattr(module, attr_name)
    if not callable(backend):
        raise TypeError(f"HypEx backend is not callable: {backend_path}")
    return backend


@dataclass
class KSTestEstimator:
    name: str = "ks_test"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        if control.size == 0 or treatment.size == 0:
            raise ValueError("Both groups need finite observations.")
        try:
            from scipy import stats as scipy_stats  # type: ignore

            test = scipy_stats.ks_2samp(control, treatment, method="auto")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        except Exception:
            statistic = _ks_statistic(control, treatment)
            p_value = _ks_asymptotic_pvalue(statistic, control.size, treatment.size)
        return EstimateResult(
            estimator=self.name,
            effect=statistic,
            standard_error=math.nan,
            statistic=statistic,
            p_value=p_value,
            ci_low=math.nan,
            ci_high=math.nan,
            n_control=int(control.size),
            n_treatment=int(treatment.size),
            diagnostics={"effect_definition": "two_sample_ks_statistic"},
        )


@dataclass
class RandomizedTTestEstimator:
    """Klartag-Sodin style randomized t-statistic used in the HSE notebook.

    The implementation is written from the statistic definition, not copied
    from the notebook. It uses deterministic hedgehog weights and Welch
    degrees of freedom.
    """

    name: str = "randomized_t_test"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        if control.size < 2 or treatment.size < 2:
            raise ValueError("Both groups need at least two finite observations.")
        mean_c, var_c, n_c = _mean_var_n(control)
        mean_t, var_t, n_t = _mean_var_n(treatment)
        theta_t = _hedgehog_weights(n_t)
        theta_c = _hedgehog_weights(n_c)
        numerator = float(theta_t @ treatment - theta_c @ control)
        denom = math.sqrt(
            var_t * float(theta_t @ theta_t) + var_c * float(theta_c @ theta_c)
        )
        statistic = safe_divide(numerator, denom)
        se2_c = var_c / n_c
        se2_t = var_t / n_t
        df_denom = (se2_c**2 / (n_c - 1)) + (se2_t**2 / (n_t - 1))
        df = ((se2_c + se2_t) ** 2) / df_denom if df_denom > 0 else math.nan
        p_value = t_two_sided_p(statistic, df)
        effect = mean_t - mean_c
        se = math.sqrt(se2_c + se2_t)
        crit = t_critical(spec.alpha, df)
        return EstimateResult(
            estimator=self.name,
            effect=effect,
            standard_error=se,
            statistic=statistic,
            p_value=p_value,
            ci_low=effect - crit * se if math.isfinite(crit) else math.nan,
            ci_high=effect + crit * se if math.isfinite(crit) else math.nan,
            n_control=n_c,
            n_treatment=n_t,
            diagnostics={"weights": "hedgehog", "df": df},
        )


def _hedgehog_weights(n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=float)
    prefix_repeats = int(math.ceil(n**0.75))
    prefix = np.tile(np.array([1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0)]), prefix_repeats)
    if prefix.size >= n:
        return prefix[:n]
    suffix_count = n - prefix.size
    suffix = np.resize(np.array([1.0, math.sqrt(2.0), -1.0, -math.sqrt(2.0)]), suffix_count)
    return np.concatenate([prefix, suffix])


def _ks_statistic(control: np.ndarray, treatment: np.ndarray) -> float:
    control_sorted = np.sort(control)
    treatment_sorted = np.sort(treatment)
    values = np.sort(np.concatenate([control_sorted, treatment_sorted]))
    cdf_c = np.searchsorted(control_sorted, values, side="right") / control_sorted.size
    cdf_t = np.searchsorted(treatment_sorted, values, side="right") / treatment_sorted.size
    return float(np.max(np.abs(cdf_t - cdf_c)))


def _ks_asymptotic_pvalue(statistic: float, n_control: int, n_treatment: int) -> float:
    if statistic < 0 or n_control <= 0 or n_treatment <= 0:
        return math.nan
    effective_n = n_control * n_treatment / (n_control + n_treatment)
    lam = (math.sqrt(effective_n) + 0.12 + 0.11 / math.sqrt(effective_n)) * statistic
    terms = [(-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101)]
    return float(min(max(2.0 * sum(terms), 0.0), 1.0))


@dataclass
class PermutationMeanTestEstimator:
    name: str = "permutation_mean_test"
    n_resamples: int = 999
    seed: int = 20260709

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        if control.size < 2 or treatment.size < 2:
            raise ValueError("Both groups need at least two finite observations.")
        observed = float(treatment.mean() - control.mean())
        pooled = np.concatenate([control, treatment])
        n_control = int(control.size)
        n_treatment = int(treatment.size)
        rng = np.random.default_rng(self.seed)
        shuffled = pooled.copy()
        extreme = 0
        permuted_effects = np.empty(self.n_resamples, dtype=float)
        for i in range(self.n_resamples):
            rng.shuffle(shuffled)
            perm_effect = float(shuffled[n_control:].mean() - shuffled[:n_control].mean())
            permuted_effects[i] = perm_effect
            extreme += abs(perm_effect) >= abs(observed)
        p_value = (extreme + 1.0) / (self.n_resamples + 1.0)
        se = float(permuted_effects.std(ddof=1)) if self.n_resamples > 1 else math.nan
        statistic = safe_divide(observed, se)
        return EstimateResult(
            estimator=self.name,
            effect=observed,
            standard_error=se,
            statistic=statistic,
            p_value=p_value,
            ci_low=math.nan,
            ci_high=math.nan,
            n_control=n_control,
            n_treatment=n_treatment,
            diagnostics={"n_resamples": self.n_resamples},
        )


@dataclass
class TrimmedMeanEstimator:
    name: str = "trimmed_mean_diff"
    trim_fraction: float = 0.1

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        control_trimmed = _trim_values(control, self.trim_fraction)
        treatment_trimmed = _trim_values(treatment, self.trim_fraction)
        mean_c, var_c, n_c = _mean_var_n(control_trimmed)
        mean_t, var_t, n_t = _mean_var_n(treatment_trimmed)
        if n_c < 2 or n_t < 2:
            raise ValueError("Trimmed groups need at least two observations.")
        effect = mean_t - mean_c
        se2_c = var_c / n_c
        se2_t = var_t / n_t
        se = math.sqrt(se2_c + se2_t)
        denom = (se2_c**2 / (n_c - 1)) + (se2_t**2 / (n_t - 1))
        df = ((se2_c + se2_t) ** 2) / denom if denom > 0 else math.nan
        return _build_result(
            self.name,
            effect,
            se,
            df,
            n_c,
            n_t,
            spec.alpha,
            {"trim_fraction": self.trim_fraction},
        )


@dataclass
class WinsorizedMeanEstimator:
    name: str = "winsorized_mean_diff"
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        c = _winsorize_values(control, self.lower_quantile, self.upper_quantile)
        t = _winsorize_values(treatment, self.lower_quantile, self.upper_quantile)
        mean_c, var_c, n_c = _mean_var_n(c)
        mean_t, var_t, n_t = _mean_var_n(t)
        if n_c < 2 or n_t < 2:
            raise ValueError("Winsorized groups need at least two observations.")
        effect = mean_t - mean_c
        se2_c = var_c / n_c
        se2_t = var_t / n_t
        se = math.sqrt(se2_c + se2_t)
        denom = (se2_c**2 / (n_c - 1)) + (se2_t**2 / (n_t - 1))
        df = ((se2_c + se2_t) ** 2) / denom if denom > 0 else math.nan
        return _build_result(
            self.name,
            effect,
            se,
            df,
            n_c,
            n_t,
            spec.alpha,
            {
                "lower_quantile": self.lower_quantile,
                "upper_quantile": self.upper_quantile,
            },
        )


def _trim_values(values: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 <= fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5).")
    sorted_values = np.sort(values)
    k = int(math.floor(sorted_values.size * fraction))
    if k == 0:
        return sorted_values
    return sorted_values[k:-k]


def _winsorize_values(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not 0 <= lower < upper <= 1:
        raise ValueError("Winsorization quantiles must satisfy 0 <= lower < upper <= 1.")
    lo, hi = np.quantile(values, [lower, upper])
    return np.clip(values, lo, hi)


@dataclass
class RatioDeltaMethodEstimator:
    name: str = "ratio_delta"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        require_columns(data, [spec.treatment_col, spec.numerator_col, spec.denominator_col])
        if spec.numerator_col is None or spec.denominator_col is None:
            raise ValueError("Ratio estimator requires numerator_col and denominator_col.")

        treatment = np.asarray(data[spec.treatment_col])
        num = _float_column(data, spec.numerator_col)
        den = _float_column(data, spec.denominator_col)
        valid = np.isfinite(num) & np.isfinite(den)

        def group(mask: np.ndarray) -> tuple[float, float, int]:
            n_values = num[mask]
            d_values = den[mask]
            n_obs = int(n_values.size)
            if n_obs < 2:
                raise ValueError("Each ratio group needs at least two observations.")
            mean_den = float(d_values.mean())
            if mean_den == 0:
                raise ValueError("Mean denominator is zero.")
            ratio = float(n_values.mean() / mean_den)
            residual = n_values - ratio * d_values
            var = float(residual.var(ddof=1) / (n_obs * mean_den**2))
            return ratio, var, n_obs

        ratio_c, var_c, n_c = group((treatment == spec.control_value) & valid)
        ratio_t, var_t, n_t = group((treatment == spec.treatment_value) & valid)
        effect = ratio_t - ratio_c
        se = math.sqrt(var_c + var_t)
        df = float(n_c + n_t - 2)
        return _build_result(
            self.name,
            effect,
            se,
            df,
            n_c,
            n_t,
            spec.alpha,
            {"metric": "ratio_delta_method"},
        )


@dataclass
class StratifiedDifferenceInMeansEstimator:
    name: str = "stratified_mean_diff"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        require_columns(data, [spec.treatment_col, spec.target_col, spec.stratum_col])
        if spec.stratum_col is None:
            raise ValueError("Stratified estimator requires stratum_col.")

        y = _float_column(data, spec.target_col)
        treatment = np.asarray(data[spec.treatment_col])
        strata = np.asarray(data[spec.stratum_col])
        valid = np.isfinite(y)
        total = int(valid.sum())
        if total == 0:
            raise ValueError("No finite target values.")

        effect = 0.0
        variance = 0.0
        n_c_total = 0
        n_t_total = 0
        skipped = 0
        used = 0
        for stratum in np.unique(strata[valid]):
            mask = valid & (strata == stratum)
            weight = float(mask.sum() / total)
            c = y[mask & (treatment == spec.control_value)]
            t = y[mask & (treatment == spec.treatment_value)]
            if c.size < 2 or t.size < 2:
                skipped += 1
                continue
            mean_c, var_c, n_c = _mean_var_n(c)
            mean_t, var_t, n_t = _mean_var_n(t)
            effect += weight * (mean_t - mean_c)
            variance += weight**2 * (var_c / n_c + var_t / n_t)
            n_c_total += n_c
            n_t_total += n_t
            used += 1

        if used == 0:
            raise ValueError("No stratum has enough observations in both groups.")

        se = math.sqrt(variance)
        df = float(max(n_c_total + n_t_total - 2 * used, 1))
        return _build_result(
            self.name,
            effect,
            se,
            df,
            n_c_total,
            n_t_total,
            spec.alpha,
            {"strata_used": used, "strata_skipped": skipped},
        )


@dataclass
class VarianceWeightedStratifiedEstimator:
    name: str = "variance_weighted_stratified"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        require_columns(data, [spec.treatment_col, spec.target_col, spec.stratum_col])
        if spec.stratum_col is None:
            raise ValueError("Variance-weighted estimator requires stratum_col.")

        y = _float_column(data, spec.target_col)
        treatment = np.asarray(data[spec.treatment_col])
        strata = np.asarray(data[spec.stratum_col])
        valid = np.isfinite(y)
        effects: list[float] = []
        variances: list[float] = []
        n_control = 0
        n_treatment = 0
        skipped = 0
        for stratum in np.unique(strata[valid]):
            mask = valid & (strata == stratum)
            c = y[mask & (treatment == spec.control_value)]
            t = y[mask & (treatment == spec.treatment_value)]
            if c.size < 2 or t.size < 2:
                skipped += 1
                continue
            mean_c, var_c, n_c = _mean_var_n(c)
            mean_t, var_t, n_t = _mean_var_n(t)
            var_effect = var_c / n_c + var_t / n_t
            if var_effect <= 0 or not math.isfinite(var_effect):
                skipped += 1
                continue
            effects.append(mean_t - mean_c)
            variances.append(var_effect)
            n_control += n_c
            n_treatment += n_t

        if not effects:
            raise ValueError("No stratum has a finite effect variance.")
        inv_var = 1.0 / np.asarray(variances, dtype=float)
        weights = inv_var / inv_var.sum()
        effect = float(np.dot(weights, np.asarray(effects, dtype=float)))
        se = math.sqrt(float(np.sum((weights**2) * np.asarray(variances, dtype=float))))
        df = float(max(n_control + n_treatment - 2 * len(effects), 1))
        return _build_result(
            self.name,
            effect,
            se,
            df,
            n_control,
            n_treatment,
            spec.alpha,
            {"strata_used": len(effects), "strata_skipped": skipped},
        )


@dataclass
class WaldEstimator:
    name: str = "wald_late"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        require_columns(
            data,
            [spec.instrument_col, spec.received_treatment_col, spec.target_col],
        )
        if spec.instrument_col is None or spec.received_treatment_col is None:
            raise ValueError("Wald estimator requires instrument and received treatment columns.")

        z = np.asarray(data[spec.instrument_col])
        y = _float_column(data, spec.target_col)
        d = _float_column(data, spec.received_treatment_col)
        valid = np.isfinite(y) & np.isfinite(d)
        z0 = (z == spec.control_value) & valid
        z1 = (z == spec.treatment_value) & valid
        if z0.sum() < 2 or z1.sum() < 2:
            raise ValueError("Both instrument groups need at least two observations.")

        y0, y1 = y[z0], y[z1]
        d0, d1 = d[z0], d[z1]
        itt_y = float(y1.mean() - y0.mean())
        itt_d = float(d1.mean() - d0.mean())
        if abs(itt_d) < 1e-12:
            raise ValueError("First stage is too close to zero for Wald estimation.")

        effect = itt_y / itt_d
        n0 = int(z0.sum())
        n1 = int(z1.sum())
        var_y = float(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)
        var_d = float(d1.var(ddof=1) / n1 + d0.var(ddof=1) / n0)
        cov_y_d = float(
            np.cov(y1, d1, ddof=1)[0, 1] / n1 + np.cov(y0, d0, ddof=1)[0, 1] / n0
        )
        variance = var_y / itt_d**2 + (itt_y**2 * var_d) / itt_d**4
        variance -= 2.0 * itt_y * cov_y_d / itt_d**3
        se = math.sqrt(max(variance, 0.0))
        statistic = safe_divide(effect, se)
        p_value = normal_two_sided_p(statistic)
        crit = 1.959963984540054 if abs(spec.alpha - 0.05) < 1e-12 else 1.959963984540054
        return EstimateResult(
            estimator=self.name,
            effect=effect,
            standard_error=se,
            statistic=statistic,
            p_value=p_value,
            ci_low=effect - crit * se,
            ci_high=effect + crit * se,
            n_control=n0,
            n_treatment=n1,
            diagnostics={"itt_y": itt_y, "itt_d": itt_d},
        )


@dataclass
class PlaceholderEstimator:
    name: str
    reason: str

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        raise NotImplementedError(f"{self.name} is not implemented: {self.reason}")
