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


def _feature_matrix(data: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([_float_column(data, col) for col in features])


def _cross_fitted_ridge_predictions(
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    ridge: float,
    seed: int,
) -> np.ndarray:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    folds = np.array_split(indices, min(n_splits, x.shape[0]))
    predictions = np.empty(x.shape[0], dtype=float)
    all_indices = np.arange(x.shape[0])
    for fold in folds:
        train_mask = np.ones(x.shape[0], dtype=bool)
        train_mask[fold] = False
        train_idx = all_indices[train_mask]
        predictions[fold] = _ridge_predict(
            x[train_idx],
            y[train_idx],
            x[fold],
            ridge,
        )
    return predictions


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    ridge: float,
) -> np.ndarray:
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0, ddof=0)
    x_std = np.where(x_std > 0.0, x_std, 1.0)
    x_centered = (x_train - x_mean) / x_std
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    xtx = x_centered.T @ x_centered
    penalty = max(float(ridge), 0.0) * np.eye(xtx.shape[0], dtype=float)
    coef = np.linalg.solve(xtx + penalty, x_centered.T @ y_centered)
    return y_mean + ((x_predict - x_mean) / x_std) @ coef


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


@dataclass
class MLRateEstimator:
    """Cross-fitted machine-learning regression adjustment.

    The nuisance model is a ridge predictor over numeric covariates. The final
    stage regresses the target on treatment, out-of-fold prediction, and the
    treatment-by-prediction interaction with HC1 inference.
    """

    name: str = "ml_rate"
    n_splits: int = 5
    ridge: float = 1e-6
    seed: int = 20260710
    include_interaction: bool = True

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        features = tuple(spec.pre_treatment_features())
        require_columns(data, [spec.treatment_col, spec.target_col, *features])
        if not features:
            raise ValueError("MLRateEstimator requires numeric pre-treatment features.")

        y = _float_column(data, spec.target_col)
        treatment_raw = np.asarray(data[spec.treatment_col])
        treatment = (treatment_raw == spec.treatment_value).astype(float)
        x = _feature_matrix(data, features)
        valid = (
            np.isfinite(y)
            & np.isfinite(x).all(axis=1)
            & (
                (treatment_raw == spec.control_value)
                | (treatment_raw == spec.treatment_value)
            )
        )
        y_valid = y[valid]
        treatment_valid = treatment[valid]
        x_valid = x[valid]
        if y_valid.size <= max(self.n_splits, x_valid.shape[1] + 2):
            raise ValueError("MLRateEstimator needs more rows for cross-fitting.")
        predictions = _cross_fitted_ridge_predictions(
            x_valid,
            y_valid,
            n_splits=self.n_splits,
            ridge=self.ridge,
            seed=self.seed,
        )
        centered_pred = predictions - float(predictions.mean())
        columns = [
            np.ones(y_valid.size, dtype=float),
            treatment_valid,
            centered_pred,
        ]
        if self.include_interaction:
            columns.append(treatment_valid * centered_pred)
        design = np.column_stack(columns)
        beta, *_ = np.linalg.lstsq(design, y_valid, rcond=None)
        residuals = y_valid - design @ beta
        xtx_inv = np.linalg.pinv(design.T @ design)
        meat = design.T @ ((residuals**2)[:, None] * design)
        n, p = design.shape
        hc1 = n / max(n - p, 1)
        cov = hc1 * xtx_inv @ meat @ xtx_inv
        effect = float(beta[1])
        se = math.sqrt(max(float(cov[1, 1]), 0.0))
        n_control = int((treatment_valid == 0.0).sum())
        n_treatment = int((treatment_valid == 1.0).sum())
        return _build_result(
            self.name,
            effect,
            se,
            float(n - p),
            n_control,
            n_treatment,
            spec.alpha,
            {
                "features": ",".join(features),
                "n_splits": self.n_splits,
                "ridge": self.ridge,
                "include_interaction": self.include_interaction,
                "prediction_variance": float(predictions.var(ddof=1)),
                "se": "HC1",
            },
        )


class HypExABTestEstimator(DifferenceInMeansEstimator):
    """Race slot for the HypEx A/B baseline.

    The adapter prefers the public PyPI HypEx API. If the optional backend is
    unavailable, it falls back to the same estimand and Welch inference
    implemented locally. A project-specific callable can also be injected.
    """

    def __init__(
        self,
        backend: Callable[[pd.DataFrame, ExperimentSpec], EstimateResult] | None = None,
        backend_path: str | None = None,
        prefer_pypi: bool = True,
        require_pypi: bool = False,
        name: str = "hypex_ab",
    ) -> None:
        super().__init__(name=name, variance="welch")
        self.backend = backend or _load_backend(backend_path)
        self.prefer_pypi = prefer_pypi
        self.require_pypi = require_pypi
        if self.require_pypi and not self.prefer_pypi and self.backend is None:
            raise ValueError("require_pypi=True requires prefer_pypi=True.")

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
                if self.require_pypi:
                    raise RuntimeError(
                        f"Required PyPI HypEx backend failed: {exc}"
                    ) from exc
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
class HypExCupedEstimator:
    """PyPI HypEx CUPED followed by its default Welch A/B test.

    HypEx estimates the CUPED coefficient on the complete experiment dataset.
    The local fallback reproduces that data policy explicitly.
    """

    name: str = "hypex_cuped"
    covariate_col: str | None = None
    prefer_pypi: bool = True
    require_pypi: bool = False

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        covariate = self.covariate_col or spec.pre_target_col
        if covariate is None:
            raise ValueError("HypEx CUPED requires a pre-treatment covariate.")
        try:
            if not self.prefer_pypi:
                raise HypExBackendUnavailable("pypi backend disabled")
            result = estimate_with_hypex_abtest(
                data,
                spec,
                cuped_features={spec.target_col: covariate},
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
                    "cuped_covariate": covariate,
                    "cuped_fit_data": "experiment_all",
                },
            )
        except HypExBackendUnavailable as exc:
            if self.require_pypi:
                raise RuntimeError(
                    f"Required PyPI HypEx CUPED backend failed: {exc}"
                ) from exc
            from varwar.core import WrappedEstimator
            from varwar.transforms import CupedTransformer

            local = WrappedEstimator(
                CupedTransformer(
                    covariate_col=covariate,
                    fit_data="experiment_all",
                    name="hypex_cuped_native_policy",
                ),
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
                    "cuped_fit_data": "experiment_all",
                },
            )


@dataclass
class HypExCupacEstimator:
    """PyPI HypEx temporal linear CUPAC followed by Welch inference."""

    name: str = "hypex_cupac_temporal"
    cupac_features: tuple[str, ...] | None = None
    prefer_pypi: bool = True
    require_pypi: bool = False

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        features = self.cupac_features or spec.pre_treatment_features()
        try:
            if not self.prefer_pypi:
                raise HypExBackendUnavailable("pypi backend disabled")
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
                    "cupac_contract": "temporal_lag2_to_lag1_then_lag1_to_current",
                    "current_outcome_used_for_model_fit": False,
                    "current_outcome_used_for_theta": True,
                    "cupac_features": ",".join(features),
                },
            )
        except HypExBackendUnavailable as exc:
            if self.require_pypi:
                raise RuntimeError(
                    f"Required PyPI HypEx CUPAC backend failed: {exc}"
                ) from exc
            from varwar.core import WrappedEstimator
            from varwar.transforms import TemporalLinearCupacTransformer

            local = WrappedEstimator(
                TemporalLinearCupacTransformer(
                    feature_cols=features,
                    name="cupac_temporal_linear_fallback",
                ),
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
                    "cupac_contract": "temporal_lag2_to_lag1_then_lag1_to_current",
                    "current_outcome_used_for_model_fit": False,
                    "current_outcome_used_for_theta": True,
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
class ApproxKSTestEstimator:
    """Fast histogram approximation of the two-sample KS statistic.

    This is an explicit approximation for large synthetic sweeps. It avoids
    sorting the full pooled sample and uses a fixed number of bins instead.
    """

    name: str = "approx_ks_fast"
    n_bins: int = 2048

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        if control.size == 0 or treatment.size == 0:
            raise ValueError("Both groups need finite observations.")
        statistic = _approx_ks_statistic(control, treatment, self.n_bins)
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
            diagnostics={
                "effect_definition": "histogram_approx_two_sample_ks",
                "n_bins": self.n_bins,
            },
        )


@dataclass
class EnergyDistanceTestEstimator:
    """One-dimensional energy two-sample test with permutation p-value."""

    name: str = "energy_test"
    n_resamples: int = 199
    seed: int = 20260710

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        if control.size < 2 or treatment.size < 2:
            raise ValueError("Both groups need at least two finite observations.")
        statistic = _energy_distance(control, treatment)
        pooled = np.concatenate([control, treatment])
        n_control = int(control.size)
        rng = np.random.default_rng(self.seed)
        shuffled = pooled.copy()
        permuted = np.empty(self.n_resamples, dtype=float)
        extreme = 0
        for i in range(self.n_resamples):
            rng.shuffle(shuffled)
            perm_stat = _energy_distance(shuffled[:n_control], shuffled[n_control:])
            permuted[i] = perm_stat
            extreme += perm_stat >= statistic
        p_value = (extreme + 1.0) / (self.n_resamples + 1.0)
        se = float(permuted.std(ddof=1)) if self.n_resamples > 1 else math.nan
        return EstimateResult(
            estimator=self.name,
            effect=statistic,
            standard_error=se,
            statistic=safe_divide(statistic, se),
            p_value=p_value,
            ci_low=math.nan,
            ci_high=math.nan,
            n_control=n_control,
            n_treatment=int(treatment.size),
            diagnostics={
                "effect_definition": "energy_distance_1d",
                "n_resamples": self.n_resamples,
            },
        )


@dataclass
class RandomizedTTestEstimator:
    """Randomized t-statistic with deterministic hedgehog weights.

    The statistic uses alternating normalized weights within each arm and
    Welch degrees of freedom for its two-sided p-value.
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


@dataclass
class RandomizedWeightedMeanDiffEstimator:
    """Оценивает разность средних с воспроизводимыми ёжиковыми весами.

    В каждой экспериментальной группе наблюдения сначала перемешиваются с
    контролируемым seed. Затем к ним применяются те же знакопеременные
    коэффициенты, что и в ``randomized_t_test``, но после нормировки до суммы
    единица. Поэтому точечная оценка, стандартная ошибка, p-value и доверительный
    интервал относятся к одной и той же взвешенной разности средних:

    ``mu_g = sum_i w_gi * Y_gi``, где ``sum_i w_gi = 1``.

    При весах, не зависящих от target, оценка сохраняет математическое ожидание
    среднего группы. Данные не изменяются на месте. Если задан ``unit_col``,
    он используется для стабильного порядка перед перемешиванием; иначе при
    уникальном индексе сначала используется индекс DataFrame.
    """

    name: str = "randomized_weighted_mean_diff"
    seed: int = 20260731
    shuffle: bool = True

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        if spec.unit_col is not None:
            require_columns(data, [spec.unit_col])
            ordered = data.sort_values(spec.unit_col, kind="mergesort")
            order_policy = "stable_unit_sort_then_seeded_shuffle"
        elif data.index.is_unique:
            ordered = data.sort_index(kind="stable")
            order_policy = "stable_index_sort_then_seeded_shuffle"
        else:
            ordered = data
            order_policy = "input_order_then_seeded_shuffle"

        control, treatment = _group_arrays(ordered, spec)
        if control.size < 2 or treatment.size < 2:
            raise ValueError("Both groups need at least two finite observations.")

        rng = np.random.default_rng(self.seed)
        if self.shuffle:
            control = control[rng.permutation(control.size)]
            treatment = treatment[rng.permutation(treatment.size)]

        raw_weights_c = _hedgehog_weights(control.size)
        raw_weights_t = _hedgehog_weights(treatment.size)
        sum_weights_c = float(raw_weights_c.sum())
        sum_weights_t = float(raw_weights_t.sum())
        if sum_weights_c == 0.0 or sum_weights_t == 0.0:
            raise ValueError("Hedgehog weights must have non-zero sums.")
        weights_c = raw_weights_c / sum_weights_c
        weights_t = raw_weights_t / sum_weights_t

        mean_c, var_c, n_c = _mean_var_n(control)
        mean_t, var_t, n_t = _mean_var_n(treatment)
        effect = float(weights_t @ treatment - weights_c @ control)
        term_c = var_c * float(weights_c @ weights_c)
        term_t = var_t * float(weights_t @ weights_t)
        variance = term_c + term_t
        se = math.sqrt(max(variance, 0.0))
        denominator = (term_c**2 / (n_c - 1)) + (term_t**2 / (n_t - 1))
        df = variance**2 / denominator if denominator > 0.0 else math.nan
        return _build_result(
            self.name,
            effect,
            se,
            df,
            n_c,
            n_t,
            spec.alpha,
            {
                "weights": "normalized_hedgehog",
                "seed": self.seed,
                "shuffle": self.shuffle,
                "order_policy": order_policy,
                "weight_sum_control": sum_weights_c,
                "weight_sum_treatment": sum_weights_t,
                "effective_n_control": float(1.0 / (weights_c @ weights_c)),
                "effective_n_treatment": float(1.0 / (weights_t @ weights_t)),
                "unweighted_mean_difference": float(mean_t - mean_c),
                "estimation_target": "difference_of_normalized_hedgehog_weighted_means",
            },
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


def _approx_ks_statistic(control: np.ndarray, treatment: np.ndarray, n_bins: int) -> float:
    if n_bins < 8:
        raise ValueError("n_bins must be at least 8.")
    lower = float(min(control.min(), treatment.min()))
    upper = float(max(control.max(), treatment.max()))
    if lower == upper:
        return 0.0
    edges = np.linspace(lower, upper, num=n_bins + 1, dtype=float)
    hist_c, _ = np.histogram(control, bins=edges)
    hist_t, _ = np.histogram(treatment, bins=edges)
    cdf_c = np.cumsum(hist_c, dtype=float) / control.size
    cdf_t = np.cumsum(hist_t, dtype=float) / treatment.size
    return float(np.max(np.abs(cdf_t - cdf_c)))


def _ks_asymptotic_pvalue(statistic: float, n_control: int, n_treatment: int) -> float:
    if statistic < 0 or n_control <= 0 or n_treatment <= 0:
        return math.nan
    effective_n = n_control * n_treatment / (n_control + n_treatment)
    lam = (math.sqrt(effective_n) + 0.12 + 0.11 / math.sqrt(effective_n)) * statistic
    terms = [(-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101)]
    return float(min(max(2.0 * sum(terms), 0.0), 1.0))


def _energy_distance(control: np.ndarray, treatment: np.ndarray) -> float:
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(scipy_stats.energy_distance(control, treatment))
    except Exception:
        cross = _mean_abs_cross(control, treatment)
        within_c = _mean_abs_cross(control, control)
        within_t = _mean_abs_cross(treatment, treatment)
        return math.sqrt(max(2.0 * cross - within_c - within_t, 0.0))


def _mean_abs_cross(left: np.ndarray, right: np.ndarray) -> float:
    right_sorted = np.sort(right)
    prefix = np.concatenate([[0.0], np.cumsum(right_sorted, dtype=float)])
    positions = np.searchsorted(right_sorted, left, side="right")
    sum_le = prefix[positions]
    count_le = positions.astype(float)
    total = prefix[-1]
    count_right = float(right_sorted.size)
    distances = left * count_le - sum_le
    distances += (total - sum_le) - left * (count_right - count_le)
    return float(distances.sum() / (left.size * right_sorted.size))


def _permutation_mean_differences(
    control: np.ndarray,
    treatment: np.ndarray,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """Return a bounded-memory Monte Carlo randomization distribution."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    pooled = np.concatenate([control, treatment])
    n_control = int(control.size)
    rng = np.random.default_rng(seed)
    shuffled = pooled.copy()
    effects = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        rng.shuffle(shuffled)
        effects[index] = float(
            shuffled[n_control:].mean() - shuffled[:n_control].mean()
        )
    return effects


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
        n_control = int(control.size)
        n_treatment = int(treatment.size)
        permuted_effects = _permutation_mean_differences(
            control,
            treatment,
            self.n_resamples,
            self.seed,
        )
        extreme = int(np.count_nonzero(np.abs(permuted_effects) >= abs(observed)))
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
class PermutationMeanDiffEstimator:
    """Estimate a mean difference with permutation-calibrated inference.

    The confidence interval is obtained by inverting the two-sided
    randomization test: null-distribution quantiles are reflected around the
    observed effect. This is justified for exchangeable outcomes under the
    sharp null and, for an effect interval, under a constant additive effect.
    """

    name: str = "permutation_mean_diff"
    n_resamples: int = 999
    seed: int = 20260709

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        mean_c, _, n_c = _mean_var_n(control)
        mean_t, _, n_t = _mean_var_n(treatment)
        if n_c < 2 or n_t < 2:
            raise ValueError("Both groups need at least two finite observations.")

        effect = mean_t - mean_c
        permuted = _permutation_mean_differences(
            control,
            treatment,
            self.n_resamples,
            self.seed,
        )
        extreme = int(np.count_nonzero(np.abs(permuted) >= abs(effect)))
        permutation_p = (extreme + 1.0) / (self.n_resamples + 1.0)
        se = float(permuted.std(ddof=1))
        lower_null, upper_null = np.quantile(
            permuted,
            [spec.alpha / 2.0, 1.0 - spec.alpha / 2.0],
        )
        return EstimateResult(
            estimator=self.name,
            effect=effect,
            standard_error=se,
            statistic=safe_divide(effect, se),
            p_value=permutation_p,
            ci_low=float(effect - upper_null),
            ci_high=float(effect - lower_null),
            n_control=n_c,
            n_treatment=n_t,
            diagnostics={
                "n_resamples": self.n_resamples,
                "seed": self.seed,
                "p_value_method": "permutation_randomization",
                "ci_method": "permutation_test_inversion_shifted",
                "exchangeability_assumption": "sharp_null_or_constant_additive_effect",
            },
        )


@dataclass
class BootstrapMeanDiffEstimator:
    """Bootstrap mean-difference estimator with percentile confidence interval."""

    name: str = "bootstrap_mean_diff"
    n_resamples: int = 999
    seed: int = 20260709

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        if control.size < 2 or treatment.size < 2:
            raise ValueError("Both groups need at least two finite observations.")
        if self.n_resamples < 2:
            raise ValueError("n_resamples must be at least two.")

        effect = float(treatment.mean() - control.mean())
        rng = np.random.default_rng(self.seed)
        bootstrap_effects = np.empty(self.n_resamples, dtype=float)
        for index in range(self.n_resamples):
            control_indices = rng.integers(0, control.size, size=control.size)
            treatment_indices = rng.integers(0, treatment.size, size=treatment.size)
            bootstrap_effects[index] = float(
                treatment[treatment_indices].mean()
                - control[control_indices].mean()
            )
        se = float(bootstrap_effects.std(ddof=1))
        lower, upper = np.quantile(
            bootstrap_effects,
            [spec.alpha / 2.0, 1.0 - spec.alpha / 2.0],
        )
        return EstimateResult(
            estimator=self.name,
            effect=effect,
            standard_error=se,
            statistic=safe_divide(effect, se),
            p_value=normal_two_sided_p(safe_divide(effect, se)),
            ci_low=float(lower),
            ci_high=float(upper),
            n_control=int(control.size),
            n_treatment=int(treatment.size),
            diagnostics={
                "n_resamples": self.n_resamples,
                "seed": self.seed,
                "ci_method": "percentile_bootstrap",
                "p_value_method": "normal_approximation_from_bootstrap_se",
            },
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


def _nearest_neighbor_indices(
    query: np.ndarray,
    reference: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        _, indices = cKDTree(reference).query(query, k=1, workers=1)
        return np.asarray(indices, dtype=int)
    except Exception:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        result = np.empty(query.shape[0], dtype=int)
        for start in range(0, query.shape[0], chunk_size):
            stop = min(start + chunk_size, query.shape[0])
            chunk = query[start:stop]
            distances = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            result[start:stop] = np.argmin(distances, axis=1)
        return result


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
    sparse_policy: str = "error"
    stratum_weights: Mapping[Any, float] | None = None

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

        components: list[tuple[float, float, float, float, float, int, int]] = []
        n_c_total = 0
        n_t_total = 0
        skipped = 0
        used = 0
        for stratum in np.unique(strata[valid]):
            mask = valid & (strata == stratum)
            if self.stratum_weights is None:
                weight = float(mask.sum() / total)
            else:
                if stratum not in self.stratum_weights:
                    raise ValueError(f"Missing reference weight for stratum {stratum!r}.")
                weight = float(self.stratum_weights[stratum])
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError("Stratum weights must be finite and non-negative.")
            c = y[mask & (treatment == spec.control_value)]
            t = y[mask & (treatment == spec.treatment_value)]
            if c.size < 2 or t.size < 2:
                skipped += 1
                continue
            mean_c, var_c, n_c = _mean_var_n(c)
            mean_t, var_t, n_t = _mean_var_n(t)
            term_c = var_c / n_c
            term_t = var_t / n_t
            components.append(
                (
                    weight,
                    mean_t - mean_c,
                    term_c + term_t,
                    term_c,
                    term_t,
                    n_c,
                    n_t,
                )
            )
            n_c_total += n_c
            n_t_total += n_t
            used += 1

        if used == 0:
            raise ValueError("No stratum has enough observations in both groups.")
        if self.sparse_policy not in {"error", "drop_renormalize"}:
            raise ValueError("sparse_policy must be 'error' or 'drop_renormalize'.")
        if skipped and self.sparse_policy == "error":
            raise ValueError(
                f"{skipped} strata lack two finite observations in both groups."
            )

        used_weight = sum(component[0] for component in components)
        if used_weight <= 0.0:
            raise ValueError("Used stratum weights must have a positive sum.")
        if self.stratum_weights is not None and skipped == 0:
            if not math.isclose(used_weight, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Reference stratum weights must sum to one.")
        normalized = [
            (
                weight / used_weight,
                stratum_effect,
                stratum_variance,
                term_c,
                term_t,
                n_c,
                n_t,
            )
            for (
                weight,
                stratum_effect,
                stratum_variance,
                term_c,
                term_t,
                n_c,
                n_t,
            ) in components
        ]
        effect = sum(weight * value for weight, value, *_ in normalized)
        variance = sum(
            weight**2 * stratum_variance
            for weight, _, stratum_variance, *_ in normalized
        )

        se = math.sqrt(variance)
        df_denominator = sum(
            (weight**2 * term_c) ** 2 / (n_c - 1)
            + (weight**2 * term_t) ** 2 / (n_t - 1)
            for weight, _, _, term_c, term_t, n_c, n_t in normalized
        )
        df = (
            float(variance**2 / df_denominator)
            if df_denominator > 0.0
            else float(max(n_c_total + n_t_total - 2 * used, 1))
        )
        return _build_result(
            self.name,
            effect,
            se,
            df,
            n_c_total,
            n_t_total,
            spec.alpha,
            {
                "strata_used": used,
                "strata_skipped": skipped,
                "sparse_policy": self.sparse_policy,
                "used_weight": used_weight,
                "degrees_of_freedom": df,
                "weight_source": (
                    "reference" if self.stratum_weights is not None else "observed_sample"
                ),
            },
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
class NearestNeighborMatchingEstimator:
    """Covariate nearest-neighbor ATT estimator for baseline comparison."""

    name: str = "nearest_neighbor_matching"
    chunk_size: int = 1024

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        features = tuple(spec.pre_treatment_features())
        require_columns(data, [spec.treatment_col, spec.target_col, *features])
        if not features:
            raise ValueError("Nearest-neighbor matching requires numeric covariates.")
        y = _float_column(data, spec.target_col)
        treatment = np.asarray(data[spec.treatment_col])
        x = _feature_matrix(data, features)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        control_mask = valid & (treatment == spec.control_value)
        treatment_mask = valid & (treatment == spec.treatment_value)
        if control_mask.sum() < 2 or treatment_mask.sum() < 2:
            raise ValueError("Both groups need at least two matched observations.")
        x_control = x[control_mask]
        x_treatment = x[treatment_mask]
        y_control = y[control_mask]
        y_treatment = y[treatment_mask]
        pooled = np.vstack([x_control, x_treatment])
        center = pooled.mean(axis=0)
        scale = pooled.std(axis=0, ddof=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        x_control = (x_control - center) / scale
        x_treatment = (x_treatment - center) / scale
        match_index = _nearest_neighbor_indices(x_treatment, x_control, self.chunk_size)
        matched_y = y_control[match_index]
        diffs = y_treatment - matched_y
        n_pairs = int(diffs.size)
        effect = float(diffs.mean())
        se = float(diffs.std(ddof=1) / math.sqrt(n_pairs))
        diagnostics = {
            "estimand": "ATT",
            "features": ",".join(features),
            "matched_pairs": n_pairs,
            "unique_controls": int(np.unique(match_index).size),
        }
        return _build_result(
            self.name,
            effect,
            se,
            float(max(n_pairs - 1, 1)),
            int(control_mask.sum()),
            int(treatment_mask.sum()),
            spec.alpha,
            diagnostics,
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
