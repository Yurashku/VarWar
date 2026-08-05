from __future__ import annotations

import math
import os
import tempfile
import threading
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from varwar.core import EstimateResult, ExperimentSpec, require_columns
from varwar.hypex_backend import HypExBackendUnavailable, estimate_with_hypex_abtest
from varwar.stats import (
    normal_critical,
    normal_two_sided_p,
    safe_divide,
    t_critical,
    t_two_sided_p,
)


_HYPEX_MATCHING_CWD_LOCK = threading.Lock()


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


def _robust_covariance(
    design: np.ndarray,
    residuals: np.ndarray,
    kind: str = "HC2",
) -> np.ndarray:
    """Closed-form heteroskedastic covariance with leverage correction."""

    inverse = np.linalg.pinv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    if kind == "HC2":
        omega = residuals**2 / np.maximum(1.0 - leverage, 1e-12)
    elif kind == "HC3":
        omega = residuals**2 / np.maximum((1.0 - leverage) ** 2, 1e-12)
    elif kind == "HC1":
        omega = residuals**2 * design.shape[0] / max(design.shape[0] - design.shape[1], 1)
    else:
        raise ValueError("kind must be HC1, HC2 or HC3.")
    return inverse @ (design.T @ (omega[:, None] * design)) @ inverse


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
    ridge_grid: tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0),
    auto_tune_ridge: bool = True,
) -> tuple[np.ndarray, tuple[float, ...]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    folds = np.array_split(indices, min(n_splits, x.shape[0]))
    predictions = np.empty(x.shape[0], dtype=float)
    selected_ridges: list[float] = []
    all_indices = np.arange(x.shape[0])
    for fold_number, fold in enumerate(folds):
        train_mask = np.ones(x.shape[0], dtype=bool)
        train_mask[fold] = False
        train_idx = all_indices[train_mask]
        selected = float(ridge)
        if auto_tune_ridge and train_idx.size >= x.shape[1] + 8:
            holdout_size = max(2, int(np.ceil(train_idx.size * 0.2)))
            inner_train = train_idx[:-holdout_size]
            inner_valid = train_idx[-holdout_size:]
            if inner_train.size > x.shape[1] + 1:
                candidates = tuple(
                    sorted({float(value) for value in ridge_grid if float(value) >= 0.0})
                ) or (selected,)
                scores: dict[float, float] = {}
                for candidate in candidates:
                    prediction = _ridge_predict(
                        x[inner_train], y[inner_train], x[inner_valid], candidate
                    )
                    scores[candidate] = float(
                        np.mean((y[inner_valid] - prediction) ** 2)
                    )
                selected = min(candidates, key=lambda value: (scores[value], value))
        selected_ridges.append(selected)
        predictions[fold] = _ridge_predict(
            x[train_idx],
            y[train_idx],
            x[fold],
            selected,
        )
    return predictions, tuple(selected_ridges)


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
    penalty_value = max(float(ridge), 0.0)
    if penalty_value == 0.0:
        coef, *_ = np.linalg.lstsq(xtx, x_centered.T @ y_centered, rcond=None)
    else:
        penalty = penalty_value * np.eye(xtx.shape[0], dtype=float)
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

    Uses centered covariates, treatment interactions and HC2 inference.  This
    is the assumption-lean randomized-experiment form: interactions protect
    the treatment coefficient against effect heterogeneity and unequal arm
    sizes while keeping the calculation closed-form.
    """

    name: str = "regression_adjustment"
    include_pre_target: bool = True
    covariance_type: str = "HC2"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        features = list(spec.pre_treatment_features())
        if self.include_pre_target and spec.pre_target_col and spec.pre_target_col not in features:
            features.append(spec.pre_target_col)
        require_columns(data, [spec.treatment_col, spec.target_col, *features])

        y = _float_column(data, spec.target_col)
        treatment = (np.asarray(data[spec.treatment_col]) == spec.treatment_value).astype(float)
        columns = [np.ones(len(data), dtype=float), treatment]
        centered_covariates = []
        for col in features:
            values = _float_column(data, col)
            centered = values - float(np.nanmean(values))
            centered_covariates.append(centered)
            columns.append(centered)
        columns.extend(treatment * values for values in centered_covariates)
        x = np.column_stack(columns)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        x = x[valid]
        y = y[valid]
        if x.shape[0] <= x.shape[1]:
            raise ValueError("Regression adjustment needs more rows than coefficients.")

        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ beta
        n, p = x.shape
        if self.covariance_type not in {"HC2", "HC3"}:
            raise ValueError("covariance_type must be HC2 or HC3.")
        cov = _robust_covariance(x, residuals, self.covariance_type)
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
            {
                "features": ",".join(features),
                "se": self.covariance_type,
                "interactions": True,
                "covariates_centered": True,
            },
        )


@dataclass
class MLRateEstimator:
    """Cross-fitted machine-learning regression adjustment.

    The nuisance model is a ridge predictor over numeric covariates. The final
    stage regresses the target on treatment, out-of-fold prediction, and the
    treatment-by-prediction interaction with HC2 inference. Ridge strength is
    selected inside each training fold, so outcome rows in the held-out fold
    cannot influence their own nuisance prediction.
    """

    name: str = "ml_rate"
    n_splits: int = 5
    ridge: float = 1e-6
    ridge_grid: tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)
    auto_tune_ridge: bool = True
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
        predictions, selected_ridges = _cross_fitted_ridge_predictions(
            x_valid,
            y_valid,
            n_splits=self.n_splits,
            ridge=self.ridge,
            seed=self.seed,
            ridge_grid=self.ridge_grid,
            auto_tune_ridge=self.auto_tune_ridge,
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
        n, p = design.shape
        cov = _robust_covariance(design, residuals, "HC2")
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
                "ridge_auto_tuned": self.auto_tune_ridge,
                "selected_ridges": ",".join(
                    f"{value:.12g}" for value in selected_ridges
                ),
                "include_interaction": self.include_interaction,
                "prediction_variance": float(predictions.var(ddof=1)),
                "se": "HC2",
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

    The statistic uses the same hedgehog weights in equally sized arms. Rows
    are shuffled reproducibly before weighting, so the result is not an
    accidental function of input order. This object is a homogeneity test,
    not an ATE estimator; effect and confidence-interval fields are therefore
    intentionally undefined.
    """

    name: str = "randomized_t_test"
    seed: int = 20260731

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        if control.size < 2 or treatment.size < 2:
            raise ValueError("Both groups need at least two finite observations.")
        original_n_control = int(control.size)
        original_n_treatment = int(treatment.size)
        rng = np.random.default_rng(self.seed)
        n_used = int(min(control.size, treatment.size))
        control = control[rng.permutation(control.size)[:n_used]]
        treatment = treatment[rng.permutation(treatment.size)[:n_used]]
        _, var_c, n_c = _mean_var_n(control)
        _, var_t, n_t = _mean_var_n(treatment)
        theta_t = _hedgehog_weights(n_t)
        theta_c = _hedgehog_weights(n_c)
        numerator = float(theta_t @ treatment - theta_c @ control)
        term_t = var_t * float(theta_t @ theta_t)
        term_c = var_c * float(theta_c @ theta_c)
        denom = math.sqrt(term_t + term_c)
        statistic = safe_divide(numerator, denom)
        df_denom = (term_c**2 / (n_c - 1)) + (term_t**2 / (n_t - 1))
        df = ((term_c + term_t) ** 2) / df_denom if df_denom > 0 else math.nan
        p_value = t_two_sided_p(statistic, df)
        return EstimateResult(
            estimator=self.name,
            effect=math.nan,
            standard_error=math.nan,
            statistic=statistic,
            p_value=p_value,
            ci_low=math.nan,
            ci_high=math.nan,
            n_control=n_c,
            n_treatment=n_t,
            diagnostics={
                "weights": "hedgehog",
                "df": df,
                "seed": self.seed,
                "row_order": "seeded_shuffle_then_equal_arm_subsample",
                "original_n_control": original_n_control,
                "original_n_treatment": original_n_treatment,
                "effect_definition": "not_an_ATE_estimator",
                "theory_reference": "Klartag_and_Sodin_2012",
            },
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
    среднего группы. Однако конструкция Klartag--Sodin обосновывает скорость
    нормальной аппроксимации центрированной взвешенной суммы, а не выигрыш по
    дисперсии оценки среднего. Для используемого положительного префикса
    эффективный размер выборки растёт лишь порядка ``sqrt(n)``. Поэтому метод
    является исследовательским контролем и не должен заменять Welch baseline.

    Данные не изменяются на месте. Если задан ``unit_col``, он используется
    для стабильного порядка перед перемешиванием; иначе при уникальном индексе
    сначала используется индекс DataFrame.
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
                "theory_scope": "weighted_sum_normal_approximation_not_efficiency",
                "theory_reference": "Klartag_and_Sodin_2012",
                "recommended_role": "research_control_not_primary_ATE_baseline",
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


def _permutation_studentized_statistics(
    control: np.ndarray,
    treatment: np.ndarray,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """Monte Carlo permutation distribution of the Welch studentized statistic."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    pooled = np.concatenate([control, treatment])
    n_control = int(control.size)
    n_treatment = int(treatment.size)
    rng = np.random.default_rng(seed)
    shuffled = pooled.copy()
    statistics = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        rng.shuffle(shuffled)
        left = shuffled[:n_control]
        right = shuffled[n_control:]
        se2 = left.var(ddof=1) / n_control + right.var(ddof=1) / n_treatment
        statistics[index] = (
            (right.mean() - left.mean()) / math.sqrt(se2)
            if se2 > 0.0
            else math.nan
        )
    return statistics[np.isfinite(statistics)]


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

    The p-value uses a Welch-studentized permutation statistic.  The interval
    uses the same empirical pivotal scale around the observed difference; it
    is a studentized permutation interval, not an exact inversion of a sharp
    null test for arbitrary equality-of-means alternatives.
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
        var_c = float(control.var(ddof=1))
        var_t = float(treatment.var(ddof=1))
        observed_se = math.sqrt(var_c / n_c + var_t / n_t)
        if observed_se <= 0.0 or not math.isfinite(observed_se):
            raise ValueError("Permutation studentization requires positive variance.")
        observed_statistic = effect / observed_se
        permuted = _permutation_studentized_statistics(
            control,
            treatment,
            self.n_resamples,
            self.seed,
        )
        if permuted.size < max(20, self.n_resamples // 2):
            raise ValueError("Too few finite permutation studentized replications.")
        extreme = int(np.count_nonzero(np.abs(permuted) >= abs(observed_statistic)))
        permutation_p = (extreme + 1.0) / (self.n_resamples + 1.0)
        lower_null, upper_null = np.quantile(
            permuted,
            [spec.alpha / 2.0, 1.0 - spec.alpha / 2.0],
        )
        return EstimateResult(
            estimator=self.name,
            effect=effect,
            standard_error=observed_se,
            statistic=observed_statistic,
            p_value=permutation_p,
            ci_low=float(effect - upper_null * observed_se),
            ci_high=float(effect - lower_null * observed_se),
            n_control=n_c,
            n_treatment=n_t,
            diagnostics={
                "n_resamples": self.n_resamples,
                "seed": self.seed,
                "p_value_method": "permutation_randomization",
                "ci_method": "studentized_permutation_interval",
                "exchangeability_assumption": "randomized_assignment",
                "studentized_statistic": "welch",
            },
        )


@dataclass
class BootstrapMeanDiffEstimator:
    """Разность средних со studentized bootstrap-t выводом.

    В каждой репликации контроль и тест ресэмплируются независимо, с
    возвращением и с сохранением исходных размеров групп. Для репликации
    строится pivotal-статистика

    ``t* = ((mean_t* - mean_c*) - effect) / se*``.

    Квантили одной и той же bootstrap-t статистики используются и для
    доверительного интервала, и для двустороннего p-value. Это устраняет
    прежнюю несогласованность percentile-интервала с нормальным p-value.
    Репликации обрабатываются блоками: память зависит от ``batch_size``, а не
    от произведения размера выборки на ``n_resamples``.
    """

    name: str = "bootstrap_mean_diff"
    n_resamples: int = 999
    seed: int = 20260709
    batch_size: int = 128

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        if control.size < 2 or treatment.size < 2:
            raise ValueError("Both groups need at least two finite observations.")
        if self.n_resamples < 2:
            raise ValueError("n_resamples must be at least two.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")

        effect = float(treatment.mean() - control.mean())
        var_c = float(control.var(ddof=1))
        var_t = float(treatment.var(ddof=1))
        observed_se = math.sqrt(var_c / control.size + var_t / treatment.size)
        if not math.isfinite(observed_se) or observed_se <= 0.0:
            raise ValueError("Studentized bootstrap requires a positive standard error.")

        rng = np.random.default_rng(self.seed)
        bootstrap_effects = np.empty(self.n_resamples, dtype=float)
        pivots = np.empty(self.n_resamples, dtype=float)
        valid_pivots = np.ones(self.n_resamples, dtype=bool)
        for start in range(0, self.n_resamples, self.batch_size):
            stop = min(start + self.batch_size, self.n_resamples)
            size = stop - start
            control_sample = control[
                rng.integers(0, control.size, size=(size, control.size))
            ]
            treatment_sample = treatment[
                rng.integers(0, treatment.size, size=(size, treatment.size))
            ]
            effects = treatment_sample.mean(axis=1) - control_sample.mean(axis=1)
            replicate_se = np.sqrt(
                control_sample.var(axis=1, ddof=1) / control.size
                + treatment_sample.var(axis=1, ddof=1) / treatment.size
            )
            bootstrap_effects[start:stop] = effects
            finite = np.isfinite(replicate_se) & (replicate_se > 0.0)
            valid_pivots[start:stop] = finite
            block_pivots = np.full(size, np.nan, dtype=float)
            block_pivots[finite] = (effects[finite] - effect) / replicate_se[finite]
            pivots[start:stop] = block_pivots

        pivots = pivots[valid_pivots]
        if pivots.size < max(20, self.n_resamples // 2):
            raise ValueError("Too few finite studentized bootstrap replications.")
        lower_t, upper_t = np.quantile(
            pivots,
            [spec.alpha / 2.0, 1.0 - spec.alpha / 2.0],
        )
        statistic = effect / observed_se
        p_value = (1.0 + float(np.count_nonzero(np.abs(pivots) >= abs(statistic)))) / (
            pivots.size + 1.0
        )
        bootstrap_se = float(bootstrap_effects.std(ddof=1))
        return EstimateResult(
            estimator=self.name,
            effect=effect,
            standard_error=observed_se,
            statistic=statistic,
            p_value=p_value,
            ci_low=float(effect - upper_t * observed_se),
            ci_high=float(effect - lower_t * observed_se),
            n_control=int(control.size),
            n_treatment=int(treatment.size),
            diagnostics={
                "n_resamples": self.n_resamples,
                "valid_resamples": int(pivots.size),
                "seed": self.seed,
                "batch_size": self.batch_size,
                "ci_method": "studentized_bootstrap_t",
                "p_value_method": "studentized_bootstrap_t",
                "standard_error_method": "welch_plugin_studentizer",
                "bootstrap_effect_standard_error": bootstrap_se,
                "resampling": "independent_within_arm_with_replacement",
            },
        )


@dataclass
class TrimmedMeanEstimator:
    """Yuen two-sample estimator and test for symmetrically trimmed means."""

    name: str = "trimmed_mean_diff"
    trim_fraction: float = 0.1

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        control, treatment = _group_arrays(data, spec)
        mean_c, component_c, effective_c = _yuen_group_statistics(
            control,
            self.trim_fraction,
        )
        mean_t, component_t, effective_t = _yuen_group_statistics(
            treatment,
            self.trim_fraction,
        )
        if effective_c < 2 or effective_t < 2:
            raise ValueError("Trimmed groups need at least two observations.")
        effect = mean_t - mean_c
        se = math.sqrt(component_c + component_t)
        denom = (component_c**2 / (effective_c - 1)) + (
            component_t**2 / (effective_t - 1)
        )
        df = (
            ((component_c + component_t) ** 2) / denom
            if denom > 0
            else math.nan
        )
        return _build_result(
            self.name,
            effect,
            se,
            df,
            int(control.size),
            int(treatment.size),
            spec.alpha,
            {
                "trim_fraction": self.trim_fraction,
                "inference": "yuen_welch_trimmed_mean_test",
                "effective_n_control": effective_c,
                "effective_n_treatment": effective_t,
            },
        )


def _yuen_group_statistics(
    values: np.ndarray,
    fraction: float,
) -> tuple[float, float, int]:
    """Return trimmed mean, Yuen variance component, and effective size."""

    if not 0 <= fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5).")
    ordered = np.sort(np.asarray(values, dtype=float))
    n = int(ordered.size)
    removed_per_tail = int(math.floor(n * fraction))
    effective_n = n - 2 * removed_per_tail
    if effective_n < 2:
        raise ValueError("Trim fraction leaves fewer than two observations.")
    trimmed = ordered[removed_per_tail : n - removed_per_tail]
    if removed_per_tail:
        winsorized = ordered.copy()
        winsorized[:removed_per_tail] = ordered[removed_per_tail]
        winsorized[n - removed_per_tail :] = ordered[n - removed_per_tail - 1]
    else:
        winsorized = ordered
    winsorized_variance = float(
        winsorized.var(ddof=2 * removed_per_tail + 1)
    )
    return (
        float(trimmed.mean()),
        winsorized_variance / effective_n,
        effective_n,
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


def _matching_weight_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """Estimate a weighted-mean variance while retaining match reuse.

    The normalized weights may be non-uniform because one unit can be reused as
    another unit's nearest neighbour.  Treating matched differences as
    independent loses that information.  This HC-style linear-weight variance
    is an Abadie-Imbens-like finite-sample approximation conditional on the
    covariates and the selected matches; it is not the exact Abadie-Imbens
    bias-corrected variance estimator.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError(
            "Matching values and weights must be one-dimensional and aligned."
        )
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Matching weights must have a positive finite sum.")
    normalized = weights / total
    mean = float(np.dot(normalized, values))
    squared_weight_sum = float(np.dot(normalized, normalized))
    correction = 1.0 - squared_weight_sum
    if correction <= 0.0:
        return math.nan
    return float(
        np.dot(normalized**2, (values - mean) ** 2) / correction
    )


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
                    "reference"
                    if self.stratum_weights is not None
                    else "experiment_sample"
                ),
            },
        )


@dataclass
class PopulationWeightedStratifiedEstimator:
    """Постстратифицированная оценка эффекта для известной популяции.

    Внутри каждой страты эффект оценивается только по наблюдаемым A/B-группам,
    но затем эффекты усредняются с заранее известными долями целевой популяции.
    Веса не оцениваются по экспериментальному target и не нормируются по
    составу экспериментальной выборки.

    Такой метод оценивает population ATE и имеет информационное преимущество
    перед обычным ``stratified_mean_diff``. Поэтому в R&D-таблицах он должен
    находиться в отдельном режиме с явно отмеченными внешними весами.
    """

    population_weights: Mapping[Any, float]
    name: str = "population_weighted_stratified"
    sparse_policy: str = "error"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        result = StratifiedDifferenceInMeansEstimator(
            name=self.name,
            sparse_policy=self.sparse_policy,
            stratum_weights=self.population_weights,
        ).estimate(data, spec)
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
                "weight_source": "known_target_population",
                "estimand": "population_ATE",
            },
        )


@dataclass
class CalibrationWeightedATEEstimator:
    """Entropy-calibration estimator целевого population ATE.

    В каждой A/B-группе положительные веса минимально отклоняются от
    равномерных в смысле относительной энтропии и точно воспроизводят известные
    средние pre-treatment признаков целевой популяции. После этого оценивается
    разность двух калиброванных средних target.

    Реализация следует постановке calibration/entropy balancing. Она требует
    внешних population moments и оценивает эффект именно для этой популяции.
    Стандартная ошибка является робастной для фиксированных нормированных
    весов; неопределённость самих population moments в неё не входит.
    """

    population_moments: Mapping[str, float]
    name: str = "calibration_reweighting"
    tolerance: float = 1e-8
    max_iterations: int = 500

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        features = tuple(self.population_moments)
        require_columns(data, [spec.treatment_col, spec.target_col, *features])
        if not features:
            raise ValueError("Calibration weighting requires population moments.")
        if self.tolerance <= 0.0 or self.max_iterations < 1:
            raise ValueError("Invalid calibration solver settings.")

        y = _float_column(data, spec.target_col)
        treatment = np.asarray(data[spec.treatment_col])
        x = _feature_matrix(data, features)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        target = np.asarray([self.population_moments[col] for col in features], dtype=float)
        if not np.isfinite(target).all():
            raise ValueError("Population moments must be finite.")

        group_results: list[tuple[float, float, int, float, float]] = []
        max_balance_error = 0.0
        for group_value in (spec.control_value, spec.treatment_value):
            mask = valid & (treatment == group_value)
            if mask.sum() < 2:
                raise ValueError("Both groups need at least two calibration rows.")
            group_x = x[mask]
            group_y = y[mask]
            weights, balance_error = _entropy_calibration_weights(
                group_x,
                target,
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
            )
            mean = float(weights @ group_y)
            sum_squared = float(weights @ weights)
            denominator = 1.0 - sum_squared
            if denominator <= 0.0:
                raise ValueError("Calibration weights have effective sample size one.")
            variance = float(
                np.sum((weights**2) * ((group_y - mean) ** 2)) / denominator
            )
            group_results.append(
                (mean, variance, int(group_y.size), 1.0 / sum_squared, float(weights.max()))
            )
            max_balance_error = max(max_balance_error, balance_error)

        mean_c, var_c, n_c, ess_c, max_weight_c = group_results[0]
        mean_t, var_t, n_t, ess_t, max_weight_t = group_results[1]
        effect = mean_t - mean_c
        se = math.sqrt(max(var_c + var_t, 0.0))
        df = float(max(min(ess_c, ess_t) - 1.0, 1.0))
        return _build_result(
            self.name,
            effect,
            se,
            df,
            n_c,
            n_t,
            spec.alpha,
            {
                "estimand": "population_ATE",
                "weight_source": "known_target_population_moments",
                "calibration": "entropy_balancing_separately_within_arm",
                "features": ",".join(features),
                "max_balance_error": max_balance_error,
                "effective_n_control": ess_c,
                "effective_n_treatment": ess_t,
                "max_weight_control": max_weight_c,
                "max_weight_treatment": max_weight_t,
                "variance": "fixed_weight_hajek_robust",
            },
        )


def _entropy_calibration_weights(
    x: np.ndarray,
    target_mean: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    """Solve the entropy-balancing dual with stable standardized features."""

    center = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    constant = scale <= np.finfo(float).eps
    if np.any(constant & (np.abs(target_mean - center) > tolerance)):
        raise ValueError("A population moment is outside a constant sample feature.")
    active = ~constant
    if not np.any(active):
        weights = np.full(x.shape[0], 1.0 / x.shape[0], dtype=float)
        return weights, float(np.max(np.abs(weights @ x - target_mean)))

    z = (x[:, active] - center[active]) / scale[active]
    target_z = (target_mean[active] - center[active]) / scale[active]
    try:
        from scipy.optimize import minimize
        from scipy.special import logsumexp
    except Exception as exc:  # pragma: no cover - SciPy is a core dependency
        raise RuntimeError(f"SciPy calibration solver is unavailable: {exc}") from exc

    def objective(coef: np.ndarray) -> float:
        scores = z @ coef
        return float(logsumexp(scores) - math.log(z.shape[0]) - target_z @ coef)

    def gradient(coef: np.ndarray) -> np.ndarray:
        scores = z @ coef
        scores -= float(scores.max())
        weights = np.exp(scores)
        weights /= weights.sum()
        return weights @ z - target_z

    result = minimize(
        objective,
        np.zeros(z.shape[1], dtype=float),
        jac=gradient,
        method="BFGS",
        options={"gtol": tolerance, "maxiter": max_iterations},
    )
    if not result.success and np.linalg.norm(gradient(result.x), ord=np.inf) > tolerance * 10:
        raise ValueError(f"Entropy calibration did not converge: {result.message}")
    scores = z @ np.asarray(result.x, dtype=float)
    scores -= float(scores.max())
    weights = np.exp(scores)
    weights /= weights.sum()
    balance_error = float(np.max(np.abs(weights @ x - target_mean)))
    scaled_tolerance = tolerance * max(1.0, float(np.max(np.abs(target_mean)))) * 10
    if balance_error > scaled_tolerance:
        raise ValueError(
            f"Entropy calibration balance error {balance_error:.3g} exceeds tolerance."
        )
    return weights, balance_error


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
    """Nearest-neighbour matching with explicit ATT or randomized ATE mode."""

    name: str = "nearest_neighbor_matching"
    chunk_size: int = 1024
    estimand: str = "att"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        if self.estimand not in {"att", "ate"}:
            raise ValueError("NearestNeighborMatchingEstimator estimand must be 'att' or 'ate'.")
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
        treatment_to_control = _nearest_neighbor_indices(
            x_treatment, x_control, self.chunk_size
        )
        control_reuse = np.bincount(
            treatment_to_control, minlength=y_control.size
        ).astype(float)

        if self.estimand == "ate":
            control_to_treatment = _nearest_neighbor_indices(
                x_control, x_treatment, self.chunk_size
            )
            treatment_reuse = np.bincount(
                control_to_treatment, minlength=y_treatment.size
            ).astype(float)
            treatment_weights = 1.0 + treatment_reuse
            control_weights = 1.0 + control_reuse
            treatment_weights /= treatment_weights.sum()
            control_weights /= control_weights.sum()
            effect = float(
                np.dot(treatment_weights, y_treatment)
                - np.dot(control_weights, y_control)
            )
            variance = _matching_weight_variance(
                y_treatment, treatment_weights
            ) + _matching_weight_variance(y_control, control_weights)
            se = math.sqrt(variance) if variance >= 0.0 else math.nan
            n_total = int(y_treatment.size + y_control.size)
            att_effect = float(
                np.mean(y_treatment - y_control[treatment_to_control])
            )
            atc_effect = float(
                np.mean(y_treatment[control_to_treatment] - y_control)
            )
            diagnostics = {
                "estimand": "ATE",
                "matching_mode": "symmetric_imputation",
                "features": ",".join(features),
                "att_effect": att_effect,
                "atc_effect": atc_effect,
                "unique_controls": int(np.count_nonzero(control_reuse)),
                "unique_treatments": int(np.count_nonzero(treatment_reuse)),
                "max_control_reuse": int(control_reuse.max(initial=0.0)),
                "max_treatment_reuse": int(treatment_reuse.max(initial=0.0)),
                "inference": "reuse_aware_linear_weight_sandwich",
                "inference_scope": "conditional_on_covariates_and_matches",
                "variance_family": "abadie_imbens_like_not_exact_bias_corrected",
                "inference_quality": "approximate",
                "bias_correction": False,
                "att_atc_se_combination": "not_used_joint_linear_weights",
            }
            return _build_result(
                self.name,
                effect,
                se,
                float(max(n_total - 2, 1)),
                int(y_control.size),
                int(y_treatment.size),
                spec.alpha,
                diagnostics,
            )

        matched_y = y_control[treatment_to_control]
        diffs = y_treatment - matched_y
        n_pairs = int(diffs.size)
        treatment_weights = np.full(y_treatment.size, 1.0 / y_treatment.size)
        control_weights = control_reuse / control_reuse.sum()
        effect = float(
            np.dot(treatment_weights, y_treatment)
            - np.dot(control_weights, y_control)
        )
        variance = _matching_weight_variance(
            y_treatment, treatment_weights
        ) + _matching_weight_variance(y_control, control_weights)
        se = math.sqrt(variance) if variance >= 0.0 else math.nan
        diagnostics = {
            "estimand": "ATT",
            "features": ",".join(features),
            "matched_pairs": n_pairs,
            "unique_controls": int(np.count_nonzero(control_reuse)),
            "max_control_reuse": int(control_reuse.max(initial=0.0)),
            "inference": "reuse_aware_linear_weight_sandwich",
            "inference_scope": "conditional_on_covariates_and_matches",
            "variance_family": "abadie_imbens_like_not_exact_bias_corrected",
            "inference_quality": "approximate",
            "bias_correction": False,
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
class HypExMatchingEstimator:
    """Адаптер к matching-эстиматору из PyPI HypEx.

    HypEx 1.0.6 строит matching в обе стороны и возвращает ATT, ATC и ATE.
    Его стандартный профиль использует расстояние Махаланобиса, FAISS,
    matching с возвращением и поправку смещения. VarWar извлекает строку ATE
    из публичного ``Matching`` API без локального воспроизведения алгоритма.
    """

    name: str = "hypex_matching"
    distance: str = "mahalanobis"
    n_neighbors: int = 1
    bias_estimation: bool = True
    faiss_mode: str = "auto"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        features = tuple(spec.pre_treatment_features())
        require_columns(data, [spec.treatment_col, spec.target_col, *features])
        if not features:
            raise ValueError("HypEx matching requires pre-treatment covariates.")
        try:
            from hypex import Matching
            from hypex.dataset import (
                Dataset,
                ExperimentData,
                FeatureRole,
                TargetRole,
                TreatmentRole,
            )
        except Exception as exc:  # pragma: no cover - optional integration
            raise RuntimeError(f"PyPI HypEx matching is unavailable: {exc}") from exc

        roles: dict[str, Any] = {
            spec.treatment_col: TreatmentRole(int),
            spec.target_col: TargetRole(float),
        }
        roles.update({feature: FeatureRole(float) for feature in features})
        y = _float_column(data, spec.target_col)
        x = _feature_matrix(data, features)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        clean = data.loc[
            valid,
            [spec.treatment_col, spec.target_col, *features],
        ].copy(deep=False)
        treatment_values = np.asarray(clean[spec.treatment_col])
        if (
            np.count_nonzero(treatment_values == spec.control_value) < 2
            or np.count_nonzero(treatment_values == spec.treatment_value) < 2
        ):
            raise ValueError("Both groups need at least two finite matching rows.")
        try:
            # HypEx 1.0.6 writes matched_indexes.csv to the process cwd and
            # rounds its public UI resume.  Execute the same experiment core
            # in a private cwd and read the raw MatchingAnalyzer table.
            with _HYPEX_MATCHING_CWD_LOCK:
                previous_cwd = os.getcwd()
                try:
                    isolated_cwd = tempfile.mkdtemp(prefix="varwar_hypex_matching_")
                    os.chdir(isolated_cwd)
                    output = Matching(
                        distance=self.distance,
                        bias_estimation=self.bias_estimation,
                        quality_tests=[],
                        faiss_mode=self.faiss_mode,
                        n_neighbors=self.n_neighbors,
                        encode_categories=False,
                    ).experiment.execute(
                        ExperimentData(Dataset(roles=roles, data=clean))
                    )
                    analysis_tables = output.analysis_tables
                finally:
                    os.chdir(previous_cwd)
            if not analysis_tables:
                raise RuntimeError("HypEx matching returned no analysis table.")
            analysis = next(iter(analysis_tables.values())).data
            row = analysis.loc["ATE"]
            effect = float(row["Effect Size"])
            se = float(row["Standard Error"])
            p_value = float(row["P-value"])
            ci_low = float(row["CI Lower"])
            ci_high = float(row["CI Upper"])
        except Exception as exc:  # pragma: no cover - depends on HypEx internals
            raise RuntimeError(f"PyPI HypEx matching failed: {exc}") from exc

        treatment = np.asarray(clean[spec.treatment_col])
        return EstimateResult(
            estimator=self.name,
            effect=effect,
            standard_error=se,
            statistic=safe_divide(effect, se),
            p_value=p_value,
            ci_low=ci_low,
            ci_high=ci_high,
            n_control=int(np.count_nonzero(treatment == spec.control_value)),
            n_treatment=int(np.count_nonzero(treatment == spec.treatment_value)),
            diagnostics={
                "backend": "pypi_hypex",
                "estimand": "ATE",
                "distance": self.distance,
                "n_neighbors": self.n_neighbors,
                "bias_estimation": self.bias_estimation,
                "matching_direction": "two_sided_ATT_and_ATC",
                "features": ",".join(features),
                "raw_analysis_table": True,
                "side_effects_isolated": True,
            },
        )


@dataclass
class OptimalTransportMatchingEstimator:
    """Глобальное matching как дискретный optimal transport.

    Все treated-объекты одновременно сопоставляются control-объектам так,
    чтобы минимизировать суммарную квадратичную transport cost в
    стандартизированном пространстве pre-treatment признаков. Массы treated
    одинаковы, регуляризация отсутствует. Если control не меньше treated,
    получается точный частичный Monge plan без возвращения. Иначе каждому
    control разрешается минимально необходимая одинаковая capacity. Это
    сохраняет ATT, но явно ослабляет ограничение уникальности пар.

    Реализация использует точный solver SciPy ``linear_sum_assignment``.
    Матрица стоимостей имеет размер
    ``n_treatment * n_control * control_capacity``; защитный лимит
    предотвращает случайное исчерпание памяти. Вывод является paired-t
    inference, условным на найденном matching, и не учитывает неопределённость
    выбора matching. Поэтому метод остаётся observational/exploratory.
    """

    name: str = "optimal_transport_matching"
    # 10M float distances are large but bounded; callers can lower this guard
    # for tighter production memory budgets.
    max_cost_matrix_entries: int = 10_000_000
    caliper: float | None = None
    max_control_reuse: int | None = None
    estimand: str = "att"

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        if self.estimand not in {"att", "ate"}:
            raise ValueError("OptimalTransportMatchingEstimator estimand must be 'att' or 'ate'.")
        if self.estimand == "ate":
            treated = self.__class__(
                name=self.name,
                max_cost_matrix_entries=self.max_cost_matrix_entries,
                caliper=self.caliper,
                max_control_reuse=self.max_control_reuse,
                estimand="att",
            ).estimate(data, spec)
            reversed_spec = replace(spec, control_value=spec.treatment_value, treatment_value=spec.control_value)
            control = self.__class__(
                name=self.name,
                max_cost_matrix_entries=self.max_cost_matrix_entries,
                caliper=self.caliper,
                max_control_reuse=self.max_control_reuse,
                estimand="att",
            ).estimate(data, reversed_spec)
            n_total = treated.n_treatment + treated.n_control
            weight_t = treated.n_treatment / n_total
            weight_c = treated.n_control / n_total
            atc_effect = -control.effect
            effect = weight_t * treated.effect + weight_c * atc_effect
            return EstimateResult(
                estimator=self.name,
                effect=effect,
                standard_error=math.nan,
                statistic=math.nan,
                p_value=math.nan,
                ci_low=math.nan,
                ci_high=math.nan,
                n_control=treated.n_control,
                n_treatment=treated.n_treatment,
                diagnostics={
                    "estimand": "ATE",
                    "matching_mode": "symmetric_transport_imputation",
                    "att_effect": treated.effect,
                    "atc_effect": atc_effect,
                    "att_standard_error": treated.standard_error,
                    "atc_standard_error": control.standard_error,
                    "inference": "point_only",
                    "inference_reason": (
                        "ATT and ATC are dependent and no justified general-ATE "
                        "variance for the fitted transport plan is available."
                    ),
                    "required_for_inference": (
                        "assignment-replay randomization inference under a sharp "
                        "constant-effect null"
                    ),
                    "constant_effect_assumption": False,
                    "transport_plan_uncertainty_accounted": False,
                },
            )
        features = tuple(spec.pre_treatment_features())
        require_columns(data, [spec.treatment_col, spec.target_col, *features])
        if not features:
            raise ValueError("Optimal-transport matching requires numeric covariates.")
        y = _float_column(data, spec.target_col)
        treatment = np.asarray(data[spec.treatment_col])
        x = _feature_matrix(data, features)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        control_mask = valid & (treatment == spec.control_value)
        treatment_mask = valid & (treatment == spec.treatment_value)
        n_control = int(control_mask.sum())
        n_treatment = int(treatment_mask.sum())
        if n_treatment < 2 or n_control < 2:
            raise ValueError("Both groups need at least two observations.")
        minimum_capacity = int(math.ceil(n_treatment / n_control))
        capacity = minimum_capacity if self.max_control_reuse is None else int(
            self.max_control_reuse
        )
        if capacity < minimum_capacity:
            raise ValueError(
                "max_control_reuse is too small to match every treated unit."
            )
        matrix_entries = n_treatment * n_control * capacity
        if matrix_entries > self.max_cost_matrix_entries:
            raise ValueError(
                "Optimal-transport cost matrix would contain "
                f"{matrix_entries:,} entries; limit is "
                f"{self.max_cost_matrix_entries:,}."
            )

        x_control = x[control_mask]
        x_treatment = x[treatment_mask]
        pooled = np.vstack([x_control, x_treatment])
        center = pooled.mean(axis=0)
        scale = pooled.std(axis=0, ddof=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        x_control = (x_control - center) / scale
        x_treatment = (x_treatment - center) / scale
        try:
            from scipy.optimize import linear_sum_assignment
            from scipy.spatial.distance import cdist
        except Exception as exc:  # pragma: no cover - SciPy is a core dependency
            raise RuntimeError(f"SciPy matching solver is unavailable: {exc}") from exc

        base_cost = cdist(x_treatment, x_control, metric="sqeuclidean")
        cost = np.repeat(base_cost, capacity, axis=1)
        treated_index, control_slot = linear_sum_assignment(cost)
        control_index = control_slot // capacity
        matched_cost = base_cost[treated_index, control_index]
        if self.caliper is not None:
            if self.caliper <= 0.0:
                raise ValueError("caliper must be positive.")
            keep = np.sqrt(matched_cost) <= self.caliper
            treated_index = treated_index[keep]
            control_index = control_index[keep]
            matched_cost = matched_cost[keep]
        if treated_index.size < 2:
            raise ValueError("Fewer than two pairs remain after optimal matching.")

        y_control = y[control_mask]
        y_treatment = y[treatment_mask]
        differences = y_treatment[treated_index] - y_control[control_index]
        effect = float(differences.mean())
        se = float(differences.std(ddof=1) / math.sqrt(differences.size))
        return _build_result(
            self.name,
            effect,
            se,
            float(max(differences.size - 1, 1)),
            n_control,
            n_treatment,
            spec.alpha,
            {
                "estimand": "ATT" if self.caliper is None else "overlap_ATT",
                "solver": "scipy_linear_sum_assignment",
                "transport": "exact_unregularized_capacitated_monge_assignment",
                "treated_mass": "uniform",
                "regularization": "none",
                "control_reuse_capacity": capacity,
                "minimum_required_capacity": minimum_capacity,
                "unique_controls": int(np.unique(control_index).size),
                "distance": "pooled_standardized_squared_euclidean",
                "features": ",".join(features),
                "matched_pairs": int(differences.size),
                "total_transport_cost": float(matched_cost.sum()),
                "mean_match_distance": float(np.sqrt(matched_cost).mean()),
                "max_match_distance": float(np.sqrt(matched_cost).max()),
                "cost_matrix_entries": matrix_entries,
                "inference": "paired_t_conditional_on_matching",
            },
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
        crit = normal_critical(spec.alpha)
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
