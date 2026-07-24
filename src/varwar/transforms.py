from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec, require_columns
from varwar.feature_sets import CupacFeaturePolicy


def _as_matrix(data: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(pd.to_numeric(data[column], errors="coerce"), dtype=float)
            for column in columns
        ]
    )


def _select_fit_frame(
    data: pd.DataFrame,
    spec: ExperimentSpec,
    auxiliary_data: pd.DataFrame | None,
    fit_data: str,
) -> tuple[pd.DataFrame, str]:
    if fit_data == "reference":
        if auxiliary_data is None:
            raise ValueError(
                "This transformer requires an independent historical/reference sample."
            )
        return auxiliary_data, "reference"
    if fit_data == "experiment_control":
        require_columns(data, [spec.treatment_col])
        mask = np.asarray(data[spec.treatment_col]) == spec.control_value
        return data.loc[mask], "experiment_control"
    if fit_data == "experiment_all":
        return data, "experiment_all"
    raise ValueError(
        "fit_data must be 'reference', 'experiment_control', or 'experiment_all'."
    )


@dataclass(frozen=True)
class _RidgeState:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    coefficients: np.ndarray
    prediction_mean: float
    rank: int

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        standardized = (matrix - self.x_mean) / self.x_scale
        return self.y_mean + standardized @ self.coefficients


def _fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> tuple[_RidgeState, np.ndarray]:
    valid = np.isfinite(target) & np.isfinite(matrix).all(axis=1)
    if int(valid.sum()) <= matrix.shape[1] + 1:
        raise ValueError("CUPAC needs more finite reference rows than model columns.")
    x = matrix[valid]
    y = target[valid]
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0, ddof=0)
    x_scale = np.where(x_scale > 0.0, x_scale, 1.0)
    standardized = (x - x_mean) / x_scale
    y_mean = float(y.mean())
    centered_y = y - y_mean
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative.")
    if ridge == 0.0:
        coefficients, _, rank, _ = np.linalg.lstsq(
            standardized,
            centered_y,
            rcond=None,
        )
    else:
        gram = standardized.T @ standardized
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(gram.shape[0], dtype=float),
            standardized.T @ centered_y,
        )
        rank = int(np.linalg.matrix_rank(standardized))
    fitted = y_mean + standardized @ coefficients
    state = _RidgeState(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        coefficients=coefficients,
        prediction_mean=float(fitted.mean()),
        rank=int(rank),
    )
    return state, valid


@dataclass
class CupedTransformer:
    """CUPED по одной pre-experiment ковариате.

    ``fit_data`` явно задаёт источник коэффициента theta. Вариант
    ``experiment_control`` оценивает связь target и ковариаты только в
    контроле текущего эксперимента. Вариант ``reference`` использует
    независимую историческую выборку. Значение по умолчанию не зависит от
    наличия случайно переданного ``auxiliary_data``.
    """

    covariate_col: str | None = None
    fit_data: str = "experiment_control"
    name: str | None = None
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.name is None:
            suffix = {
                "experiment_control": "control",
                "experiment_all": "experiment_all",
                "reference": "reference",
            }.get(self.fit_data, self.fit_data)
            self.name = f"cuped_{suffix}"

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        covariate = self.covariate_col or spec.pre_target_col
        if covariate is None:
            raise ValueError("CUPED requires covariate_col or spec.pre_target_col.")
        fit_frame, source = _select_fit_frame(
            data,
            spec,
            auxiliary_data,
            self.fit_data,
        )
        require_columns(fit_frame, [spec.target_col, covariate])
        require_columns(data, [spec.target_col, covariate])

        y_fit = np.asarray(
            pd.to_numeric(fit_frame[spec.target_col], errors="coerce"),
            dtype=float,
        )
        x_fit = np.asarray(
            pd.to_numeric(fit_frame[covariate], errors="coerce"),
            dtype=float,
        )
        valid = np.isfinite(y_fit) & np.isfinite(x_fit)
        if int(valid.sum()) < 2:
            raise ValueError("CUPED needs at least two finite fit rows.")
        y_valid = y_fit[valid]
        x_valid = x_fit[valid]
        x_mean = float(x_valid.mean())
        x_centered = x_valid - x_mean
        denominator = float(np.dot(x_centered, x_centered))
        theta = (
            float(np.dot(y_valid - float(y_valid.mean()), x_centered) / denominator)
            if denominator > 0.0
            else 0.0
        )

        y = np.asarray(pd.to_numeric(data[spec.target_col], errors="coerce"), dtype=float)
        x = np.asarray(pd.to_numeric(data[covariate], errors="coerce"), dtype=float)
        adjusted = y - theta * (x - x_mean)
        self._diagnostics = {
            "covariate": covariate,
            "theta": theta,
            "fit_data": source,
            "fit_rows": int(valid.sum()),
            "centering_mean": x_mean,
        }
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class LinearCupacTransformer:
    """Temporal linear CUPAC внутри одного доступного датасета.

    Модель обучается на переходе ``lag2 features -> lag1 pre-target`` и
    применяется как ``lag1 features -> current target``. Текущий outcome
    не участвует в обучении модели.
    """

    feature_cols: tuple[str, ...] | None = None
    ridge: float = 1e-8
    name: str = "cupac_linear"
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def _resolve_features(self, spec: ExperimentSpec) -> tuple[str, ...]:
        features = self.feature_cols or spec.pre_treatment_features()
        if not features:
            raise ValueError(
                "Linear CUPAC requires feature_cols or pre-treatment features."
            )
        return tuple(features)

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        del auxiliary_data
        if spec.pre_target_col is None:
            raise ValueError("Linear CUPAC requires spec.pre_target_col.")
        features = self._resolve_features(spec)
        if spec.pre_target_col not in features:
            features = (spec.pre_target_col, *features)
        features = tuple(dict.fromkeys(features))
        lag2_features = tuple(f"{column}__lag2" for column in features)
        require_columns(
            data,
            [spec.target_col, spec.pre_target_col, *features, *lag2_features],
        )

        state, valid = _fit_ridge(
            _as_matrix(data, lag2_features),
            np.asarray(
                pd.to_numeric(data[spec.pre_target_col], errors="coerce"),
                dtype=float,
            ),
            self.ridge,
        )
        prediction = state.predict(_as_matrix(data, features))
        target = np.asarray(
            pd.to_numeric(data[spec.target_col], errors="coerce"),
            dtype=float,
        )
        prediction_mean = float(np.nanmean(prediction))
        adjusted = target - (prediction - prediction_mean)
        self._diagnostics = {
            "features_lag1": ",".join(features),
            "features_lag2": ",".join(lag2_features),
            "feature_count": len(features),
            "ridge": self.ridge,
            "fit_data": "same_dataset_pre_period_lag2_to_lag1",
            "fit_rows": int(valid.sum()),
            "model_rank": state.rank,
            "adjustment_coefficient": 1.0,
            "prediction_variance": float(np.nanvar(prediction, ddof=1)),
            "prediction_center": prediction_mean,
            "current_outcome_used_for_model_fit": False,
        }
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class LinearInTimeAdjustmentTransformer:
    """Линейная second-stage корректировка по разрешённым in-time признакам."""

    feature_cols: tuple[str, ...]
    fit_data: str = "experiment_control"
    ridge: float = 1e-8
    name: str = "linear_intime_adjustment"
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        del auxiliary_data
        if not self.feature_cols:
            raise ValueError("In-time adjustment requires at least one feature.")
        fit_frame, source = _select_fit_frame(
            data,
            spec,
            auxiliary_data=None,
            fit_data=self.fit_data,
        )
        require_columns(fit_frame, [spec.target_col, *self.feature_cols])
        require_columns(data, [spec.target_col, *self.feature_cols])
        state, valid = _fit_ridge(
            _as_matrix(fit_frame, self.feature_cols),
            np.asarray(
                pd.to_numeric(fit_frame[spec.target_col], errors="coerce"),
                dtype=float,
            ),
            self.ridge,
        )
        prediction = state.predict(_as_matrix(data, self.feature_cols))
        prediction_mean = float(np.nanmean(prediction))
        target = np.asarray(
            pd.to_numeric(data[spec.target_col], errors="coerce"),
            dtype=float,
        )
        adjusted = target - (prediction - prediction_mean)
        self._diagnostics = {
            "features": ",".join(self.feature_cols),
            "feature_count": len(self.feature_cols),
            "fit_data": source,
            "fit_rows": int(valid.sum()),
            "ridge": self.ridge,
            "model_rank": state.rank,
            "prediction_variance": float(np.nanvar(prediction, ddof=1)),
            "prediction_center": prediction_mean,
            "current_outcome_used_for_model_fit": True,
        }
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class SequentialTargetTransformer:
    """Последовательно применить несколько target-transformers."""

    transformers: tuple[Any, ...]
    name: str
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        working = data.copy(deep=False)
        current_spec = spec
        diagnostics: dict[str, Any] = {"stage_count": len(self.transformers)}
        adjusted: pd.Series | None = None
        for index, transformer in enumerate(self.transformers, start=1):
            adjusted = transformer.fit_transform(
                working,
                current_spec,
                auxiliary_data,
            )
            stage_col = f"__varwar_stage_{index}_{self.name}"
            working = working.copy(deep=False)
            working[stage_col] = adjusted
            current_spec = current_spec.with_target(stage_col)
            diagnostics[f"stage_{index}"] = transformer.name
            diagnostics.update(
                {
                    f"stage_{index}_{key}": value
                    for key, value in transformer.diagnostics().items()
                }
            )
        if adjusted is None:
            raise ValueError("SequentialTargetTransformer requires at least one stage.")
        self._diagnostics = diagnostics
        return pd.Series(
            np.asarray(adjusted, dtype=float),
            index=data.index,
            name=f"{spec.target_col}_{self.name}",
        )

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class ZonedCupacTransformer:
    """RnD6-цепочка: temporal CUPAC A, затем linear B, затем gated linear C."""

    policy: CupacFeaturePolicy
    ridge: float = 1e-8
    name: str | None = None
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.policy.method_id

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        _, policy_diagnostics = self.policy.resolve(spec, data)
        a_features = spec.pre_treatment_features()
        if not a_features:
            raise ValueError(f"{self.policy.method_id} requires zone A features.")
        stages: list[Any] = [
            LinearCupacTransformer(
                feature_cols=a_features,
                ridge=self.ridge,
                name="cupac_A",
            )
        ]
        if "B" in self.policy.zones and spec.expert_safe_in_time_cols:
            stages.append(
                LinearInTimeAdjustmentTransformer(
                    feature_cols=spec.expert_safe_in_time_cols,
                    ridge=self.ridge,
                    name="linear_adjustment_B",
                )
            )
        admitted_c = tuple(
            value
            for value in str(policy_diagnostics.get("zone_c_admitted", "")).split(",")
            if value
        )
        if "C" in self.policy.zones and admitted_c:
            stages.append(
                LinearInTimeAdjustmentTransformer(
                    feature_cols=admitted_c,
                    ridge=self.ridge,
                    name="linear_adjustment_C",
                )
            )
        delegate = SequentialTargetTransformer(
            transformers=tuple(stages),
            name=self.name or self.policy.method_id,
        )
        adjusted = delegate.fit_transform(data, spec, auxiliary_data)
        self._diagnostics = {
            **delegate.diagnostics(),
            **policy_diagnostics,
        }
        return adjusted

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


def _quadratic_expand(matrix: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    columns = [matrix]
    pairs: list[tuple[int, int]] = []
    for left in range(matrix.shape[1]):
        for right in range(left, matrix.shape[1]):
            columns.append((matrix[:, left] * matrix[:, right])[:, None])
            pairs.append((left, right))
    return np.column_stack(columns), tuple(pairs)


@dataclass
class QuadraticCupacTransformer:
    """CUPAC с линейными, квадратными и попарными interaction-признаками.

    Это ограниченная и воспроизводимая нелинейная модель без тяжёлой
    зависимости: число модельных колонок равно ``p + p*(p+1)/2``.
    """

    feature_cols: tuple[str, ...] | None = None
    ridge: float = 1e-4
    name: str = "cupac_quadratic_A"
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        del auxiliary_data
        if spec.pre_target_col is None:
            raise ValueError("Quadratic CUPAC requires spec.pre_target_col.")
        features = self.feature_cols or spec.pre_treatment_features()
        if not features:
            raise ValueError("Quadratic CUPAC requires pre-treatment features.")
        if spec.pre_target_col not in features:
            features = (spec.pre_target_col, *features)
        features = tuple(dict.fromkeys(features))
        lag2_features = tuple(f"{column}__lag2" for column in features)
        require_columns(
            data,
            [spec.target_col, spec.pre_target_col, *features, *lag2_features],
        )

        x_reference, pairs = _quadratic_expand(_as_matrix(data, lag2_features))
        x_apply, _ = _quadratic_expand(_as_matrix(data, features))
        state, valid = _fit_ridge(
            x_reference,
            np.asarray(
                pd.to_numeric(data[spec.pre_target_col], errors="coerce"),
                dtype=float,
            ),
            self.ridge,
        )
        prediction = state.predict(x_apply)
        target = np.asarray(
            pd.to_numeric(data[spec.target_col], errors="coerce"),
            dtype=float,
        )
        prediction_mean = float(np.nanmean(prediction))
        adjusted = target - (prediction - prediction_mean)
        self._diagnostics = {
            "features_lag1": ",".join(features),
            "features_lag2": ",".join(lag2_features),
            "raw_feature_count": len(features),
            "model_feature_count": x_reference.shape[1],
            "interaction_count": len(pairs),
            "ridge": self.ridge,
            "fit_data": "same_dataset_pre_period_lag2_to_lag1",
            "fit_rows": int(valid.sum()),
            "model_rank": state.rank,
            "adjustment_coefficient": 1.0,
            "prediction_variance": float(np.nanvar(prediction, ddof=1)),
            "prediction_center": prediction_mean,
            "current_outcome_used_for_model_fit": False,
        }
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class TemporalLinearCupacTransformer:
    """Локальный совместимый вариант temporal CUPAC из HypEx.

    Модель обучается внутри experiment dataset: признаки lag-2 прогнозируют
    pre-target lag-1, после чего та же модель применяется к признакам lag-1.
    Как и HypEx, transformer оценивает по всем текущим строкам коэффициент
    ``theta = Cov(target, prediction) / Var(prediction)``. Текущий outcome
    не используется для обучения прогностической модели, но используется для
    оценки theta.
    """

    feature_cols: tuple[str, ...] | None = None
    ridge: float = 1e-8
    name: str = "cupac_temporal_linear"
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        del auxiliary_data
        if spec.pre_target_col is None:
            raise ValueError("Temporal CUPAC requires spec.pre_target_col.")
        features = self.feature_cols or spec.pre_treatment_features()
        if not features:
            raise ValueError("Temporal CUPAC requires pre-treatment features.")
        if spec.pre_target_col not in features:
            features = (spec.pre_target_col, *features)
        features = tuple(dict.fromkeys(features))
        lag2_features = tuple(f"{column}__lag2" for column in features)
        require_columns(
            data,
            [spec.target_col, spec.pre_target_col, *features, *lag2_features],
        )

        state, valid_model = _fit_ridge(
            _as_matrix(data, lag2_features),
            np.asarray(
                pd.to_numeric(data[spec.pre_target_col], errors="coerce"),
                dtype=float,
            ),
            self.ridge,
        )
        prediction = state.predict(_as_matrix(data, features))
        target = np.asarray(
            pd.to_numeric(data[spec.target_col], errors="coerce"),
            dtype=float,
        )
        valid_theta = np.isfinite(target) & np.isfinite(prediction)
        if int(valid_theta.sum()) < 2:
            raise ValueError("Temporal CUPAC needs two finite theta-fit rows.")
        prediction_mean = float(prediction[valid_theta].mean())
        centered_prediction = prediction[valid_theta] - prediction_mean
        denominator = float(np.dot(centered_prediction, centered_prediction))
        theta = (
            float(
                np.dot(
                    target[valid_theta] - float(target[valid_theta].mean()),
                    centered_prediction,
                )
                / denominator
            )
            if denominator > 0.0
            else 0.0
        )
        adjusted = target - theta * (prediction - prediction_mean)
        self._diagnostics = {
            "features_lag1": ",".join(features),
            "features_lag2": ",".join(lag2_features),
            "feature_count": len(features),
            "ridge": self.ridge,
            "fit_data": "same_dataset_pre_period_lag2_to_lag1",
            "fit_rows": int(valid_model.sum()),
            "model_rank": state.rank,
            "adjustment_coefficient": theta,
            "theta_fit_data": "experiment_all",
            "theta_fit_rows": int(valid_theta.sum()),
            "prediction_variance": float(np.nanvar(prediction, ddof=1)),
            "prediction_center": prediction_mean,
            "current_outcome_used_for_model_fit": False,
            "current_outcome_used_for_theta": True,
        }
        return pd.Series(
            adjusted,
            index=data.index,
            name=f"{spec.target_col}_{self.name}",
        )

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class WinsorizeTransformer:
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    fit_data: str = "experiment_control"
    name: str = "winsorized"
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Winsorization quantiles must satisfy 0 <= lower < upper <= 1.")
        fit_frame, source = _select_fit_frame(
            data,
            spec,
            auxiliary_data,
            self.fit_data,
        )
        require_columns(fit_frame, [spec.target_col])
        require_columns(data, [spec.target_col])
        fit_target = np.asarray(
            pd.to_numeric(fit_frame[spec.target_col], errors="coerce"),
            dtype=float,
        )
        fit_target = fit_target[np.isfinite(fit_target)]
        if fit_target.size < 2:
            raise ValueError("Winsorization needs at least two finite reference rows.")
        lower, upper = np.quantile(
            fit_target,
            [self.lower_quantile, self.upper_quantile],
        )
        target = np.asarray(
            pd.to_numeric(data[spec.target_col], errors="coerce"),
            dtype=float,
        )
        adjusted = np.clip(target, lower, upper)
        self._diagnostics = {
            "fit_data": source,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "lower_value": float(lower),
            "upper_value": float(upper),
            "clipped_fraction": float(
                np.mean((target < lower) | (target > upper))
            ),
        }
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics
