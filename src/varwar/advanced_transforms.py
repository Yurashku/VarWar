from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec, require_columns


def _sklearn_components() -> dict[str, Any]:
    try:
        from sklearn.base import clone
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import HuberRegressor, Ridge
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Advanced target transformers require the 'ml' extra: "
            "pip install 'varwar[ml]'."
        ) from exc
    return locals()


def _numeric_matrix(data: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(pd.to_numeric(data[column], errors="coerce"), dtype=float)
            for column in columns
        ]
    )


@dataclass
class AutoMLTemporalCupacTransformer:
    """CUPAC A с автоматическим выбором pre-treatment модели.

    При обучении используются только пары ``признаки lag2 -> target lag1``.
    Текущий экспериментальный outcome не передаётся ни одной ML-модели и не
    участвует в выборе кандидата. После ``fit`` объект можно многократно
    применять через ``transform`` к данным с теми же lag1-признаками.

    Parameters
    ----------
    feature_cols:
        Имена признаков в зоне lag1. Обучающие колонки получаются добавлением
        ``lag2_suffix``. Если не заданы, используются pre-treatment признаки
        из ``ExperimentSpec``.
    candidate_ids:
        Подмножество допустимых моделей. Порядок задаёт детерминированное
        разрешение ничьей по CV.
    """

    feature_cols: tuple[str, ...] | None = None
    candidate_ids: tuple[str, ...] = (
        "ridge",
        "quadratic_ridge",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    )
    lag2_suffix: str = "__lag2"
    cv_splits: int = 5
    random_state: int = 42
    ridge_alpha: float = 1.0
    n_estimators: int = 96
    max_tree_depth: int | None = 8
    min_samples_leaf: int = 5
    hist_max_iter: int = 100
    name: str = "cupac_automl_A"
    _model: Any | None = field(default=None, init=False, repr=False)
    _features: tuple[str, ...] = field(default=(), init=False, repr=False)
    _lag2_features: tuple[str, ...] = field(default=(), init=False, repr=False)
    _prediction_center: float | None = field(default=None, init=False, repr=False)
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def _resolve_features(self, spec: ExperimentSpec) -> tuple[str, ...]:
        features = self.feature_cols or spec.pre_treatment_features()
        if spec.pre_target_col and spec.pre_target_col not in features:
            features = (spec.pre_target_col, *features)
        features = tuple(dict.fromkeys(features))
        if not features:
            raise ValueError("AutoML temporal CUPAC requires pre-treatment features.")
        return features

    def _candidate_models(self) -> dict[str, Any]:
        sk = _sklearn_components()
        pipeline = sk["Pipeline"]
        imputer = sk["SimpleImputer"]
        scaler = sk["StandardScaler"]
        ridge = sk["Ridge"]
        polynomial = sk["PolynomialFeatures"]
        candidates = {
            "ridge": pipeline(
                [
                    ("imputer", imputer(strategy="median")),
                    ("scale", scaler()),
                    ("model", ridge(alpha=self.ridge_alpha)),
                ]
            ),
            "quadratic_ridge": pipeline(
                [
                    ("imputer", imputer(strategy="median")),
                    ("poly", polynomial(degree=2, include_bias=False)),
                    ("scale", scaler()),
                    ("model", ridge(alpha=self.ridge_alpha)),
                ]
            ),
            "random_forest": pipeline(
                [
                    ("imputer", imputer(strategy="median")),
                    (
                        "model",
                        sk["RandomForestRegressor"](
                            n_estimators=self.n_estimators,
                            max_depth=self.max_tree_depth,
                            min_samples_leaf=self.min_samples_leaf,
                            max_features="sqrt",
                            n_jobs=1,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
            "extra_trees": pipeline(
                [
                    ("imputer", imputer(strategy="median")),
                    (
                        "model",
                        sk["ExtraTreesRegressor"](
                            n_estimators=self.n_estimators,
                            max_depth=self.max_tree_depth,
                            min_samples_leaf=self.min_samples_leaf,
                            max_features="sqrt",
                            n_jobs=1,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
            "hist_gradient_boosting": pipeline(
                [
                    ("imputer", imputer(strategy="median")),
                    (
                        "model",
                        sk["HistGradientBoostingRegressor"](
                            max_iter=self.hist_max_iter,
                            max_depth=self.max_tree_depth,
                            min_samples_leaf=self.min_samples_leaf,
                            l2_regularization=1.0,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
        }
        unknown = [name for name in self.candidate_ids if name not in candidates]
        if unknown:
            raise ValueError(f"Unknown AutoML CUPAC candidates: {unknown}")
        if not self.candidate_ids:
            raise ValueError("candidate_ids must contain at least one model.")
        return {name: candidates[name] for name in self.candidate_ids}

    def fit(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
    ) -> "AutoMLTemporalCupacTransformer":
        if spec.pre_target_col is None:
            raise ValueError("AutoML temporal CUPAC requires spec.pre_target_col.")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least 2.")
        features = self._resolve_features(spec)
        lag2_features = tuple(f"{column}{self.lag2_suffix}" for column in features)
        require_columns(data, [spec.pre_target_col, *lag2_features])

        matrix = _numeric_matrix(data, lag2_features)
        target = np.asarray(
            pd.to_numeric(data[spec.pre_target_col], errors="coerce"), dtype=float
        )
        valid = np.isfinite(target) & np.isfinite(matrix).any(axis=1)
        x_fit = matrix[valid]
        y_fit = target[valid]
        if len(y_fit) < max(12, self.cv_splits * 2):
            raise ValueError("AutoML temporal CUPAC needs more finite pre-period rows.")

        sk = _sklearn_components()
        cv = sk["KFold"](
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        scores: dict[str, float] = {}
        models = self._candidate_models()
        for candidate_id, model in models.items():
            fold_scores = sk["cross_val_score"](
                model,
                x_fit,
                y_fit,
                cv=cv,
                scoring="neg_mean_squared_error",
                n_jobs=1,
                error_score="raise",
            )
            scores[candidate_id] = float(-np.mean(fold_scores))

        selected_id = min(
            self.candidate_ids,
            key=lambda candidate_id: (
                scores[candidate_id],
                self.candidate_ids.index(candidate_id),
            ),
        )
        self._model = sk["clone"](models[selected_id]).fit(x_fit, y_fit)
        self._features = features
        self._lag2_features = lag2_features
        self._prediction_center = float(np.mean(y_fit))
        self._diagnostics = {
            "selected_model": selected_id,
            "cv_mse": dict(scores),
            "cv_splits": self.cv_splits,
            "random_state": self.random_state,
            "fit_rows": int(len(y_fit)),
            "features_lag1": ",".join(features),
            "features_lag2": ",".join(lag2_features),
            "fit_data": "pre_period_lag2_to_lag1",
            "current_outcome_used_for_model_fit": False,
            "resource_n_jobs": 1,
            "resource_n_estimators": self.n_estimators,
        }
        return self

    def transform(self, data: pd.DataFrame, spec: ExperimentSpec) -> pd.Series:
        if self._model is None or self._prediction_center is None:
            raise RuntimeError("AutoMLTemporalCupacTransformer must be fitted first.")
        require_columns(data, [spec.target_col, *self._features])
        prediction = np.asarray(
            self._model.predict(_numeric_matrix(data, self._features)), dtype=float
        )
        target = np.asarray(
            pd.to_numeric(data[spec.target_col], errors="coerce"), dtype=float
        )
        adjusted = target - (prediction - self._prediction_center)
        self._diagnostics.update(
            {
                "prediction_variance": float(np.nanvar(prediction, ddof=1)),
                "prediction_center": self._prediction_center,
            }
        )
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        fit_data = data if auxiliary_data is None else auxiliary_data
        self.fit(fit_data, spec)
        self._diagnostics["fit_source"] = (
            "same_dataset_pre_period" if auxiliary_data is None else "auxiliary_pre_period"
        )
        return self.transform(data, spec)

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class RobustHistoricalCupedTransformer:
    """CUPED с Huber-наклоном, оценённым только по историческим лагам.

    Коэффициент ``theta`` оценивается из зависимости
    ``pre_target lag1 ~ pre_target lag2``. Экспериментальный outcome нужен
    лишь методу ``transform`` для вычисления ``Y - theta * (X - mean(X))``.
    Поэтому выбросы текущего outcome не могут изменить коэффициент.
    """

    covariate_col: str | None = None
    lag2_suffix: str = "__lag2"
    epsilon: float = 1.35
    alpha: float = 1e-4
    max_iter: int = 200
    name: str = "cuped_huber_historical"
    _theta: float | None = field(default=None, init=False, repr=False)
    _center: float | None = field(default=None, init=False, repr=False)
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def fit(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
    ) -> "RobustHistoricalCupedTransformer":
        covariate = self.covariate_col or spec.pre_target_col
        if covariate is None:
            raise ValueError("Robust historical CUPED requires a pre-target covariate.")
        lag2_col = f"{covariate}{self.lag2_suffix}"
        require_columns(data, [covariate, lag2_col])
        x = np.asarray(pd.to_numeric(data[lag2_col], errors="coerce"), dtype=float)
        y = np.asarray(pd.to_numeric(data[covariate], errors="coerce"), dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 8:
            raise ValueError("Robust historical CUPED needs at least eight lag pairs.")
        x_fit = x[valid]
        y_fit = y[valid]
        x_mean = float(x_fit.mean())
        x_scale = float(x_fit.std(ddof=0))
        if x_scale <= 0.0:
            theta = 0.0
            outlier_fraction = 0.0
            iterations = 0
        else:
            huber = _sklearn_components()["HuberRegressor"](
                epsilon=self.epsilon,
                alpha=self.alpha,
                max_iter=self.max_iter,
                fit_intercept=True,
            )
            huber.fit(((x_fit - x_mean) / x_scale).reshape(-1, 1), y_fit)
            theta = float(huber.coef_[0] / x_scale)
            outlier_fraction = float(np.mean(huber.outliers_))
            iterations = int(huber.n_iter_)
        self._theta = theta
        self._center = float(y_fit.mean())
        self._diagnostics = {
            "theta": theta,
            "covariate": covariate,
            "lag2_covariate": lag2_col,
            "centering_mean": self._center,
            "fit_rows": int(valid.sum()),
            "huber_epsilon": self.epsilon,
            "huber_alpha": self.alpha,
            "huber_outlier_fraction": outlier_fraction,
            "huber_iterations": iterations,
            "fit_data": "pre_period_lag_pairs",
            "current_outcome_used_for_model_fit": False,
        }
        return self

    def transform(self, data: pd.DataFrame, spec: ExperimentSpec) -> pd.Series:
        if self._theta is None or self._center is None:
            raise RuntimeError("RobustHistoricalCupedTransformer must be fitted first.")
        covariate = self.covariate_col or spec.pre_target_col
        require_columns(data, [spec.target_col, covariate])
        target = np.asarray(
            pd.to_numeric(data[spec.target_col], errors="coerce"), dtype=float
        )
        x = np.asarray(pd.to_numeric(data[covariate], errors="coerce"), dtype=float)
        adjusted = target - self._theta * (x - self._center)
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        fit_data = data if auxiliary_data is None else auxiliary_data
        self.fit(fit_data, spec)
        self._diagnostics["fit_source"] = (
            "same_dataset_pre_period" if auxiliary_data is None else "auxiliary_pre_period"
        )
        return self.transform(data, spec)

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics
