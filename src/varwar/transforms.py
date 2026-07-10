from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec, require_columns


def _as_matrix(data: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([np.asarray(data[col], dtype=float) for col in columns])


@dataclass
class CupedTransformer:
    covariate_col: str | None = None
    name: str = "cuped"
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        covariate = self.covariate_col or spec.pre_target_col
        if covariate is None:
            raise ValueError("CUPED requires covariate_col or spec.pre_target_col.")
        fit_data = auxiliary_data if auxiliary_data is not None else data
        require_columns(fit_data, [spec.target_col, covariate])
        require_columns(data, [spec.target_col, covariate])

        y_fit = np.asarray(fit_data[spec.target_col], dtype=float)
        x_fit = np.asarray(fit_data[covariate], dtype=float)
        valid_fit = np.isfinite(y_fit) & np.isfinite(x_fit)
        if valid_fit.sum() < 2:
            raise ValueError("CUPED needs at least two finite fit rows.")
        yv = y_fit[valid_fit]
        xv = x_fit[valid_fit]
        x_mean = float(xv.mean())
        x_centered = xv - x_mean
        var_x = float(np.dot(x_centered, x_centered) / max(xv.size - 1, 1))
        if var_x <= 0:
            theta = 0.0
        else:
            y_centered = yv - float(yv.mean())
            cov_yx = float(np.dot(y_centered, x_centered) / max(xv.size - 1, 1))
            theta = cov_yx / var_x

        y = np.asarray(data[spec.target_col], dtype=float)
        x = np.asarray(data[covariate], dtype=float)
        adjusted = y - theta * (x - x_mean)
        self._diagnostics = {
            "covariate": covariate,
            "theta": theta,
            "fit_rows": int(valid_fit.sum()),
        }
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class LinearCupacTransformer:
    feature_cols: tuple[str, ...] | None = None
    ridge: float = 1e-8
    fit_on: str = "control"
    name: str = "cupac_linear"
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def _resolve_features(self, spec: ExperimentSpec) -> tuple[str, ...]:
        if self.feature_cols is not None:
            return self.feature_cols
        features = list(spec.pre_treatment_features())
        if not features:
            raise ValueError("Linear CUPAC requires feature_cols, covariate_cols, or pre_target_col.")
        return tuple(features)

    def _fit_frame(self, data: pd.DataFrame, spec: ExperimentSpec) -> pd.DataFrame:
        if self.fit_on == "all":
            return data
        if self.fit_on == "control":
            require_columns(data, [spec.treatment_col])
            return data.loc[np.asarray(data[spec.treatment_col]) == spec.control_value]
        raise ValueError(f"Unknown fit_on policy: {self.fit_on}")

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        features = self._resolve_features(spec)
        fit_data = auxiliary_data if auxiliary_data is not None else self._fit_frame(data, spec)
        require_columns(fit_data, [spec.target_col, *features])
        require_columns(data, [spec.target_col, *features])

        x_fit = _as_matrix(fit_data, features)
        y_fit = np.asarray(fit_data[spec.target_col], dtype=float)
        valid = np.isfinite(y_fit) & np.isfinite(x_fit).all(axis=1)
        if valid.sum() <= len(features) + 1:
            raise ValueError("Linear CUPAC needs more finite fit rows than features.")

        x_valid = x_fit[valid]
        y_valid = y_fit[valid]
        x_mean = x_valid.mean(axis=0)
        y_mean = float(y_valid.mean())
        x_centered = x_valid - x_mean
        y_centered = y_valid - y_mean

        xtx = x_centered.T @ x_centered
        if self.ridge > 0:
            xtx = xtx + np.eye(xtx.shape[0]) * self.ridge
        beta = np.linalg.solve(xtx, x_centered.T @ y_centered)
        pred_fit = y_mean + (x_centered @ beta)
        pred_mean = float(pred_fit.mean())
        pred_centered = pred_fit - pred_mean
        pred_var = float(np.dot(pred_centered, pred_centered) / max(pred_fit.size - 1, 1))
        if pred_var <= 0 or not math.isfinite(pred_var):
            theta = 0.0
        else:
            cov = float(np.dot(y_valid - y_mean, pred_centered) / max(pred_fit.size - 1, 1))
            theta = cov / pred_var

        x_apply = _as_matrix(data, features)
        pred = y_mean + ((x_apply - x_mean) @ beta)
        y = np.asarray(data[spec.target_col], dtype=float)
        adjusted = y - theta * (pred - pred_mean)
        self._diagnostics = {
            "features": ",".join(features),
            "ridge": self.ridge,
            "fit_on": "auxiliary" if auxiliary_data is not None else self.fit_on,
            "fit_rows": int(valid.sum()),
            "theta": theta,
            "prediction_variance": pred_var,
        }
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class WinsorizeTransformer:
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
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
        fit_data = auxiliary_data if auxiliary_data is not None else data
        require_columns(fit_data, [spec.target_col])
        require_columns(data, [spec.target_col])
        fit_y = np.asarray(fit_data[spec.target_col], dtype=float)
        fit_y = fit_y[np.isfinite(fit_y)]
        if fit_y.size < 2:
            raise ValueError("WinsorizeTransformer needs at least two finite fit rows.")
        lo, hi = np.quantile(fit_y, [self.lower_quantile, self.upper_quantile])
        y = np.asarray(data[spec.target_col], dtype=float)
        adjusted = np.clip(y, lo, hi)
        self._diagnostics = {
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "lower_value": float(lo),
            "upper_value": float(hi),
        }
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics
