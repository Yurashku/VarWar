from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec, require_columns


@dataclass
class PrognosticScoreBlockBuilder:
    """Build randomization blocks from an honest pre-treatment prognostic score.

    A ridge model is fitted on a historical pre-treatment target. Its encoding,
    coefficients and score cut points are frozen and reused by ``transform``.
    Neither the experiment outcome nor treatment labels are accepted as inputs.
    """

    feature_cols: tuple[str, ...] | None = None
    reference_target_col: str | None = None
    n_blocks: int = 10
    ridge: float = 1e-6
    output_name: str = "prognostic_block"
    numeric_cols_: tuple[str, ...] = field(default=(), init=False)
    categorical_cols_: tuple[str, ...] = field(default=(), init=False)
    numeric_location_: dict[str, float] = field(default_factory=dict, init=False)
    numeric_scale_: dict[str, float] = field(default_factory=dict, init=False)
    categorical_levels_: dict[str, tuple[Any, ...]] = field(default_factory=dict, init=False)
    coefficients_: np.ndarray | None = field(default=None, init=False, repr=False)
    score_edges_: np.ndarray | None = field(default=None, init=False, repr=False)
    diagnostics_: dict[str, Any] = field(default_factory=dict, init=False)
    last_transform_diagnostics_: dict[str, Any] = field(default_factory=dict, init=False)

    @staticmethod
    def _categorical_values(series: pd.Series) -> pd.Series:
        return series.astype("object").where(series.notna(), "__missing__")

    def _resolve_contract(
        self, data: pd.DataFrame, spec: ExperimentSpec
    ) -> tuple[tuple[str, ...], str]:
        allowed = tuple(dict.fromkeys(spec.pre_treatment_features()))
        target = self.reference_target_col or spec.pre_target_col
        if target is None:
            raise ValueError("A historical pre-treatment target column is required.")
        if target == spec.target_col:
            raise ValueError("The experiment outcome cannot train a randomization structure.")
        if target not in allowed:
            raise ValueError(
                "reference_target_col must be declared in ExperimentSpec pre-treatment columns."
            )
        columns = self.feature_cols or tuple(column for column in allowed if column != target)
        if not columns:
            raise ValueError("At least one pre-treatment predictor is required.")
        unknown = sorted(set(columns).difference(allowed))
        if unknown:
            raise ValueError(f"Only pre-treatment predictors are allowed: {unknown}.")
        forbidden = {target, spec.target_col, spec.treatment_col}
        overlap = sorted(set(columns).intersection(forbidden))
        if overlap:
            raise ValueError(f"Target/treatment columns cannot be predictors: {overlap}.")
        require_columns(data, (*columns, target))
        return tuple(columns), target

    def _fit_matrix(self, data: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        numeric = tuple(
            column for column in columns if pd.api.types.is_numeric_dtype(data[column])
        )
        categorical = tuple(column for column in columns if column not in numeric)
        self.numeric_cols_ = numeric
        self.categorical_cols_ = categorical
        arrays: list[np.ndarray] = [np.ones((len(data), 1), dtype=float)]

        for column in numeric:
            values = np.asarray(pd.to_numeric(data[column], errors="coerce"), dtype=float)
            finite = np.isfinite(values)
            location = float(np.mean(values[finite])) if finite.any() else 0.0
            filled = np.where(finite, values, location)
            scale = float(np.std(filled, ddof=0))
            if not np.isfinite(scale) or scale == 0.0:
                scale = 1.0
            self.numeric_location_[column] = location
            self.numeric_scale_[column] = scale
            arrays.append(((filled - location) / scale)[:, None])
            arrays.append((~finite).astype(float)[:, None])

        for column in categorical:
            values = self._categorical_values(data[column])
            levels = tuple(pd.unique(values))
            self.categorical_levels_[column] = levels
            for level in levels[1:]:
                arrays.append((values.to_numpy() == level).astype(float)[:, None])
        return np.column_stack(arrays)

    def _transform_matrix(self, data: pd.DataFrame) -> tuple[np.ndarray, int]:
        columns = (*self.numeric_cols_, *self.categorical_cols_)
        require_columns(data, columns)
        arrays: list[np.ndarray] = [np.ones((len(data), 1), dtype=float)]
        unknown_rows = np.zeros(len(data), dtype=bool)
        for column in self.numeric_cols_:
            values = np.asarray(pd.to_numeric(data[column], errors="coerce"), dtype=float)
            finite = np.isfinite(values)
            location = self.numeric_location_[column]
            filled = np.where(finite, values, location)
            arrays.append(((filled - location) / self.numeric_scale_[column])[:, None])
            arrays.append((~finite).astype(float)[:, None])
        for column in self.categorical_cols_:
            values = self._categorical_values(data[column]).to_numpy()
            levels = self.categorical_levels_[column]
            unknown_rows |= ~np.isin(values, np.asarray(levels, dtype=object))
            for level in levels[1:]:
                arrays.append((values == level).astype(float)[:, None])
        return np.column_stack(arrays), int(unknown_rows.sum())

    def fit(
        self, reference_data: pd.DataFrame, spec: ExperimentSpec
    ) -> "PrognosticScoreBlockBuilder":
        """Fit a frozen score model and quantile cut points on historical data."""

        if self.n_blocks < 2:
            raise ValueError("n_blocks must be at least 2.")
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative.")
        columns, target = self._resolve_contract(reference_data, spec)
        y = np.asarray(pd.to_numeric(reference_data[target], errors="coerce"), dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("The historical target must contain only finite numeric values.")
        if len(reference_data) < 2:
            raise ValueError("At least two historical rows are required.")

        self.numeric_location_.clear()
        self.numeric_scale_.clear()
        self.categorical_levels_.clear()
        matrix = self._fit_matrix(reference_data, columns)
        penalty = np.eye(matrix.shape[1], dtype=float) * self.ridge
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.pinv(matrix.T @ matrix + penalty) @ (matrix.T @ y)
        scores = matrix @ self.coefficients_
        score_scale = max(float(np.max(np.abs(scores))), 1.0)
        tolerance = np.finfo(float).eps * score_scale * 64.0
        if float(np.ptp(scores)) <= tolerance:
            self.score_edges_ = np.empty(0, dtype=float)
        else:
            quantiles = np.linspace(0.0, 1.0, self.n_blocks + 1)[1:-1]
            raw_edges = np.sort(np.quantile(scores, quantiles))
            kept: list[float] = []
            for edge in raw_edges:
                if not kept or float(edge) - kept[-1] > tolerance:
                    kept.append(float(edge))
            self.score_edges_ = np.asarray(kept, dtype=float)
        actual_blocks = int(self.score_edges_.size + 1)
        self.diagnostics_ = {
            "feature_cols": columns,
            "reference_target_col": target,
            "requested_blocks": self.n_blocks,
            "actual_blocks": actual_blocks,
            "collapsed_boundaries": self.n_blocks - actual_blocks,
            "n_reference_rows": len(reference_data),
            "n_model_columns": matrix.shape[1],
            "ridge": self.ridge,
            "training_score_variance": float(np.var(scores, ddof=1)),
        }
        self.last_transform_diagnostics_.clear()
        return self

    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Apply the frozen score model and return stable categorical block labels."""

        if self.coefficients_ is None or self.score_edges_ is None:
            raise RuntimeError("PrognosticScoreBlockBuilder must be fitted first.")
        matrix, unknown_rows = self._transform_matrix(data)
        scores = matrix @ self.coefficients_
        codes = np.searchsorted(self.score_edges_, scores, side="right")
        categories = tuple(
            f"block_{index:03d}" for index in range(self.score_edges_.size + 1)
        )
        labels = pd.Categorical.from_codes(codes, categories=categories, ordered=True)
        self.last_transform_diagnostics_ = {
            "n_rows": len(data),
            "unknown_category_rows": unknown_rows,
            "score_min": float(np.min(scores)) if len(scores) else np.nan,
            "score_max": float(np.max(scores)) if len(scores) else np.nan,
        }
        return pd.Series(labels, index=data.index, name=self.output_name)

    def fit_transform(self, reference_data: pd.DataFrame, spec: ExperimentSpec) -> pd.Series:
        return self.fit(reference_data, spec).transform(reference_data)

    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy(deep=False)
        result[self.output_name] = self.transform(data)
        return result

    def diagnostics(self) -> Mapping[str, Any]:
        return {**self.diagnostics_, **self.last_transform_diagnostics_}
