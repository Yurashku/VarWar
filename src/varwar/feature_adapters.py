from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from varwar.core import require_columns


@dataclass
class NumericFeatureBinner:
    """Преобразует числовые признаки в стабильные категориальные интервалы.

    Границы обучаются только на reference-данных и затем без переобучения
    применяются к экспериментальным данным. ``transform`` возвращает только
    новые категориальные колонки, а ``augment`` добавляет их в копию входного
    DataFrame, не меняя оригинал.
    """

    columns: tuple[str, ...]
    strategy: str = "quantile"
    n_bins: int | Mapping[str, int] = 10
    custom_edges: Mapping[str, Sequence[float]] | None = None
    output_suffix: str = "__bin"
    missing_label: str = "missing"
    edges_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    categories_: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)
    diagnostics_: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def _requested_bins(self, column: str) -> int:
        value = self.n_bins[column] if isinstance(self.n_bins, Mapping) else self.n_bins
        bins = int(value)
        if bins < 1:
            raise ValueError(f"n_bins for {column!r} must be at least 1.")
        return bins

    def _fit_edges(self, values: np.ndarray, column: str) -> tuple[np.ndarray, int]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"Column {column!r} has no finite numeric values.")

        if self.strategy == "custom":
            if self.custom_edges is None or column not in self.custom_edges:
                raise ValueError(f"custom_edges must contain {column!r}.")
            internal = np.asarray(self.custom_edges[column], dtype=float)
            if internal.ndim != 1 or not np.isfinite(internal).all():
                raise ValueError(f"Custom edges for {column!r} must be finite and one-dimensional.")
            if internal.size and np.any(np.diff(internal) <= 0):
                raise ValueError(f"Custom edges for {column!r} must be strictly increasing.")
            requested = int(internal.size + 1)
        elif self.strategy == "quantile":
            requested = self._requested_bins(column)
            if requested == 1 or float(finite.min()) == float(finite.max()):
                internal = np.empty(0, dtype=float)
            else:
                quantiles = np.linspace(0.0, 1.0, requested + 1)[1:-1]
                internal = np.unique(np.quantile(finite, quantiles))
        elif self.strategy == "uniform":
            requested = self._requested_bins(column)
            lo = float(finite.min())
            hi = float(finite.max())
            if requested == 1 or lo == hi:
                internal = np.empty(0, dtype=float)
            else:
                internal = np.linspace(lo, hi, requested + 1)[1:-1]
        else:
            raise ValueError("strategy must be 'quantile', 'uniform', or 'custom'.")

        edges = np.concatenate(([-np.inf], np.asarray(internal, dtype=float), [np.inf]))
        return edges, requested

    def fit(self, reference_data: pd.DataFrame) -> "NumericFeatureBinner":
        """Обучить границы интервалов на исторических данных."""

        if not self.columns:
            raise ValueError("NumericFeatureBinner requires at least one column.")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("NumericFeatureBinner columns must be unique.")
        if self.missing_label.startswith("bin_"):
            raise ValueError("missing_label must not collide with generated bin labels.")
        require_columns(reference_data, self.columns)
        output_columns = [f"{column}{self.output_suffix}" for column in self.columns]
        duplicates = set(output_columns).intersection(reference_data.columns)
        if duplicates:
            raise ValueError(f"Output columns already exist: {sorted(duplicates)}")

        self.edges_.clear()
        self.categories_.clear()
        self.diagnostics_.clear()
        for column in self.columns:
            numeric = pd.to_numeric(reference_data[column], errors="coerce")
            values = np.asarray(numeric, dtype=float)
            edges, requested = self._fit_edges(values, column)
            n_actual = int(edges.size - 1)
            labels = tuple(f"bin_{index:03d}" for index in range(n_actual))
            self.edges_[column] = edges
            self.categories_[column] = labels + (self.missing_label,)
            self.diagnostics_[column] = {
                "strategy": self.strategy,
                "requested_bins": requested,
                "actual_bins": n_actual,
                "finite_rows": int(np.isfinite(values).sum()),
                "missing_rows": int((~np.isfinite(values)).sum()),
                "collapsed_edges": int(max(requested - n_actual, 0)),
                "edges": tuple(float(value) for value in edges),
            }
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Преобразовать данные, используя ранее зафиксированные границы."""

        if not self.edges_:
            raise RuntimeError("NumericFeatureBinner must be fitted before transform().")
        require_columns(data, self.columns)
        transformed: dict[str, pd.Categorical] = {}
        for column in self.columns:
            values = np.asarray(pd.to_numeric(data[column], errors="coerce"), dtype=float)
            finite = np.isfinite(values)
            internal = self.edges_[column][1:-1]
            codes = np.searchsorted(internal, values, side="right")
            labels = self.categories_[column][:-1]
            output = np.full(values.size, self.missing_label, dtype=object)
            if finite.any():
                label_array = np.asarray(labels, dtype=object)
                output[finite] = label_array[codes[finite]]
            transformed[f"{column}{self.output_suffix}"] = pd.Categorical(
                output,
                categories=self.categories_[column],
                ordered=False,
            )
        return pd.DataFrame(transformed, index=data.index)

    def fit_transform(self, reference_data: pd.DataFrame) -> pd.DataFrame:
        """Обучить adapter и вернуть только новые категориальные колонки."""

        return self.fit(reference_data).transform(reference_data)

    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Вернуть shallow copy данных с добавленными категориальными колонками."""

        result = data.copy(deep=False)
        transformed = self.transform(data)
        for column in transformed.columns:
            result[column] = transformed[column]
        return result

    @property
    def output_columns_(self) -> tuple[str, ...]:
        if not self.edges_:
            raise RuntimeError("NumericFeatureBinner has not been fitted.")
        return tuple(f"{column}{self.output_suffix}" for column in self.columns)

    def diagnostics(self) -> Mapping[str, Mapping[str, Any]]:
        """Вернуть сведения о выученных границах и потерянных интервалах."""

        return {column: dict(values) for column, values in self.diagnostics_.items()}
