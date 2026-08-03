"""Aggregated historical-panel information for leakage-safe RCT adjustment.

The objects in this module deliberately contain sufficient statistics rather
than entity identifiers or historical rows.  They can therefore be produced
by a SQL aggregation over an ``entity_id x report_date`` panel and passed to a
target transformer without turning the historical pool into extra experiment
observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _numeric_stats(frame: pd.DataFrame, features: tuple[str, ...], target: str) -> dict[str, Any]:
    values = frame.loc[:, [*features, target]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    values = values.loc[valid]
    x = values.loc[:, features].to_numpy(dtype=float)
    y = values[target].to_numpy(dtype=float)
    return {
        "count": int(y.size),
        "x_sum": x.sum(axis=0).tolist(),
        "xx": (x.T @ x).tolist(),
        "y_sum": float(y.sum()),
        "xy": (x.T @ y).tolist(),
        "yy": float(y @ y),
    }


def _add_stats(left: Mapping[str, Any], right: Mapping[str, Any], multiplier: float = 1.0) -> dict[str, Any]:
    return {
        "count": float(left["count"]) + multiplier * float(right["count"]),
        "x_sum": (
            np.asarray(left["x_sum"], dtype=float)
            + multiplier * np.asarray(right["x_sum"], dtype=float)
        ).tolist(),
        "xx": (
            np.asarray(left["xx"], dtype=float)
            + multiplier * np.asarray(right["xx"], dtype=float)
        ).tolist(),
        "y_sum": float(left["y_sum"]) + multiplier * float(right["y_sum"]),
        "xy": (
            np.asarray(left["xy"], dtype=float)
            + multiplier * np.asarray(right["xy"], dtype=float)
        ).tolist(),
        "yy": float(left["yy"]) + multiplier * float(right["yy"]),
    }


@dataclass(frozen=True)
class PanelPriorSummary:
    """Sufficient statistics of a historical panel available before an RCT.

    The summary is treatment-blind and contains no user identifiers.  A
    cutoff is mandatory when ``date_col`` exists: rows on or after the RCT
    start are rejected to make accidental post-treatment leakage explicit.
    ``from_frame`` is intended for synthetic experiments and local checks;
    production SQL should emit the same fields.
    """

    feature_cols: tuple[str, ...]
    target_col: str
    date_col: str | None
    cutoff: str | None
    entity_col: str | None
    n_rows: int
    n_entities: int | None
    date_min: str | None
    date_max: str | None
    stats: Mapping[str, Any]
    monthly_stats: Mapping[str, Mapping[str, Any]]
    category_shares: Mapping[str, Mapping[str, float]]
    stratum_shares: Mapping[str, float]
    metadata: Mapping[str, Any]

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        feature_cols: tuple[str, ...],
        target_col: str,
        date_col: str | None = "report_date",
        cutoff: Any | None = None,
        entity_col: str | None = "unit_id",
        category_cols: tuple[str, ...] = (),
        stratum_col: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PanelPriorSummary":
        required = [*feature_cols, target_col]
        if date_col is not None:
            required.append(date_col)
        if entity_col is not None:
            required.append(entity_col)
        required.extend(category_cols)
        if stratum_col is not None:
            required.append(stratum_col)
        missing = [column for column in dict.fromkeys(required) if column not in frame.columns]
        if missing:
            raise ValueError(f"Panel prior is missing columns: {missing}")
        if not feature_cols:
            raise ValueError("PanelPriorSummary requires at least one feature.")

        working = frame.copy(deep=False)
        parsed_dates: pd.Series | None = None
        cutoff_value: pd.Timestamp | None = None
        if date_col is not None:
            parsed_dates = pd.to_datetime(working[date_col], errors="coerce")
            if parsed_dates.isna().any():
                raise ValueError("Panel prior date column contains invalid dates.")
            if cutoff is None:
                raise ValueError("cutoff is required when date_col is provided.")
            cutoff_value = pd.Timestamp(cutoff)
            if not (parsed_dates < cutoff_value).all():
                raise ValueError(
                    "Panel prior contains rows on or after the experiment cutoff; "
                    "post-treatment or future rows must be removed before aggregation."
                )
            if working.empty:
                raise ValueError("No historical rows remain before the cutoff.")

        stats = _numeric_stats(working, feature_cols, target_col)
        if int(stats["count"]) < len(feature_cols) + 2:
            raise ValueError("Panel prior needs more finite rows than model columns.")

        monthly: dict[str, Mapping[str, Any]] = {}
        if date_col is not None and parsed_dates is not None:
            periods = parsed_dates.loc[working.index].dt.to_period("M").astype(str)
            for month, indices in periods.groupby(periods).groups.items():
                monthly[str(month)] = _numeric_stats(
                    working.loc[indices], feature_cols, target_col
                )

        shares: dict[str, Mapping[str, float]] = {}
        for column in category_cols:
            counts = working[column].astype("string").fillna("<NA>").value_counts(normalize=True)
            shares[column] = {str(key): float(value) for key, value in counts.items()}
        stratum_shares: Mapping[str, float] = {}
        if stratum_col is not None:
            counts = working[stratum_col].astype("string").fillna("<NA>").value_counts(normalize=True)
            stratum_shares = {str(key): float(value) for key, value in counts.items()}

        dates = parsed_dates.loc[working.index] if parsed_dates is not None else None
        return cls(
            feature_cols=tuple(feature_cols),
            target_col=target_col,
            date_col=date_col,
            cutoff=None if cutoff_value is None else cutoff_value.isoformat(),
            entity_col=entity_col,
            n_rows=int(len(working)),
            n_entities=(int(working[entity_col].nunique()) if entity_col is not None else None),
            date_min=(dates.min().isoformat() if dates is not None else None),
            date_max=(dates.max().isoformat() if dates is not None else None),
            stats=stats,
            monthly_stats=monthly,
            category_shares=shares,
            stratum_shares=stratum_shares,
            metadata=dict(metadata or {}),
        )

    def stats_for(self, feature_cols: tuple[str, ...]) -> Mapping[str, Any]:
        if tuple(feature_cols) != self.feature_cols:
            raise ValueError(
                "Panel prior feature order differs from the fitted summary: "
                f"expected {self.feature_cols}, got {tuple(feature_cols)}."
            )
        return self.stats

    def validation_month(self) -> str | None:
        return max(self.monthly_stats) if self.monthly_stats else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_cols": list(self.feature_cols),
            "target_col": self.target_col,
            "date_col": self.date_col,
            "cutoff": self.cutoff,
            "entity_col": self.entity_col,
            "n_rows": self.n_rows,
            "n_entities": self.n_entities,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "stats": self.stats,
            "monthly_stats": self.monthly_stats,
            "category_shares": self.category_shares,
            "stratum_shares": self.stratum_shares,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PanelPriorSummary":
        return cls(
            feature_cols=tuple(payload["feature_cols"]),
            target_col=str(payload["target_col"]),
            date_col=payload.get("date_col"),
            cutoff=payload.get("cutoff"),
            entity_col=payload.get("entity_col"),
            n_rows=int(payload["n_rows"]),
            n_entities=(None if payload.get("n_entities") is None else int(payload["n_entities"])),
            date_min=payload.get("date_min"),
            date_max=payload.get("date_max"),
            stats=payload["stats"],
            monthly_stats=payload.get("monthly_stats", {}),
            category_shares=payload.get("category_shares", {}),
            stratum_shares=payload.get("stratum_shares", {}),
            metadata=payload.get("metadata", {}),
        )

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    @classmethod
    def from_json(cls, path_or_text: str | Path) -> "PanelPriorSummary":
        candidate = str(path_or_text)
        try:
            is_path = Path(candidate).exists()
        except OSError:
            is_path = False
        value = Path(candidate).read_text(encoding="utf-8") if is_path else candidate
        return cls.from_dict(json.loads(value))

    def to_parquet(self, path: str | Path) -> None:
        pd.DataFrame({"panel_prior_json": [self.to_json()]}).to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "PanelPriorSummary":
        value = pd.read_parquet(path).loc[0, "panel_prior_json"]
        return cls.from_dict(json.loads(str(value)))


def combine_linear_stats(
    current: Mapping[str, Any],
    prior: Mapping[str, Any],
    prior_strength: float,
) -> dict[str, Any]:
    """Combine experiment sufficient statistics with a discounted prior pool."""

    if prior_strength < 0.0 or not np.isfinite(prior_strength):
        raise ValueError("prior_strength must be finite and non-negative.")
    return _add_stats(current, prior, prior_strength)


def fit_linear_from_stats(
    stats: Mapping[str, Any],
    *,
    ridge: float = 1e-8,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit an intercept plus ridge-regularized slopes from raw sufficient stats."""

    n = float(stats["count"])
    if n <= 1.0:
        raise ValueError("Linear sufficient statistics need at least two observations.")
    x_sum = np.asarray(stats["x_sum"], dtype=float)
    xx = np.asarray(stats["xx"], dtype=float)
    y_sum = float(stats["y_sum"])
    xy = np.asarray(stats["xy"], dtype=float)
    x_mean = x_sum / n
    y_mean = y_sum / n
    centered_xx = xx - np.outer(x_sum, x_sum) / n
    centered_xy = xy - x_sum * y_sum / n
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative.")
    penalty = ridge * np.eye(centered_xx.shape[0], dtype=float)
    beta = np.linalg.solve(centered_xx + penalty, centered_xy)
    intercept = float(y_mean - x_mean @ beta)
    return intercept, beta, x_mean, np.sqrt(np.maximum(np.diag(centered_xx) / n, 1e-12))


def linear_stats_from_frame(
    frame: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
) -> dict[str, Any]:
    return _numeric_stats(frame, feature_cols, target_col)


def linear_mse_from_stats(
    stats: Mapping[str, Any],
    intercept: float,
    beta: np.ndarray,
) -> float:
    n = float(stats["count"])
    x_sum = np.asarray(stats["x_sum"], dtype=float)
    xx = np.asarray(stats["xx"], dtype=float)
    y_sum = float(stats["y_sum"])
    xy = np.asarray(stats["xy"], dtype=float)
    yy = float(stats["yy"])
    value = (
        yy
        + n * intercept**2
        + beta @ xx @ beta
        + 2.0 * intercept * (beta @ x_sum - y_sum)
        - 2.0 * beta @ xy
    )
    return float(max(value / n, 0.0))
