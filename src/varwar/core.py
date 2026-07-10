from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    """Column contract shared by synthetic and real-data estimators."""

    treatment_col: str = "treatment"
    target_col: str = "y"
    control_value: Any = 0
    treatment_value: Any = 1
    unit_col: str | None = None
    covariate_cols: tuple[str, ...] = ()
    pre_treatment_cols: tuple[str, ...] = ()
    expert_safe_in_time_cols: tuple[str, ...] = ()
    gated_in_time_cols: tuple[str, ...] = ()
    unsafe_demo_cols: tuple[str, ...] = ()
    pre_target_col: str | None = "y_pre"
    numerator_col: str | None = None
    denominator_col: str | None = None
    stratum_col: str | None = None
    instrument_col: str | None = None
    received_treatment_col: str | None = None
    alpha: float = 0.05

    def with_target(self, target_col: str) -> "ExperimentSpec":
        return replace(self, target_col=target_col)

    def pre_treatment_features(self) -> tuple[str, ...]:
        if self.pre_treatment_cols:
            return self.pre_treatment_cols
        features = list(self.covariate_cols)
        if self.pre_target_col and self.pre_target_col not in features:
            features.insert(0, self.pre_target_col)
        return tuple(features)


@dataclass(frozen=True)
class EstimateResult:
    estimator: str
    effect: float
    standard_error: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    n_control: int
    n_treatment: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        row = {
            "estimator": self.estimator,
            "effect": self.effect,
            "standard_error": self.standard_error,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
        }
        row.update({f"diag_{key}": value for key, value in self.diagnostics.items()})
        return row


class Estimator(Protocol):
    name: str

    def estimate(self, data: pd.DataFrame, spec: ExperimentSpec) -> EstimateResult:
        ...


class TargetTransformer(Protocol):
    name: str

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        ...

    def diagnostics(self) -> Mapping[str, Any]:
        ...


@dataclass
class WrappedEstimator:
    """Compose a target adjustment with a base estimator."""

    transformer: TargetTransformer
    base_estimator: Estimator
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"{self.transformer.name}+{self.base_estimator.name}"

    def estimate(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> EstimateResult:
        adjusted_col = f"__varwar_{self.name}_target"
        adjusted = self.transformer.fit_transform(data, spec, auxiliary_data)
        adjusted_data = data.copy(deep=False)
        adjusted_data[adjusted_col] = adjusted
        result = self.base_estimator.estimate(
            adjusted_data,
            spec.with_target(adjusted_col),
        )
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "base_estimator": self.base_estimator.name,
                "transformer": self.transformer.name,
                **{
                    f"transformer_{key}": value
                    for key, value in self.transformer.diagnostics().items()
                },
            }
        )
        return EstimateResult(
            estimator=self.name or "",
            effect=result.effect,
            standard_error=result.standard_error,
            statistic=result.statistic,
            p_value=result.p_value,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            n_control=result.n_control,
            n_treatment=result.n_treatment,
            diagnostics=diagnostics,
        )


def require_columns(data: pd.DataFrame, columns: Sequence[str | None]) -> None:
    missing = [col for col in columns if col is not None and col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
