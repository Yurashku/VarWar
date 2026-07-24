from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec, require_columns


@dataclass(frozen=True)
class CupacFeaturePolicy:
    """Правило выбора признаков для конкретного варианта CUPAC.

    Зона A содержит только признаки, измеренные до эксперимента. Зона B
    добавляет признаки, заранее признанные нечувствительными к воздействию.
    Зона C добавляет признаки, которые допускаются только после проверки
    баланса между группами.
    """

    method_id: str
    zones: tuple[str, ...]
    safety_status: str
    description: str
    predecessor: str | None = None
    max_abs_smd: float = 0.1
    max_missing_rate_difference: float = 0.02
    max_ks_distance: float = 0.1
    failed_gate: str = "drop"

    def resolve(
        self,
        spec: ExperimentSpec,
        experiment_data: pd.DataFrame | None = None,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        """Вернуть допустимые признаки и диагностику проверки зоны C."""

        zone_set = set(self.zones)
        columns: list[str] = []
        if "A" in zone_set:
            columns.extend(spec.pre_treatment_features())
        if "B" in zone_set:
            columns.extend(spec.expert_safe_in_time_cols)

        admitted_c: list[str] = []
        rejected_c: list[str] = []
        smd_by_feature: dict[str, float] = {}
        missing_by_feature: dict[str, float] = {}
        ks_by_feature: dict[str, float] = {}
        if "C" in zone_set:
            if experiment_data is None:
                raise ValueError(
                    f"{self.method_id} requires experiment data for the zone C gate."
                )
            for column in spec.gated_in_time_cols:
                smd = absolute_standardized_mean_difference(
                    experiment_data,
                    treatment_col=spec.treatment_col,
                    feature_col=column,
                    control_value=spec.control_value,
                    treatment_value=spec.treatment_value,
                )
                smd_by_feature[column] = smd
                missing_difference = absolute_missing_rate_difference(
                    experiment_data,
                    treatment_col=spec.treatment_col,
                    feature_col=column,
                    control_value=spec.control_value,
                    treatment_value=spec.treatment_value,
                )
                ks_distance = two_sample_ks_distance(
                    experiment_data,
                    treatment_col=spec.treatment_col,
                    feature_col=column,
                    control_value=spec.control_value,
                    treatment_value=spec.treatment_value,
                )
                missing_by_feature[column] = missing_difference
                ks_by_feature[column] = ks_distance
                if (
                    np.isfinite(smd)
                    and smd <= self.max_abs_smd
                    and missing_difference <= self.max_missing_rate_difference
                    and ks_distance <= self.max_ks_distance
                ):
                    admitted_c.append(column)
                else:
                    rejected_c.append(column)
            if rejected_c and self.failed_gate == "error":
                raise ValueError(
                    "Zone C balance gate rejected features: "
                    + ", ".join(rejected_c)
                )
            if self.failed_gate not in {"drop", "error"}:
                raise ValueError("failed_gate must be 'drop' or 'error'.")
            columns.extend(admitted_c)

        resolved = _dedupe(tuple(columns))
        diagnostics: dict[str, Any] = {
            "feature_policy": self.method_id,
            "zones": "+".join(self.zones),
            "safety_status": self.safety_status,
            "features_requested": len(
                _dedupe(
                    spec.pre_treatment_features()
                    + (
                        spec.expert_safe_in_time_cols
                        if "B" in zone_set
                        else ()
                    )
                    + (spec.gated_in_time_cols if "C" in zone_set else ())
                )
            ),
            "features_admitted": len(resolved),
            "zone_c_admitted": ",".join(admitted_c),
            "zone_c_rejected": ",".join(rejected_c),
            "zone_c_max_abs_smd": self.max_abs_smd if "C" in zone_set else None,
            "zone_c_max_missing_rate_difference": (
                self.max_missing_rate_difference if "C" in zone_set else None
            ),
            "zone_c_max_ks_distance": (
                self.max_ks_distance if "C" in zone_set else None
            ),
        }
        diagnostics.update(
            {f"zone_c_smd_{column}": value for column, value in smd_by_feature.items()}
        )
        diagnostics.update(
            {
                f"zone_c_missing_rate_difference_{column}": value
                for column, value in missing_by_feature.items()
            }
        )
        diagnostics.update(
            {
                f"zone_c_ks_distance_{column}": value
                for column, value in ks_by_feature.items()
            }
        )
        return resolved, diagnostics


# Совместимое имя для старого публичного импорта.
CupacFeatureSet = CupacFeaturePolicy


def default_cupac_feature_sets(
    spec: ExperimentSpec,
) -> list[CupacFeaturePolicy]:
    """Построить цепочку политик CUPAC: A, A+B и A+B+C.

    Объекты являются самостоятельными примитивами. Зона A задаёт temporal
    CUPAC, а наличие B/C включает отдельные последовательные линейные этапы.
    """

    result = [
        CupacFeaturePolicy(
            method_id="cupac_linear_A",
            zones=("A",),
            safety_status="confirmatory_pre_experiment",
            description="A: только признаки, измеренные до эксперимента",
            predecessor="hypex_ab",
        ),
        CupacFeaturePolicy(
            method_id="cupac_linear_A_plus_B",
            zones=("A", "B"),
            safety_status="exploratory_expert_safe_in_time",
            description=(
                "A+B: temporal CUPAC A, затем отдельная корректировка по "
                "заранее одобренным нечувствительным in-time признакам"
            ),
            predecessor="cupac_linear_A",
        ),
        CupacFeaturePolicy(
            method_id="cupac_linear_A_plus_B_plus_C",
            zones=("A", "B", "C"),
            safety_status="exploratory_balance_gated",
            description=(
                "A+B+C: после A и B отдельная корректировка по признакам, "
                "прошедшим safety-, balance- и missingness-gate"
            ),
            predecessor="cupac_linear_A_plus_B",
        ),
    ]
    return result


def absolute_standardized_mean_difference(
    data: pd.DataFrame,
    *,
    treatment_col: str,
    feature_col: str,
    control_value: Any,
    treatment_value: Any,
) -> float:
    """Вычислить абсолютную стандартизованную разность средних.

    Признак не проходит фильтр, если в одной из групп недостаточно конечных
    значений или объединённое стандартное отклонение равно нулю.
    """

    require_columns(data, [treatment_col, feature_col])
    treatment = np.asarray(data[treatment_col])
    values = np.asarray(pd.to_numeric(data[feature_col], errors="coerce"), dtype=float)
    control = values[(treatment == control_value) & np.isfinite(values)]
    treated = values[(treatment == treatment_value) & np.isfinite(values)]
    if control.size < 2 or treated.size < 2:
        return float("inf")
    variance = (
        (control.size - 1) * control.var(ddof=1)
        + (treated.size - 1) * treated.var(ddof=1)
    ) / (control.size + treated.size - 2)
    if not np.isfinite(variance) or variance <= 0.0:
        return 0.0 if float(control.mean()) == float(treated.mean()) else float("inf")
    return float(abs(treated.mean() - control.mean()) / np.sqrt(variance))


def absolute_missing_rate_difference(
    data: pd.DataFrame,
    *,
    treatment_col: str,
    feature_col: str,
    control_value: Any,
    treatment_value: Any,
) -> float:
    """Вернуть абсолютную разность долей пропусков между группами."""

    require_columns(data, [treatment_col, feature_col])
    treatment = np.asarray(data[treatment_col])
    values = np.asarray(pd.to_numeric(data[feature_col], errors="coerce"), dtype=float)
    control_mask = treatment == control_value
    treated_mask = treatment == treatment_value
    if not control_mask.any() or not treated_mask.any():
        return float("inf")
    control_missing = float(np.mean(~np.isfinite(values[control_mask])))
    treated_missing = float(np.mean(~np.isfinite(values[treated_mask])))
    return abs(treated_missing - control_missing)


def two_sample_ks_distance(
    data: pd.DataFrame,
    *,
    treatment_col: str,
    feature_col: str,
    control_value: Any,
    treatment_value: Any,
) -> float:
    """Посчитать KS-расстояние без калибровки p-value."""

    require_columns(data, [treatment_col, feature_col])
    treatment = np.asarray(data[treatment_col])
    values = np.asarray(pd.to_numeric(data[feature_col], errors="coerce"), dtype=float)
    control = np.sort(values[(treatment == control_value) & np.isfinite(values)])
    treated = np.sort(values[(treatment == treatment_value) & np.isfinite(values)])
    if control.size == 0 or treated.size == 0:
        return float("inf")
    pooled = np.sort(np.concatenate([control, treated]))
    control_cdf = np.searchsorted(control, pooled, side="right") / control.size
    treated_cdf = np.searchsorted(treated, pooled, side="right") / treated.size
    return float(np.max(np.abs(treated_cdf - control_cdf)))


def _dedupe(columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(columns))
