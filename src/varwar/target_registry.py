from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from varwar.advanced_transforms import (
    AutoMLTemporalCupacTransformer,
    RobustHistoricalCupedTransformer,
)
from varwar.analysis import CausalSafety, InformationMode, TemporalAccess, TheoryGrade
from varwar.core import ExperimentSpec, require_columns
from varwar.feature_sets import CupacFeaturePolicy
from varwar.hypex_backend import hypex_package_version, hypex_requirement
from varwar.transforms import (
    CupedTransformer,
    LinearCupacTransformer,
    LinearInTimeAdjustmentTransformer,
    PanelPriorCupacTransformer,
    QuadraticCupacTransformer,
    SequentialTargetTransformer,
    TemporalLinearCupacTransformer,
    target_variance_ratio,
)


TargetFactory = Callable[[], Any]


@dataclass(frozen=True)
class TargetTransformDefinition:
    """Machine-readable contract of one target transformation."""

    method_id: str
    title_ru: str
    information_modes: frozenset[InformationMode]
    minimum_temporal_access: TemporalAccess
    causal_safety: CausalSafety
    theory_grade: TheoryGrade
    required_columns: tuple[str, ...]
    factory: TargetFactory | None
    available: bool = True
    unavailable_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def create(self) -> Any:
        """Create a fresh transformer or fail explicitly for unavailable entries."""

        if not self.available or self.factory is None:
            reason = self.unavailable_reason or "target transformer is unavailable"
            raise RuntimeError(f"{self.method_id}: {reason}")
        return self.factory()


@dataclass
class _IdentityTargetTransformer:
    name: str = "identity"

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: Any | None = None,
    ) -> pd.Series:
        del auxiliary_data
        require_columns(data, [spec.target_col])
        return pd.Series(data[spec.target_col], index=data.index, name=spec.target_col)

    def diagnostics(self) -> Mapping[str, Any]:
        return {"target_changed": False}


@dataclass
class _HistoricalCupedTransformer:
    """Ordinary CUPED slope fitted on the pre-period lag transition only."""

    covariate_col: str | None = None
    lag2_suffix: str = "__lag2"
    name: str = "cuped_historical"
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        fit_frame = data if auxiliary_data is None else auxiliary_data
        covariate = self.covariate_col or spec.pre_target_col
        if covariate is None:
            raise ValueError("Historical CUPED requires spec.pre_target_col.")
        lag2_col = f"{covariate}{self.lag2_suffix}"
        require_columns(fit_frame, [covariate, lag2_col])
        require_columns(data, [spec.target_col, covariate])
        lag2 = pd.to_numeric(fit_frame[lag2_col], errors="coerce").to_numpy(float)
        lag1 = pd.to_numeric(fit_frame[covariate], errors="coerce").to_numpy(float)
        valid = np.isfinite(lag2) & np.isfinite(lag1)
        if int(valid.sum()) < 2:
            raise ValueError("Historical CUPED needs at least two finite lag pairs.")
        x = lag2[valid]
        y = lag1[valid]
        x_centered = x - float(x.mean())
        denominator = float(x_centered @ x_centered)
        theta = (
            float(x_centered @ (y - float(y.mean())) / denominator)
            if denominator > 0.0
            else 0.0
        )
        center = float(y.mean())
        target = pd.to_numeric(data[spec.target_col], errors="coerce").to_numpy(float)
        current_pre = pd.to_numeric(data[covariate], errors="coerce").to_numpy(float)
        adjusted = target - theta * (current_pre - center)
        self._diagnostics = {
            "theta": theta,
            "fit_rows": int(valid.sum()),
            "fit_data": "pre_period_lag2_to_lag1",
            "current_outcome_used_for_model_fit": False,
            **target_variance_ratio(data, spec, target, adjusted),
        }
        return pd.Series(adjusted, index=data.index, name=f"{spec.target_col}_{self.name}")

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


@dataclass
class _ZonedCupacComposition:
    """Compose a family-specific A model with separate control-fitted B/C stages."""

    method_id: str
    a_factory: TargetFactory
    zones: tuple[str, ...]
    ridge: float = 1e-8
    name: str = field(init=False)
    _delegate: SequentialTargetTransformer | None = field(default=None, init=False)
    _diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.name = self.method_id

    def fit_transform(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        auxiliary_data: Any | None = None,
    ) -> pd.Series:
        policy = CupacFeaturePolicy(
            method_id=self.method_id,
            zones=self.zones,
            safety_status=(
                "post_treatment_risk"
                if "C" in self.zones
                else "conditional" if "B" in self.zones else "pre_treatment"
            ),
            description="Registry-defined sequential CUPAC composition.",
        )
        _, policy_diagnostics = policy.resolve(spec, data)
        stages: list[Any] = [self.a_factory()]
        if "B" in self.zones:
            if not spec.expert_safe_in_time_cols:
                raise ValueError(f"{self.method_id} requires zone B columns.")
            stages.append(
                LinearInTimeAdjustmentTransformer(
                    feature_cols=tuple(spec.expert_safe_in_time_cols),
                    fit_data="experiment_control",
                    ridge=self.ridge,
                    name=f"{self.method_id}__stage_B",
                )
            )
        if "C" in self.zones:
            gate_admitted = tuple(
                column
                for column in str(policy_diagnostics["zone_c_admitted"]).split(",")
                if column
            )
            # The gate is diagnostic, not a proof of causal safety.  The
            # full-timeline research track deliberately runs declared C
            # features even when an empirical balance check warns about them.
            admitted = tuple(spec.gated_in_time_cols)
            if not admitted:
                raise ValueError(f"{self.method_id} requires declared zone C columns.")
            policy_diagnostics["zone_c_gate_admitted"] = ",".join(gate_admitted)
            policy_diagnostics["zone_c_executed_despite_gate"] = bool(
                set(admitted).difference(gate_admitted)
            )
            stages.append(
                LinearInTimeAdjustmentTransformer(
                    feature_cols=admitted,
                    fit_data="experiment_control",
                    ridge=self.ridge,
                    name=f"{self.method_id}__stage_C",
                )
            )
        self._delegate = SequentialTargetTransformer(tuple(stages), self.method_id)
        result = self._delegate.fit_transform(data, spec, auxiliary_data)
        self._diagnostics = {
            **policy_diagnostics,
            **dict(self._delegate.diagnostics()),
            "stages_are_separate": True,
            "control_fitted_stages": "+".join(zone for zone in self.zones if zone != "A"),
        }
        return result

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


def _pooled_cuped_factory() -> CupedTransformer:
    return CupedTransformer(fit_data="experiment_all", name="cuped_pooled_theta")


def _pooled_cupac_factory() -> LinearCupacTransformer:
    return LinearCupacTransformer(theta_policy="pooled", name="cupac_linear_pooled_theta_A")


def _zoned_factory(method_id: str, a_factory: TargetFactory, zones: tuple[str, ...]) -> TargetFactory:
    return lambda: _ZonedCupacComposition(method_id, a_factory, zones)


def _cupac_definition(
    *,
    family_id: str,
    family_title: str,
    a_factory: TargetFactory | None,
    information_modes: frozenset[InformationMode],
    base_grade: TheoryGrade,
    base_safety: CausalSafety = CausalSafety.SAFE,
    available: bool = True,
    unavailable_reason: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> list[TargetTransformDefinition]:
    definitions: list[TargetTransformDefinition] = []
    variants = (
        ("A", ("A",), TemporalAccess.PRE_TREATMENT_ONLY, base_safety, base_grade),
        (
            "A_plus_B",
            ("A", "B"),
            TemporalAccess.FULL_TIMELINE,
            CausalSafety.CONDITIONAL,
            max(base_grade, TheoryGrade.C),
        ),
        (
            "A_plus_B_plus_C",
            ("A", "B", "C"),
            TemporalAccess.FULL_TIMELINE,
            CausalSafety.POST_TREATMENT_RISK,
            TheoryGrade.D,
        ),
    )
    for suffix, zones, temporal_access, safety, grade in variants:
        method_id = (
            f"panel_prior_cupac_{suffix}"
            if family_id == "panel_prior"
            else f"cupac_{family_id}_{suffix}"
        )
        factory = (
            _zoned_factory(method_id, a_factory, zones)
            if available and a_factory is not None
            else None
        )
        required = ["target", "pre_target", "pre_treatment_features", "lag2_pre_features"]
        if "B" in zones:
            required.extend(("treatment", "expert_safe_in_time_features"))
        if "C" in zones:
            required.append("gated_in_time_features")
        if family_id == "panel_prior":
            required.append("panel_prior_summary")
        definitions.append(
            TargetTransformDefinition(
                method_id=method_id,
                title_ru=f"CUPAC {family_title}, зоны {'+'.join(zones)}",
                information_modes=information_modes,
                minimum_temporal_access=temporal_access,
                causal_safety=safety,
                theory_grade=grade,
                required_columns=tuple(required),
                factory=factory,
                available=available,
                unavailable_reason=unavailable_reason,
                metadata=MappingProxyType(
                    {
                        "family_id": family_id,
                        "zones": zones,
                        "b_c_are_separate_control_fitted_stages": True,
                        **dict(metadata or {}),
                    }
                ),
            )
        )
    return definitions


def build_target_registry() -> Mapping[str, TargetTransformDefinition]:
    """Build the canonical target-transform registry with stable method IDs."""

    experiment_modes = frozenset(InformationMode)
    panel_only = frozenset({InformationMode.PANEL_PRIOR})
    definitions = [
        TargetTransformDefinition(
            method_id="identity",
            title_ru="Без преобразования target",
            information_modes=experiment_modes,
            minimum_temporal_access=TemporalAccess.PRE_TREATMENT_ONLY,
            causal_safety=CausalSafety.SAFE,
            theory_grade=TheoryGrade.A,
            required_columns=("target",),
            factory=_IdentityTargetTransformer,
            metadata=MappingProxyType({"baseline": True}),
        ),
        TargetTransformDefinition(
            method_id="cuped_historical",
            title_ru="Исторический CUPED",
            information_modes=experiment_modes,
            minimum_temporal_access=TemporalAccess.PRE_TREATMENT_ONLY,
            causal_safety=CausalSafety.SAFE,
            theory_grade=TheoryGrade.B,
            required_columns=("target", "pre_target", "pre_target__lag2"),
            factory=_HistoricalCupedTransformer,
            metadata=MappingProxyType(
                {"theta_fit": "pre_target_lag2_to_lag1", "trial_outcome_fit": False}
            ),
        ),
        TargetTransformDefinition(
            method_id="cuped_huber_historical",
            title_ru="Исторический CUPED с Huber-наклоном",
            information_modes=experiment_modes,
            minimum_temporal_access=TemporalAccess.PRE_TREATMENT_ONLY,
            causal_safety=CausalSafety.SAFE,
            theory_grade=TheoryGrade.C,
            required_columns=("target", "pre_target", "pre_target__lag2"),
            factory=RobustHistoricalCupedTransformer,
            metadata=MappingProxyType(
                {"theta_fit": "Huber_pre_target_lag2_to_lag1", "trial_outcome_fit": False}
            ),
        ),
        TargetTransformDefinition(
            method_id="cuped_pooled_theta",
            title_ru="CUPED с theta по текущему эксперименту (обе группы)",
            information_modes=experiment_modes,
            minimum_temporal_access=TemporalAccess.PRE_TREATMENT_ONLY,
            causal_safety=CausalSafety.CONDITIONAL,
            theory_grade=TheoryGrade.B,
            required_columns=("target", "pre_target"),
            factory=_pooled_cuped_factory,
            metadata=MappingProxyType(
                {
                    "theta_fit": "Cov(Y, pre_target) / Var(pre_target), experiment_all",
                    "trial_outcome_fit": True,
                    "trial_outcome_used_for_theta": True,
                    "variance_guarantee": "asymptotically never above Var(Y)",
                }
            ),
        ),
        TargetTransformDefinition(
            method_id="cupac_linear_pooled_theta_A",
            title_ru="CUPAC A с theta = Cov(Y, g)/Var(g) по эксперименту",
            information_modes=experiment_modes,
            minimum_temporal_access=TemporalAccess.PRE_TREATMENT_ONLY,
            causal_safety=CausalSafety.CONDITIONAL,
            theory_grade=TheoryGrade.B,
            required_columns=("target", "pre_target", "pre_treatment_features", "lag2_pre_features"),
            factory=_pooled_cupac_factory,
            metadata=MappingProxyType(
                {
                    "family_id": "linear_pooled_theta",
                    "zones": ("A",),
                    "model_fit": "lag2_to_lag1_ridge",
                    "theta_fit": "Cov(Y, g) / Var(g), experiment_all",
                    "trial_outcome_used_for_theta": True,
                    "variance_guarantee": "asymptotically never above Var(Y)",
                }
            ),
        ),
    ]
    definitions.extend(
        _cupac_definition(
            family_id="linear",
            family_title="с линейной моделью",
            a_factory=LinearCupacTransformer,
            information_modes=experiment_modes,
            base_grade=TheoryGrade.B,
        )
    )
    definitions.extend(
        _cupac_definition(
            family_id="quadratic",
            family_title="с квадратичной ridge-моделью",
            a_factory=QuadraticCupacTransformer,
            information_modes=experiment_modes,
            base_grade=TheoryGrade.B,
        )
    )
    definitions.extend(
        _cupac_definition(
            family_id="automl",
            family_title="с автоматическим выбором ML-модели",
            a_factory=AutoMLTemporalCupacTransformer,
            information_modes=experiment_modes,
            base_grade=TheoryGrade.C,
        )
    )
    definitions.extend(
        _cupac_definition(
            family_id="panel_prior",
            family_title="с априорной информацией панели",
            a_factory=PanelPriorCupacTransformer,
            information_modes=panel_only,
            base_grade=TheoryGrade.B,
            metadata={"requires_aggregate_prior": True},
        )
    )
    definitions.extend(
        _cupac_definition(
            family_id="hypex_compatible",
            family_title="локально совместимый с семантикой HypEx",
            a_factory=TemporalLinearCupacTransformer,
            information_modes=experiment_modes,
            base_grade=TheoryGrade.C,
            base_safety=CausalSafety.CONDITIONAL,
            metadata={
                "official_hypex_target_api": False,
                "implementation": "VarWar TemporalLinearCupacTransformer",
                "trial_outcome_used_for_theta": True,
            },
        )
    )
    definitions.extend(
        _cupac_definition(
            family_id="hypex_official_target",
            family_title="официальный target-only API HypEx",
            a_factory=None,
            information_modes=experiment_modes,
            base_grade=TheoryGrade.D,
            base_safety=CausalSafety.CONDITIONAL,
            available=False,
            unavailable_reason=(
                f"PyPI HypEx {hypex_package_version()} is integrated as a complete analysis method; "
                "VarWar has no verified public HypEx target-only API to register here."
            ),
            metadata={"integration_layer": "AnalysisMethod", "package": hypex_requirement()},
        )
    )
    registry = {definition.method_id: definition for definition in definitions}
    if len(registry) != len(definitions):
        raise RuntimeError("Duplicate target-transform method IDs.")
    return MappingProxyType(registry)


TARGET_REGISTRY_METADATA: Mapping[str, Any] = MappingProxyType(
    {
        "excluded_from_target_layer": MappingProxyType(
            {
                "cuped_trial_lin": (
                    "Lin-style trial adjustment couples point estimation and HC2/HC3 "
                    "inference, so it is registered as AnalysisMethod rather than a "
                    "target transformer."
                )
            }
        ),
        "zone_semantics": MappingProxyType(
            {
                "A": "lag2 features predict lag1 pre-target before outcome analysis",
                "B": "separate control-fitted stage on expert-approved in-time features",
                "C": "separate control-fitted stage after the configured balance gate",
            }
        ),
    }
)


TARGET_REGISTRY = build_target_registry()
