from __future__ import annotations

from dataclasses import dataclass

from varwar.core import ExperimentSpec


@dataclass(frozen=True)
class CupacFeatureSet:
    name: str
    columns: tuple[str, ...]
    safety_status: str
    description: str
    predecessor: str | None = None
    reference_only: bool = False


def default_cupac_feature_sets(
    spec: ExperimentSpec,
    include_unsafe_demo: bool = False,
) -> list[CupacFeatureSet]:
    """Build the RnD6 CUPAC feature chain: A, A+B, A+B+C."""

    a = spec.pre_treatment_features()
    b = spec.expert_safe_in_time_cols
    c = spec.gated_in_time_cols
    result = [
        CupacFeatureSet(
            name="sklearn_cupac_A",
            columns=a,
            safety_status="safe_pre_treatment",
            description="A: pre-treatment features only",
            predecessor="hypex_ab",
        ),
        CupacFeatureSet(
            name="sklearn_cupac_A_plus_B",
            columns=_dedupe(a + b),
            safety_status="safe_pre_treatment_plus_expert_safe_intime",
            description="A+B: pre-treatment plus expert-safe in-time features",
            predecessor="sklearn_cupac_A",
        ),
        CupacFeatureSet(
            name="sklearn_cupac_A_plus_B_plus_C",
            columns=_dedupe(a + b + c),
            safety_status="safe_plus_balance_gated_intime",
            description="A+B+C: plus balance-gated in-time features",
            predecessor="sklearn_cupac_A_plus_B",
        ),
    ]
    if include_unsafe_demo and spec.unsafe_demo_cols:
        result.append(
            CupacFeatureSet(
                name="unsafe_demo_cupac",
                columns=_dedupe(a + b + c + spec.unsafe_demo_cols),
                safety_status="forbidden_demo",
                description="Unsafe demo only; not a candidate estimator",
                predecessor="sklearn_cupac_A_plus_B_plus_C",
            )
        )
    return result


def _dedupe(columns: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column not in seen:
            result.append(column)
            seen.add(column)
    return tuple(result)
