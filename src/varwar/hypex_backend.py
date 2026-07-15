from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from varwar.core import EstimateResult, ExperimentSpec
from varwar.stats import t_critical


class HypExBackendUnavailable(RuntimeError):
    pass


def estimate_with_hypex_abtest(
    data: pd.DataFrame,
    spec: ExperimentSpec,
    *,
    cuped_features: dict[str, str] | None = None,
    cupac_features: dict[str, list[str]] | None = None,
    additional_tests: list[str] | None = None,
) -> EstimateResult:
    """Run PyPI HypEx ABTest and convert its output into EstimateResult."""

    try:
        from hypex import ABTest
        from hypex.dataset import Dataset, FeatureRole, InfoRole, TargetRole, TreatmentRole
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise HypExBackendUnavailable(str(exc)) from exc

    try:
        roles: dict[str, Any] = {
            spec.treatment_col: TreatmentRole(int),
            spec.target_col: TargetRole(float),
        }
        if spec.unit_col and spec.unit_col in data.columns:
            roles[spec.unit_col] = InfoRole()

        feature_cols = set(spec.pre_treatment_features())
        feature_cols.update(spec.expert_safe_in_time_cols)
        feature_cols.update(spec.gated_in_time_cols)
        if spec.pre_target_col and spec.pre_target_col in data.columns:
            feature_cols.add(spec.pre_target_col)
        for column in feature_cols:
            if column in data.columns and column not in roles:
                roles[column] = FeatureRole(float)

        clean = data[[column for column in roles if column in data.columns]].copy(deep=False)
        dataset = Dataset(roles=roles, data=clean)
        ab_test = ABTest(
            additional_tests=additional_tests or ["t-test"],
            multitest_method="holm",
            cuped_features=cuped_features,
            cupac_features=cupac_features,
            enable_cupac=bool(cupac_features),
            cupac_models="linear",
        )
        output = ab_test.execute(dataset)
        resume = getattr(output, "resume", None)
        resume_df = getattr(resume, "data", resume)
        if not isinstance(resume_df, pd.DataFrame):
            raise HypExBackendUnavailable("HypEx ABTest resume is not a pandas DataFrame.")
        row = _select_target_row(resume_df, spec.target_col)
        effect = _first_present(row, ["difference", "Difference", "mean difference"])
        p_value = _first_present(row, ["TTest p-value 1", "TTest p-value", "p-value"])
        n_control, n_treatment, raw_se, df = _group_statistics(data, spec)
        variance_reduced = bool(cuped_features or cupac_features)
        if variance_reduced:
            statistic, se = _infer_statistic_and_se(effect, p_value, df)
            se_source = "p_value_inversion"
        else:
            se = raw_se
            statistic = effect / se if se > 0.0 else float("nan")
            se_source = "welch_group_variance"
        critical = t_critical(spec.alpha, df)
        ci_low = float(effect - critical * se) if math.isfinite(se) else float("nan")
        ci_high = float(effect + critical * se) if math.isfinite(se) else float("nan")
        diagnostics = {
            "backend": "pypi_hypex",
            "standard_error_source": se_source,
            "degrees_of_freedom": df,
        }
        if variance_reduced:
            diagnostics["variance_reduction"] = (
                "cupac" if cupac_features else "cuped"
            )
    except HypExBackendUnavailable:
        raise
    except Exception as exc:  # pragma: no cover - depends on optional install/API
        raise HypExBackendUnavailable(f"PyPI HypEx ABTest failed: {exc}") from exc
    return EstimateResult(
        estimator="hypex_ab",
        effect=float(effect),
        standard_error=se,
        statistic=float(statistic),
        p_value=float(p_value),
        ci_low=ci_low,
        ci_high=ci_high,
        n_control=n_control,
        n_treatment=n_treatment,
        diagnostics=diagnostics,
    )


def _select_target_row(resume_df: pd.DataFrame, target_col: str) -> pd.Series:
    if "feature" in resume_df.columns:
        exact = resume_df[resume_df["feature"] == target_col]
        if len(exact):
            return exact.iloc[0]
    if len(resume_df) == 0:
        raise HypExBackendUnavailable("HypEx ABTest resume is empty.")
    return resume_df.iloc[0]


def _first_present(row: pd.Series, names: list[str], default: float | None = None) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    if default is not None:
        return default
    raise HypExBackendUnavailable(f"HypEx ABTest output lacks columns: {names}")


def _group_statistics(
    data: pd.DataFrame,
    spec: ExperimentSpec,
) -> tuple[int, int, float, float]:
    target = np.asarray(pd.to_numeric(data[spec.target_col], errors="coerce"), dtype=float)
    treatment = np.asarray(data[spec.treatment_col])
    control = target[np.isfinite(target) & (treatment == spec.control_value)]
    treated = target[np.isfinite(target) & (treatment == spec.treatment_value)]
    if control.size < 2 or treated.size < 2:
        raise HypExBackendUnavailable("Both HypEx groups need at least two finite targets.")
    variance_control = float(control.var(ddof=1))
    variance_treated = float(treated.var(ddof=1))
    term_control = variance_control / control.size
    term_treated = variance_treated / treated.size
    standard_error = math.sqrt(term_control + term_treated)
    denominator = (
        term_control**2 / (control.size - 1)
        + term_treated**2 / (treated.size - 1)
    )
    degrees_of_freedom = (
        (term_control + term_treated) ** 2 / denominator
        if denominator > 0.0
        else float(control.size + treated.size - 2)
    )
    return int(control.size), int(treated.size), standard_error, degrees_of_freedom


def _infer_statistic_and_se(
    effect: float,
    p_value: float,
    degrees_of_freedom: float,
) -> tuple[float, float]:
    try:
        from scipy import stats as scipy_stats  # type: ignore

        magnitude = float(scipy_stats.t.isf(p_value / 2.0, degrees_of_freedom))
        statistic = math.copysign(magnitude, effect)
        standard_error = abs(effect / statistic) if statistic != 0.0 else float("nan")
        return statistic, standard_error
    except Exception as exc:
        raise HypExBackendUnavailable(
            "Cannot derive variance-reduced standard error from HypEx p-value."
        ) from exc
