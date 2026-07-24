from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from varwar.core import EstimateResult, ExperimentSpec
from varwar.stats import t_critical, t_two_sided_p


class HypExBackendUnavailable(RuntimeError):
    pass


def estimate_with_hypex_abtest(
    data: pd.DataFrame,
    spec: ExperimentSpec,
    *,
    auxiliary_data: pd.DataFrame | None = None,
    cuped_features: dict[str, str] | None = None,
    cupac_features: dict[str, list[str]] | None = None,
    additional_tests: list[str] | None = None,
) -> EstimateResult:
    """Run PyPI HypEx ABTest and convert its output into EstimateResult."""

    try:
        from hypex import ABTest
        from hypex.dataset import (
            Dataset,
            ExperimentData,
            FeatureRole,
            InfoRole,
            PreTargetRole,
            TargetRole,
            TreatmentRole,
        )
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise HypExBackendUnavailable(str(exc)) from exc

    try:
        target_cupac_features = (
            list(cupac_features.get(spec.target_col, []))
            if cupac_features
            else []
        )
        roles: dict[str, Any] = {spec.treatment_col: TreatmentRole(int)}
        roles[spec.target_col] = TargetRole(
            float,
            cofounders=target_cupac_features,
        )
        if spec.unit_col and spec.unit_col in data.columns:
            roles[spec.unit_col] = InfoRole()

        clean = data[[column for column in roles if column in data.columns]].copy(
            deep=False
        )
        if target_cupac_features:
            if spec.pre_target_col is None:
                raise HypExBackendUnavailable(
                    "HypEx CUPAC requires spec.pre_target_col."
                )
            pre_target = spec.pre_target_col
            pre_target_lag2 = f"{pre_target}__lag2"
            if pre_target not in data or pre_target_lag2 not in data:
                raise HypExBackendUnavailable(
                    "HypEx CUPAC requires two aligned pre-period target columns: "
                    f"{pre_target!r} and {pre_target_lag2!r}."
                )
            clean[pre_target] = np.asarray(data[pre_target], dtype=float)
            clean[pre_target_lag2] = np.asarray(data[pre_target_lag2], dtype=float)
            roles[pre_target] = PreTargetRole(
                float,
                parent=spec.target_col,
                lag=1,
            )
            roles[pre_target_lag2] = PreTargetRole(
                float,
                parent=spec.target_col,
                lag=2,
            )
            cofounders = [
                column for column in target_cupac_features if column != pre_target
            ]
            roles[spec.target_col] = TargetRole(float, cofounders=cofounders)
            for column in cofounders:
                lag2_column = f"{column}__lag2"
                if column not in data or lag2_column not in data:
                    raise HypExBackendUnavailable(
                        "HypEx CUPAC feature requires aligned lag-1/lag-2 columns: "
                        f"{column!r}, {lag2_column!r}."
                    )
                roles[column] = FeatureRole(float, parent=column, lag=1)
                clean[column] = np.asarray(data[column], dtype=float)
                clean[lag2_column] = np.asarray(data[lag2_column], dtype=float)
                roles[lag2_column] = FeatureRole(
                    float,
                    parent=column,
                    lag=2,
                )
        elif cuped_features:
            for covariate in dict.fromkeys(cuped_features.values()):
                if covariate not in data.columns:
                    raise HypExBackendUnavailable(
                        f"HypEx CUPED covariate {covariate!r} is missing."
                    )
                roles[covariate] = FeatureRole(float)
                clean[covariate] = np.asarray(data[covariate], dtype=float)

        clean = clean[[column for column in roles if column in clean.columns]]
        dataset = Dataset(roles=roles, data=clean)
        ab_test = ABTest(
            additional_tests=additional_tests or ["t-test"],
            multitest_method="holm",
            cuped_features=cuped_features,
            enable_cupac=bool(cupac_features),
            cupac_models="linear",
        )
        if cupac_features:
            output_target = f"{spec.target_col}_cupac"
        elif cuped_features:
            output_target = f"{spec.target_col}_cuped"
        else:
            output_target = spec.target_col

        experiment_data = ab_test.experiment.execute(ExperimentData(dataset))
        if output_target in experiment_data.ds.data.columns:
            transformed = experiment_data.ds.data[output_target]
        else:
            additional = experiment_data.additional_fields.data
            if output_target not in additional.columns:
                raise HypExBackendUnavailable(
                    f"HypEx did not produce transformed target {output_target!r}."
                )
            transformed = additional[output_target]

        analysis_target = "__varwar_hypex_output_target"
        analysis_data = pd.DataFrame(
            {
                spec.treatment_col: np.asarray(data[spec.treatment_col]),
                analysis_target: np.asarray(transformed, dtype=float),
            },
            index=data.index,
        )
        analysis_spec = spec.with_target(analysis_target)
        n_control, n_treatment, se, df = _group_statistics(
            analysis_data,
            analysis_spec,
        )
        target = np.asarray(analysis_data[analysis_target], dtype=float)
        treatment = np.asarray(analysis_data[spec.treatment_col])
        control = target[treatment == spec.control_value]
        treated = target[treatment == spec.treatment_value]
        effect = float(np.mean(treated) - np.mean(control))
        statistic = effect / se if se > 0.0 else float("nan")
        p_value = t_two_sided_p(statistic, df)
        critical = t_critical(spec.alpha, df)
        ci_low = float(effect - critical * se) if math.isfinite(se) else float("nan")
        ci_high = float(effect + critical * se) if math.isfinite(se) else float("nan")
        diagnostics = {
            "backend": "pypi_hypex",
            "standard_error_source": "welch_on_hypex_output_target",
            "degrees_of_freedom": df,
            "hypex_output_target": output_target,
        }
        if cuped_features or cupac_features:
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
