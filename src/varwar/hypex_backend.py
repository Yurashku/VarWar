from __future__ import annotations

from typing import Any

import pandas as pd

from varwar.core import EstimateResult, ExperimentSpec


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
        n_control = int(_first_present(row, ["control size", "control_size"], default=0))
        n_treatment = int(_first_present(row, ["test size", "test_size"], default=0))
        se = _derive_se(row, effect)
        statistic = _first_present(
            row,
            ["TTest statistic 1", "TTest statistic", "statistic"],
            default=float("nan"),
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
        ci_low=float("nan"),
        ci_high=float("nan"),
        n_control=n_control,
        n_treatment=n_treatment,
        diagnostics={"backend": "pypi_hypex"},
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


def _derive_se(row: pd.Series, effect: float) -> float:
    statistic = _first_present(
        row,
        ["TTest statistic 1", "TTest statistic", "statistic"],
        default=float("nan"),
    )
    if statistic and pd.notna(statistic):
        return abs(float(effect) / float(statistic))
    return float("nan")
