from __future__ import annotations

import time
import tracemalloc
from collections.abc import Iterable
from typing import Any

import pandas as pd

from varwar.core import Estimator, ExperimentSpec, WrappedEstimator
from varwar.registry import default_estimators


def build_default_race(
    include_optional: bool = True,
    spec: ExperimentSpec | None = None,
    include_unsafe_demo: bool = False,
) -> list[Estimator]:
    return default_estimators(
        include_optional=include_optional,
        spec=spec,
        include_unsafe_demo=include_unsafe_demo,
    )


def _estimate_one(
    estimator: Any,
    data: pd.DataFrame,
    spec: ExperimentSpec,
    auxiliary_data: pd.DataFrame | None,
) -> Any:
    if isinstance(estimator, WrappedEstimator):
        return estimator.estimate(data, spec, auxiliary_data=auxiliary_data)
    return estimator.estimate(data, spec)


def run_estimator_race(
    data: pd.DataFrame,
    spec: ExperimentSpec,
    estimators: Iterable[Any] | None = None,
    auxiliary_data: pd.DataFrame | None = None,
    measure_memory: bool = False,
) -> pd.DataFrame:
    """Run estimators sequentially and return a compact result table."""

    rows: list[dict[str, Any]] = []
    race_estimators = (
        list(estimators) if estimators is not None else build_default_race(spec=spec)
    )
    for estimator in race_estimators:
        name = getattr(estimator, "name", estimator.__class__.__name__)
        start = time.perf_counter()
        peak_memory = None
        if measure_memory:
            tracemalloc.start()
        try:
            result = _estimate_one(estimator, data, spec, auxiliary_data)
            row = result.to_record()
            row["status"] = "ok"
        except NotImplementedError as exc:
            row = {"estimator": name, "status": "not_implemented", "error": str(exc)}
        except Exception as exc:
            row = {"estimator": name, "status": "failed", "error": str(exc)}
        finally:
            if measure_memory:
                _, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
        row["runtime_ms"] = (time.perf_counter() - start) * 1000.0
        if peak_memory is not None:
            row["peak_memory_bytes"] = peak_memory
        rows.append(row)
    return pd.DataFrame(rows)
