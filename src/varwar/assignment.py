from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from varwar.core import ExperimentSpec, require_columns


@dataclass(frozen=True)
class AssignmentReplayTrace:
    """Serializable information required to reproduce an assignment design."""

    design: str
    seeds: Mapping[str, Any]
    threshold: float | None
    attempts: int
    accepted_attempt: int | None
    selected_candidate_attempt: int | None
    fallback_policy: str | None
    fallback_used: bool
    fallback_design: str | None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "seeds": dict(self.seeds),
            "threshold": self.threshold,
            "attempts": self.attempts,
            "accepted_attempt": self.accepted_attempt,
            "selected_candidate_attempt": self.selected_candidate_attempt,
            "fallback_policy": self.fallback_policy,
            "fallback_used": self.fallback_used,
            "fallback_design": self.fallback_design,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class AssignmentResult:
    """Результат назначения воздействия и диагностика механизма.

    ``labels`` индексируется так же, как входной ``DataFrame``. Диагностика не
    содержит исходные строки и поэтому пригодна для сохранения в R&D-отчёте.
    """

    labels: pd.Series
    mechanism: str
    n_control: int
    n_treatment: int
    attempts: int = 1
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    replay_trace: AssignmentReplayTrace | None = None

    def augment(self, data: pd.DataFrame, treatment_col: str = "treatment") -> pd.DataFrame:
        """Вернуть поверхностную копию данных с добавленным назначением."""

        if not self.labels.index.equals(data.index):
            raise ValueError("Assignment labels and data must have identical indexes.")
        result = data.copy(deep=False)
        result[treatment_col] = self.labels
        return result


def _validate_share(treatment_share: float) -> None:
    if not 0.0 < treatment_share < 1.0:
        raise ValueError("treatment_share must be strictly between 0 and 1.")


def _target_size(n_rows: int, treatment_share: float) -> int:
    if n_rows < 2:
        raise ValueError("At least two rows are required for assignment.")
    return min(max(int(round(n_rows * treatment_share)), 1), n_rows - 1)


def _candidate(n_rows: int, n_treatment: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.zeros(n_rows, dtype=np.int8)
    labels[rng.choice(n_rows, size=n_treatment, replace=False)] = 1
    return labels


def _draw_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))


def _to_result_labels(
    binary: np.ndarray,
    data: pd.DataFrame,
    spec: ExperimentSpec,
) -> pd.Series:
    values = np.where(binary == 1, spec.treatment_value, spec.control_value)
    return pd.Series(values, index=data.index, name=spec.treatment_col)


def _resolve_feature_cols(
    data: pd.DataFrame,
    spec: ExperimentSpec,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    allowed = tuple(
        dict.fromkeys((*spec.pre_treatment_features(), *spec.covariate_cols))
    )
    columns = tuple(requested) if requested is not None else allowed
    if not columns:
        raise ValueError("At least one pre-treatment feature is required.")
    unknown = sorted(set(columns).difference(allowed))
    if unknown:
        raise ValueError(
            "Assignment mechanisms may use only ExperimentSpec pre-treatment "
            f"columns; forbidden columns: {unknown}."
        )
    forbidden = {spec.treatment_col, spec.target_col}
    overlap = sorted(forbidden.intersection(columns))
    if overlap:
        raise ValueError(f"Treatment/outcome columns cannot be used for assignment: {overlap}.")
    require_columns(data, columns)
    return columns


def _numeric_values(series: pd.Series) -> np.ndarray:
    return np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)


def _standardized_numeric_matrix(data: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(data), 0), dtype=float)
    arrays: list[np.ndarray] = []
    for column in columns:
        values = _numeric_values(data[column])
        finite = np.isfinite(values)
        fill = float(np.mean(values[finite])) if finite.any() else 0.0
        values = np.where(finite, values, fill)
        scale = float(np.std(values, ddof=0))
        if scale > 0.0:
            arrays.append((values - float(np.mean(values))) / scale)
        if not finite.all():
            missing = (~finite).astype(float)
            missing_scale = float(np.std(missing, ddof=0))
            if missing_scale > 0.0:
                arrays.append((missing - float(np.mean(missing))) / missing_scale)
    return np.column_stack(arrays) if arrays else np.empty((len(data), 0), dtype=float)


def _categorical_indicator_matrix(data: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(data), 0), dtype=float)
    frames: list[np.ndarray] = []
    for column in columns:
        values = data[column].astype("object").where(data[column].notna(), "__missing__")
        codes, uniques = pd.factorize(values, sort=True)
        if len(uniques) <= 1:
            continue
        # One level is redundant. uint8 limits temporary memory before stacking.
        indicators = np.equal.outer(codes, np.arange(len(uniques) - 1)).astype(np.uint8)
        prevalence = indicators.mean(axis=0)
        scales = np.sqrt(prevalence * (1.0 - prevalence))
        keep = scales > 0.0
        if keep.any():
            frames.append((indicators[:, keep] - prevalence[keep]) / scales[keep])
    return np.column_stack(frames) if frames else np.empty((len(data), 0), dtype=float)


def _mean_difference(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=float)
    n_treatment = int(labels.sum())
    n_control = labels.size - n_treatment
    treatment_sum = labels @ matrix
    return treatment_sum / n_treatment - (matrix.sum(axis=0) - treatment_sum) / n_control


def _stable_key(value: Any, column: str, seed: int) -> int:
    digest = blake2b(
        f"{seed}|{column}|{value!r}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _spread_ties(
    values: np.ndarray,
    identities: np.ndarray,
    column: str,
    seed: int,
) -> np.ndarray:
    """Детерминированно разомкнуть ties, не используя назначение treatment."""

    result = values.copy()
    finite = np.isfinite(values)
    unique = np.unique(values[finite])
    if unique.size <= 1:
        base_gap = max(abs(float(unique[0])) if unique.size else 1.0, 1.0)
    else:
        base_gap = float(np.min(np.diff(unique)))
    epsilon = max(base_gap * 0.125, np.finfo(float).eps * 16.0)
    for value in unique:
        positions = np.flatnonzero(finite & (values == value))
        if positions.size <= 1:
            continue
        ordered = sorted(
            positions.tolist(),
            key=lambda position: _stable_key(identities[position], column, seed),
        )
        offsets = np.linspace(-epsilon, epsilon, positions.size, dtype=float)
        result[np.asarray(ordered, dtype=int)] = float(value) + offsets
    return result


def _ks_orders(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    anti_nuclear: bool,
    identities: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, ...]:
    orders: list[np.ndarray] = []
    for column in columns:
        values = _numeric_values(data[column])
        finite = np.isfinite(values)
        if not finite.any():
            continue
        fill = float(np.max(values[finite])) + max(float(np.ptp(values[finite])), 1.0)
        sortable = np.where(finite, values, fill)
        if anti_nuclear:
            sortable = _spread_ties(sortable, identities, column, seed)
        orders.append(np.argsort(sortable, kind="stable"))
    return tuple(orders)


def _max_ks_distance(
    labels: np.ndarray,
    orders: Sequence[np.ndarray],
    n_treatment: int,
) -> float:
    if not orders:
        return 0.0
    n_control = labels.size - n_treatment
    largest = 0.0
    for order in orders:
        ordered = labels[order]
        treated_cdf = np.cumsum(ordered, dtype=float) / n_treatment
        control_cdf = (np.arange(1, ordered.size + 1) - treated_cdf * n_treatment) / n_control
        largest = max(largest, float(np.max(np.abs(treated_cdf - control_cdf))))
    return largest


@dataclass
class CompleteRandomization:
    """Полная рандомизация с заранее фиксированным точным размером treatment."""

    treatment_share: float = 0.5
    name: str = field(default="complete_randomization", init=False)

    def assign(
        self,
        design_data: pd.DataFrame,
        spec: ExperimentSpec,
        rng: np.random.Generator,
    ) -> AssignmentResult:
        _validate_share(self.treatment_share)
        started = perf_counter()
        n_treatment = _target_size(len(design_data), self.treatment_share)
        assignment_seed = _draw_seed(rng)
        binary = _candidate(
            len(design_data), n_treatment, np.random.default_rng(assignment_seed)
        )
        trace = AssignmentReplayTrace(
            design=self.name,
            seeds={"assignment_seed": assignment_seed},
            threshold=None,
            attempts=1,
            accepted_attempt=1,
            selected_candidate_attempt=1,
            fallback_policy=None,
            fallback_used=False,
            fallback_design=None,
            parameters={"treatment_share": self.treatment_share},
        )
        return AssignmentResult(
            labels=_to_result_labels(binary, design_data, spec),
            mechanism=self.name,
            n_control=len(design_data) - n_treatment,
            n_treatment=n_treatment,
            diagnostics={
                "requested_treatment_share": self.treatment_share,
                "actual_treatment_share": n_treatment / len(design_data),
                "runtime_seconds": perf_counter() - started,
            },
            replay_trace=trace,
        )


@dataclass
class StratifiedRandomization:
    """Назначение внутри заранее построенных страт с точным общим размером групп.

    Число treatment-наблюдений по стратам выбирается методом наибольших остатков.
    Это сохраняет общий плановый размер и максимально близкие к заданной доли
    размеры внутри каждой страты.
    """

    treatment_share: float = 0.5
    stratum_col: str | None = None
    min_per_arm: int = 1
    name: str = field(default="stratified_randomization", init=False)

    def assign(
        self,
        design_data: pd.DataFrame,
        spec: ExperimentSpec,
        rng: np.random.Generator,
    ) -> AssignmentResult:
        _validate_share(self.treatment_share)
        started = perf_counter()
        column = self.stratum_col or spec.stratum_col
        if column is None:
            raise ValueError("stratum_col must be specified for stratified randomization.")
        require_columns(design_data, (column,))
        if design_data[column].isna().any():
            raise ValueError("Stratum column must not contain missing values.")

        codes, levels = pd.factorize(design_data[column], sort=True)
        counts = np.bincount(codes, minlength=len(levels)).astype(int)
        if self.min_per_arm < 1:
            raise ValueError("min_per_arm must be positive.")
        if np.any(counts < 2 * self.min_per_arm):
            bad = [levels[index] for index in np.flatnonzero(counts < 2 * self.min_per_arm)]
            raise ValueError(
                "Every stratum must contain at least two units; "
                f"this configuration requires {2 * self.min_per_arm} for the per-arm minimum; "
                f"invalid strata: {bad}."
            )
        total_treatment = _target_size(len(design_data), self.treatment_share)
        lower = np.full_like(counts, self.min_per_arm)
        upper = counts - self.min_per_arm
        if total_treatment < int(lower.sum()) or total_treatment > int(upper.sum()):
            raise ValueError(
                "The requested treatment share is incompatible with assigning both arms "
                "inside every stratum."
            )
        quotas = counts * self.treatment_share
        allocated = np.clip(np.floor(quotas).astype(int), lower, upper)
        while int(allocated.sum()) < total_treatment:
            eligible = np.flatnonzero(allocated < upper)
            group = int(eligible[np.argmax(quotas[eligible] - allocated[eligible])])
            allocated[group] += 1
        while int(allocated.sum()) > total_treatment:
            eligible = np.flatnonzero(allocated > lower)
            group = int(eligible[np.argmax(allocated[eligible] - quotas[eligible])])
            allocated[group] -= 1

        assignment_seed = _draw_seed(rng)
        assignment_rng = np.random.default_rng(assignment_seed)
        binary = np.zeros(len(design_data), dtype=np.int8)
        for group, group_treatment in enumerate(allocated):
            positions = np.flatnonzero(codes == group)
            binary[
                assignment_rng.choice(positions, size=int(group_treatment), replace=False)
            ] = 1
        trace = AssignmentReplayTrace(
            design=self.name,
            seeds={"assignment_seed": assignment_seed},
            threshold=None,
            attempts=1,
            accepted_attempt=1,
            selected_candidate_attempt=1,
            fallback_policy=None,
            fallback_used=False,
            fallback_design=None,
            parameters={
                "treatment_share": self.treatment_share,
                "stratum_col": column,
                "min_per_arm": self.min_per_arm,
                "stratum_treatment_sizes": tuple(int(value) for value in allocated),
            },
        )
        return AssignmentResult(
            labels=_to_result_labels(binary, design_data, spec),
            mechanism=self.name,
            n_control=len(design_data) - total_treatment,
            n_treatment=total_treatment,
            diagnostics={
                "requested_treatment_share": self.treatment_share,
                "actual_treatment_share": total_treatment / len(design_data),
                "stratum_col": column,
                "n_strata": len(levels),
                "stratum_sizes": tuple(int(value) for value in counts),
                "stratum_treatment_sizes": tuple(int(value) for value in allocated),
                "runtime_seconds": perf_counter() - started,
            },
            replay_trace=trace,
        )


@dataclass
class _RerandomizationBase:
    treatment_share: float = 0.5
    feature_cols: tuple[str, ...] | None = None
    acceptance_probability: float = 0.1
    n_pilot: int = 256
    max_attempts: int = 1000
    fallback_policy: str = "best"

    name: str = field(default="rerandomization", init=False)

    def _validate(self) -> None:
        _validate_share(self.treatment_share)
        if not 0.0 < self.acceptance_probability <= 1.0:
            raise ValueError("acceptance_probability must be in (0, 1].")
        if self.n_pilot < 20:
            raise ValueError("n_pilot must be at least 20 for threshold calibration.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        if self.fallback_policy not in {"best", "complete_randomization", "error"}:
            raise ValueError(
                "fallback_policy must be 'best', 'complete_randomization', or 'error'."
            )

    def _prepare(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        columns: tuple[str, ...],
        tie_seed: int,
    ) -> Any:
        raise NotImplementedError

    def _score(
        self,
        labels: np.ndarray,
        prepared: Any,
        n_treatment: int,
        score_parameters: Any = None,
    ) -> tuple[float, Mapping[str, float]]:
        raise NotImplementedError

    def _calibrate(
        self,
        pilot_rng: np.random.Generator,
        n_rows: int,
        prepared: Any,
        n_treatment: int,
    ) -> tuple[float, Mapping[str, Any], Any]:
        scores = np.empty(self.n_pilot, dtype=float)
        for index in range(self.n_pilot):
            labels = _candidate(n_rows, n_treatment, pilot_rng)
            scores[index] = self._score(labels, prepared, n_treatment)[0]
        threshold = float(np.quantile(scores, self.acceptance_probability, method="higher"))
        return (
            threshold,
            {
                "pilot_score_mean": float(np.mean(scores)),
                "pilot_score_min": float(np.min(scores)),
            },
            None,
        )

    def assign(
        self,
        design_data: pd.DataFrame,
        spec: ExperimentSpec,
        rng: np.random.Generator,
    ) -> AssignmentResult:
        """Выбрать первое назначение, прошедшее независимо откалиброванный порог."""

        self._validate()
        started = perf_counter()
        columns = _resolve_feature_cols(design_data, spec, self.feature_cols)
        n_rows = len(design_data)
        n_treatment = _target_size(n_rows, self.treatment_share)

        # Independent child streams make every phase replayable and keep fallback
        # assignments invariant to the number of rejected candidates.
        pilot_seed = _draw_seed(rng)
        candidate_seed = _draw_seed(rng)
        tie_seed = _draw_seed(rng)
        fallback_seed = _draw_seed(rng)
        pilot_rng = np.random.default_rng(pilot_seed)
        candidate_rng = np.random.default_rng(candidate_seed)
        prepared = self._prepare(design_data, spec, columns, tie_seed)
        threshold, calibration, score_parameters = self._calibrate(
            pilot_rng, n_rows, prepared, n_treatment
        )

        accepted: np.ndarray | None = None
        accepted_score = float("inf")
        accepted_components: Mapping[str, float] = {}
        best: np.ndarray | None = None
        best_score = float("inf")
        best_components: Mapping[str, float] = {}
        best_attempt: int | None = None
        attempts = 0
        for attempts in range(1, self.max_attempts + 1):
            current = _candidate(n_rows, n_treatment, candidate_rng)
            score, components = self._score(
                current, prepared, n_treatment, score_parameters
            )
            if score < best_score:
                best, best_score, best_components = current, score, components
                best_attempt = attempts
            if score <= threshold:
                accepted, accepted_score, accepted_components = current, score, components
                break

        accepted_attempt = attempts if accepted is not None else None
        fallback_used = accepted is None
        fallback_design: str | None = None
        result_mechanism = self.name
        selected_candidate_attempt = accepted_attempt
        if accepted is None:
            if self.fallback_policy == "error":
                raise RuntimeError("No assignment passed the calibrated acceptance threshold.")
            if self.fallback_policy == "complete_randomization":
                fallback_design = "complete_randomization"
                result_mechanism = f"{self.name}__complete_randomization_fallback"
                accepted = _candidate(
                    n_rows, n_treatment, np.random.default_rng(fallback_seed)
                )
                accepted_score, accepted_components = self._score(
                    accepted, prepared, n_treatment, score_parameters
                )
                selected_candidate_attempt = None
            else:
                assert best is not None
                fallback_design = "best_of_k"
                result_mechanism = f"{self.name}__best_of_k_fallback"
                accepted, accepted_score, accepted_components = best, best_score, best_components
                selected_candidate_attempt = best_attempt

        trace = AssignmentReplayTrace(
            design=result_mechanism,
            seeds={
                "pilot_seed": pilot_seed,
                "candidate_seed": candidate_seed,
                "tie_seed": tie_seed,
                "fallback_seed": fallback_seed,
            },
            threshold=threshold,
            attempts=attempts,
            accepted_attempt=accepted_attempt,
            selected_candidate_attempt=selected_candidate_attempt,
            fallback_policy=self.fallback_policy if fallback_used else None,
            fallback_used=fallback_used,
            fallback_design=fallback_design,
            parameters={
                "treatment_share": self.treatment_share,
                "feature_cols": columns,
                "acceptance_probability": self.acceptance_probability,
                "n_pilot": self.n_pilot,
                "max_attempts": self.max_attempts,
            },
        )

        diagnostics: dict[str, Any] = {
            "requested_treatment_share": self.treatment_share,
            "actual_treatment_share": n_treatment / n_rows,
            "feature_cols": columns,
            "n_pilot": self.n_pilot,
            "acceptance_probability": self.acceptance_probability,
            "acceptance_threshold": threshold,
            "accepted_score": accepted_score,
            "accepted": not fallback_used,
            "fallback_used": fallback_used,
            "fallback_policy": self.fallback_policy if fallback_used else None,
            "fallback_design": fallback_design,
            "selected_candidate_attempt": selected_candidate_attempt,
            "replay_seeds": dict(trace.seeds),
            "empirical_attempt_acceptance_rate": (0.0 if fallback_used else 1.0 / attempts),
            "runtime_seconds": perf_counter() - started,
            **calibration,
            **{f"accepted_{key}": value for key, value in accepted_components.items()},
        }
        return AssignmentResult(
            labels=_to_result_labels(accepted, design_data, spec),
            mechanism=result_mechanism,
            n_control=n_rows - n_treatment,
            n_treatment=n_treatment,
            attempts=attempts,
            diagnostics=diagnostics,
            replay_trace=trace,
        )


@dataclass
class RerandomizationMahalanobis(_RerandomizationBase):
    """Рерандомизация по расстоянию Махаланобиса средних pre-treatment признаков."""

    ridge: float = 1e-10
    name: str = field(default="rerandomization_mahalanobis", init=False)

    def _prepare(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        columns: tuple[str, ...],
        tie_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        numeric = tuple(column for column in columns if pd.api.types.is_numeric_dtype(data[column]))
        categorical = tuple(column for column in columns if column not in numeric)
        matrix = np.column_stack(
            (_standardized_numeric_matrix(data, numeric), _categorical_indicator_matrix(data, categorical))
        )
        if matrix.shape[1] == 0:
            raise ValueError("Mahalanobis rerandomization has no non-constant usable features.")
        covariance = np.cov(matrix, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)
        inverse = np.linalg.pinv(covariance + self.ridge * np.eye(covariance.shape[0]))
        return matrix, inverse

    def _score(
        self,
        labels: np.ndarray,
        prepared: tuple[np.ndarray, np.ndarray],
        n_treatment: int,
        score_parameters: Any = None,
    ) -> tuple[float, Mapping[str, float]]:
        matrix, inverse = prepared
        difference = _mean_difference(matrix, labels)
        score = float(difference @ inverse @ difference)
        return score, {"mahalanobis": score}


@dataclass
class RerandomizationMaxSMD(_RerandomizationBase):
    """Рерандомизация по максимальному модулю SMD среди всех признаков."""

    name: str = field(default="rerandomization_max_smd", init=False)

    def _prepare(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        columns: tuple[str, ...],
        tie_seed: int,
    ) -> np.ndarray:
        numeric = tuple(column for column in columns if pd.api.types.is_numeric_dtype(data[column]))
        categorical = tuple(column for column in columns if column not in numeric)
        matrix = np.column_stack(
            (_standardized_numeric_matrix(data, numeric), _categorical_indicator_matrix(data, categorical))
        )
        if matrix.shape[1] == 0:
            raise ValueError("Max-SMD rerandomization has no non-constant usable features.")
        return matrix

    def _score(
        self,
        labels: np.ndarray,
        prepared: np.ndarray,
        n_treatment: int,
        score_parameters: Any = None,
    ) -> tuple[float, Mapping[str, float]]:
        score = float(np.max(np.abs(_mean_difference(prepared, labels))))
        return score, {"max_smd": score}


@dataclass
class RerandomizationMixedBalance(_RerandomizationBase):
    """Рерандомизация по совместному контролю средних, долей, missingness и формы.

    Числовые признаки контролируются SMD и расстоянием между эмпирическими
    функциями распределения, категориальные — SMD долей уровней. Пропуски входят
    отдельными индикаторами. По умолчанию KS-расстояние корректно учитывает ties
    без джиттера. ``anti_nuclear=True`` включает исследовательское, полностью
    treatment-blind размыкание совпадений, фиксируемое до пилотных назначений.
    """

    numeric_cols: tuple[str, ...] | None = None
    categorical_cols: tuple[str, ...] | None = None
    shape_cols: tuple[str, ...] | None = None
    anti_nuclear: bool = False
    name: str = field(default="rerandomization_mixed_balance", init=False)

    def _prepare(
        self,
        data: pd.DataFrame,
        spec: ExperimentSpec,
        columns: tuple[str, ...],
        tie_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        inferred_numeric = tuple(
            column for column in columns if pd.api.types.is_numeric_dtype(data[column])
        )
        numeric = self.numeric_cols if self.numeric_cols is not None else inferred_numeric
        categorical = (
            self.categorical_cols
            if self.categorical_cols is not None
            else tuple(column for column in columns if column not in numeric)
        )
        shape = self.shape_cols if self.shape_cols is not None else numeric
        selected = set(numeric) | set(categorical) | set(shape)
        invalid = sorted(selected.difference(columns))
        if invalid:
            raise ValueError(f"Mixed-balance subcolumns must be pre-treatment features: {invalid}.")
        numeric_matrix = _standardized_numeric_matrix(data, numeric)
        categorical_matrix = _categorical_indicator_matrix(data, categorical)
        identities = (
            np.asarray(data[spec.unit_col], dtype=object)
            if spec.unit_col is not None and spec.unit_col in data
            else np.asarray(data.index, dtype=object)
        )
        orders = _ks_orders(
            data,
            shape,
            anti_nuclear=self.anti_nuclear,
            identities=identities,
            seed=tie_seed,
        )
        if numeric_matrix.shape[1] + categorical_matrix.shape[1] == 0 and not orders:
            raise ValueError("Mixed-balance rerandomization has no usable features.")
        return numeric_matrix, categorical_matrix, orders

    @staticmethod
    def _components(
        labels: np.ndarray,
        prepared: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]],
        n_treatment: int,
    ) -> dict[str, float]:
        numeric, categorical, orders = prepared
        numeric_diff = _mean_difference(numeric, labels)
        category_diff = _mean_difference(categorical, labels)
        return {
            "numeric_max_smd": float(np.max(np.abs(numeric_diff))) if numeric_diff.size else 0.0,
            "categorical_max_smd": (
                float(np.max(np.abs(category_diff))) if category_diff.size else 0.0
            ),
            "max_ks_distance": _max_ks_distance(labels, orders, n_treatment),
        }

    def _calibrate(
        self,
        pilot_rng: np.random.Generator,
        n_rows: int,
        prepared: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]],
        n_treatment: int,
    ) -> tuple[float, Mapping[str, Any], np.ndarray]:
        component_names = ("numeric_max_smd", "categorical_max_smd", "max_ks_distance")
        values = np.empty((self.n_pilot, len(component_names)), dtype=float)
        for index in range(self.n_pilot):
            labels = _candidate(n_rows, n_treatment, pilot_rng)
            components = self._components(labels, prepared, n_treatment)
            values[index] = [components[name] for name in component_names]
        scales = np.quantile(values, 0.5, axis=0)
        positive = values[values > 0.0]
        global_floor = float(np.min(positive)) if positive.size else 1.0
        scales = np.where(scales > 0.0, scales, global_floor)
        scores = np.max(values / scales, axis=1)
        threshold = float(np.quantile(scores, self.acceptance_probability, method="higher"))
        return (
            threshold,
            {
                "pilot_score_mean": float(np.mean(scores)),
                "pilot_score_min": float(np.min(scores)),
                "component_scales": tuple(float(value) for value in scales),
                "anti_nuclear": self.anti_nuclear,
            },
            scales,
        )

    def _score(
        self,
        labels: np.ndarray,
        prepared: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]],
        n_treatment: int,
        score_parameters: np.ndarray | None = None,
    ) -> tuple[float, Mapping[str, float]]:
        components = self._components(labels, prepared, n_treatment)
        values = np.asarray(tuple(components.values()), dtype=float)
        score = float(
            np.max(values if score_parameters is None else values / score_parameters)
        )
        return score, components


@dataclass
class HypExAABestOfK:
    """Adapter for the best split selected by the PyPI ``hypex.AATest`` API.

    The adapter never substitutes a local random split when HypEx cannot return a
    valid result. Such a failure is explicit because a substitute would describe a
    different assignment mechanism.
    """

    treatment_share: float = 0.5
    feature_cols: tuple[str, ...] | None = None
    n_iterations: int = 10
    early_stopping: bool = False
    name: str = field(default="hypex_aa_best_of_k", init=False)

    def assign(
        self,
        design_data: pd.DataFrame,
        spec: ExperimentSpec,
        rng: np.random.Generator,
    ) -> AssignmentResult:
        _validate_share(self.treatment_share)
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be positive.")
        columns = _resolve_feature_cols(design_data, spec, self.feature_cols)
        started = perf_counter()
        seed_stream = _draw_seed(rng)
        state_rng = np.random.default_rng(seed_stream)
        random_states = tuple(
            int(value)
            for value in state_rng.integers(
                0, np.iinfo(np.int32).max, size=self.n_iterations, dtype=np.int64
            )
        )

        try:
            import hypex
            from hypex.dataset import Dataset, FeatureRole
        except ImportError as error:  # pragma: no cover - dependency is required by VarWar
            raise RuntimeError("HypExAABestOfK requires the PyPI hypex package.") from error

        roles = {column: FeatureRole() for column in columns}
        hypex_frame = design_data.loc[:, list(columns)].copy()
        try:
            output = hypex.AATest(
                control_size=1.0 - self.treatment_share,
                n_iterations=self.n_iterations,
                random_states=random_states,
                early_stopping=self.early_stopping,
            ).execute(Dataset(roles, hypex_frame))
            best = output.best_split.data
        except Exception as error:
            raise RuntimeError("PyPI HypEx AATest failed to produce an assignment.") from error

        if "split" not in best.columns or not best.index.equals(design_data.index):
            raise RuntimeError("PyPI HypEx AATest returned an invalid best-split table.")
        split = best["split"].astype(str)
        observed_groups = set(split.unique())
        if observed_groups != {"control", "test_1"}:
            raise RuntimeError(
                "PyPI HypEx AATest returned unexpected group labels: "
                f"{sorted(observed_groups)}."
            )
        binary = (split == "test_1").to_numpy(dtype=np.int8)
        n_treatment = int(binary.sum())
        if n_treatment == 0 or n_treatment == len(binary):
            raise RuntimeError("PyPI HypEx AATest returned an empty assignment arm.")

        selected_splitter = None
        experiment_data = getattr(output.main_output, "_experiment_data", None)
        variables = getattr(experiment_data, "variables", {})
        for value in variables.values():
            if isinstance(value, Mapping) and "best splitter" in value:
                selected_splitter = str(value["best splitter"])
                break
        experiments = getattr(output, "experiments", None)
        experiments_data = getattr(experiments, "data", None)
        executed_iterations = (
            int(len(experiments_data))
            if isinstance(experiments_data, pd.DataFrame)
            else self.n_iterations
        )
        selected_candidate_attempt = None
        if selected_splitter is not None:
            selected_candidate_attempt = next(
                (
                    index + 1
                    for index, state in enumerate(random_states[:executed_iterations])
                    if f"rs {state}" in selected_splitter
                ),
                None,
            )

        hypex_version = str(getattr(hypex, "__version__", "unknown"))
        trace = AssignmentReplayTrace(
            design=self.name,
            seeds={
                "random_state_stream_seed": seed_stream,
                "hypex_random_states": random_states,
            },
            threshold=None,
            attempts=executed_iterations,
            accepted_attempt=None,
            selected_candidate_attempt=selected_candidate_attempt,
            fallback_policy=None,
            fallback_used=False,
            fallback_design=None,
            parameters={
                "treatment_share": self.treatment_share,
                "control_size": 1.0 - self.treatment_share,
                "feature_cols": columns,
                "n_iterations": self.n_iterations,
                "executed_iterations": executed_iterations,
                "early_stopping": self.early_stopping,
                "hypex_version": hypex_version,
                "selected_splitter": selected_splitter,
            },
        )
        return AssignmentResult(
            labels=_to_result_labels(binary, design_data, spec),
            mechanism=self.name,
            n_control=len(binary) - n_treatment,
            n_treatment=n_treatment,
            attempts=executed_iterations,
            diagnostics={
                "requested_treatment_share": self.treatment_share,
                "actual_treatment_share": n_treatment / len(binary),
                "feature_cols": columns,
                "n_iterations": self.n_iterations,
                "executed_iterations": executed_iterations,
                "random_states": random_states,
                "selected_splitter": selected_splitter,
                "hypex_version": hypex_version,
                "runtime_seconds": perf_counter() - started,
            },
            replay_trace=trace,
        )
