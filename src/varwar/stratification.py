from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from varwar.core import require_columns


@dataclass(frozen=True)
class AtomicStrataSummary:
    """Достаточные статистики атомарных страт без хранения строк и ID."""

    keys: pd.DataFrame
    counts: np.ndarray
    means: np.ndarray
    m2: np.ndarray
    category_cols: tuple[str, ...]
    target_col: str
    source_rows: int
    dropped_target_rows: int

    @classmethod
    def from_frame(
        cls,
        data: pd.DataFrame,
        category_cols: tuple[str, ...],
        target_col: str,
    ) -> "AtomicStrataSummary":
        """Сжать reference-данные до count, mean и центральной суммы квадратов."""

        if not category_cols:
            raise ValueError("At least one categorical column is required.")
        if target_col in category_cols:
            raise ValueError("reference_target_col cannot also be a category column.")
        require_columns(data, [*category_cols, target_col])
        target = np.asarray(pd.to_numeric(data[target_col], errors="coerce"), dtype=float)
        valid = np.isfinite(target)
        if not valid.any():
            raise ValueError("Reference target has no finite observations.")

        frame = data.loc[valid, [*category_cols, target_col]].copy(deep=False)
        frame[target_col] = target[valid]
        grouped = (
            frame.groupby(
                list(category_cols),
                observed=True,
                sort=False,
                dropna=False,
            )[target_col]
            .agg(["count", "mean", "var"])
            .reset_index()
        )
        counts = np.asarray(grouped.pop("count"), dtype=np.int64)
        means = np.asarray(grouped.pop("mean"), dtype=float)
        variances = np.asarray(grouped.pop("var").fillna(0.0), dtype=float)
        m2 = np.maximum(variances, 0.0) * np.maximum(counts - 1, 0)
        if counts.size == 0:
            raise ValueError("No atomic strata were produced.")
        return cls(
            keys=grouped.reset_index(drop=True),
            counts=counts,
            means=means,
            m2=m2,
            category_cols=category_cols,
            target_col=target_col,
            source_rows=int(len(data)),
            dropped_target_rows=int((~valid).sum()),
        )

    @property
    def n_atomic_strata(self) -> int:
        return int(self.counts.size)

    @property
    def total_count(self) -> int:
        return int(self.counts.sum())


@dataclass(frozen=True)
class ObjectiveEvaluation:
    score: float
    continuous_score: float
    allocation: np.ndarray | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class StratificationObjective(Protocol):
    """Контракт целевой функции для произвольного разбиения на макростраты."""

    name: str

    def evaluate(self, counts: np.ndarray, variances: np.ndarray) -> ObjectiveEvaluation:
        ...


@dataclass(frozen=True)
class ABProportionalVarianceObjective:
    """Взвешенная внутристратовая дисперсия при естественных долях трафика."""

    planned_sample_size: int | None = None
    treatment_share: float = 0.5
    name: str = "ab_proportional"

    def evaluate(self, counts: np.ndarray, variances: np.ndarray) -> ObjectiveEvaluation:
        counts = np.asarray(counts, dtype=float)
        variances = np.maximum(np.asarray(variances, dtype=float), 0.0)
        total = float(counts.sum())
        if counts.ndim != 1 or counts.size == 0 or total <= 0:
            raise ValueError("Objective requires non-empty positive stratum counts.")
        if np.any(counts <= 0) or not np.isfinite(variances).all():
            raise ValueError("All macro strata must have positive counts and finite variances.")
        score = float(np.dot(counts / total, variances))
        diagnostics: dict[str, Any] = {}
        if self.planned_sample_size is not None:
            if self.planned_sample_size <= 0:
                raise ValueError("planned_sample_size must be positive.")
            if not 0.0 < self.treatment_share < 1.0:
                raise ValueError("treatment_share must be between zero and one.")
            diagnostics["predicted_effect_variance"] = score / (
                self.planned_sample_size
                * self.treatment_share
                * (1.0 - self.treatment_share)
            )
        return ObjectiveEvaluation(score, score, diagnostics=diagnostics)


def canonicalize_partition(labels: np.ndarray) -> np.ndarray:
    """Сделать метки групп инвариантными к их исходным числовым именам."""

    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("Partition labels must be one-dimensional.")
    mapping: dict[int, int] = {}
    result = np.empty(values.size, dtype=np.int32)
    next_label = 0
    for index, raw in enumerate(values):
        value = int(raw)
        if value < 0:
            raise ValueError("Partition labels must be non-negative.")
        if value not in mapping:
            mapping[value] = next_label
            next_label += 1
        result[index] = mapping[value]
    return result


def _aggregate_atomic_arrays(
    summary: AtomicStrataSummary,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = canonicalize_partition(labels)
    if labels.size != summary.n_atomic_strata:
        raise ValueError("Partition length must equal the number of atomic strata.")
    n_groups = int(labels.max()) + 1
    counts = np.bincount(
        labels,
        weights=summary.counts,
        minlength=n_groups,
    )
    weighted_sum = np.bincount(
        labels,
        weights=summary.counts * summary.means,
        minlength=n_groups,
    )
    means = np.divide(weighted_sum, counts, out=np.zeros_like(weighted_sum), where=counts > 0)
    pooled_m2 = np.bincount(
        labels,
        weights=summary.m2 + summary.counts * (summary.means - means[labels]) ** 2,
        minlength=n_groups,
    )
    variances = np.divide(
        pooled_m2,
        counts - 1.0,
        out=np.zeros_like(pooled_m2),
        where=counts > 1.0,
    )
    return counts.astype(np.int64), means, pooled_m2, np.maximum(variances, 0.0)


def aggregate_atomic_strata(
    summary: AtomicStrataSummary,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Агрегировать достаточные статистики атомов по заданному разбиению."""

    counts, means, pooled_m2, variances = _aggregate_atomic_arrays(summary, labels)
    n_groups = counts.size
    return pd.DataFrame(
        {
            "stratum": np.arange(n_groups, dtype=np.int32),
            "count": counts,
            "mean": means,
            "m2": pooled_m2,
            "variance": variances,
            "std": np.sqrt(variances),
        }
    )


@dataclass(frozen=True)
class StrataReductionResult:
    mapping: pd.DataFrame
    macro_summary: pd.DataFrame
    objective_name: str
    train_score: float
    continuous_score: float
    validation_score: float
    history: pd.DataFrame
    n_atomic_strata: int
    n_strata: int
    generations_run: int
    converged: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class GeneticStrataReducer:
    """Укрупняет атомарные страты специализированным grouping GA.

    Объект обучается на историческом target. После ``fit`` метод ``transform``
    использует только категориальные признаки и не читает экспериментальный
    outcome, что позволяет безопасно применять mapping перед A/B-анализом.
    """

    category_cols: tuple[str, ...]
    reference_target_col: str
    n_strata: int
    objective: StratificationObjective = field(
        default_factory=ABProportionalVarianceObjective
    )
    min_stratum_size: int = 4
    population_size: int = 64
    max_generations: int = 250
    elite_size: int = 8
    mutation_probability: float = 0.2
    patience: int = 40
    tolerance: float = 1e-9
    random_state: int = 42
    unknown_policy: str = "error"
    output_col: str = "stratum"
    summary_: AtomicStrataSummary | None = field(default=None, init=False)
    mapping_: pd.DataFrame | None = field(default=None, init=False)
    macro_summary_: pd.DataFrame | None = field(default=None, init=False)
    result_: StrataReductionResult | None = field(default=None, init=False)
    labels_: np.ndarray | None = field(default=None, init=False)

    def _minimum_group_size(self) -> int:
        allocation_minimum = int(getattr(self.objective, "min_sample_per_stratum", 1))
        return max(self.min_stratum_size, allocation_minimum)

    def _validate_parameters(self) -> None:
        if self.n_strata < 1:
            raise ValueError("n_strata must be at least one.")
        if self.min_stratum_size < 1:
            raise ValueError("min_stratum_size must be at least one.")
        if self.population_size < 2:
            raise ValueError("population_size must be at least two.")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be in [1, population_size).")
        if self.max_generations < 0 or self.patience < 1:
            raise ValueError("max_generations must be non-negative and patience positive.")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between zero and one.")
        if self.unknown_policy not in {"error", "largest"}:
            raise ValueError("unknown_policy must be 'error' or 'largest'.")

    def _macro_arrays(self, labels: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if self.summary_ is None:
            raise RuntimeError("Reducer has no atomic summary.")
        macro = aggregate_atomic_strata(self.summary_, labels)
        return (
            macro,
            np.asarray(macro["count"], dtype=np.int64),
            np.asarray(macro["variance"], dtype=float),
        )

    def _fitness(
        self,
        labels: np.ndarray,
        cache: dict[bytes, ObjectiveEvaluation],
    ) -> ObjectiveEvaluation:
        canonical = canonicalize_partition(labels)
        key = canonical.tobytes()
        if key not in cache:
            if self.summary_ is None:
                raise RuntimeError("Reducer has no atomic summary.")
            counts, _, _, variances = _aggregate_atomic_arrays(self.summary_, canonical)
            cache[key] = self.objective.evaluate(counts, variances)
        return cache[key]

    def _merge_closest(self, labels: np.ndarray, source: int) -> np.ndarray:
        if self.summary_ is None:
            raise RuntimeError("Reducer has no atomic summary.")
        _, means, _, _ = _aggregate_atomic_arrays(self.summary_, labels)
        candidates = np.arange(means.size) != source
        if not candidates.any():
            return labels
        distance = np.where(candidates, np.abs(means - means[source]), np.inf)
        destination = int(np.argmin(distance))
        merged = labels.copy()
        merged[merged == source] = destination
        return canonicalize_partition(merged)

    def _split_once(self, labels: np.ndarray, n_groups: int) -> np.ndarray | None:
        if self.summary_ is None:
            raise RuntimeError("Reducer has no atomic summary.")
        counts, _, _, variances = _aggregate_atomic_arrays(self.summary_, labels)
        priority = np.argsort(
            -counts.astype(float) * variances
        )
        for group in priority:
            atoms = np.flatnonzero(labels == group)
            if atoms.size < 2:
                continue
            ordered = atoms[np.argsort(self.summary_.means[atoms], kind="stable")]
            cumulative = np.cumsum(self.summary_.counts[ordered])
            total = int(cumulative[-1])
            feasible = np.flatnonzero(
                (cumulative[:-1] >= self._minimum_group_size())
                & ((total - cumulative[:-1]) >= self._minimum_group_size())
            )
            if feasible.size == 0:
                continue
            split = int(feasible[np.argmin(np.abs(cumulative[feasible] - total / 2.0))]) + 1
            result = labels.copy()
            result[ordered[split:]] = n_groups
            return canonicalize_partition(result)
        return None

    def _repair(self, labels: np.ndarray, target_groups: int) -> np.ndarray | None:
        labels = canonicalize_partition(labels)
        limit = max(4 * target_groups + self.summary_.n_atomic_strata, 16)  # type: ignore[union-attr]
        for _ in range(limit):
            if self.summary_ is None:
                raise RuntimeError("Reducer has no atomic summary.")
            counts, _, _, _ = _aggregate_atomic_arrays(self.summary_, labels)
            n_groups = counts.size
            if n_groups > target_groups:
                source = int(np.argmin(counts))
                labels = self._merge_closest(labels, source)
                continue
            undersized = np.flatnonzero(counts < self._minimum_group_size())
            if undersized.size:
                if n_groups == 1:
                    return None
                labels = self._merge_closest(labels, int(undersized[0]))
                continue
            if n_groups < target_groups:
                split = self._split_once(labels, n_groups)
                if split is None:
                    return None
                labels = split
                continue
            return canonicalize_partition(labels)
        return None

    def _mean_sorted_partition(self, target_groups: int) -> np.ndarray | None:
        assert self.summary_ is not None
        order = np.argsort(self.summary_.means, kind="stable")
        labels = np.empty(self.summary_.n_atomic_strata, dtype=np.int32)
        for group, atoms in enumerate(np.array_split(order, target_groups)):
            labels[atoms] = group
        return self._repair(labels, target_groups)

    def _balanced_partition(
        self,
        target_groups: int,
        rng: np.random.Generator,
        randomize: bool,
    ) -> np.ndarray | None:
        assert self.summary_ is not None
        if randomize:
            order = rng.permutation(self.summary_.n_atomic_strata)
        else:
            order = np.argsort(-self.summary_.counts, kind="stable")
        labels = np.empty(self.summary_.n_atomic_strata, dtype=np.int32)
        group_counts = np.zeros(target_groups, dtype=np.int64)
        for group, atom in enumerate(order[:target_groups]):
            labels[atom] = group
            group_counts[group] += self.summary_.counts[atom]
        for atom in order[target_groups:]:
            if randomize and rng.random() < 0.35:
                cutoff = np.quantile(group_counts, 0.5)
                candidates = np.flatnonzero(group_counts <= cutoff)
                group = int(rng.choice(candidates))
            else:
                group = int(np.argmin(group_counts))
            labels[atom] = group
            group_counts[group] += self.summary_.counts[atom]
        return self._repair(labels, target_groups)

    def _initial_population(
        self,
        target_groups: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        population: list[np.ndarray] = []
        for candidate in (
            self._mean_sorted_partition(target_groups),
            self._balanced_partition(target_groups, rng, randomize=False),
        ):
            if candidate is not None:
                population.append(candidate)
        attempts = 0
        while len(population) < self.population_size and attempts < self.population_size * 30:
            attempts += 1
            candidate = self._balanced_partition(target_groups, rng, randomize=True)
            if candidate is not None:
                population.append(candidate)
        if not population:
            raise ValueError(
                "Unable to construct a feasible partition; reduce n_strata or min_stratum_size."
            )
        while len(population) < self.population_size:
            population.append(population[len(population) % len(population)].copy())
        return population[: self.population_size]

    @staticmethod
    def _tournament(
        population: list[np.ndarray],
        scores: np.ndarray,
        rng: np.random.Generator,
        size: int = 3,
    ) -> np.ndarray:
        competitors = rng.integers(0, len(population), size=min(size, len(population)))
        winner = int(competitors[np.argmin(scores[competitors])])
        return population[winner]

    def _crossover(
        self,
        first: np.ndarray,
        second: np.ndarray,
        target_groups: int,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        child = second.copy()
        groups = np.unique(first)
        maximum = max(1, min(len(groups), max(1, target_groups // 3)))
        count = int(rng.integers(1, maximum + 1))
        selected = rng.choice(groups, size=count, replace=False)
        next_label = int(child.max()) + 1
        for group in selected:
            child[first == group] = next_label
            next_label += 1
        return self._repair(child, target_groups)

    def _mutate(
        self,
        labels: np.ndarray,
        target_groups: int,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        result = labels.copy()
        if rng.random() < 0.5:
            atom = int(rng.integers(0, result.size))
            destinations = np.setdiff1d(np.arange(target_groups), [result[atom]])
            if destinations.size:
                result[atom] = int(rng.choice(destinations))
        else:
            groups = np.unique(result)
            if groups.size >= 2:
                first, second = rng.choice(groups, size=2, replace=False)
                result[result == second] = first
        return self._repair(result, target_groups)

    def _validation_score(self, validation_data: pd.DataFrame) -> tuple[float, str | None]:
        try:
            require_columns(validation_data, [*self.category_cols, self.reference_target_col])
            labels = self.transform(validation_data)
        except (KeyError, TypeError, ValueError) as exc:
            return math.nan, str(exc)
        target = np.asarray(
            pd.to_numeric(validation_data[self.reference_target_col], errors="coerce"),
            dtype=float,
        )
        codes = labels.cat.codes.to_numpy()
        valid = np.isfinite(target) & (codes >= 0)
        if not valid.any():
            return math.nan, "validation target has no finite mapped rows"
        frame = pd.DataFrame({"stratum": codes[valid], "target": target[valid]})
        grouped = frame.groupby("stratum", sort=True)["target"].agg(["count", "var"])
        if len(grouped) != self.result_.n_strata or (grouped["count"] < 2).any():  # type: ignore[union-attr]
            return math.nan, "validation data does not contain enough rows in every macro stratum"
        try:
            evaluation = self.objective.evaluate(
                np.asarray(grouped["count"], dtype=np.int64),
                np.asarray(grouped["var"].fillna(0.0), dtype=float),
            )
        except ValueError as exc:
            return math.nan, str(exc)
        return evaluation.score, None

    def fit(
        self,
        reference_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
    ) -> "GeneticStrataReducer":
        """Обучить mapping атомарных комбинаций в укрупнённые страты."""

        summary = AtomicStrataSummary.from_frame(
            reference_data,
            self.category_cols,
            self.reference_target_col,
        )
        return self.fit_summary(summary, validation_data)

    def fit_summary(
        self,
        summary: AtomicStrataSummary,
        validation_data: pd.DataFrame | None = None,
    ) -> "GeneticStrataReducer":
        """Оптимизировать mapping по заранее рассчитанным атомарным статистикам.

        Метод полезен для R&D-прогонов с несколькими objective или seed: один
        ``AtomicStrataSummary`` можно переиспользовать без повторного groupby по
        исходным строкам.
        """

        self._validate_parameters()
        if tuple(summary.category_cols) != tuple(self.category_cols):
            raise ValueError("Summary category columns do not match reducer.category_cols.")
        if summary.target_col != self.reference_target_col:
            raise ValueError("Summary target column does not match reference_target_col.")
        self.summary_ = summary
        target_groups = min(self.n_strata, self.summary_.n_atomic_strata)
        if self.summary_.total_count < target_groups * self._minimum_group_size():
            raise ValueError("Reference population is too small for the requested macro strata.")
        sample_size = getattr(self.objective, "sample_size", None)
        allocation_minimum = int(getattr(self.objective, "min_sample_per_stratum", 1))
        if sample_size is not None:
            if int(sample_size) < target_groups * allocation_minimum:
                raise ValueError("Objective sample_size is too small for all macro strata.")
            if int(sample_size) > self.summary_.total_count:
                raise ValueError("Objective sample_size exceeds the reference population.")
        rng = np.random.default_rng(self.random_state)
        population = self._initial_population(target_groups, rng)
        cache: dict[bytes, ObjectiveEvaluation] = {}
        evaluations = [self._fitness(candidate, cache) for candidate in population]
        scores = np.asarray([value.score for value in evaluations], dtype=float)
        if not np.isfinite(scores).any():
            raise ValueError("No initial partition has a finite objective value.")
        best_index = int(np.nanargmin(scores))
        best = population[best_index].copy()
        best_evaluation = evaluations[best_index]
        initial_best = float(best_evaluation.score)
        history = [
            {
                "generation": 0,
                "best_score": initial_best,
                "median_score": float(np.nanmedian(scores)),
                "unique_partitions": len({candidate.tobytes() for candidate in population}),
            }
        ]
        stale = 0
        converged = False

        for generation in range(1, self.max_generations + 1):
            order = np.argsort(scores)
            new_population = [population[index].copy() for index in order[: self.elite_size]]
            while len(new_population) < self.population_size:
                first = self._tournament(population, scores, rng)
                second = self._tournament(population, scores, rng)
                child = self._crossover(first, second, target_groups, rng)
                if child is None:
                    child = first.copy()
                if rng.random() < self.mutation_probability:
                    mutated = self._mutate(child, target_groups, rng)
                    if mutated is not None:
                        child = mutated
                new_population.append(canonicalize_partition(child))
            population = new_population
            evaluations = [self._fitness(candidate, cache) for candidate in population]
            scores = np.asarray([value.score for value in evaluations], dtype=float)
            candidate_index = int(np.nanargmin(scores))
            candidate_evaluation = evaluations[candidate_index]
            threshold = self.tolerance * max(1.0, abs(best_evaluation.score))
            if candidate_evaluation.score < best_evaluation.score - threshold:
                best = population[candidate_index].copy()
                best_evaluation = candidate_evaluation
                stale = 0
            else:
                stale += 1
            history.append(
                {
                    "generation": generation,
                    "best_score": float(best_evaluation.score),
                    "median_score": float(np.nanmedian(scores)),
                    "unique_partitions": len(
                        {candidate.tobytes() for candidate in population}
                    ),
                }
            )
            if stale >= self.patience:
                converged = True
                break

        self.labels_ = canonicalize_partition(best)
        macro, counts, variances = self._macro_arrays(self.labels_)
        final_evaluation = self.objective.evaluate(counts, variances)
        if final_evaluation.allocation is not None:
            macro["allocation"] = final_evaluation.allocation
        mapping = self.summary_.keys.copy()
        mapping[self.output_col] = self.labels_
        self.mapping_ = mapping
        self.macro_summary_ = macro

        history_frame = pd.DataFrame(history)
        provisional = StrataReductionResult(
            mapping=mapping.copy(),
            macro_summary=macro.copy(),
            objective_name=self.objective.name,
            train_score=float(final_evaluation.score),
            continuous_score=float(final_evaluation.continuous_score),
            validation_score=math.nan,
            history=history_frame,
            n_atomic_strata=self.summary_.n_atomic_strata,
            n_strata=target_groups,
            generations_run=int(history_frame["generation"].iloc[-1]),
            converged=converged,
            diagnostics={
                "largest_stratum": int(macro.loc[macro["count"].idxmax(), "stratum"]),
            },
        )
        self.result_ = provisional
        validation_score = math.nan
        validation_error = None
        if validation_data is not None:
            validation_score, validation_error = self._validation_score(validation_data)
        diagnostics: dict[str, Any] = {
            "initial_best_score": initial_best,
            "fitness_evaluations": len(cache),
            "dropped_target_rows": self.summary_.dropped_target_rows,
            "largest_stratum": int(macro.loc[macro["count"].idxmax(), "stratum"]),
            **dict(final_evaluation.diagnostics),
        }
        if validation_error is not None:
            diagnostics["validation_error"] = validation_error
        self.result_ = StrataReductionResult(
            mapping=mapping.copy(),
            macro_summary=macro.copy(),
            objective_name=self.objective.name,
            train_score=float(final_evaluation.score),
            continuous_score=float(final_evaluation.continuous_score),
            validation_score=float(validation_score),
            history=history_frame,
            n_atomic_strata=self.summary_.n_atomic_strata,
            n_strata=target_groups,
            generations_run=int(history_frame["generation"].iloc[-1]),
            converged=converged,
            diagnostics=diagnostics,
        )
        return self

    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Назначить ранее обученные макростраты без использования target."""

        if self.mapping_ is None or self.result_ is None:
            raise RuntimeError("GeneticStrataReducer must be fitted before transform().")
        require_columns(data, self.category_cols)
        position_col = "__varwar_row_position"
        while position_col in data.columns:
            position_col += "_"
        left = data.loc[:, self.category_cols].copy()
        left[position_col] = np.arange(len(data), dtype=np.int64)
        merged = left.merge(
            self.mapping_,
            how="left",
            on=list(self.category_cols),
            sort=False,
            validate="many_to_one",
        ).sort_values(position_col, kind="stable")
        labels = merged[self.output_col].to_numpy(dtype=float, copy=True)
        unknown = ~np.isfinite(labels)
        if unknown.any():
            if self.unknown_policy == "error":
                raise ValueError(
                    f"Found {int(unknown.sum())} rows with category combinations absent from fit data."
                )
            labels[unknown] = int(self.result_.diagnostics["largest_stratum"])
        categorical = pd.Categorical(
            labels.astype(np.int32),
            categories=np.arange(self.result_.n_strata, dtype=np.int32),
            ordered=False,
        )
        return pd.Series(categorical, index=data.index, name=self.output_col)

    def fit_transform(
        self,
        reference_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Обучить reducer и назначить страты тем же reference-строкам."""

        return self.fit(reference_data, validation_data).transform(reference_data)

    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Вернуть shallow copy DataFrame с назначенной колонкой макрострат."""

        result = data.copy(deep=False)
        result[self.output_col] = self.transform(data)
        return result

    def diagnostics(self) -> Mapping[str, Any]:
        """Вернуть компактную диагностику оптимизации и качества mapping."""

        if self.result_ is None:
            raise RuntimeError("GeneticStrataReducer has not been fitted.")
        return {
            "objective": self.result_.objective_name,
            "train_score": self.result_.train_score,
            "continuous_score": self.result_.continuous_score,
            "validation_score": self.result_.validation_score,
            "n_atomic_strata": self.result_.n_atomic_strata,
            "n_strata": self.result_.n_strata,
            "generations_run": self.result_.generations_run,
            "converged": self.result_.converged,
            **dict(self.result_.diagnostics),
        }
