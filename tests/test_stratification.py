from __future__ import annotations

import itertools
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from varwar.benchmark import run_stratification_race, summarize_stratification_trials
from varwar.core import ExperimentSpec
from varwar.feature_adapters import NumericFeatureBinner
from varwar.estimators import StratifiedDifferenceInMeansEstimator
from varwar.stratification import (
    ABProportionalVarianceObjective,
    AtomicStrataSummary,
    GeneticStrataReducer,
    NeymanVarianceObjective,
    aggregate_atomic_strata,
    canonicalize_partition,
)
from varwar.synthetic import SyntheticConfig, generate_historical_controls, generate_synthetic_ab


def _grouped_reference() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    means = np.array([0.0, 0.15, 0.3, 4.0, 4.15, 4.3])
    atom = np.repeat(np.arange(means.size), 24)
    target = np.repeat(means, 24) + rng.normal(0.0, 0.2, size=atom.size)
    return pd.DataFrame({"atom": atom, "y_pre": target})


def _binary_partition_optimum(
    summary: AtomicStrataSummary,
    objective: ABProportionalVarianceObjective | NeymanVarianceObjective,
) -> float:
    best = math.inf
    for remainder in itertools.product((0, 1), repeat=summary.n_atomic_strata - 1):
        labels = np.asarray((0, *remainder), dtype=np.int32)
        if np.unique(labels).size != 2:
            continue
        macro = aggregate_atomic_strata(summary, labels)
        evaluation = objective.evaluate(
            np.asarray(macro["count"], dtype=np.int64),
            np.asarray(macro["variance"], dtype=float),
        )
        best = min(best, evaluation.score)
    return best


class StratificationMathTests(unittest.TestCase):
    def test_pooled_statistics_match_direct_calculation(self) -> None:
        data = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y", "z", "z", "z"],
                "b": [0, 0, 0, 0, 1, 1, 1],
                "target": [1.0, 2.0, 4.0, 6.0, 8.0, 9.0, 12.0],
            }
        )
        summary = AtomicStrataSummary.from_frame(data, ("a", "b"), "target")
        labels = np.array([0, 0, 1], dtype=np.int32)
        macro = aggregate_atomic_strata(summary, labels)

        atom_to_macro = summary.keys.copy()
        atom_to_macro["macro"] = labels
        direct = data.merge(atom_to_macro, on=["a", "b"]).groupby("macro")["target"]
        direct_stats = direct.agg(["count", "mean", "var"])
        np.testing.assert_array_equal(macro["count"], direct_stats["count"])
        np.testing.assert_allclose(macro["mean"], direct_stats["mean"])
        np.testing.assert_allclose(macro["variance"], direct_stats["var"])

    def test_ab_objective_matches_manual_formula(self) -> None:
        counts = np.array([10, 30])
        variances = np.array([1.0, 4.0])
        result = ABProportionalVarianceObjective(
            planned_sample_size=100,
            treatment_share=0.4,
        ).evaluate(counts, variances)
        expected = (10 * 1.0 + 30 * 4.0) / 40
        self.assertAlmostEqual(result.score, expected)
        self.assertAlmostEqual(
            result.diagnostics["predicted_effect_variance"],
            expected / (100 * 0.4 * 0.6),
        )

    def test_integer_neyman_allocation_is_globally_optimal_on_small_case(self) -> None:
        counts = np.array([5, 7, 9])
        variances = np.array([1.0, 4.0, 2.0])
        objective = NeymanVarianceObjective(sample_size=9, min_sample_per_stratum=1)
        result = objective.evaluate(counts, variances)
        self.assertEqual(int(result.allocation.sum()), 9)  # type: ignore[union-attr]
        self.assertTrue(np.all(result.allocation >= 1))  # type: ignore[operator]
        self.assertTrue(np.all(result.allocation <= counts))  # type: ignore[operator]

        brute_force = math.inf
        for first in range(1, counts[0] + 1):
            for second in range(1, counts[1] + 1):
                third = 9 - first - second
                if not 1 <= third <= counts[2]:
                    continue
                allocation = np.array([first, second, third], dtype=float)
                score = objective._score(counts, variances, allocation)
                brute_force = min(brute_force, score)
        self.assertAlmostEqual(result.score, brute_force)
        self.assertLessEqual(result.continuous_score, result.score + 1e-12)

    def test_neyman_allocation_rejects_infeasible_bounds_and_handles_zero_variance(self) -> None:
        counts = np.array([3, 4])
        variances = np.zeros(2)
        result = NeymanVarianceObjective(
            sample_size=5,
            min_sample_per_stratum=1,
        ).evaluate(counts, variances)
        self.assertEqual(int(result.allocation.sum()), 5)  # type: ignore[union-attr]
        self.assertTrue(np.all(result.allocation >= 1))  # type: ignore[operator]
        self.assertEqual(result.score, 0.0)
        with self.assertRaises(ValueError):
            NeymanVarianceObjective(sample_size=1).evaluate(counts, variances)
        with self.assertRaises(ValueError):
            NeymanVarianceObjective(sample_size=8, min_sample_per_stratum=1).evaluate(
                counts,
                variances,
            )

    def test_partition_canonicalization_removes_label_symmetry(self) -> None:
        first = canonicalize_partition(np.array([0, 0, 1, 2, 1]))
        second = canonicalize_partition(np.array([9, 9, 3, 7, 3]))
        np.testing.assert_array_equal(first, second)


class GeneticReducerTests(unittest.TestCase):
    def test_ga_matches_exhaustive_ab_optimum_and_is_reproducible(self) -> None:
        data = _grouped_reference()
        summary = AtomicStrataSummary.from_frame(data, ("atom",), "y_pre")
        optimum = _binary_partition_optimum(summary, ABProportionalVarianceObjective())
        parameters = dict(
            category_cols=("atom",),
            reference_target_col="y_pre",
            n_strata=2,
            population_size=16,
            max_generations=30,
            elite_size=3,
            patience=8,
            random_state=4,
        )
        first = GeneticStrataReducer(**parameters).fit(data)
        second = GeneticStrataReducer(**parameters).fit(data)
        self.assertAlmostEqual(first.result_.train_score, optimum)  # type: ignore[union-attr]
        np.testing.assert_array_equal(first.labels_, second.labels_)
        pd.testing.assert_frame_equal(first.result_.history, second.result_.history)  # type: ignore[union-attr]
        self.assertEqual(first.result_.n_strata, 2)  # type: ignore[union-attr]
        self.assertTrue((first.macro_summary_["count"] >= 4).all())  # type: ignore[index]

    def test_ga_neyman_mode_matches_small_partition_optimum(self) -> None:
        data = _grouped_reference()
        summary = AtomicStrataSummary.from_frame(data, ("atom",), "y_pre")
        objective = NeymanVarianceObjective(sample_size=48, min_sample_per_stratum=2)
        optimum = _binary_partition_optimum(summary, objective)
        reducer = GeneticStrataReducer(
            category_cols=("atom",),
            reference_target_col="y_pre",
            n_strata=2,
            objective=objective,
            population_size=18,
            max_generations=35,
            elite_size=3,
            patience=10,
            random_state=6,
        ).fit(data)
        self.assertAlmostEqual(reducer.result_.train_score, optimum)  # type: ignore[union-attr]
        allocation = np.asarray(reducer.macro_summary_["allocation"], dtype=int)  # type: ignore[index]
        self.assertEqual(int(allocation.sum()), 48)
        self.assertTrue(np.all(allocation >= 2))
        self.assertTrue(np.all(allocation <= reducer.macro_summary_["count"]))  # type: ignore[index]

    def test_preaggregated_summary_produces_the_same_mapping(self) -> None:
        data = _grouped_reference()
        summary = AtomicStrataSummary.from_frame(data, ("atom",), "y_pre")
        parameters = dict(
            category_cols=("atom",),
            reference_target_col="y_pre",
            n_strata=2,
            population_size=12,
            max_generations=12,
            elite_size=2,
            patience=4,
            random_state=9,
        )
        direct = GeneticStrataReducer(**parameters).fit(data)
        aggregated = GeneticStrataReducer(**parameters).fit_summary(summary)
        np.testing.assert_array_equal(direct.labels_, aggregated.labels_)
        self.assertAlmostEqual(
            direct.result_.train_score,  # type: ignore[union-attr]
            aggregated.result_.train_score,  # type: ignore[union-attr]
        )

    def test_transform_does_not_read_outcome_and_unknown_policy_is_explicit(self) -> None:
        data = _grouped_reference()
        reducer = GeneticStrataReducer(
            ("atom",),
            "y_pre",
            2,
            population_size=10,
            max_generations=8,
            elite_size=2,
            patience=3,
        ).fit(data)
        categories_only = data[["atom"]].copy()
        first = reducer.transform(categories_only)
        changed_outcome = categories_only.assign(y_pre=np.arange(len(data)) * 1000.0)
        second = reducer.transform(changed_outcome)
        pd.testing.assert_series_equal(first, second)
        with self.assertRaises(ValueError):
            reducer.transform(pd.DataFrame({"atom": [999]}))

        fallback = GeneticStrataReducer(
            ("atom",),
            "y_pre",
            2,
            population_size=10,
            max_generations=8,
            elite_size=2,
            patience=3,
            unknown_policy="largest",
        ).fit(data)
        assigned = fallback.transform(pd.DataFrame({"atom": [999]}))
        self.assertEqual(len(assigned), 1)
        self.assertGreaterEqual(int(assigned.cat.codes.iloc[0]), 0)

    def test_validation_failure_is_diagnostic_and_does_not_change_fit(self) -> None:
        data = _grouped_reference()
        validation = pd.concat(
            [data, pd.DataFrame({"atom": [999], "y_pre": [1.0]})],
            ignore_index=True,
        )
        reducer = GeneticStrataReducer(
            ("atom",),
            "y_pre",
            2,
            population_size=10,
            max_generations=8,
            elite_size=2,
            patience=3,
            unknown_policy="error",
        ).fit(data, validation_data=validation)
        self.assertTrue(math.isnan(reducer.result_.validation_score))  # type: ignore[union-attr]
        self.assertIn("validation_error", reducer.result_.diagnostics)  # type: ignore[union-attr]
        self.assertEqual(reducer.result_.n_strata, 2)  # type: ignore[union-attr]


class FeatureAdapterTests(unittest.TestCase):
    def test_quantile_binner_handles_duplicates_missing_and_future_extremes(self) -> None:
        reference = pd.DataFrame(
            {
                "x": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, np.nan],
                "constant": [5.0] * 7,
            }
        )
        binner = NumericFeatureBinner(
            columns=("x", "constant"),
            strategy="quantile",
            n_bins={"x": 5, "constant": 4},
        ).fit(reference)
        transformed = binner.transform(
            pd.DataFrame(
                {
                    "x": [-100.0, 100.0, np.nan, "bad"],
                    "constant": [5.0, 6.0, np.nan, 5.0],
                }
            )
        )
        self.assertEqual(str(transformed["x__bin"].dtype), "category")
        self.assertEqual(transformed["x__bin"].iloc[2], "missing")
        self.assertEqual(transformed["x__bin"].iloc[3], "missing")
        self.assertEqual(binner.diagnostics()["constant"]["actual_bins"], 1)
        self.assertGreater(binner.diagnostics()["x"]["collapsed_edges"], 0)
        augmented = binner.augment(reference)
        self.assertNotIn("x__bin", reference.columns)
        self.assertIn("x__bin", augmented.columns)

    def test_uniform_and_custom_binning_have_stable_labels(self) -> None:
        data = pd.DataFrame({"x": np.arange(10, dtype=float)})
        uniform = NumericFeatureBinner(("x",), strategy="uniform", n_bins=2).fit(data)
        self.assertEqual(list(uniform.transform(data)["x__bin"].cat.categories), [
            "bin_000",
            "bin_001",
            "missing",
        ])
        custom = NumericFeatureBinner(
            ("x",),
            strategy="custom",
            custom_edges={"x": [2.0, 7.0]},
        ).fit(data)
        values = custom.transform(pd.DataFrame({"x": [1.0, 3.0, 8.0]}))["x__bin"]
        self.assertEqual(list(values.astype(str)), ["bin_000", "bin_001", "bin_002"])


class StratificationRaceTests(unittest.TestCase):
    def test_reference_weights_are_used_for_disproportionate_samples(self) -> None:
        data = pd.DataFrame(
            {
                "treatment": [0, 0, 1, 1, 0, 0, 1, 1],
                "y": [0.0, 0.0, 2.0, 2.0, 10.0, 10.0, 14.0, 14.0],
                "stratum": [0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        estimator = StratifiedDifferenceInMeansEstimator(
            stratum_weights={0: 0.75, 1: 0.25}
        )
        result = estimator.estimate(data, ExperimentSpec(stratum_col="stratum"))
        self.assertAlmostEqual(result.effect, 2.5)
        self.assertEqual(result.diagnostics["weight_source"], "reference")

    def test_race_contains_ab_and_neyman_preprocessors(self) -> None:
        config = SyntheticConfig(n=900, effect=0.2, seed=21)
        data = generate_synthetic_ab(config)
        reference = generate_historical_controls(config)
        common = dict(
            category_cols=("region", "device", "activity_band"),
            reference_target_col="y_pre",
            n_strata=5,
            population_size=10,
            max_generations=8,
            elite_size=2,
            patience=3,
            unknown_policy="largest",
            output_col="__varwar_race_stratum",
        )
        reducers = {
            "ga_ab": GeneticStrataReducer(
                objective=ABProportionalVarianceObjective(),
                random_state=2,
                **common,
            ),
            "ga_neyman": GeneticStrataReducer(
                objective=NeymanVarianceObjective(sample_size=180),
                random_state=2,
                **common,
            ),
        }
        results = run_stratification_race(
            data,
            reference,
            ExperimentSpec(
                pre_target_col="y_pre",
                pre_treatment_cols=("y_pre", "x1", "x2"),
                expert_safe_in_time_cols=("calendar_index", "market_index"),
                gated_in_time_cols=("operational_load",),
            ),
            ("region", "device", "activity_band"),
            "y_pre",
            5,
            reducers=reducers,
            include_target_transformers=True,
            validation_data=reference,
            true_effect=config.effect,
        )
        self.assertIn("ab_proportional", set(results["objective"]))
        self.assertIn("neyman", set(results["objective"]))
        ga_rows = results[results["preprocessor"].isin(reducers)]
        self.assertTrue((ga_rows["status"] == "ok").all())
        self.assertTrue(np.isfinite(ga_rows["validation_score"]).all())
        np.testing.assert_allclose(
            ga_rows["bias"],
            ga_rows["effect"] - config.effect,
        )
        pipelines = set(results["pipeline"])
        self.assertIn("cuped+hypex_ab", pipelines)
        self.assertIn("hypex_cupac", pipelines)
        self.assertTrue(any("sklearn_cupac_A_plus_B_plus_C" in name for name in pipelines))
        neyman_rows = results[results["preprocessor"] == "ga_neyman"]
        self.assertTrue((neyman_rows["analysis_rows"] == 180).all())
        self.assertTrue(neyman_rows["neyman_allocation_applied"].all())
        self.assertTrue(
            (neyman_rows["diag_weight_source"] == "reference").all()
        )

    def test_trial_summary_labels_fpr_and_power(self) -> None:
        trials = pd.DataFrame(
            {
                "pipeline": ["p"] * 3,
                "preprocessor": ["ga"] * 3,
                "objective": ["ab_proportional"] * 3,
                "estimator": ["stratified_mean_diff"] * 3,
                "status": ["ok"] * 3,
                "effect": [0.0, 0.1, -0.1],
                "standard_error": [0.1] * 3,
                "p_value": [0.9, 0.01, 0.5],
            }
        )
        null = summarize_stratification_trials(trials, true_effect=0.0)
        alternative = summarize_stratification_trials(trials, true_effect=0.2)
        self.assertEqual(null.loc[0, "rate_kind"], "false_positive_rate")
        self.assertEqual(alternative.loc[0, "rate_kind"], "power")
        self.assertAlmostEqual(null.loc[0, "rejection_rate"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
