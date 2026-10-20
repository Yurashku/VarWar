from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from varwar.analysis import PipelineStatus, RaceContext
from varwar.analysis_registry import build_analysis_method_registry
from varwar.core import ExperimentSpec
from varwar.estimators import (
    BootstrapMeanDiffEstimator,
    MLRateEstimator,
    RegressionAdjustmentEstimator,
    _bell_mccaffrey_df,
    _cross_fitted_ridge_predictions,
)
from varwar.hypex_backend import hypex_package_version, hypex_requirement
from varwar.robust_estimators import _full_refit_bootstrap, resampling_failure_diagnostics
from varwar.synthetic_v3 import (
    generate_world,
    get_world_spec,
    world_covariate_cols,
    world_pre_treatment_cols,
)


ROOT = Path(__file__).resolve().parents[1]
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None


def _spec(features: tuple[str, ...] = ("x1", "x2")) -> ExperimentSpec:
    return ExperimentSpec(pre_treatment_cols=features, pre_target_col=None)


def _frame(n: int = 240, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    treatment = np.zeros(n, dtype=int)
    treatment[rng.choice(n, size=n // 3, replace=False)] = 1
    y = 0.8 * x1 - 0.3 * x2 + treatment * (0.4 + 0.5 * x1) + rng.normal(0, 1 + treatment, n)
    return pd.DataFrame({"treatment": treatment, "x1": x1, "x2": x2, "y": y})


def _world_frame(world_id: str, seed: int, **overrides: object) -> tuple[pd.DataFrame, ExperimentSpec]:
    frame, experiment_spec, _ = _world_frame_and_world(world_id, seed, **overrides)
    return frame, experiment_spec


def _world_frame_and_world(world_id: str, seed: int, **overrides: object):
    spec = get_world_spec(world_id, **overrides)
    world = generate_world(spec, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = len(world.assignment_data)
    treatment = np.zeros(n, dtype=np.int8)
    treatment[rng.choice(n, size=int(round(n * spec.treatment_share)), replace=False)] = 1
    frame = world.assignment_data.copy()
    cohort = world.experiment_cohort
    frame["treatment"] = treatment
    frame["y"] = np.where(treatment == 1, cohort["y1"], cohort["y0"])
    frame["pre_target_lag1__lag2"] = frame["pre_target_lag2"]
    for column in ("x1", "x2", "x3", "x4"):
        frame[f"{column}__lag2"] = frame[column]
    experiment_spec = ExperimentSpec(
        unit_col="entity_id",
        covariate_cols=world_covariate_cols(spec),
        pre_treatment_cols=world_pre_treatment_cols(spec),
        pre_target_col="pre_target_lag1",
    )
    return frame, experiment_spec, world


class LinRegressionTests(unittest.TestCase):
    @unittest.skipUnless(HAS_STATSMODELS, "statsmodels is not installed")
    def test_centering_uses_valid_rows_and_matches_statsmodels(self) -> None:
        import statsmodels.api as sm

        data = _frame()
        data.loc[data.index[:30], "x2"] = np.nan
        data.loc[data.index[30:35], "y"] = np.nan
        result = RegressionAdjustmentEstimator().estimate(data, _spec())
        valid = data.dropna()
        centered = valid[["x1", "x2"]] - valid[["x1", "x2"]].mean()
        design = pd.DataFrame(
            {
                "const": 1.0,
                "treatment": valid["treatment"].astype(float),
                "x1": centered["x1"],
                "x2": centered["x2"],
                "t_x1": valid["treatment"] * centered["x1"],
                "t_x2": valid["treatment"] * centered["x2"],
            }
        )
        fit = sm.OLS(valid["y"], design).fit(cov_type="HC2")
        self.assertAlmostEqual(result.effect, float(fit.params["treatment"]), places=12)
        self.assertAlmostEqual(result.standard_error, float(fit.bse["treatment"]), places=12)
        self.assertEqual(result.diagnostics["n_used"], len(valid))
        self.assertEqual(result.diagnostics["centering_rows"], "valid")

    def test_rank_is_exposed(self) -> None:
        data = _frame()
        data["x3"] = 2.0 * data["x1"]
        result = RegressionAdjustmentEstimator().estimate(data, _spec(("x1", "x2", "x3")))
        self.assertEqual(result.diagnostics["design_columns"], 8)
        self.assertEqual(result.diagnostics["design_rank"], 6)
        self.assertTrue(result.diagnostics["rank_deficient"])
        full = RegressionAdjustmentEstimator().estimate(_frame(), _spec())
        self.assertFalse(full.diagnostics["rank_deficient"])

    def test_bell_mccaffrey_df_matches_dense_formula(self) -> None:
        data = _frame(n=80)
        result = RegressionAdjustmentEstimator(df_method="bell_mccaffrey").estimate(data, _spec())
        x1 = data["x1"] - data["x1"].mean()
        x2 = data["x2"] - data["x2"].mean()
        t = data["treatment"].to_numpy(float)
        design = np.column_stack([np.ones(len(data)), t, x1, x2, t * x1, t * x2])
        inverse = np.linalg.inv(design.T @ design)
        hat = design @ inverse @ design.T
        leverage = np.diag(hat)
        contrast = np.zeros(design.shape[1])
        contrast[1] = 1.0
        weights = (design @ inverse @ contrast) ** 2 / (1.0 - leverage)
        residual_maker = np.eye(len(data)) - hat
        dm = np.diag(weights) @ residual_maker
        dense = np.trace(dm) ** 2 / np.trace(dm @ dm)
        self.assertAlmostEqual(result.diagnostics["degrees_of_freedom"], dense, places=8)
        self.assertAlmostEqual(_bell_mccaffrey_df(design, inverse, leverage, contrast), dense, places=8)
        self.assertLess(dense, len(data) - design.shape[1])
        default = RegressionAdjustmentEstimator().estimate(data, _spec())
        self.assertEqual(default.diagnostics["degrees_of_freedom"], len(data) - 6)
        self.assertAlmostEqual(default.standard_error, result.standard_error, places=14)

    def test_pate_correction_adds_gamma_sigma_gamma_term(self) -> None:
        data = _frame(n=400)
        sate = RegressionAdjustmentEstimator().estimate(data, _spec())
        pate = RegressionAdjustmentEstimator(pate_correction=True).estimate(data, _spec())
        self.assertAlmostEqual(sate.effect, pate.effect, places=14)
        x = data[["x1", "x2"]].to_numpy()
        xc = x - x.mean(axis=0)
        t = data["treatment"].to_numpy(float)
        design = np.column_stack([np.ones(len(data)), t, xc, t[:, None] * xc])
        beta = np.linalg.lstsq(design, data["y"].to_numpy(), rcond=None)[0]
        gamma = beta[4:6]
        extra = float(gamma @ np.cov(x, rowvar=False) @ gamma) / len(data)
        self.assertAlmostEqual(pate.standard_error**2, sate.standard_error**2 + extra, places=12)
        self.assertGreater(pate.diagnostics["pate_variance_term"], 0.0)

    def test_pate_variant_is_registered_only_for_population_ate(self) -> None:
        methods = build_analysis_method_registry(resamples=99, prefer_pypi_hypex=False)
        method = methods["lin_regression__hc2_pate"]
        self.assertEqual(method.contract.estimand_id, "population_ate")
        status, _ = method.compatibility(RaceContext(ranking_estimand_id="sample_ate"))
        self.assertEqual(status, PipelineStatus.ESTIMAND_MISMATCH)
        status, _ = method.compatibility(RaceContext(ranking_estimand_id="population_ate"))
        self.assertEqual(status, PipelineStatus.ELIGIBLE)

    def test_pate_lin_covers_population_ate_under_iid_sampling(self) -> None:
        # heterogeneous_effect samples the cohort iid from the population, so
        # X-bar estimates the population covariate mean; population_shift is
        # the scenario where this variant must fail (biased for PATE).
        hits = 0
        reps = 60
        for seed in range(reps):
            frame, spec, world = _world_frame_and_world(
                "heterogeneous_effect",
                seed,
                n_experiment=400,
                population_size=4_000,
                effect=0.0,
                effect_modifier_scale=0.8,
            )
            result = RegressionAdjustmentEstimator(pate_correction=True).estimate(frame, spec)
            hits += result.ci_low <= world.true_population_ate <= result.ci_high
        self.assertGreaterEqual(hits / reps, 0.85)
        frame, spec, world = _world_frame_and_world("population_shift", 1, effect=0.0)
        shifted = RegressionAdjustmentEstimator(pate_correction=True).estimate(frame, spec)
        self.assertGreater(
            abs(shifted.effect - world.true_population_ate), 4 * shifted.standard_error
        )


class MLRateTests(unittest.TestCase):
    def test_inner_ridge_split_is_shuffled_by_fold_rng(self) -> None:
        rng = np.random.default_rng(1)
        n = 300
        x = np.sort(rng.normal(size=(n, 2)), axis=0)
        # The tail of the sorted frame is an outlier block: a tail holdout
        # would validate only on it.
        y = x @ np.array([1.0, -0.5]) + rng.normal(size=n)
        y[-40:] += 25.0
        predictions, ridges = _cross_fitted_ridge_predictions(x, y, 5, 1e-6, 11)
        again, ridges_again = _cross_fitted_ridge_predictions(x, y, 5, 1e-6, 11)
        np.testing.assert_array_equal(predictions, again)
        self.assertEqual(ridges, ridges_again)

    def test_sorted_and_shuffled_frames_give_the_same_distribution(self) -> None:
        sorted_errors = []
        shuffled_errors = []
        for seed in range(40):
            frame, spec = _world_frame("gaussian_r2_04_n200", seed)
            truth = 0.2
            sorted_frame = frame.sort_values("x1").reset_index(drop=True)
            shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            sorted_errors.append(MLRateEstimator().estimate(sorted_frame, spec).effect - truth)
            shuffled_errors.append(MLRateEstimator().estimate(shuffled, spec).effect - truth)
        ratio = np.mean(np.square(sorted_errors)) / np.mean(np.square(shuffled_errors))
        self.assertGreater(ratio, 0.75)
        self.assertLess(ratio, 1.33)

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed")
    def test_gbm_ml_rate_is_deterministic_and_uses_four_column_final_stage(self) -> None:
        frame, spec = _world_frame("nonlinear_panel", 2, n_experiment=600, population_size=3_000)
        estimator = MLRateEstimator(learner="hist_gbm", seed=5)
        first = estimator.estimate(frame, spec)
        second = MLRateEstimator(learner="hist_gbm", seed=5).estimate(frame, spec)
        self.assertEqual(first.effect, second.effect)
        self.assertEqual(first.standard_error, second.standard_error)
        self.assertEqual(first.diagnostics["learner"], "hist_gbm")
        self.assertNotIn("treatment", first.diagnostics["features"].split(","))
        self.assertEqual(first.diagnostics["design_rank"], 4)
        # df = n - 4 for the interacted final stage.
        from varwar.stats import t_critical

        crit = (first.ci_high - first.effect) / first.standard_error
        self.assertAlmostEqual(crit, t_critical(spec.alpha, len(frame) - 4), places=10)

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed")
    def test_gbm_beats_linear_lin_in_nonlinear_world(self) -> None:
        lin_se = []
        gbm_se = []
        for seed in range(3):
            frame, spec = _world_frame("nonlinear_panel", seed)
            lin_se.append(RegressionAdjustmentEstimator().estimate(frame, spec).standard_error)
            gbm_se.append(MLRateEstimator(learner="hist_gbm", seed=seed).estimate(frame, spec).standard_error)
        self.assertLess(np.mean(gbm_se) ** 2, 0.92 * np.mean(lin_se) ** 2)

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed")
    def test_gbm_ml_rate_is_calibrated_under_h0_small_mc(self) -> None:
        # Small paired MC: GBM ML-RATE and Lin see the same draws, so the
        # ratio of z-score dispersions is far less noisy than either alone.
        gbm_z = []
        lin_z = []
        for seed in range(20):
            frame, spec = _world_frame(
                "nonlinear_panel", 100 + seed, n_experiment=500, population_size=2_500, effect=0.0
            )
            gbm = MLRateEstimator(learner="hist_gbm", seed=seed).estimate(frame, spec)
            lin = RegressionAdjustmentEstimator().estimate(frame, spec)
            gbm_z.append(gbm.effect / gbm.standard_error)
            lin_z.append(lin.effect / lin.standard_error)
        gbm_z = np.asarray(gbm_z)
        self.assertLessEqual(np.mean(np.abs(gbm_z) > 1.96), 0.2)
        self.assertLess(abs(np.std(gbm_z) / np.std(lin_z) - 1.0), 0.3)

    def test_gbm_method_is_registered(self) -> None:
        methods = build_analysis_method_registry(resamples=99, prefer_pypi_hypex=False)
        method = methods["ml_rate__crossfit_gbm_hc2"]
        self.assertEqual(method.contract.estimand_id, "sample_ate")
        self.assertTrue(method.contract.primary_ranking_eligible)


class BootstrapFailureTests(unittest.TestCase):
    def test_failure_share_is_recorded_and_flagged(self) -> None:
        self.assertFalse(resampling_failure_diagnostics(1000, 960)["inference_failure"])
        flagged = resampling_failure_diagnostics(1000, 940)
        self.assertTrue(flagged["inference_failure"])
        self.assertAlmostEqual(flagged["failed_resample_share"], 0.06)

        calls = {"count": 0}

        def flaky(control, treatment, rng):
            calls["count"] += 1
            if calls["count"] % 10 == 0:
                raise ValueError("synthetic failure")
            return float(treatment.mean() - control.mean()), {}

        rng = np.random.default_rng(0)
        control = rng.normal(size=50)
        treatment = rng.normal(size=50)
        *_, diagnostics = _full_refit_bootstrap(
            control,
            treatment,
            flaky,
            observed_effect=float(treatment.mean() - control.mean()),
            n_resamples=200,
            confidence_level=0.95,
            random_state=1,
        )
        self.assertEqual(diagnostics["failed_resamples"], 20)
        self.assertTrue(diagnostics["inference_failure"])

    def test_bootstrap_mean_diff_reports_failure_share(self) -> None:
        data = _frame(n=60)
        result = BootstrapMeanDiffEstimator(n_resamples=199).estimate(
            data, ExperimentSpec(pre_target_col=None)
        )
        self.assertEqual(result.diagnostics["failed_resamples"], 0)
        self.assertFalse(result.diagnostics["inference_failure"])


class HypExVersionTests(unittest.TestCase):
    def test_version_strings_come_from_package_metadata(self) -> None:
        try:
            installed = importlib.metadata.version("hypex")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("hypex is not installed")
        self.assertEqual(hypex_package_version(), installed)
        self.assertEqual(hypex_requirement(), f"hypex=={installed}")
        from varwar.target_registry import TARGET_REGISTRY

        definition = TARGET_REGISTRY["cupac_hypex_official_target_A"]
        self.assertEqual(definition.metadata["package"], f"hypex=={installed}")
        self.assertIn(installed, definition.unavailable_reason)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"hypex=={installed}", readme)

    @unittest.skipUnless(importlib.util.find_spec("hypex") is not None, "hypex is not installed")
    def test_hypex_ab_test_is_called_with_explicit_welch(self) -> None:
        import hypex

        from varwar.hypex_backend import estimate_with_hypex_abtest

        captured: dict[str, object] = {}
        original = hypex.ABTest

        class RecordingABTest(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        hypex.ABTest = RecordingABTest
        try:
            result = estimate_with_hypex_abtest(_frame(n=80), ExperimentSpec(pre_target_col=None))
        finally:
            hypex.ABTest = original
        self.assertIs(captured.get("equal_variance"), False)
        self.assertFalse(result.diagnostics["hypex_equal_variance"])
        self.assertTrue(math.isfinite(result.standard_error))


if __name__ == "__main__":
    unittest.main()
