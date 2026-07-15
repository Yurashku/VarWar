# VarWar R&D Plan

## Summary

The repository is reorganized around estimator comparison. Historical CUPAC
work is archived in `legacy/`; active code lives in a package with a benchmark
runner, composable estimator wrappers, synthetic data generation, and tests.

## Implementation Plan

- Keep `welch_mean_diff` as the canonical statistical baseline. Preserve
  `hypex_ab` as a PyPI-backed integration/parity slot and compose wrapper
  estimators on top of it.
- Treat wrappers as target transformers: they adjust the outcome and then call
  an ordinary estimator.
- Support synthetic data now and real internal data later through one
  `ExperimentSpec` interface.
- Derive implementations from explicit mathematical specifications and
  document only the behavior and guarantees provided by VarWar.
- Optimize default estimators for runtime and memory: vectorized arrays,
  closed-form variances, no default bootstrap, no unnecessary deep copies.

## Implemented In This Pass

- Welch and pooled difference in means.
- HypEx-compatible A/B baseline slot, with PyPI `hypex` dependency.
- Optional external HypEx backend hook via `VARWAR_HYPEX_BACKEND`.
- CUPED target adjustment.
- Linear/Ridge CUPAC target adjustment, including RnD6 A, A+B, and A+B+C
  feature-set variants.
- `hypex_cupac` reference/parity slot.
- KS two-sample test, randomized/hedgehog t-test, and permutation mean test.
- Genetic reduction of atomic strata with A/B-proportional and Neyman
  objectives, plus target-free numeric feature binning.
- Fast histogram KS approximation and one-dimensional energy-distance test.
- Regression adjustment with HC1 standard error.
- ML-rate-style cross-fitted ridge adjustment.
- Nearest-neighbor matching baseline.
- Ratio metric delta-method estimator.
- Stratified and variance-weighted difference in means.
- Robust trimmed and winsorized mean estimators.
- Wald/LATE estimator for instrumented treatment take-up.
- Bonferroni, Holm-Bonferroni, and Romano-Wolf correction helpers.

## Next R&D Candidates

- Calibration and reweighting estimators.
- Optimal-transport distribution checks.
- Doubly robust / orthogonalized estimators for covariate-rich settings.

## Acceptance Checks

- `python -m unittest discover -s tests`
- `python examples/run_synthetic_race.py`
- `python examples/run_stratification_simulation.py --trials 2`
- Default race output starts with `welch_mean_diff` and contains `hypex_ab`,
  `cuped+hypex_ab`, `hypex_cupac`,
  `sklearn_cupac_A+hypex_ab`, `sklearn_cupac_A_plus_B+hypex_ab`, and
  `sklearn_cupac_A_plus_B_plus_C+hypex_ab`.
