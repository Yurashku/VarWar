# VarWar R&D Plan

## Summary

The repository is reorganized around estimator comparison. Historical CUPAC
work is archived in `legacy/`; active code lives in a package with a benchmark
runner, composable estimator wrappers, synthetic data generation, and tests.

## Implementation Plan

- Keep `hypex_ab` as the baseline race slot, backed by PyPI `hypex` when
  available, and compose wrapper estimators on top of it.
- Treat wrappers as target transformers: they adjust the outcome and then call
  an ordinary estimator.
- Support synthetic data now and real internal data later through one
  `ExperimentSpec` interface.
- Use HSE materials only to identify candidate methods. Reimplement cleanly.
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
- KS two-sample test, HSE randomized/hedgehog t-test, and permutation mean
  test.
- Regression adjustment with HC1 standard error.
- Ratio metric delta-method estimator.
- Stratified and variance-weighted difference in means.
- Robust trimmed and winsorized mean estimators.
- Wald/LATE estimator for instrumented treatment take-up.
- Bonferroni and Holm-Bonferroni correction helpers.

## Next R&D Candidates

- ML-Rate variance reduction.
- Calibration and reweighting estimators.
- Optimal-transport distribution checks.
- Randomized test statistics.
- Doubly robust / orthogonalized estimators for covariate-rich settings.

## Acceptance Checks

- `python -m unittest discover -s tests`
- `python examples/run_synthetic_race.py`
- Default race output contains `hypex_ab`, `cuped+hypex_ab`, `hypex_cupac`,
  `sklearn_cupac_A+hypex_ab`, `sklearn_cupac_A_plus_B+hypex_ab`, and
  `sklearn_cupac_A_plus_B_plus_C+hypex_ab`.
