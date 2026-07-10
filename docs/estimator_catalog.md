# Estimator Catalog

## Ready

| Name | Type | Notes |
| --- | --- | --- |
| `hypex_ab` | baseline | PyPI `hypex` AB baseline when installed, native-compatible fallback otherwise. |
| `welch_mean_diff` | estimator | Difference in means with Welch standard error. |
| `pooled_mean_diff` | estimator | Difference in means with pooled variance. |
| `ks_test` | test | Two-sample KS statistic; SciPy when available, asymptotic fallback otherwise. |
| `approx_ks_fast` | test | Histogram/sketch approximation of KS for large sweeps. |
| `energy_test` | test | One-dimensional energy-distance two-sample test with permutation p-value. |
| `randomized_t_test` | test | HSE randomized/hedgehog t-statistic, reimplemented from the statistic definition. |
| `permutation_mean_test` | test | Memory-bounded randomization test for mean difference. |
| `regression_adjustment` | estimator | OLS treatment coefficient with HC1 standard error. |
| `ml_rate` | estimator | Cross-fitted ridge nuisance prediction followed by robust regression adjustment. |
| `nearest_neighbor_matching` | estimator | Covariate nearest-neighbor ATT estimate for matching baselines. |
| `cuped+hypex_ab` | wrapper + baseline | CUPED target adjustment followed by `hypex_ab`. |
| `hypex_cupac` | reference | PyPI HypEx CUPAC reference/parity slot; fallback is local linear CUPAC A. |
| `sklearn_cupac_A+hypex_ab` | wrapper + baseline | RnD6 A: pre-treatment features. |
| `sklearn_cupac_A_plus_B+hypex_ab` | wrapper + baseline | RnD6 A+B: add expert-safe in-time features. |
| `sklearn_cupac_A_plus_B_plus_C+hypex_ab` | wrapper + baseline | RnD6 A+B+C: add balance-gated in-time features. |
| `winsorized+hypex_ab` | wrapper + baseline | Winsorized target followed by `hypex_ab`. |
| `ratio_delta` | estimator | Ratio metric difference by delta method. |
| `stratified_mean_diff` | estimator | Weighted stratum-level differences. |
| `variance_weighted_stratified` | estimator | Stratum estimates combined by inverse variance. |
| `trimmed_mean_diff` | estimator | Difference of trimmed means with Welch-style SE. |
| `winsorized_mean_diff` | estimator | Difference of winsorized means with Welch-style SE. |
| `wald_late` | estimator | Wald ratio for instrumented treatment take-up. |
| `bonferroni` | correction | Multiple-testing correction helper. |
| `holm_bonferroni` | correction | Step-down Bonferroni helper. |
| `romano_wolf_stepdown` | correction | Step-down max-statistic correction over caller-supplied bootstrap statistics. |

## Placeholders

| Name | Reason |
| --- | --- |
| `anti_nuclear_ks_test` | The table name is not enough to identify the exact statistic. |
| `hypex_matcher` | Requires internal HypEx matcher API and matching policy. |
| `calibration_reweighting` | Needs constraints, calibration totals, and variance estimator policy. |
| `optimal_transport_check` | Better treated as a distribution diagnostic before becoming an effect estimator. |
| `genetic_algorithm_hse` | HSE code is not copied; needs a fresh optimization objective. |
| `optimal_transport_matching` | Needs matching distance, regularization, and inference policy. |
| `bayesian_cupac` | Needs Bayesian model and posterior decision rule. |

## Additional Candidates

- Trimmed or winsorized mean estimators for heavy-tailed metrics.
- CUPED with robust base estimators.
- Cross-fitted CUPAC for larger feature sets.
- Doubly robust / orthogonalized ATE estimators when real data contains rich
  pre-treatment covariates.
- Cluster-robust estimators for switchback or grouped randomization.
