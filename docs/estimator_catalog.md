# Estimator Catalog

> Историческая таблица. Актуальная классификация, нумерация, статусы и решения
> об исключении находятся в [docs/methods/all_methods.md](methods/all_methods.md).

This is a compact compatibility index. The current taxonomy and method
documentation live in [`methods/README.md`](methods/README.md).
The complete status registry is [`methods/all_methods.md`](methods/all_methods.md).

## Ready

| Name | Type | Notes |
| --- | --- | --- |
| `welch_mean_diff` | canonical baseline | Difference in means with Welch standard error. |
| `hypex_ab` | integration baseline | PyPI `hypex` AB baseline when installed, native-compatible fallback otherwise. |
| `pooled_mean_diff` | estimator | Difference in means with pooled variance. |
| `ks_test` | test | Two-sample KS statistic; SciPy when available, asymptotic fallback otherwise. |
| `approx_ks_fast` | test | Histogram/sketch approximation of KS for large sweeps. |
| `energy_test` | test | One-dimensional energy-distance two-sample test with permutation p-value. |
| `randomized_t_test` | test | Randomized/hedgehog weighted t-statistic with deterministic weights. |
| `permutation_mean_test` | test | Memory-bounded randomization test for mean difference. |
| `regression_adjustment` | estimator | OLS treatment coefficient with HC1 standard error. |
| `ml_rate` | estimator | Cross-fitted ridge nuisance prediction followed by robust regression adjustment. |
| `nearest_neighbor_matching` | estimator | Covariate nearest-neighbor ATT estimate for matching baselines. |
| `hypex_cuped` | integration pipeline | Native PyPI HypEx CUPED followed by its A/B inference. |
| `cuped_control__hypex_ab` | pipeline | CUPED fitted on current control followed by `hypex_ab`. |
| `hypex_cupac_temporal` | integration pipeline | PyPI HypEx temporal linear CUPAC parity slot. |
| `cupac_linear_A__hypex_ab` | pipeline | Temporal CUPAC on pre-treatment lag2/lag1 features. |
| `cupac_linear_A_plus_B__hypex_ab` | pipeline | CUPAC A followed by a separate control-fitted B stage. |
| `cupac_linear_A_plus_B_plus_C__hypex_ab` | pipeline | A and B followed by a separately gated C stage. |
| `winsorized__hypex_ab` | changed-metric pipeline | Control-defined winsorization followed by `hypex_ab`. |
| `ratio_delta` | estimator | Ratio metric difference by delta method. |
| `stratified_mean_diff` | estimator | Weighted stratum-level differences. |
| `variance_weighted_stratified` | estimator | Stratum estimates combined by inverse variance. |
| `trimmed_mean_diff` | estimator | Difference of trimmed means with Welch-style SE. |
| `winsorized_mean_diff` | estimator | Difference of winsorized means with Welch-style SE. |
| `wald_late` | estimator | Wald ratio for instrumented treatment take-up. |
| `bonferroni` | correction | Multiple-testing correction helper. |
| `holm_bonferroni` | correction | Step-down Bonferroni helper. |
| `romano_wolf_stepdown` | correction | Step-down max-statistic correction over caller-supplied bootstrap statistics. |

## Design Preprocessors

| Name | Type | Notes |
| --- | --- | --- |
| `genetic_strata_reducer` | design preprocessor | Reduces atomic categorical combinations to exactly `U` macrostrata using the A/B-proportional variance objective. |
| `numeric_feature_binner` | feature adapter | Learns fixed categorical bins for numeric features without using target values. |

## Placeholders

| Name | Reason |
| --- | --- |
| `anti_nuclear_ks_test` | The table name is not enough to identify the exact statistic. |
| `hypex_matcher` | Requires internal HypEx matcher API and matching policy. |
| `calibration_reweighting` | Needs constraints, calibration totals, and variance estimator policy. |
| `optimal_transport_homogeneity_test` | Needs a statistic, calibration policy, and scalable solver contract. |
| `optimal_transport_check` | Excluded ambiguous historical label. |
| `optimal_transport_matching` | Needs matching distance, regularization, and inference policy. |
| `bayesian_cupac` | Target-transformer placeholder; needs a Bayesian model and posterior decision rule. |
| `nonlinear_cupac` | Selected target-transformer candidate; needs an honest model and tuning contract. |
| `prognostic_score_block_randomization` | Selected prospective design candidate; needs a randomized block-assignment contract. |

## Additional Candidates

- Trimmed or winsorized mean estimators for heavy-tailed metrics.
- CUPED with robust base estimators.
- Cross-fitted CUPAC for larger feature sets.
- Doubly robust / orthogonalized ATE estimators when real data contains rich
  pre-treatment covariates.
- Cluster-robust estimators for switchback or grouped randomization.
