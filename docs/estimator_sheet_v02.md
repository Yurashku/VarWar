# Estimator Sheet V02 Mapping

Source: second worksheet, `Estimators V02`, from the user-provided workbook
`C:/Users/ivan/Downloads/Estimators (1).xlsx`.

| Sheet row | Sheet name | VarWar status |
| --- | --- | --- |
| 2 | t-test | `hypex_ab`, `welch_mean_diff`, `pooled_mean_diff` |
| 3 | ks-test | `ks_test` |
| 4 | hedgehog t-test | `randomized_t_test` |
| 5 | anti-nuclear ks-test | placeholder: `anti_nuclear_ks_test` |
| 6 | approx ks-test fast | `approx_ks_fast` |
| 7 | e-test | `energy_test` interpretation; other e-value meanings still need clarification |
| 8 | Bonferroni method | `bonferroni`, `holm_bonferroni` helpers |
| 9 | Romano-Wolf | `romano_wolf_stepdown` helper over supplied bootstrap/null statistics |
| 10 | HypEx matcher | placeholder: `hypex_matcher` |
| 11 | regression adjustment estimator | `regression_adjustment` |
| 12 | permutation test | `permutation_mean_test` |
| 13 | robust estimation | `trimmed_mean_diff`, `winsorized_mean_diff` |
| 14 | stratified estimation | `stratified_mean_diff` |
| 15 | variance-weighted estimator | `variance_weighted_stratified` |
| 16 | MLRATE | `ml_rate` |
| 17 | winsorization | `winsorized+hypex_ab`, `winsorized_mean_diff` |
| 18 | genetic strata reduction | implemented as the separate `genetic_strata_reducer` design preprocessor |
| 19 | stratified summary | `stratified_mean_diff` |
| 20 | optimal-transport matching | placeholder: `optimal_transport_matching`; added `nearest_neighbor_matching` as a lightweight matching baseline |
| 21 | CUPED | `cuped+hypex_ab` |
| 22 | CUPAC on pretreatment features C0 | `sklearn_cupac_A+hypex_ab`, `hypex_cupac` reference |
| 23 | CUPAC on pretreatment features C0 + C1 | `sklearn_cupac_A_plus_B+hypex_ab` |
| 24 | CUPAC on pretreatment features C0 + C1 + C2 | `sklearn_cupac_A_plus_B_plus_C+hypex_ab` |
| 25 | Bayesian CUPAC | placeholder: `bayesian_cupac` |

The worksheet is preserved as a historical requirements mapping. Current
method categories and behavior are documented in
[`methods/README.md`](methods/README.md).
