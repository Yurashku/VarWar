# Estimator Sheet V02 Mapping

> Историческое сопоставление. Актуальная классификация и нумерация находятся в
> [docs/methods/all_methods.md](methods/all_methods.md).

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
| 17 | winsorization | `winsorized__hypex_ab`; `winsorized_mean_diff` retained as legacy |
| 18 | genetic strata reduction | implemented as the separate `genetic_strata_reducer` design preprocessor |
| 19 | stratified summary | `stratified_mean_diff` |
| 20 | optimal-transport matching | placeholder: `optimal_transport_matching`; added `nearest_neighbor_matching` as a lightweight matching baseline |
| 21 | CUPED | `hypex_cuped`, `cuped_control__hypex_ab` |
| 22 | CUPAC on pretreatment features C0 | `cupac_linear_A__hypex_ab`, `hypex_cupac_temporal` parity |
| 23 | CUPAC on pretreatment features C0 + C1 | `cupac_linear_A_plus_B__hypex_ab`; B is a separate stage |
| 24 | CUPAC on pretreatment features C0 + C1 + C2 | `cupac_linear_A_plus_B_plus_C__hypex_ab`; C is separately gated |
| 25 | Bayesian CUPAC | `TargetTransformer` placeholder: `bayesian_cupac` |

The worksheet is preserved as a historical requirements mapping. Current
method categories and behavior are documented in
[`methods/README.md`](methods/README.md).
The full implementation-status list is
[`methods/all_methods.md`](methods/all_methods.md).
