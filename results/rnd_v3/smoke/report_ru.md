# Отчёт VarWar R&D v3

**Этап:** `smoke`.

Единица сравнения — полная процедура назначения и анализа. Рейтинг строится только внутри одного estimand и только после численного экрана статистического вывода.

## Полнота запуска

- Строк в полной матрице совместимости: **762,048**.
- Исполненных trial-level строк: **280**.
- Успешно: **280**; только точка: **0**; ошибок: **0**.

## Как проверяется статистический вывод

Под H0 измеряются FPR с интервалом Уилсона и super-uniformity p-value. Под H1 измеряются bias, RMSE, empirical SD, средний SE, SE/SD, покрытие CI, power и ошибка второго рода. Значение FPR=0 на малом числе повторов помечается как неопределённое, а не идеальное.

> Smoke нужен для проверки исполнимости. Пять повторений не дают научного рейтинга и не позволяют заключать о FPR или power.

## Сводная таблица

| Пайплайн | Мир | Estimand | Вывод | Bias | RMSE | FPR | Coverage | Power | SE/SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adapt.identity__structure.atomic_strata__assign.stratified_randomization__target.identity__analysis.stratified_ate__native | gaussian_r2_04_n200 | sample_ate | warning | 0.02317 | 0.07149 | 0.000 | 1.000 | 0.200 | 1.643 |
| adapt.identity__structure.none__assign.rerandomization_mixed_balance__target.identity__analysis.lin_regression__hc3 | gaussian_r2_04_n200 | sample_ate | warning | -0.03861 | 0.0784 | 0.000 | 1.000 | 0.400 | 1.486 |
| adapt.identity__structure.none__assign.rerandomization_mahalanobis__target.cuped_huber_historical__analysis.diff_means__design_based_normal | gaussian_r2_04_n200 | sample_ate | warning | -0.01511 | 0.08045 | 0.000 | 1.000 | 0.000 | 1.623 |
| adapt.identity__structure.atomic_strata__assign.stratified_randomization__target.identity__analysis.population_weighted_stratified_ate__native | gaussian_r2_04_n200 | population_ate | warning | 0.05821 | 0.09062 | 0.000 | 1.000 | 0.200 | 1.824 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__student_pooled | gaussian_r2_04_n200 | sample_ate | warning | 0.05209 | 0.09122 | 0.200 | 1.000 | 0.400 | 1.712 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.05209 | 0.09122 | 0.200 | 1.000 | 0.400 | 1.712 |
| adapt.identity__structure.prognostic_blocks__assign.stratified_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0562 | 0.09193 | 0.000 | 1.000 | 0.000 | 1.390 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.trimmed_mean__yuen | gaussian_r2_04_n200 | trimmed_mean_effect | warning | 0.06248 | 0.09787 | 0.200 | 1.000 | 0.200 | 1.783 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.hedgehog_weighted_ate__native | gaussian_r2_04_n200 | sample_ate | warning | 0.05451 | 0.1007 | 0.200 | 1.000 | 0.400 | 1.537 |
| adapt.identity__structure.genetic_strata__assign.stratified_randomization__target.identity__analysis.hypex_ab__integration | gaussian_r2_04_n200 | sample_ate | warning | -0.02296 | 0.1022 | 0.000 | 1.000 | 0.200 | 1.286 |
| adapt.identity__structure.none__assign.rerandomization_max_smd__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0491 | 0.1035 | 0.200 | 1.000 | 0.400 | 1.087 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_hypex_compatible_A_plus_B__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.08945 | 0.1316 | 0.000 | 1.000 | 0.400 | 1.328 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.hypex_cuped__integration | gaussian_r2_04_n200 | sample_ate | warning | 0.06633 | 0.1334 | 0.000 | 1.000 | 0.400 | 0.998 |
| adapt.identity__structure.none__assign.complete_randomization__target.panel_prior_cupac_A_plus_B__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.08858 | 0.1341 | 0.000 | 1.000 | 0.400 | 1.273 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_quadratic_A_plus_B__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.097 | 0.1345 | 0.000 | 1.000 | 0.400 | 1.375 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A_plus_B__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.09261 | 0.1346 | 0.000 | 1.000 | 0.400 | 1.312 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.calibration_weighted_population_ate__native | gaussian_r2_04_n200 | population_ate | warning | 0.1174 | 0.1351 | 0.000 | 1.000 | 0.600 | 1.963 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.hypex_cupac_A__integration | gaussian_r2_04_n200 | sample_ate | warning | 0.08065 | 0.1357 | 0.000 | 0.800 | 0.400 | 0.968 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_hypex_compatible_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.08065 | 0.1357 | 0.000 | 1.000 | 0.400 | 1.174 |
| adapt.identity__structure.atomic_strata__assign.stratified_randomization__target.identity__analysis.common_effect_inverse_variance_stratified__native | gaussian_r2_04_n200 | common_effect | warning | 0.007665 | 0.1493 | 0.000 | 0.600 | 0.400 | 0.630 |
| adapt.numeric_binner__structure.atomic_strata__assign.stratified_randomization__target.cuped_historical__analysis.diff_means__student_pooled | gaussian_r2_04_n200 | sample_ate | warning | -0.01279 | 0.1561 | 0.000 | 1.000 | 0.200 | 0.823 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_quadratic_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.1028 | 0.1578 | 0.000 | 0.800 | 0.400 | 1.070 |
| adapt.identity__structure.none__assign.complete_randomization__target.panel_prior_cupac_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.08948 | 0.158 | 0.000 | 0.800 | 0.400 | 0.985 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.09374 | 0.1586 | 0.000 | 0.800 | 0.400 | 1.002 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_hypex_compatible_A_plus_B_plus_C__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.1163 | 0.164 | 0.000 | 0.800 | 0.400 | 1.108 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_quadratic_A_plus_B_plus_C__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.1236 | 0.1707 | 0.000 | 0.800 | 0.600 | 1.088 |
| adapt.identity__structure.none__assign.complete_randomization__target.panel_prior_cupac_A_plus_B_plus_C__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.1165 | 0.1708 | 0.000 | 0.800 | 0.400 | 1.026 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A_plus_B_plus_C__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.1211 | 0.1716 | 0.000 | 0.800 | 0.400 | 1.054 |

## Правило выбора

1. Совпадают estimand, режим информации и временной доступ.
2. Проходят bias, FPR, p-value, coverage и SE/SD.
3. Среди прошедших сравниваются RMSE, variance reduction, power, MDE, время и RSS.
4. Full-timeline и post-treatment-risk кандидаты ранжируются отдельно от causal-safe.
