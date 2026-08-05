# Отчёт VarWar R&D v3

**Этап:** `screening`.

Единица сравнения — полная процедура назначения и анализа. Рейтинг строится только внутри одного estimand и только после численного экрана статистического вывода.

## Полнота запуска

- Строк в полной матрице совместимости: **762,048**.
- Исполненных trial-level строк: **16,800**.
- Успешно: **16,800**; только точка: **0**; ошибок: **0**.

## Как проверяется статистический вывод

Под H0 измеряются FPR с интервалом Уилсона и super-uniformity p-value. Под H1 измеряются bias, RMSE, empirical SD, средний SE, SE/SD, покрытие CI, power и ошибка второго рода. Значение FPR=0 на малом числе повторов помечается как неопределённое, а не идеальное.

## Сводная таблица

| Пайплайн | Мир | Estimand | Вывод | Bias | RMSE | FPR | Coverage | Power | SE/SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0004933 | 0.1099 | 0.085 | 0.960 | 0.130 | 1.004 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0004933 | 0.1099 | 0.085 | 0.960 | 0.130 | 1.004 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0004933 | 0.1099 | 0.085 | 0.960 | 0.130 | 1.004 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0004933 | 0.1099 | 0.085 | 0.960 | 0.130 | 1.004 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.006067 | 0.1104 | 0.085 | 0.950 | 0.400 | 1.002 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.006067 | 0.1104 | 0.085 | 0.950 | 0.400 | 1.002 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.006067 | 0.1104 | 0.085 | 0.950 | 0.400 | 1.002 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.006067 | 0.1104 | 0.085 | 0.950 | 0.400 | 1.002 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.004155 | 0.111 | 0.075 | 0.945 | 0.405 | 1.006 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.004155 | 0.111 | 0.075 | 0.945 | 0.405 | 1.006 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.004155 | 0.111 | 0.075 | 0.945 | 0.405 | 1.006 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.004155 | 0.111 | 0.075 | 0.945 | 0.405 | 1.006 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0005899 | 0.1123 | 0.075 | 0.945 | 0.130 | 0.996 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0005899 | 0.1123 | 0.075 | 0.945 | 0.130 | 0.996 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0005899 | 0.1123 | 0.075 | 0.945 | 0.130 | 0.996 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | warning | -0.0005899 | 0.1123 | 0.075 | 0.945 | 0.130 | 0.996 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A_plus_B__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.002987 | 0.1136 | 0.010 | 0.990 | 0.245 | 1.239 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A_plus_B__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.002987 | 0.1136 | 0.010 | 0.990 | 0.245 | 1.239 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.005609 | 0.1151 | 0.020 | 0.990 | 0.240 | 1.225 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.005609 | 0.1151 | 0.020 | 0.990 | 0.240 | 1.225 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.005609 | 0.1151 | 0.020 | 0.990 | 0.240 | 1.225 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.005609 | 0.1151 | 0.020 | 0.990 | 0.240 | 1.225 |
| adapt.identity__structure.none__assign.complete_randomization__target.panel_prior_cupac_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.006842 | 0.1152 | 0.015 | 0.985 | 0.240 | 1.224 |
| adapt.identity__structure.none__assign.complete_randomization__target.panel_prior_cupac_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.006842 | 0.1152 | 0.015 | 0.985 | 0.240 | 1.224 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A_plus_B_plus_C__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.003203 | 0.1158 | 0.025 | 0.985 | 0.225 | 1.216 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A_plus_B_plus_C__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | -0.003203 | 0.1158 | 0.025 | 0.985 | 0.225 | 1.216 |
| adapt.identity__structure.none__assign.complete_randomization__target.panel_prior_cupac_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.0007388 | 0.1183 | 0.015 | 0.985 | 0.085 | 1.187 |
| adapt.identity__structure.none__assign.complete_randomization__target.panel_prior_cupac_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.0007388 | 0.1183 | 0.015 | 0.985 | 0.085 | 1.187 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.0005671 | 0.1191 | 0.020 | 0.980 | 0.080 | 1.180 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | warning | 0.0005671 | 0.1191 | 0.020 | 0.980 | 0.080 | 1.180 |

## Правило выбора

1. Совпадают estimand, режим информации и временной доступ.
2. Проходят bias, FPR, p-value, coverage и SE/SD.
3. Среди прошедших сравниваются RMSE, variance reduction, power, MDE, время и RSS.
4. Full-timeline и post-treatment-risk кандидаты ранжируются отдельно от causal-safe.
