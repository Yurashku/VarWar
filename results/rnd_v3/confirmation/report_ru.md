# Отчёт VarWar R&D v3

**Этап:** `confirmation`.

Единица сравнения — полная процедура назначения и анализа. Рейтинг строится только внутри одного estimand и только после численного экрана статистического вывода.

## Полнота запуска

- Строк в полной матрице совместимости: **762,048**.
- Исполненных trial-level строк: **60,000**.
- Успешно: **60,000**; только точка: **0**; ошибок: **0**.

## Как проверяется статистический вывод

Под H0 измеряются FPR с интервалом Уилсона и super-uniformity p-value. Под H1 измеряются bias, RMSE, empirical SD, средний SE, SE/SD, покрытие CI, power и ошибка второго рода. Значение FPR=0 на малом числе повторов помечается как неопределённое, а не идеальное.

## Границы численного результата

- Реально исполнено уникальных процедур: **5**.
- Реально исполнено синтетических миров: **1**.
- Число H0-реализаций на сравнимую строку: **[1000]**.
- Полная матрица совместимости шире исполненного этапа; неисполненная строка не считается проверенным методом.

## Сводная таблица

| Пайплайн | Мир | Estimand | Информация | Время | Эффект, σ | Вывод | Bias | RMSE | FPR | Coverage | Power | SE/SD | RSS, MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | panel_prior | pre_treatment_only | 0.2 | good | -0.004403 | 0.1078 | 0.048 | 0.954 | 0.404 | 1.029 | 1390.3 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | panel_prior | full_timeline | 0.2 | good | -0.004403 | 0.1078 | 0.048 | 0.954 | 0.404 | 1.029 | 1390.3 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | experiment_only | full_timeline | 0.2 | good | -0.004403 | 0.1078 | 0.048 | 0.954 | 0.404 | 1.029 | 1390.3 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | experiment_only | pre_treatment_only | 0.2 | good | -0.004403 | 0.1078 | 0.048 | 0.954 | 0.404 | 1.029 | 1390.3 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | panel_prior | pre_treatment_only | 0.1 | good | 0.003528 | 0.1083 | 0.048 | 0.949 | 0.134 | 1.022 | 1390.3 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | panel_prior | full_timeline | 0.1 | good | 0.003528 | 0.1083 | 0.048 | 0.949 | 0.134 | 1.022 | 1390.3 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | experiment_only | full_timeline | 0.1 | good | 0.003528 | 0.1083 | 0.048 | 0.949 | 0.134 | 1.022 | 1390.3 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | gaussian_r2_04_n200 | sample_ate | experiment_only | pre_treatment_only | 0.1 | good | 0.003528 | 0.1083 | 0.048 | 0.949 | 0.134 | 1.022 | 1390.3 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | panel_prior | pre_treatment_only | 0.2 | good | -0.0024 | 0.1092 | 0.047 | 0.955 | 0.405 | 1.027 | 1381.7 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | experiment_only | full_timeline | 0.2 | good | -0.0024 | 0.1092 | 0.047 | 0.955 | 0.405 | 1.027 | 1381.7 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | panel_prior | full_timeline | 0.2 | good | -0.0024 | 0.1092 | 0.047 | 0.955 | 0.405 | 1.027 | 1381.7 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | experiment_only | pre_treatment_only | 0.2 | good | -0.0024 | 0.1092 | 0.047 | 0.955 | 0.405 | 1.027 | 1381.7 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | panel_prior | pre_treatment_only | 0.1 | good | 0.004617 | 0.1102 | 0.047 | 0.948 | 0.142 | 1.019 | 1381.7 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | experiment_only | full_timeline | 0.1 | good | 0.004617 | 0.1102 | 0.047 | 0.948 | 0.142 | 1.019 | 1381.7 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | panel_prior | full_timeline | 0.1 | good | 0.004617 | 0.1102 | 0.047 | 0.948 | 0.142 | 1.019 | 1381.7 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | gaussian_r2_04_n200 | sample_ate | experiment_only | pre_treatment_only | 0.1 | good | 0.004617 | 0.1102 | 0.047 | 0.948 | 0.142 | 1.019 | 1381.7 |
| adapt.identity__structure.none__assign.complete_randomization__target.cuped_historical__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | panel_prior | pre_treatment_only | 0.1 | good | 0.007301 | 0.1308 | 0.029 | 0.973 | 0.101 | 1.082 | 1370.0 |
| adapt.identity__structure.none__assign.complete_randomization__target.cuped_historical__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | experiment_only | pre_treatment_only | 0.1 | good | 0.007301 | 0.1308 | 0.029 | 0.973 | 0.101 | 1.082 | 1370.0 |
| adapt.identity__structure.none__assign.complete_randomization__target.cuped_historical__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | panel_prior | full_timeline | 0.1 | good | 0.007301 | 0.1308 | 0.029 | 0.973 | 0.101 | 1.082 | 1370.0 |
| adapt.identity__structure.none__assign.complete_randomization__target.cuped_historical__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | experiment_only | full_timeline | 0.1 | good | 0.007301 | 0.1308 | 0.029 | 0.973 | 0.101 | 1.082 | 1370.0 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | panel_prior | pre_treatment_only | 0.2 | good | -0.003453 | 0.1354 | 0.047 | 0.955 | 0.265 | 1.041 | 1395.1 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | panel_prior | full_timeline | 0.2 | good | -0.003453 | 0.1354 | 0.047 | 0.955 | 0.265 | 1.041 | 1395.1 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | experiment_only | full_timeline | 0.2 | good | -0.003453 | 0.1354 | 0.047 | 0.955 | 0.265 | 1.041 | 1395.1 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | experiment_only | pre_treatment_only | 0.2 | good | -0.003453 | 0.1354 | 0.047 | 0.955 | 0.265 | 1.041 | 1395.1 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | panel_prior | pre_treatment_only | 0.1 | good | 0.007439 | 0.1408 | 0.047 | 0.953 | 0.108 | 1.005 | 1395.1 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | panel_prior | full_timeline | 0.1 | good | 0.007439 | 0.1408 | 0.047 | 0.953 | 0.108 | 1.005 | 1395.1 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | experiment_only | full_timeline | 0.1 | good | 0.007439 | 0.1408 | 0.047 | 0.953 | 0.108 | 1.005 | 1395.1 |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | experiment_only | pre_treatment_only | 0.1 | good | 0.007439 | 0.1408 | 0.047 | 0.953 | 0.108 | 1.005 | 1395.1 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | panel_prior | pre_treatment_only | 0.2 | warning | -0.005157 | 0.115 | 0.017 | 0.985 | 0.237 | 1.227 | 1385.0 |
| adapt.identity__structure.none__assign.complete_randomization__target.cupac_linear_A__analysis.diff_means__welch | gaussian_r2_04_n200 | sample_ate | experiment_only | pre_treatment_only | 0.2 | warning | -0.005157 | 0.115 | 0.017 | 0.985 | 0.237 | 1.227 | 1385.0 |

## Правило выбора

1. Совпадают estimand, режим информации и временной доступ.
2. Проходят bias, FPR, p-value, coverage и SE/SD.
3. Среди прошедших сравниваются RMSE, variance reduction, power, MDE, время и RSS.
4. Full-timeline и post-treatment-risk кандидаты ранжируются отдельно от causal-safe.
