# Отчёт VarWar R&D v3

**Этап:** `screening`.

Единица сравнения — полная процедура назначения и анализа. Рейтинг строится только внутри одного estimand и только после численного экрана статистического вывода.

## Полнота запуска

- Строк в полной матрице совместимости: **762,048**.
- Исполненных trial-level строк: **288,000**.
- Успешно: **288,000**; только точка: **0**; ошибок: **0**.

## Как проверяется статистический вывод

Под H0 измеряются FPR с интервалом Уилсона и super-uniformity p-value. Под H1 измеряются bias, RMSE, empirical SD, средний SE, SE/SD, покрытие CI, power и ошибка второго рода. Значение FPR=0 на малом числе повторов помечается как неопределённое, а не идеальное.

## Границы численного результата

- Реально исполнено уникальных процедур: **5**.
- Реально исполнено синтетических миров: **24**.
- Число H0-реализаций на сравнимую строку: **[200]**.
- Полная матрица совместимости шире исполненного этапа; неисполненная строка не считается проверенным методом.

## Сводная таблица

| Пайплайн | Мир | Estimand | Информация | Время | Эффект, σ | Вывод | Bias | RMSE | FPR | Coverage | Power | SE/SD | RSS, MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | population_shift | sample_ate | panel_prior | pre_treatment_only | 0.1 | bad | -0.0024 | 0.03337 | 1.000 | 0.950 | 1.000 | 1.032 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | population_shift | sample_ate | panel_prior | full_timeline | 0.1 | bad | -0.0024 | 0.03337 | 1.000 | 0.950 | 1.000 | 1.032 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | population_shift | sample_ate | experiment_only | full_timeline | 0.1 | bad | -0.0024 | 0.03337 | 1.000 | 0.950 | 1.000 | 1.032 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | population_shift | sample_ate | experiment_only | pre_treatment_only | 0.1 | bad | -0.0024 | 0.03337 | 1.000 | 0.950 | 1.000 | 1.032 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | population_shift | sample_ate | panel_prior | pre_treatment_only | 0.2 | bad | -0.0024 | 0.03337 | 1.000 | 0.950 | 1.000 | 1.032 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | population_shift | sample_ate | panel_prior | full_timeline | 0.2 | bad | -0.0024 | 0.03337 | 1.000 | 0.950 | 1.000 | 1.032 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | population_shift | sample_ate | experiment_only | full_timeline | 0.2 | bad | -0.0024 | 0.03337 | 1.000 | 0.950 | 1.000 | 1.032 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | population_shift | sample_ate | experiment_only | pre_treatment_only | 0.2 | bad | -0.0024 | 0.03337 | 1.000 | 0.950 | 1.000 | 1.032 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | population_shift | sample_ate | panel_prior | pre_treatment_only | 0.1 | bad | -0.002054 | 0.03338 | 1.000 | 0.955 | 1.000 | 1.037 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | population_shift | sample_ate | experiment_only | pre_treatment_only | 0.1 | bad | -0.002054 | 0.03338 | 1.000 | 0.955 | 1.000 | 1.037 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | population_shift | sample_ate | experiment_only | full_timeline | 0.1 | bad | -0.002054 | 0.03338 | 1.000 | 0.955 | 1.000 | 1.037 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | population_shift | sample_ate | panel_prior | full_timeline | 0.1 | bad | -0.002054 | 0.03338 | 1.000 | 0.955 | 1.000 | 1.037 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | population_shift | sample_ate | panel_prior | pre_treatment_only | 0.2 | bad | -0.00195 | 0.03339 | 1.000 | 0.955 | 1.000 | 1.037 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | population_shift | sample_ate | experiment_only | pre_treatment_only | 0.2 | bad | -0.00195 | 0.03339 | 1.000 | 0.955 | 1.000 | 1.037 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | population_shift | sample_ate | experiment_only | full_timeline | 0.2 | bad | -0.00195 | 0.03339 | 1.000 | 0.955 | 1.000 | 1.037 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | population_shift | sample_ate | panel_prior | full_timeline | 0.2 | bad | -0.00195 | 0.03339 | 1.000 | 0.955 | 1.000 | 1.037 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | heteroskedastic | sample_ate | panel_prior | pre_treatment_only | 0.2 | bad | 0.00429 | 0.03342 | 0.060 | 0.940 | 1.000 | 1.047 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | heteroskedastic | sample_ate | panel_prior | full_timeline | 0.2 | bad | 0.00429 | 0.03342 | 0.060 | 0.940 | 1.000 | 1.047 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | heteroskedastic | sample_ate | experiment_only | full_timeline | 0.2 | bad | 0.00429 | 0.03342 | 0.060 | 0.940 | 1.000 | 1.047 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | heteroskedastic | sample_ate | experiment_only | pre_treatment_only | 0.2 | bad | 0.00429 | 0.03342 | 0.060 | 0.940 | 1.000 | 1.047 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | heteroskedastic | sample_ate | panel_prior | pre_treatment_only | 0.1 | bad | 0.00429 | 0.03342 | 0.060 | 0.940 | 0.850 | 1.047 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | heteroskedastic | sample_ate | panel_prior | full_timeline | 0.1 | bad | 0.00429 | 0.03342 | 0.060 | 0.940 | 0.850 | 1.047 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | heteroskedastic | sample_ate | experiment_only | full_timeline | 0.1 | bad | 0.00429 | 0.03342 | 0.060 | 0.940 | 0.850 | 1.047 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.lin_regression__hc2 | heteroskedastic | sample_ate | experiment_only | pre_treatment_only | 0.1 | bad | 0.00429 | 0.03342 | 0.060 | 0.940 | 0.850 | 1.047 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | heteroskedastic | sample_ate | panel_prior | pre_treatment_only | 0.1 | bad | 0.004424 | 0.03354 | 0.050 | 0.950 | 0.850 | 1.046 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | heteroskedastic | sample_ate | panel_prior | full_timeline | 0.1 | bad | 0.004424 | 0.03354 | 0.050 | 0.950 | 0.850 | 1.046 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | heteroskedastic | sample_ate | experiment_only | pre_treatment_only | 0.1 | bad | 0.004424 | 0.03354 | 0.050 | 0.950 | 0.850 | 1.046 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | heteroskedastic | sample_ate | experiment_only | full_timeline | 0.1 | bad | 0.004424 | 0.03354 | 0.050 | 0.950 | 0.850 | 1.046 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | heteroskedastic | sample_ate | panel_prior | pre_treatment_only | 0.2 | bad | 0.004522 | 0.03355 | 0.050 | 0.950 | 1.000 | 1.046 |  |
| adapt.identity__structure.none__assign.complete_randomization__target.identity__analysis.ml_rate__crossfit_hc2 | heteroskedastic | sample_ate | panel_prior | full_timeline | 0.2 | bad | 0.004522 | 0.03355 | 0.050 | 0.950 | 1.000 | 1.046 |  |

## Правило выбора

1. Совпадают estimand, режим информации и временной доступ.
2. Проходят bias, FPR, p-value, coverage и SE/SD.
3. Среди прошедших сравниваются RMSE, variance reduction, power, MDE, время и RSS.
4. Full-timeline и post-treatment-risk кандидаты ранжируются отдельно от causal-safe.
