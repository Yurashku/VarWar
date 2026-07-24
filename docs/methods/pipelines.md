# Готовые конвейеры

В итоговой R&D-гонке сравниваются конвейеры с одинаковым estimand, допустимыми
данными и одной политикой статистического вывода. Сравнение времени или
дисперсии отдельных компонентов полезно для профилирования, но не определяет
лучший пользовательский метод.

## Основная гонка ATE

| ID строки | Состав | Назначение |
| --- | --- | --- |
| `welch_mean_diff` | Welch | статистический baseline без коррекции |
| `hypex_ab` | HypEx A/B, default Welch | интеграционная проверка |
| `cuped+hypex_ab` | CUPED VarWar → HypEx Welch | один pre-target |
| `hypex_cupac` | temporal linear CUPAC HypEx → HypEx Welch | lag-2 → lag-1 → текущий target |
| `sklearn_cupac_A+hypex_ab` | local linear CUPAC(A) → HypEx Welch | только pre-treatment признаки |
| `sklearn_cupac_A_plus_B+hypex_ab` | local linear CUPAC(A+B) → HypEx Welch | плюс безопасные in-time признаки |
| `sklearn_cupac_A_plus_B_plus_C+hypex_ab` | local linear CUPAC(A+B+C) → HypEx Welch | плюс признаки после balance gate |
| `regression_adjustment` | OLS + HC1 | линейная ковариатная коррекция внутри estimator |
| `ml_rate` | cross-fitted ridge → interacted OLS + HC1 | гибкая оценка без отдельного transformed target |

## Гонка стратификации

- `atomic_strata → stratified_mean_diff`
- `random_reduction → stratified_mean_diff`
- `mean_sorted_reduction → stratified_mean_diff`
- `genetic_strata_reducer → stratified_mean_diff`
- `numeric_feature_binner → genetic_strata_reducer → stratified_mean_diff`
- `genetic_strata_reducer → CUPED/CUPAC → stratified_mean_diff`

`genetic_strata_ab` и `ab_proportional` в основной таблице методов не должны
выглядеть самостоятельными estimators: это профиль и внутренняя objective.

## Отдельные гонки

Нельзя ранжировать вместе с обычным ATE без дополнительных оговорок:

- `winsorized → estimator`: эффект на winsorized metric;
- `ratio_delta`: эффект ratio-of-means;
- `ratio_linearization → ...`: будущая отдельная семья ratio-конвейеров;
- matching: ATT или другой observational estimand;
- `wald_late`: LATE;
- KS и energy: тесты различия распределений, а не ATE;
- корректировки множественных проверок: применяются после получения семейства
  статистик.

Результаты синтетической гонки сохраняются отдельно от спецификации методов в
`results/first_pipeline_race/`. Excel-каталог содержит отдельный лист
«Пайплайны» с колонками для будущего заполнения bias, RMSE, FPR, power, времени
и памяти.
