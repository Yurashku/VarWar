# Готовые конвейеры

В итоговой R&D-гонке сравниваются конвейеры с одинаковым estimand, допустимыми
данными и одной политикой статистического вывода. Сравнение времени или
дисперсии отдельных компонентов полезно для профилирования, но не определяет
лучший пользовательский метод.

## Основная гонка ATE

| ID строки | Состав | Назначение |
| --- | --- | --- |
| `welch` | `welch_mean_diff` | статистический baseline без коррекции |
| `hypex_ab` | HypEx A/B, default Welch | интеграционная проверка |
| `hypex_cuped` | нативный HypEx CUPED → HypEx Welch | интеграционная проверка |
| `cuped_control__hypex_ab` | CUPED по текущему контролю → HypEx Welch | один pre-target |
| `hypex_cupac_temporal` | temporal linear CUPAC HypEx → HypEx Welch | интеграционная проверка lag2 → lag1 |
| `cupac_linear_A__hypex_ab` | temporal CUPAC A → HypEx Welch | только pre-treatment признаки |
| `cupac_linear_A_plus_B__hypex_ab` | CUPAC A → отдельный linear B → HypEx Welch | исследовательский: экспертно safe in-time признаки |
| `cupac_linear_A_plus_B_plus_C__hypex_ab` | A → B → gated linear C → HypEx Welch | исследовательская зона C |
| `cupac_quadratic_A__hypex_ab` | quadratic temporal CUPAC A → HypEx Welch | ограниченная нелинейность зоны A |
| `regression_adjustment` | OLS + HC1 | линейная ковариатная коррекция внутри estimator |
| `ml_rate` | cross-fitted ridge → interacted OLS + HC1 | гибкая оценка без отдельного transformed target |
| `winsorized__hypex_ab` | control-winsorization → HypEx Welch | отдельный изменённый estimand |

Все локальные CUPAC-варианты работают с одним доступным датасетом. Этап A
обучается только на временном переходе `lag2 → lag1`, этапы B/C — на текущем
контроле. `hypex_cupac_temporal` является библиотечной проверкой, а не звеном
локальной predecessor-цепочки.

## Гонка стратификации

- `atomic_strata → stratified_mean_diff`
- `random_reduction → stratified_mean_diff`
- `mean_sorted_reduction → stratified_mean_diff`
- `genetic_strata_reducer → stratified_mean_diff`
- `numeric_feature_binner → genetic_strata_reducer → stratified_mean_diff`
- `genetic_strata_reducer → CUPED/CUPAC → stratified_mean_diff`

Полные строки новой гонки называются `ga_strata__stratified`,
`ga_strata__cuped_control__stratified` и
`ga_strata__cupac_linear_*__stratified`. `ab_proportional` — внутренняя
целевая функция GA, а не estimator и не способ разбить пользователей на A/B.

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

Результаты актуальной синтетической гонки сохраняются отдельно от
спецификации методов в `results/pipeline_race_v2/`. Каталог каждого запуска
фиксирует состав, estimand, контур применимости и предыдущий пайплайн цепочки.
