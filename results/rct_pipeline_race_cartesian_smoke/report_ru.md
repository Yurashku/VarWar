# Гонка аналитических пайплайнов VarWar v2

## Короткий вывод

Сравниваются полные аналитические пайплайны, а не изолированные примитивы. Сначала проверяются смещение, FPR и покрытие; только после этого рассматриваются дисперсия, мощность и стоимость.

- **Линейный сигнал, нормальный шум.** Ни один кандидат не прошёл заданный диагностический допуск.
- **Линейный сигнал, тяжёлые хвосты.** Ни один кандидат не прошёл заданный диагностический допуск.
- **Нелинейный сигнал, нормальный шум.** Ни один кандидат не прошёл заданный диагностический допуск.
- **Treatment влияет на признак зоны C.** Ни один кандидат не прошёл заданный диагностический допуск.
- **Ошибочно безопасный признак зоны B.** Ни один кандидат не прошёл заданный диагностический допуск.
- **Нерепрезентативная выборка и известная целевая популяция.** Ни один кандидат не прошёл заданный диагностический допуск.

## Постановка

- Monte Carlo повторов на H0 и H1: `1`.
- Строк в одном эксперименте: `2500`.
- Истинный средний эффект H1: `0.15`.
- Уровень значимости: `0.05`.
- H0 и H1 внутри повтора имеют одинаковые признаки, assignment и шум; отличается только добавленный treatment effect.
- Отдельной обучающей выборки нет: модель CUPAC A использует `lag2 -> lag1`, локальные CUPED/B/C оценивают коэффициенты по текущему контролю, а GA использует только pre-target тех же строк. Интеграционные HypEx CUPED/CUPAC сохраняют политику пакета и оценивают theta на всём текущем эксперименте.
- A — предэкспериментные признаки; B — экспертно безопасные признаки текущего периода и отдельная линейная корректировка; C — ещё одна линейная корректировка после смыслового отсева и проверок баланса, пропусков и распределения.
- Balance gate не доказывает причинную безопасность. Сценарий `misclassified_b` намеренно проверяет цену экспертной ошибки.

## Правило допуска

Для первичного R&D-экрана заранее заданы практические допуски: |bias| ≤ 0,02, FPR ≤ 8%, покрытие ≥ 92% и SE/SD от 0,80 до 1,20. Также требуются 100% успешных запусков.
Прохождение экрана не доказывает статистическую валидность и не является результатом множественной проверки. Точные биномиальные интервалы FPR и power приведены для оценки неопределённости Monte Carlo.
Винзоризация вынесена в отдельный estimand и не участвует в рейтинге исходного ATE.

## Каталог гонки

| pipeline | pipeline_name_ru | components | estimand | pipeline_tier | predecessor_pipeline |
| --- | --- | --- | --- | --- | --- |
| welch | Welch t-test | welch_mean_diff | ATE | confirmatory |  |
| hypex_ab | HypEx A/B (Welch) | hypex_ab | ATE | integration_parity | welch |
| pooled_mean_diff | Разность средних с общей дисперсией | pooled_mean_diff | ATE | exploratory | welch |
| randomized_weighted_mean_diff | Воспроизводимая взвешенная разность средних | randomized_weighted_mean_diff | Взвешенный ATE | exploratory | welch |
| permutation_mean_diff | Перестановочная оценка разности средних | permutation_mean_diff | ATE | exploratory | welch |
| bootstrap_mean_diff | Bootstrap-оценка разности средних | bootstrap_mean_diff | ATE | exploratory | welch |
| hypex_cuped | HypEx CUPED | hypex_cuped -> hypex_ab | ATE | integration_parity | hypex_ab |
| cuped_control__hypex_ab | CUPED по контролю -> HypEx A/B | cuped_control -> hypex_ab | ATE | confirmatory | hypex_ab |
| hypex_cupac_temporal | HypEx temporal CUPAC | hypex_cupac_temporal -> hypex_ab | ATE | integration_parity | hypex_ab |
| cupac_linear_A__hypex_ab | CUPAC A -> HypEx A/B | cupac_linear_A -> hypex_ab | ATE | confirmatory | hypex_ab |
| cupac_linear_A_plus_B__hypex_ab | CUPAC A+B -> HypEx A/B | cupac_linear_A_plus_B -> hypex_ab | ATE | exploratory | cupac_linear_A__hypex_ab |
| cupac_linear_A_plus_B_plus_C__hypex_ab | CUPAC A+B+C -> HypEx A/B | cupac_linear_A_plus_B_plus_C -> hypex_ab | ATE | exploratory | cupac_linear_A_plus_B__hypex_ab |
| cupac_quadratic_A__hypex_ab | Квадратичный CUPAC A -> HypEx A/B | cupac_quadratic_A -> hypex_ab | ATE | exploratory | cupac_linear_A__hypex_ab |
| regression_adjustment | Регрессионная корректировка | regression_adjustment | ATE | confirmatory | welch |
| ml_rate | ML-RATE | ml_rate | ATE | exploratory | welch |
| nearest_neighbor_matching | Matching ближайшего соседа | nearest_neighbor_matching | ATE | exploratory |  |
| hypex_matching | Matching HypEx | hypex_matching | ATE | integration_parity |  |
| optimal_transport_matching | Глобальный matching оптимальным транспортом | optimal_transport_matching | ATE | exploratory |  |
| panel_prior_cupac_A__hypex_ab | Panel-prior cupac_linear_A -> HypEx A/B | panel_prior_cupac_A -> hypex_ab | ATE | confirmatory | hypex_ab |
| panel_prior_cupac_A_plus_B__hypex_ab | Panel-prior cupac_linear_A_plus_B -> HypEx A/B | panel_prior_cupac_A_plus_B -> hypex_ab | ATE | exploratory | panel_prior_cupac_A |
| panel_prior_cupac_A_plus_B_plus_C__hypex_ab | Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | panel_prior_cupac_A_plus_B_plus_C -> hypex_ab | ATE | exploratory | panel_prior_cupac_A_plus_B |
| ga_strata__stratified | GA-страты -> стратифицированная оценка | genetic_strata_reducer -> stratified_mean_diff | ATE экспериментальной выборки | confirmatory | welch |
| ga_strata__cuped_control__stratified | GA-страты -> CUPED -> стратифицированная оценка | genetic_strata_reducer -> cuped_control -> stratified_mean_diff | ATE экспериментальной выборки | confirmatory | ga_strata__stratified |
| ga_strata__cupac_linear_A__stratified | GA-страты -> CUPAC A -> стратифицированная оценка | genetic_strata_reducer -> cupac_linear_A -> stratified_mean_diff | ATE экспериментальной выборки | confirmatory | ga_strata__stratified |
| ga_strata__cupac_linear_A_plus_B__stratified | GA-страты -> CUPAC A+B -> стратифицированная оценка | genetic_strata_reducer -> cupac_linear_A_plus_B -> stratified_mean_diff | ATE экспериментальной выборки | exploratory | ga_strata__cupac_linear_A__stratified |
| ga_strata__cupac_linear_A_plus_B_plus_C__stratified | GA-страты -> CUPAC A+B+C -> стратифицированная оценка | genetic_strata_reducer -> cupac_linear_A_plus_B_plus_C -> stratified_mean_diff | ATE экспериментальной выборки | exploratory | ga_strata__cupac_linear_A_plus_B__stratified |
| cartesian__welch__raw | welch + raw | raw -> welch | ATE | exploratory |  |
| cartesian__welch__cuped_control | welch + cuped_control | cuped_control -> welch | ATE | exploratory |  |
| cartesian__welch__cupac_linear_A | welch + cupac_linear_A | cupac_linear_A -> welch | ATE | exploratory |  |
| cartesian__welch__cupac_linear_A_plus_B | welch + cupac_linear_A_plus_B | cupac_linear_A_plus_B -> welch | ATE | exploratory |  |
| cartesian__welch__cupac_linear_A_plus_B_plus_C | welch + cupac_linear_A_plus_B_plus_C | cupac_linear_A_plus_B_plus_C -> welch | ATE | exploratory |  |
| cartesian__welch__cupac_quadratic_A | welch + cupac_quadratic_A | cupac_quadratic_A -> welch | ATE | exploratory |  |
| cartesian__welch__panel_prior_cupac_A | welch + panel_prior_cupac_A | panel_prior_cupac_A -> welch | ATE | exploratory |  |
| cartesian__welch__panel_prior_cupac_A_plus_B | welch + panel_prior_cupac_A_plus_B | panel_prior_cupac_A_plus_B -> welch | ATE | exploratory |  |
| cartesian__welch__panel_prior_cupac_A_plus_B_plus_C | welch + panel_prior_cupac_A_plus_B_plus_C | panel_prior_cupac_A_plus_B_plus_C -> welch | ATE | exploratory |  |
| cartesian__hypex_ab__raw | hypex_ab + raw | raw -> hypex_ab | ATE | exploratory |  |
| cartesian__hypex_ab__cuped_control | hypex_ab + cuped_control | cuped_control -> hypex_ab | ATE | exploratory |  |
| cartesian__hypex_ab__cupac_linear_A | hypex_ab + cupac_linear_A | cupac_linear_A -> hypex_ab | ATE | exploratory |  |
| cartesian__hypex_ab__cupac_linear_A_plus_B | hypex_ab + cupac_linear_A_plus_B | cupac_linear_A_plus_B -> hypex_ab | ATE | exploratory |  |
| cartesian__hypex_ab__cupac_linear_A_plus_B_plus_C | hypex_ab + cupac_linear_A_plus_B_plus_C | cupac_linear_A_plus_B_plus_C -> hypex_ab | ATE | exploratory |  |
| cartesian__hypex_ab__cupac_quadratic_A | hypex_ab + cupac_quadratic_A | cupac_quadratic_A -> hypex_ab | ATE | exploratory |  |
| cartesian__hypex_ab__panel_prior_cupac_A | hypex_ab + panel_prior_cupac_A | panel_prior_cupac_A -> hypex_ab | ATE | exploratory |  |
| cartesian__hypex_ab__panel_prior_cupac_A_plus_B | hypex_ab + panel_prior_cupac_A_plus_B | panel_prior_cupac_A_plus_B -> hypex_ab | ATE | exploratory |  |
| cartesian__hypex_ab__panel_prior_cupac_A_plus_B_plus_C | hypex_ab + panel_prior_cupac_A_plus_B_plus_C | panel_prior_cupac_A_plus_B_plus_C -> hypex_ab | ATE | exploratory |  |
| cartesian__permutation__raw | permutation + raw | raw -> permutation | ATE | exploratory |  |
| cartesian__permutation__cuped_control | permutation + cuped_control | cuped_control -> permutation | ATE | exploratory |  |
| cartesian__permutation__cupac_linear_A | permutation + cupac_linear_A | cupac_linear_A -> permutation | ATE | exploratory |  |
| cartesian__permutation__cupac_linear_A_plus_B | permutation + cupac_linear_A_plus_B | cupac_linear_A_plus_B -> permutation | ATE | exploratory |  |
| cartesian__permutation__cupac_linear_A_plus_B_plus_C | permutation + cupac_linear_A_plus_B_plus_C | cupac_linear_A_plus_B_plus_C -> permutation | ATE | exploratory |  |
| cartesian__permutation__cupac_quadratic_A | permutation + cupac_quadratic_A | cupac_quadratic_A -> permutation | ATE | exploratory |  |
| cartesian__permutation__panel_prior_cupac_A | permutation + panel_prior_cupac_A | panel_prior_cupac_A -> permutation | ATE | exploratory |  |
| cartesian__permutation__panel_prior_cupac_A_plus_B | permutation + panel_prior_cupac_A_plus_B | panel_prior_cupac_A_plus_B -> permutation | ATE | exploratory |  |
| cartesian__permutation__panel_prior_cupac_A_plus_B_plus_C | permutation + panel_prior_cupac_A_plus_B_plus_C | panel_prior_cupac_A_plus_B_plus_C -> permutation | ATE | exploratory |  |
| cartesian__bootstrap__raw | bootstrap + raw | raw -> bootstrap | ATE | exploratory |  |
| cartesian__bootstrap__cuped_control | bootstrap + cuped_control | cuped_control -> bootstrap | ATE | exploratory |  |
| cartesian__bootstrap__cupac_linear_A | bootstrap + cupac_linear_A | cupac_linear_A -> bootstrap | ATE | exploratory |  |
| cartesian__bootstrap__cupac_linear_A_plus_B | bootstrap + cupac_linear_A_plus_B | cupac_linear_A_plus_B -> bootstrap | ATE | exploratory |  |
| cartesian__bootstrap__cupac_linear_A_plus_B_plus_C | bootstrap + cupac_linear_A_plus_B_plus_C | cupac_linear_A_plus_B_plus_C -> bootstrap | ATE | exploratory |  |
| cartesian__bootstrap__cupac_quadratic_A | bootstrap + cupac_quadratic_A | cupac_quadratic_A -> bootstrap | ATE | exploratory |  |
| cartesian__bootstrap__panel_prior_cupac_A | bootstrap + panel_prior_cupac_A | panel_prior_cupac_A -> bootstrap | ATE | exploratory |  |
| cartesian__bootstrap__panel_prior_cupac_A_plus_B | bootstrap + panel_prior_cupac_A_plus_B | panel_prior_cupac_A_plus_B -> bootstrap | ATE | exploratory |  |
| cartesian__bootstrap__panel_prior_cupac_A_plus_B_plus_C | bootstrap + panel_prior_cupac_A_plus_B_plus_C | panel_prior_cupac_A_plus_B_plus_C -> bootstrap | ATE | exploratory |  |
| cartesian__regression_adjustment__raw | regression_adjustment + raw | raw -> regression_adjustment | ATE | exploratory |  |
| cartesian__regression_adjustment__cuped_control | regression_adjustment + cuped_control | cuped_control -> regression_adjustment | ATE | exploratory |  |
| cartesian__regression_adjustment__cupac_linear_A | regression_adjustment + cupac_linear_A | cupac_linear_A -> regression_adjustment | ATE | exploratory |  |
| cartesian__regression_adjustment__cupac_linear_A_plus_B | regression_adjustment + cupac_linear_A_plus_B | cupac_linear_A_plus_B -> regression_adjustment | ATE | exploratory |  |
| cartesian__regression_adjustment__cupac_linear_A_plus_B_plus_C | regression_adjustment + cupac_linear_A_plus_B_plus_C | cupac_linear_A_plus_B_plus_C -> regression_adjustment | ATE | exploratory |  |
| cartesian__regression_adjustment__cupac_quadratic_A | regression_adjustment + cupac_quadratic_A | cupac_quadratic_A -> regression_adjustment | ATE | exploratory |  |
| cartesian__regression_adjustment__panel_prior_cupac_A | regression_adjustment + panel_prior_cupac_A | panel_prior_cupac_A -> regression_adjustment | ATE | exploratory |  |
| cartesian__regression_adjustment__panel_prior_cupac_A_plus_B | regression_adjustment + panel_prior_cupac_A_plus_B | panel_prior_cupac_A_plus_B -> regression_adjustment | ATE | exploratory |  |
| cartesian__regression_adjustment__panel_prior_cupac_A_plus_B_plus_C | regression_adjustment + panel_prior_cupac_A_plus_B_plus_C | panel_prior_cupac_A_plus_B_plus_C -> regression_adjustment | ATE | exploratory |  |
| cartesian__ml_rate__raw | ml_rate + raw | raw -> ml_rate | ATE | exploratory |  |
| cartesian__ml_rate__cuped_control | ml_rate + cuped_control | cuped_control -> ml_rate | ATE | exploratory |  |
| cartesian__ml_rate__cupac_linear_A | ml_rate + cupac_linear_A | cupac_linear_A -> ml_rate | ATE | exploratory |  |
| cartesian__ml_rate__cupac_linear_A_plus_B | ml_rate + cupac_linear_A_plus_B | cupac_linear_A_plus_B -> ml_rate | ATE | exploratory |  |
| cartesian__ml_rate__cupac_linear_A_plus_B_plus_C | ml_rate + cupac_linear_A_plus_B_plus_C | cupac_linear_A_plus_B_plus_C -> ml_rate | ATE | exploratory |  |
| cartesian__ml_rate__cupac_quadratic_A | ml_rate + cupac_quadratic_A | cupac_quadratic_A -> ml_rate | ATE | exploratory |  |
| cartesian__ml_rate__panel_prior_cupac_A | ml_rate + panel_prior_cupac_A | panel_prior_cupac_A -> ml_rate | ATE | exploratory |  |
| cartesian__ml_rate__panel_prior_cupac_A_plus_B | ml_rate + panel_prior_cupac_A_plus_B | panel_prior_cupac_A_plus_B -> ml_rate | ATE | exploratory |  |
| cartesian__ml_rate__panel_prior_cupac_A_plus_B_plus_C | ml_rate + panel_prior_cupac_A_plus_B_plus_C | panel_prior_cupac_A_plus_B_plus_C -> ml_rate | ATE | exploratory |  |
| cartesian__nearest_neighbor_matching__raw | nearest_neighbor_matching + raw | raw -> nearest_neighbor_matching | ATE | exploratory |  |
| cartesian__nearest_neighbor_matching__cuped_control | nearest_neighbor_matching + cuped_control | cuped_control -> nearest_neighbor_matching | ATE | exploratory |  |
| cartesian__nearest_neighbor_matching__cupac_linear_A | nearest_neighbor_matching + cupac_linear_A | cupac_linear_A -> nearest_neighbor_matching | ATE | exploratory |  |
| cartesian__nearest_neighbor_matching__cupac_linear_A_plus_B | nearest_neighbor_matching + cupac_linear_A_plus_B | cupac_linear_A_plus_B -> nearest_neighbor_matching | ATE | exploratory |  |
| cartesian__nearest_neighbor_matching__cupac_linear_A_plus_B_plus_C | nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | cupac_linear_A_plus_B_plus_C -> nearest_neighbor_matching | ATE | exploratory |  |
| cartesian__nearest_neighbor_matching__cupac_quadratic_A | nearest_neighbor_matching + cupac_quadratic_A | cupac_quadratic_A -> nearest_neighbor_matching | ATE | exploratory |  |
| cartesian__nearest_neighbor_matching__panel_prior_cupac_A | nearest_neighbor_matching + panel_prior_cupac_A | panel_prior_cupac_A -> nearest_neighbor_matching | ATE | exploratory |  |
| cartesian__nearest_neighbor_matching__panel_prior_cupac_A_plus_B | nearest_neighbor_matching + panel_prior_cupac_A_plus_B | panel_prior_cupac_A_plus_B -> nearest_neighbor_matching | ATE | exploratory |  |
| cartesian__nearest_neighbor_matching__panel_prior_cupac_A_plus_B_plus_C | nearest_neighbor_matching + panel_prior_cupac_A_plus_B_plus_C | panel_prior_cupac_A_plus_B_plus_C -> nearest_neighbor_matching | ATE | exploratory |  |
| cartesian__optimal_transport_matching__raw | optimal_transport_matching + raw | raw -> optimal_transport_matching | ATE | exploratory |  |
| cartesian__optimal_transport_matching__cuped_control | optimal_transport_matching + cuped_control | cuped_control -> optimal_transport_matching | ATE | exploratory |  |
| cartesian__optimal_transport_matching__cupac_linear_A | optimal_transport_matching + cupac_linear_A | cupac_linear_A -> optimal_transport_matching | ATE | exploratory |  |
| cartesian__optimal_transport_matching__cupac_linear_A_plus_B | optimal_transport_matching + cupac_linear_A_plus_B | cupac_linear_A_plus_B -> optimal_transport_matching | ATE | exploratory |  |
| cartesian__optimal_transport_matching__cupac_linear_A_plus_B_plus_C | optimal_transport_matching + cupac_linear_A_plus_B_plus_C | cupac_linear_A_plus_B_plus_C -> optimal_transport_matching | ATE | exploratory |  |
| cartesian__optimal_transport_matching__cupac_quadratic_A | optimal_transport_matching + cupac_quadratic_A | cupac_quadratic_A -> optimal_transport_matching | ATE | exploratory |  |
| cartesian__optimal_transport_matching__panel_prior_cupac_A | optimal_transport_matching + panel_prior_cupac_A | panel_prior_cupac_A -> optimal_transport_matching | ATE | exploratory |  |
| cartesian__optimal_transport_matching__panel_prior_cupac_A_plus_B | optimal_transport_matching + panel_prior_cupac_A_plus_B | panel_prior_cupac_A_plus_B -> optimal_transport_matching | ATE | exploratory |  |
| cartesian__optimal_transport_matching__panel_prior_cupac_A_plus_B_plus_C | optimal_transport_matching + panel_prior_cupac_A_plus_B_plus_C | panel_prior_cupac_A_plus_B_plus_C -> optimal_transport_matching | ATE | exploratory |  |
| stratified_mean_diff | Стратификация с долями экспериментальной выборки | stratified_mean_diff | Sample ATE | confirmatory | welch |
| population_weighted_stratified | Постстратификация по известной популяции | population_weighted_stratified | Population ATE | confirmatory | stratified_mean_diff |
| calibration_reweighting | Entropy calibration по известным population moments | calibration_reweighting | Population ATE | confirmatory | stratified_mean_diff |

## Линейный сигнал, нормальный шум

Благоприятный базовый сценарий.

Ни один кандидат не прошёл заданный диагностический допуск.

### Цепочка A/B/C

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1436.99 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.0847 |  |  |  |  | 0.0 | 0.0 | 100.0 | 449.73 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 128.29 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 128.79 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 108.56 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 104.81 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.0847 |  |  |  |  | 0.0 | 0.0 | 100.0 | 238.7 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 80.12 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 69.43 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 79.81 |

### Все пайплайны

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Welch t-test | confirmatory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 945.59 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1436.99 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.27 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | -0.1304 |  |  |  |  | 0.0 | 0.0 | 100.0 | 3.81 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 116.7 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 35.86 |
| HypEx CUPED | integration_parity | смещение выше допуска | -0.0725 |  |  |  |  | 0.0 | 0.0 | 100.0 | 200.69 |
| CUPED по контролю -> HypEx A/B | confirmatory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 186.68 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.0847 |  |  |  |  | 0.0 | 0.0 | 100.0 | 449.73 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 128.29 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 128.79 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 108.56 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 103.4 |
| Регрессионная корректировка | confirmatory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 6.51 |
| ML-RATE | exploratory | смещение выше допуска | -0.0813 |  |  |  |  | 0.0 | 0.0 | 100.0 | 34.11 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1481 |  |  |  |  | 0.0 | 0.0 | 0.0 | 15.32 |
| Matching HypEx | integration_parity | смещение выше допуска | -0.0676 |  |  |  |  | 0.0 | 0.0 | 100.0 | 866.24 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1633 |  |  |  |  | 0.0 | 0.0 | 0.0 | 362.21 |
| welch + raw | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.99 |
| welch + cuped_control | exploratory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 3.93 |
| welch + cupac_linear_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 12.36 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 16.58 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 17.33 |
| welch + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 8.94 |
| hypex_ab + raw | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 95.81 |
| hypex_ab + cuped_control | exploratory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 86.86 |
| hypex_ab + cupac_linear_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 86.62 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 93.64 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 93.27 |
| hypex_ab + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 85.61 |
| permutation + raw | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 211.1 |
| permutation + cuped_control | exploratory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 210.91 |
| permutation + cupac_linear_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 188.33 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 198.55 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 218.0 |
| permutation + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 224.28 |
| bootstrap + raw | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 61.54 |
| bootstrap + cuped_control | exploratory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 61.35 |
| bootstrap + cupac_linear_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 69.79 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 71.0 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 73.61 |
| bootstrap + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 58.56 |
| regression_adjustment + raw | exploratory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 5.06 |
| regression_adjustment + cuped_control | exploratory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 7.38 |
| regression_adjustment + cupac_linear_A | exploratory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 14.36 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0822 |  |  |  |  | 100.0 | 0.0 | 0.0 | 16.85 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.09 |  |  |  |  | 100.0 | 0.0 | 0.0 | 22.94 |
| regression_adjustment + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0842 |  |  |  |  | 0.0 | 0.0 | 100.0 | 10.38 |
| ml_rate + raw | exploratory | смещение выше допуска | -0.0813 |  |  |  |  | 0.0 | 0.0 | 100.0 | 24.8 |
| ml_rate + cuped_control | exploratory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 25.56 |
| ml_rate + cupac_linear_A | exploratory | смещение выше допуска | -0.0812 |  |  |  |  | 0.0 | 0.0 | 100.0 | 39.59 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0826 |  |  |  |  | 100.0 | 0.0 | 0.0 | 40.7 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0903 |  |  |  |  | 100.0 | 0.0 | 0.0 | 48.6 |
| ml_rate + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0845 |  |  |  |  | 0.0 | 0.0 | 100.0 | 36.62 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1481 |  |  |  |  | 0.0 | 0.0 | 0.0 | 11.26 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.1459 |  |  |  |  | 0.0 | 0.0 | 0.0 | 13.08 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1468 |  |  |  |  | 0.0 | 0.0 | 0.0 | 23.27 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1585 |  |  |  |  | 0.0 | 0.0 | 0.0 | 26.43 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1576 |  |  |  |  | 0.0 | 0.0 | 0.0 | 30.76 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1497 |  |  |  |  | 0.0 | 0.0 | 0.0 | 15.39 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1633 |  |  |  |  | 0.0 | 0.0 | 0.0 | 447.36 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1486 |  |  |  |  | 0.0 | 0.0 | 0.0 | 457.04 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1499 |  |  |  |  | 0.0 | 0.0 | 0.0 | 439.55 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1518 |  |  |  |  | 0.0 | 0.0 | 0.0 | 385.45 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1468 |  |  |  |  | 0.0 | 0.0 | 0.0 | 386.81 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1503 |  |  |  |  | 0.0 | 0.0 | 0.0 | 370.09 |
| GA-страты -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0788 |  |  |  |  | 0.0 | 0.0 | 100.0 | 153.22 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.074 |  |  |  |  | 0.0 | 0.0 | 100.0 | 154.55 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0834 |  |  |  |  | 0.0 | 0.0 | 100.0 | 161.46 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | -0.084 |  |  |  |  | 100.0 | 0.0 | 0.0 | 168.01 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0914 |  |  |  |  | 100.0 | 0.0 | 0.0 | 174.69 |
| Welch t-test | confirmatory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.15 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 104.81 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.73 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | -0.1304 |  |  |  |  | 0.0 | 0.0 | 100.0 | 2.51 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 81.26 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 30.15 |
| HypEx CUPED | integration_parity | смещение выше допуска | -0.0725 |  |  |  |  | 0.0 | 0.0 | 100.0 | 137.35 |
| CUPED по контролю -> HypEx A/B | confirmatory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 56.46 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.0847 |  |  |  |  | 0.0 | 0.0 | 100.0 | 238.7 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 80.12 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 69.43 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 79.81 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 67.86 |
| Регрессионная корректировка | confirmatory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 4.16 |
| ML-RATE | exploratory | смещение выше допуска | -0.0813 |  |  |  |  | 0.0 | 0.0 | 100.0 | 19.6 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1481 |  |  |  |  | 0.0 | 0.0 | 0.0 | 10.0 |
| Matching HypEx | integration_parity | смещение выше допуска | -0.0676 |  |  |  |  | 0.0 | 0.0 | 100.0 | 661.73 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1633 |  |  |  |  | 0.0 | 0.0 | 0.0 | 328.9 |
| Panel-prior cupac_linear_A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 82.28 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 84.76 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 88.38 |
| welch + raw | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.02 |
| welch + cuped_control | exploratory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 2.86 |
| welch + cupac_linear_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 9.67 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 11.23 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 19.35 |
| welch + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 5.61 |
| welch + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 12.24 |
| welch + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 13.33 |
| welch + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 22.52 |
| hypex_ab + raw | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 67.13 |
| hypex_ab + cuped_control | exploratory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 67.51 |
| hypex_ab + cupac_linear_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 75.64 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 88.01 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 81.7 |
| hypex_ab + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 102.39 |
| hypex_ab + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 87.59 |
| hypex_ab + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 83.04 |
| hypex_ab + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 83.25 |
| permutation + raw | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 167.16 |
| permutation + cuped_control | exploratory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 175.54 |
| permutation + cupac_linear_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 163.56 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 151.69 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 185.63 |
| permutation + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 161.92 |
| permutation + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 151.26 |
| permutation + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 179.23 |
| permutation + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 198.63 |
| bootstrap + raw | exploratory | смещение выше допуска | -0.0335 |  |  |  |  | 0.0 | 0.0 | 100.0 | 54.71 |
| bootstrap + cuped_control | exploratory | смещение выше допуска | -0.0722 |  |  |  |  | 0.0 | 0.0 | 100.0 | 54.08 |
| bootstrap + cupac_linear_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 56.88 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 63.86 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 68.08 |
| bootstrap + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0715 |  |  |  |  | 0.0 | 0.0 | 100.0 | 56.15 |
| bootstrap + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0681 |  |  |  |  | 0.0 | 0.0 | 100.0 | 64.58 |
| bootstrap + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0687 |  |  |  |  | 0.0 | 0.0 | 100.0 | 72.23 |
| bootstrap + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0806 |  |  |  |  | 0.0 | 0.0 | 100.0 | 74.68 |
| regression_adjustment + raw | exploratory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 5.31 |
| regression_adjustment + cuped_control | exploratory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 6.44 |
| regression_adjustment + cupac_linear_A | exploratory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 12.56 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0822 |  |  |  |  | 100.0 | 0.0 | 0.0 | 13.69 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.09 |  |  |  |  | 100.0 | 0.0 | 0.0 | 21.3 |
| regression_adjustment + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0842 |  |  |  |  | 0.0 | 0.0 | 100.0 | 12.71 |
| regression_adjustment + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 21.34 |
| regression_adjustment + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0822 |  |  |  |  | 100.0 | 0.0 | 0.0 | 19.58 |
| regression_adjustment + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.09 |  |  |  |  | 100.0 | 0.0 | 0.0 | 23.99 |
| ml_rate + raw | exploratory | смещение выше допуска | -0.0813 |  |  |  |  | 0.0 | 0.0 | 100.0 | 19.32 |
| ml_rate + cuped_control | exploratory | смещение выше допуска | -0.0811 |  |  |  |  | 0.0 | 0.0 | 100.0 | 19.17 |
| ml_rate + cupac_linear_A | exploratory | смещение выше допуска | -0.0812 |  |  |  |  | 0.0 | 0.0 | 100.0 | 25.7 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0826 |  |  |  |  | 100.0 | 0.0 | 0.0 | 25.49 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0903 |  |  |  |  | 100.0 | 0.0 | 0.0 | 33.15 |
| ml_rate + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0845 |  |  |  |  | 0.0 | 0.0 | 100.0 | 24.12 |
| ml_rate + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0812 |  |  |  |  | 0.0 | 0.0 | 100.0 | 32.32 |
| ml_rate + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0826 |  |  |  |  | 100.0 | 0.0 | 0.0 | 35.65 |
| ml_rate + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0903 |  |  |  |  | 100.0 | 0.0 | 0.0 | 60.54 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1481 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.22 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.1459 |  |  |  |  | 0.0 | 0.0 | 0.0 | 10.31 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1468 |  |  |  |  | 0.0 | 0.0 | 0.0 | 17.41 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1585 |  |  |  |  | 0.0 | 0.0 | 0.0 | 41.07 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1576 |  |  |  |  | 0.0 | 0.0 | 0.0 | 25.17 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1497 |  |  |  |  | 0.0 | 0.0 | 0.0 | 24.43 |
| nearest_neighbor_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1468 |  |  |  |  | 0.0 | 0.0 | 0.0 | 23.39 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1585 |  |  |  |  | 0.0 | 0.0 | 0.0 | 36.61 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1576 |  |  |  |  | 0.0 | 0.0 | 0.0 | 37.97 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1633 |  |  |  |  | 0.0 | 0.0 | 0.0 | 466.69 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1486 |  |  |  |  | 0.0 | 0.0 | 0.0 | 456.87 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1499 |  |  |  |  | 0.0 | 0.0 | 0.0 | 408.9 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1518 |  |  |  |  | 0.0 | 0.0 | 0.0 | 476.17 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1468 |  |  |  |  | 0.0 | 0.0 | 0.0 | 437.07 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1503 |  |  |  |  | 0.0 | 0.0 | 0.0 | 417.94 |
| optimal_transport_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1499 |  |  |  |  | 0.0 | 0.0 | 0.0 | 433.36 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1518 |  |  |  |  | 0.0 | 0.0 | 0.0 | 509.52 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1468 |  |  |  |  | 0.0 | 0.0 | 0.0 | 488.36 |
| GA-страты -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0788 |  |  |  |  | 0.0 | 0.0 | 100.0 | 100.91 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.074 |  |  |  |  | 0.0 | 0.0 | 100.0 | 105.86 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0834 |  |  |  |  | 0.0 | 0.0 | 100.0 | 109.73 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | -0.084 |  |  |  |  | 100.0 | 0.0 | 0.0 | 115.1 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0914 |  |  |  |  | 100.0 | 0.0 | 0.0 | 119.23 |

## Линейный сигнал, тяжёлые хвосты

Проверка устойчивости SE и выигрыша по дисперсии.

Ни один кандидат не прошёл заданный диагностический допуск.

### Цепочка A/B/C

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 95.29 |
| HypEx temporal CUPAC | integration_parity | SE/SD вне допуска | 0.0054 |  |  |  |  | 0.0 | 100.0 | 100.0 | 306.91 |
| CUPAC A -> HypEx A/B | confirmatory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 106.05 |
| CUPAC A+B -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 97.59 |
| CUPAC A+B+C -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 103.25 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 59.24 |
| HypEx temporal CUPAC | integration_parity | SE/SD вне допуска | 0.0054 |  |  |  |  | 0.0 | 100.0 | 100.0 | 250.1 |
| CUPAC A -> HypEx A/B | confirmatory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 69.83 |
| CUPAC A+B -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 91.19 |
| CUPAC A+B+C -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 84.58 |

### Все пайплайны

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Welch t-test | confirmatory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 1.12 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 95.29 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 0.74 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | 0.0817 |  |  |  |  | 0.0 | 100.0 | 100.0 | 3.47 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 123.99 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 36.28 |
| HypEx CUPED | integration_parity | SE/SD вне допуска | 0.0184 |  |  |  |  | 0.0 | 100.0 | 100.0 | 172.15 |
| CUPED по контролю -> HypEx A/B | confirmatory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 89.02 |
| HypEx temporal CUPAC | integration_parity | SE/SD вне допуска | 0.0054 |  |  |  |  | 0.0 | 100.0 | 100.0 | 306.91 |
| CUPAC A -> HypEx A/B | confirmatory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 106.05 |
| CUPAC A+B -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 97.59 |
| CUPAC A+B+C -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 103.25 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 87.23 |
| Регрессионная корректировка | confirmatory | SE/SD вне допуска | -0.0014 |  |  |  |  | 0.0 | 100.0 | 100.0 | 4.36 |
| ML-RATE | exploratory | SE/SD вне допуска | -0.0032 |  |  |  |  | 0.0 | 100.0 | 100.0 | 32.84 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1448 |  |  |  |  | 0.0 | 0.0 | 0.0 | 13.08 |
| Matching HypEx | integration_parity | смещение выше допуска | -0.0262 |  |  |  |  | 0.0 | 100.0 | 100.0 | 770.89 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1259 |  |  |  |  | 0.0 | 0.0 | 0.0 | 305.71 |
| welch + raw | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 0.82 |
| welch + cuped_control | exploratory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 4.97 |
| welch + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 10.91 |
| welch + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 17.15 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 19.99 |
| welch + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 8.1 |
| hypex_ab + raw | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 92.47 |
| hypex_ab + cuped_control | exploratory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 97.81 |
| hypex_ab + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 113.45 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 105.85 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 116.7 |
| hypex_ab + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 96.04 |
| permutation + raw | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 183.56 |
| permutation + cuped_control | exploratory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 212.98 |
| permutation + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 219.13 |
| permutation + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 266.27 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 222.67 |
| permutation + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 156.74 |
| bootstrap + raw | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 40.97 |
| bootstrap + cuped_control | exploratory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 46.77 |
| bootstrap + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 81.26 |
| bootstrap + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 57.84 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 72.16 |
| bootstrap + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 50.68 |
| regression_adjustment + raw | exploratory | SE/SD вне допуска | -0.0014 |  |  |  |  | 0.0 | 100.0 | 100.0 | 3.82 |
| regression_adjustment + cuped_control | exploratory | SE/SD вне допуска | -0.0014 |  |  |  |  | 0.0 | 100.0 | 100.0 | 6.91 |
| regression_adjustment + cupac_linear_A | exploratory | SE/SD вне допуска | -0.0014 |  |  |  |  | 0.0 | 100.0 | 100.0 | 12.78 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | -0.0066 |  |  |  |  | 0.0 | 100.0 | 100.0 | 22.19 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | -0.0093 |  |  |  |  | 0.0 | 100.0 | 100.0 | 21.65 |
| regression_adjustment + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0 |  |  |  |  | 0.0 | 100.0 | 100.0 | 11.52 |
| ml_rate + raw | exploratory | SE/SD вне допуска | -0.0032 |  |  |  |  | 0.0 | 100.0 | 100.0 | 26.45 |
| ml_rate + cuped_control | exploratory | SE/SD вне допуска | -0.0025 |  |  |  |  | 0.0 | 100.0 | 100.0 | 25.84 |
| ml_rate + cupac_linear_A | exploratory | SE/SD вне допуска | -0.003 |  |  |  |  | 0.0 | 100.0 | 100.0 | 39.19 |
| ml_rate + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | -0.008 |  |  |  |  | 0.0 | 100.0 | 100.0 | 25.8 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | -0.0104 |  |  |  |  | 0.0 | 100.0 | 100.0 | 33.74 |
| ml_rate + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 23.88 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1448 |  |  |  |  | 0.0 | 0.0 | 0.0 | 8.46 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.1481 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.9 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1477 |  |  |  |  | 0.0 | 0.0 | 0.0 | 11.14 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1567 |  |  |  |  | 0.0 | 0.0 | 0.0 | 20.1 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1626 |  |  |  |  | 0.0 | 0.0 | 0.0 | 23.98 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1469 |  |  |  |  | 0.0 | 0.0 | 0.0 | 15.08 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1259 |  |  |  |  | 0.0 | 0.0 | 0.0 | 238.31 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1317 |  |  |  |  | 0.0 | 0.0 | 0.0 | 220.81 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1336 |  |  |  |  | 0.0 | 0.0 | 0.0 | 261.84 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.14 |  |  |  |  | 0.0 | 0.0 | 0.0 | 264.93 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1497 |  |  |  |  | 0.0 | 0.0 | 0.0 | 256.22 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1326 |  |  |  |  | 0.0 | 0.0 | 0.0 | 260.09 |
| GA-страты -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0055 |  |  |  |  | 0.0 | 100.0 | 100.0 | 187.44 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0086 |  |  |  |  | 0.0 | 100.0 | 100.0 | 188.85 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0034 |  |  |  |  | 0.0 | 100.0 | 100.0 | 195.08 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | SE/SD вне допуска | -0.0004 |  |  |  |  | 0.0 | 100.0 | 100.0 | 197.88 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | SE/SD вне допуска | -0.0023 |  |  |  |  | 0.0 | 100.0 | 100.0 | 200.31 |
| Welch t-test | confirmatory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 1.04 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 59.24 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 0.75 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | 0.0817 |  |  |  |  | 0.0 | 100.0 | 100.0 | 2.19 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 76.96 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 32.46 |
| HypEx CUPED | integration_parity | SE/SD вне допуска | 0.0184 |  |  |  |  | 0.0 | 100.0 | 100.0 | 126.63 |
| CUPED по контролю -> HypEx A/B | confirmatory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 73.91 |
| HypEx temporal CUPAC | integration_parity | SE/SD вне допуска | 0.0054 |  |  |  |  | 0.0 | 100.0 | 100.0 | 250.1 |
| CUPAC A -> HypEx A/B | confirmatory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 69.83 |
| CUPAC A+B -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 91.19 |
| CUPAC A+B+C -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 84.58 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 79.27 |
| Регрессионная корректировка | confirmatory | SE/SD вне допуска | -0.0014 |  |  |  |  | 0.0 | 100.0 | 100.0 | 4.58 |
| ML-RATE | exploratory | SE/SD вне допуска | -0.0032 |  |  |  |  | 0.0 | 100.0 | 100.0 | 19.12 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1448 |  |  |  |  | 0.0 | 0.0 | 0.0 | 7.78 |
| Matching HypEx | integration_parity | смещение выше допуска | -0.0262 |  |  |  |  | 0.0 | 100.0 | 100.0 | 597.38 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1259 |  |  |  |  | 0.0 | 0.0 | 0.0 | 222.09 |
| Panel-prior cupac_linear_A -> HypEx A/B | confirmatory | SE/SD вне допуска | 0.0093 |  |  |  |  | 0.0 | 100.0 | 100.0 | 74.69 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0037 |  |  |  |  | 0.0 | 100.0 | 100.0 | 77.82 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | exploratory | SE/SD вне допуска | 0.0002 |  |  |  |  | 0.0 | 100.0 | 100.0 | 86.45 |
| welch + raw | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 1.29 |
| welch + cuped_control | exploratory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 2.96 |
| welch + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 6.34 |
| welch + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 15.22 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 17.2 |
| welch + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 5.64 |
| welch + panel_prior_cupac_A | exploratory | SE/SD вне допуска | 0.0093 |  |  |  |  | 0.0 | 100.0 | 100.0 | 13.98 |
| welch + panel_prior_cupac_A_plus_B | exploratory | SE/SD вне допуска | 0.0037 |  |  |  |  | 0.0 | 100.0 | 100.0 | 14.89 |
| welch + panel_prior_cupac_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0002 |  |  |  |  | 0.0 | 100.0 | 100.0 | 23.8 |
| hypex_ab + raw | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 66.82 |
| hypex_ab + cuped_control | exploratory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 65.44 |
| hypex_ab + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 89.23 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 76.6 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 85.59 |
| hypex_ab + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 72.98 |
| hypex_ab + panel_prior_cupac_A | exploratory | SE/SD вне допуска | 0.0093 |  |  |  |  | 0.0 | 100.0 | 100.0 | 90.4 |
| hypex_ab + panel_prior_cupac_A_plus_B | exploratory | SE/SD вне допуска | 0.0037 |  |  |  |  | 0.0 | 100.0 | 100.0 | 98.63 |
| hypex_ab + panel_prior_cupac_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0002 |  |  |  |  | 0.0 | 100.0 | 100.0 | 108.18 |
| permutation + raw | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 179.34 |
| permutation + cuped_control | exploratory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 157.1 |
| permutation + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 192.39 |
| permutation + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 206.67 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 220.27 |
| permutation + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 209.4 |
| permutation + panel_prior_cupac_A | exploratory | SE/SD вне допуска | 0.0093 |  |  |  |  | 0.0 | 100.0 | 100.0 | 302.66 |
| permutation + panel_prior_cupac_A_plus_B | exploratory | SE/SD вне допуска | 0.0037 |  |  |  |  | 0.0 | 100.0 | 100.0 | 251.09 |
| permutation + panel_prior_cupac_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0002 |  |  |  |  | 0.0 | 100.0 | 100.0 | 275.91 |
| bootstrap + raw | exploratory | смещение выше допуска | 0.0256 |  |  |  |  | 0.0 | 100.0 | 100.0 | 81.58 |
| bootstrap + cuped_control | exploratory | SE/SD вне допуска | 0.0186 |  |  |  |  | 0.0 | 100.0 | 100.0 | 74.59 |
| bootstrap + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0111 |  |  |  |  | 0.0 | 100.0 | 100.0 | 72.09 |
| bootstrap + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | 0.0056 |  |  |  |  | 0.0 | 100.0 | 100.0 | 62.05 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0021 |  |  |  |  | 0.0 | 100.0 | 100.0 | 80.85 |
| bootstrap + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0126 |  |  |  |  | 0.0 | 100.0 | 100.0 | 72.43 |
| bootstrap + panel_prior_cupac_A | exploratory | SE/SD вне допуска | 0.0093 |  |  |  |  | 0.0 | 100.0 | 100.0 | 86.1 |
| bootstrap + panel_prior_cupac_A_plus_B | exploratory | SE/SD вне допуска | 0.0037 |  |  |  |  | 0.0 | 100.0 | 100.0 | 87.59 |
| bootstrap + panel_prior_cupac_A_plus_B_plus_C | exploratory | SE/SD вне допуска | 0.0002 |  |  |  |  | 0.0 | 100.0 | 100.0 | 87.87 |
| regression_adjustment + raw | exploratory | SE/SD вне допуска | -0.0014 |  |  |  |  | 0.0 | 100.0 | 100.0 | 5.46 |
| regression_adjustment + cuped_control | exploratory | SE/SD вне допуска | -0.0014 |  |  |  |  | 0.0 | 100.0 | 100.0 | 6.89 |
| regression_adjustment + cupac_linear_A | exploratory | SE/SD вне допуска | -0.0014 |  |  |  |  | 0.0 | 100.0 | 100.0 | 19.77 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | -0.0066 |  |  |  |  | 0.0 | 100.0 | 100.0 | 34.05 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | -0.0093 |  |  |  |  | 0.0 | 100.0 | 100.0 | 43.7 |
| regression_adjustment + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0 |  |  |  |  | 0.0 | 100.0 | 100.0 | 15.79 |
| regression_adjustment + panel_prior_cupac_A | exploratory | SE/SD вне допуска | -0.0014 |  |  |  |  | 0.0 | 100.0 | 100.0 | 51.81 |
| regression_adjustment + panel_prior_cupac_A_plus_B | exploratory | SE/SD вне допуска | -0.0066 |  |  |  |  | 0.0 | 100.0 | 100.0 | 46.79 |
| regression_adjustment + panel_prior_cupac_A_plus_B_plus_C | exploratory | SE/SD вне допуска | -0.0094 |  |  |  |  | 0.0 | 100.0 | 100.0 | 55.49 |
| ml_rate + raw | exploratory | SE/SD вне допуска | -0.0032 |  |  |  |  | 0.0 | 100.0 | 100.0 | 31.78 |
| ml_rate + cuped_control | exploratory | SE/SD вне допуска | -0.0025 |  |  |  |  | 0.0 | 100.0 | 100.0 | 54.91 |
| ml_rate + cupac_linear_A | exploratory | SE/SD вне допуска | -0.003 |  |  |  |  | 0.0 | 100.0 | 100.0 | 52.33 |
| ml_rate + cupac_linear_A_plus_B | exploratory | SE/SD вне допуска | -0.008 |  |  |  |  | 0.0 | 100.0 | 100.0 | 40.34 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | SE/SD вне допуска | -0.0104 |  |  |  |  | 0.0 | 100.0 | 100.0 | 46.33 |
| ml_rate + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 26.68 |
| ml_rate + panel_prior_cupac_A | exploratory | SE/SD вне допуска | -0.003 |  |  |  |  | 0.0 | 100.0 | 100.0 | 47.95 |
| ml_rate + panel_prior_cupac_A_plus_B | exploratory | SE/SD вне допуска | -0.008 |  |  |  |  | 0.0 | 100.0 | 100.0 | 59.06 |
| ml_rate + panel_prior_cupac_A_plus_B_plus_C | exploratory | SE/SD вне допуска | -0.0105 |  |  |  |  | 0.0 | 100.0 | 100.0 | 54.44 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1448 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.55 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.1481 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.62 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1477 |  |  |  |  | 0.0 | 0.0 | 0.0 | 19.47 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1567 |  |  |  |  | 0.0 | 0.0 | 0.0 | 31.13 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1626 |  |  |  |  | 0.0 | 0.0 | 0.0 | 26.99 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1469 |  |  |  |  | 0.0 | 0.0 | 0.0 | 17.16 |
| nearest_neighbor_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1478 |  |  |  |  | 0.0 | 0.0 | 0.0 | 23.18 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1567 |  |  |  |  | 0.0 | 0.0 | 0.0 | 32.21 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1628 |  |  |  |  | 0.0 | 0.0 | 0.0 | 34.26 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1259 |  |  |  |  | 0.0 | 0.0 | 0.0 | 351.74 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1317 |  |  |  |  | 0.0 | 0.0 | 0.0 | 350.45 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1336 |  |  |  |  | 0.0 | 0.0 | 0.0 | 320.76 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.14 |  |  |  |  | 0.0 | 0.0 | 0.0 | 323.7 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1497 |  |  |  |  | 0.0 | 0.0 | 0.0 | 367.52 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1326 |  |  |  |  | 0.0 | 0.0 | 0.0 | 327.64 |
| optimal_transport_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1342 |  |  |  |  | 0.0 | 0.0 | 0.0 | 330.18 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1406 |  |  |  |  | 0.0 | 0.0 | 0.0 | 344.37 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1505 |  |  |  |  | 0.0 | 0.0 | 0.0 | 274.52 |
| GA-страты -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0055 |  |  |  |  | 0.0 | 100.0 | 100.0 | 63.46 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0086 |  |  |  |  | 0.0 | 100.0 | 100.0 | 64.3 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0034 |  |  |  |  | 0.0 | 100.0 | 100.0 | 77.27 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | SE/SD вне допуска | -0.0004 |  |  |  |  | 0.0 | 100.0 | 100.0 | 79.41 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | SE/SD вне допуска | -0.0023 |  |  |  |  | 0.0 | 100.0 | 100.0 | 90.79 |

## Нелинейный сигнал, нормальный шум

Проверка линейных и квадратичных корректировок.

Ни один кандидат не прошёл заданный диагностический допуск.

### Цепочка A/B/C

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 90.15 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.0305 |  |  |  |  | 0.0 | 100.0 | 100.0 | 262.29 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 85.14 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 72.53 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 91.02 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 42.33 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.0305 |  |  |  |  | 0.0 | 100.0 | 100.0 | 155.92 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 51.18 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 67.71 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 69.91 |

### Все пайплайны

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Welch t-test | confirmatory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.47 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 90.15 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.93 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | -0.0815 |  |  |  |  | 0.0 | 0.0 | 100.0 | 2.78 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 87.92 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 34.56 |
| HypEx CUPED | integration_parity | смещение выше допуска | -0.0355 |  |  |  |  | 0.0 | 0.0 | 100.0 | 160.43 |
| CUPED по контролю -> HypEx A/B | confirmatory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 61.32 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.0305 |  |  |  |  | 0.0 | 100.0 | 100.0 | 262.29 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 85.14 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 72.53 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 91.02 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 63.6 |
| Регрессионная корректировка | confirmatory | смещение выше допуска | -0.0304 |  |  |  |  | 0.0 | 100.0 | 100.0 | 3.22 |
| ML-RATE | exploratory | смещение выше допуска | -0.032 |  |  |  |  | 0.0 | 100.0 | 100.0 | 13.71 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1253 |  |  |  |  | 0.0 | 0.0 | 0.0 | 7.12 |
| Matching HypEx | integration_parity | SE/SD вне допуска | -0.0126 |  |  |  |  | 0.0 | 0.0 | 100.0 | 714.34 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1747 |  |  |  |  | 0.0 | 0.0 | 0.0 | 366.16 |
| welch + raw | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.89 |
| welch + cuped_control | exploratory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 2.09 |
| welch + cupac_linear_A | exploratory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 11.85 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 17.23 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 18.15 |
| welch + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 5.39 |
| hypex_ab + raw | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 70.74 |
| hypex_ab + cuped_control | exploratory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 64.53 |
| hypex_ab + cupac_linear_A | exploratory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 76.69 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 74.26 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 84.58 |
| hypex_ab + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 71.8 |
| permutation + raw | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 163.99 |
| permutation + cuped_control | exploratory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 146.43 |
| permutation + cupac_linear_A | exploratory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 169.99 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 185.68 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 164.27 |
| permutation + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 161.23 |
| bootstrap + raw | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 55.76 |
| bootstrap + cuped_control | exploratory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 49.2 |
| bootstrap + cupac_linear_A | exploratory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 66.91 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 74.28 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 71.46 |
| bootstrap + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 68.15 |
| regression_adjustment + raw | exploratory | смещение выше допуска | -0.0304 |  |  |  |  | 0.0 | 100.0 | 100.0 | 6.1 |
| regression_adjustment + cuped_control | exploratory | смещение выше допуска | -0.0304 |  |  |  |  | 0.0 | 100.0 | 100.0 | 7.3 |
| regression_adjustment + cupac_linear_A | exploratory | смещение выше допуска | -0.0304 |  |  |  |  | 0.0 | 100.0 | 100.0 | 17.58 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0372 |  |  |  |  | 0.0 | 100.0 | 100.0 | 16.3 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0362 |  |  |  |  | 0.0 | 0.0 | 100.0 | 19.09 |
| regression_adjustment + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0036 |  |  |  |  | 0.0 | 100.0 | 100.0 | 6.88 |
| ml_rate + raw | exploratory | смещение выше допуска | -0.032 |  |  |  |  | 0.0 | 100.0 | 100.0 | 16.78 |
| ml_rate + cuped_control | exploratory | смещение выше допуска | -0.0324 |  |  |  |  | 0.0 | 100.0 | 100.0 | 23.56 |
| ml_rate + cupac_linear_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 35.34 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.039 |  |  |  |  | 0.0 | 0.0 | 100.0 | 35.05 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0379 |  |  |  |  | 0.0 | 0.0 | 100.0 | 33.97 |
| ml_rate + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0038 |  |  |  |  | 0.0 | 100.0 | 100.0 | 20.23 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1253 |  |  |  |  | 0.0 | 0.0 | 0.0 | 10.98 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.1284 |  |  |  |  | 0.0 | 0.0 | 0.0 | 10.5 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1273 |  |  |  |  | 0.0 | 0.0 | 0.0 | 23.72 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1212 |  |  |  |  | 0.0 | 0.0 | 0.0 | 22.99 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.109 |  |  |  |  | 0.0 | 0.0 | 0.0 | 22.22 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1441 |  |  |  |  | 0.0 | 0.0 | 0.0 | 13.02 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1747 |  |  |  |  | 0.0 | 0.0 | 0.0 | 402.64 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1334 |  |  |  |  | 0.0 | 0.0 | 0.0 | 396.78 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1519 |  |  |  |  | 0.0 | 0.0 | 0.0 | 379.75 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1543 |  |  |  |  | 0.0 | 0.0 | 0.0 | 428.68 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1467 |  |  |  |  | 0.0 | 0.0 | 0.0 | 448.11 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1475 |  |  |  |  | 0.0 | 0.0 | 0.0 | 407.45 |
| GA-страты -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0608 |  |  |  |  | 0.0 | 0.0 | 100.0 | 131.87 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0269 |  |  |  |  | 0.0 | 100.0 | 100.0 | 132.58 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0401 |  |  |  |  | 0.0 | 0.0 | 100.0 | 137.13 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0468 |  |  |  |  | 0.0 | 0.0 | 100.0 | 142.84 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0401 |  |  |  |  | 0.0 | 0.0 | 100.0 | 154.24 |
| Welch t-test | confirmatory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.46 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 42.33 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.92 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | -0.0815 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.15 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 56.64 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 18.75 |
| HypEx CUPED | integration_parity | смещение выше допуска | -0.0355 |  |  |  |  | 0.0 | 0.0 | 100.0 | 94.77 |
| CUPED по контролю -> HypEx A/B | confirmatory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 45.49 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.0305 |  |  |  |  | 0.0 | 100.0 | 100.0 | 155.92 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 51.18 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 67.71 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 69.91 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 68.06 |
| Регрессионная корректировка | confirmatory | смещение выше допуска | -0.0304 |  |  |  |  | 0.0 | 100.0 | 100.0 | 4.32 |
| ML-RATE | exploratory | смещение выше допуска | -0.032 |  |  |  |  | 0.0 | 100.0 | 100.0 | 16.31 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1253 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.59 |
| Matching HypEx | integration_parity | SE/SD вне допуска | -0.0126 |  |  |  |  | 0.0 | 0.0 | 100.0 | 465.57 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1747 |  |  |  |  | 0.0 | 0.0 | 0.0 | 301.88 |
| Panel-prior cupac_linear_A -> HypEx A/B | confirmatory | смещение выше допуска | -0.0735 |  |  |  |  | 0.0 | 0.0 | 100.0 | 77.09 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | exploratory | смещение выше допуска | -0.0797 |  |  |  |  | 0.0 | 0.0 | 100.0 | 57.33 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | exploratory | смещение выше допуска | -0.0649 |  |  |  |  | 0.0 | 0.0 | 100.0 | 83.0 |
| welch + raw | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.68 |
| welch + cuped_control | exploratory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 2.0 |
| welch + cupac_linear_A | exploratory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 6.96 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 8.63 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 9.38 |
| welch + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 3.77 |
| welch + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0735 |  |  |  |  | 0.0 | 0.0 | 100.0 | 8.54 |
| welch + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0797 |  |  |  |  | 0.0 | 0.0 | 100.0 | 15.82 |
| welch + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0649 |  |  |  |  | 0.0 | 0.0 | 100.0 | 18.46 |
| hypex_ab + raw | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 65.71 |
| hypex_ab + cuped_control | exploratory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 65.61 |
| hypex_ab + cupac_linear_A | exploratory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 75.25 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 60.1 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 89.49 |
| hypex_ab + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 71.91 |
| hypex_ab + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0735 |  |  |  |  | 0.0 | 0.0 | 100.0 | 67.72 |
| hypex_ab + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0797 |  |  |  |  | 0.0 | 0.0 | 100.0 | 80.32 |
| hypex_ab + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0649 |  |  |  |  | 0.0 | 0.0 | 100.0 | 69.17 |
| permutation + raw | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 149.35 |
| permutation + cuped_control | exploratory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 161.01 |
| permutation + cupac_linear_A | exploratory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 187.95 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 148.16 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 154.6 |
| permutation + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 152.69 |
| permutation + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0735 |  |  |  |  | 0.0 | 0.0 | 100.0 | 177.33 |
| permutation + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0797 |  |  |  |  | 0.0 | 0.0 | 100.0 | 168.99 |
| permutation + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0649 |  |  |  |  | 0.0 | 0.0 | 100.0 | 141.2 |
| bootstrap + raw | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 0.0 | 0.0 | 100.0 | 42.06 |
| bootstrap + cuped_control | exploratory | смещение выше допуска | -0.0364 |  |  |  |  | 0.0 | 0.0 | 100.0 | 44.28 |
| bootstrap + cupac_linear_A | exploratory | смещение выше допуска | -0.0739 |  |  |  |  | 0.0 | 0.0 | 100.0 | 77.6 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.08 |  |  |  |  | 0.0 | 0.0 | 100.0 | 70.12 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0651 |  |  |  |  | 0.0 | 0.0 | 100.0 | 56.66 |
| bootstrap + cupac_quadratic_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 63.51 |
| bootstrap + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0735 |  |  |  |  | 0.0 | 0.0 | 100.0 | 58.84 |
| bootstrap + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0797 |  |  |  |  | 0.0 | 0.0 | 100.0 | 61.2 |
| bootstrap + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0649 |  |  |  |  | 0.0 | 0.0 | 100.0 | 63.49 |
| regression_adjustment + raw | exploratory | смещение выше допуска | -0.0304 |  |  |  |  | 0.0 | 100.0 | 100.0 | 3.96 |
| regression_adjustment + cuped_control | exploratory | смещение выше допуска | -0.0304 |  |  |  |  | 0.0 | 100.0 | 100.0 | 3.18 |
| regression_adjustment + cupac_linear_A | exploratory | смещение выше допуска | -0.0304 |  |  |  |  | 0.0 | 100.0 | 100.0 | 9.18 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0372 |  |  |  |  | 0.0 | 100.0 | 100.0 | 11.06 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0362 |  |  |  |  | 0.0 | 0.0 | 100.0 | 12.73 |
| regression_adjustment + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0036 |  |  |  |  | 0.0 | 100.0 | 100.0 | 7.15 |
| regression_adjustment + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0304 |  |  |  |  | 0.0 | 100.0 | 100.0 | 19.89 |
| regression_adjustment + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0372 |  |  |  |  | 0.0 | 100.0 | 100.0 | 14.06 |
| regression_adjustment + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0362 |  |  |  |  | 0.0 | 0.0 | 100.0 | 24.05 |
| ml_rate + raw | exploratory | смещение выше допуска | -0.032 |  |  |  |  | 0.0 | 100.0 | 100.0 | 17.23 |
| ml_rate + cuped_control | exploratory | смещение выше допуска | -0.0324 |  |  |  |  | 0.0 | 100.0 | 100.0 | 14.69 |
| ml_rate + cupac_linear_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 30.64 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.039 |  |  |  |  | 0.0 | 0.0 | 100.0 | 28.36 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0379 |  |  |  |  | 0.0 | 0.0 | 100.0 | 36.12 |
| ml_rate + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0038 |  |  |  |  | 0.0 | 100.0 | 100.0 | 25.63 |
| ml_rate + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.0322 |  |  |  |  | 0.0 | 100.0 | 100.0 | 35.43 |
| ml_rate + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.039 |  |  |  |  | 0.0 | 0.0 | 100.0 | 33.55 |
| ml_rate + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0379 |  |  |  |  | 0.0 | 0.0 | 100.0 | 30.35 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1253 |  |  |  |  | 0.0 | 0.0 | 0.0 | 12.66 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.1284 |  |  |  |  | 0.0 | 0.0 | 0.0 | 10.34 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1273 |  |  |  |  | 0.0 | 0.0 | 0.0 | 15.6 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1212 |  |  |  |  | 0.0 | 0.0 | 0.0 | 28.65 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.109 |  |  |  |  | 0.0 | 0.0 | 0.0 | 27.51 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1441 |  |  |  |  | 0.0 | 0.0 | 0.0 | 13.78 |
| nearest_neighbor_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1273 |  |  |  |  | 0.0 | 0.0 | 0.0 | 21.5 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1213 |  |  |  |  | 0.0 | 0.0 | 0.0 | 20.62 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1091 |  |  |  |  | 0.0 | 0.0 | 0.0 | 20.57 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1747 |  |  |  |  | 0.0 | 0.0 | 0.0 | 343.96 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1334 |  |  |  |  | 0.0 | 0.0 | 0.0 | 359.46 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1519 |  |  |  |  | 0.0 | 0.0 | 0.0 | 356.93 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1543 |  |  |  |  | 0.0 | 0.0 | 0.0 | 358.3 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1467 |  |  |  |  | 0.0 | 0.0 | 0.0 | 383.89 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1475 |  |  |  |  | 0.0 | 0.0 | 0.0 | 378.65 |
| optimal_transport_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1518 |  |  |  |  | 0.0 | 0.0 | 0.0 | 351.23 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1541 |  |  |  |  | 0.0 | 0.0 | 0.0 | 424.5 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1466 |  |  |  |  | 0.0 | 0.0 | 0.0 | 418.89 |
| GA-страты -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0608 |  |  |  |  | 0.0 | 0.0 | 100.0 | 83.26 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0269 |  |  |  |  | 0.0 | 100.0 | 100.0 | 85.96 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.0401 |  |  |  |  | 0.0 | 0.0 | 100.0 | 90.68 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0468 |  |  |  |  | 0.0 | 0.0 | 100.0 | 90.51 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0401 |  |  |  |  | 0.0 | 0.0 | 100.0 | 94.27 |

## Treatment влияет на признак зоны C

Проверка, что C-gate блокирует treatment-dependent признак.

Ни один кандидат не прошёл заданный диагностический допуск.

### Цепочка A/B/C

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 67.77 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.1299 |  |  |  |  | 100.0 | 0.0 | 0.0 | 283.62 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 93.14 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 79.61 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 98.0 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 96.9 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.1299 |  |  |  |  | 100.0 | 0.0 | 0.0 | 295.5 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 84.62 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 106.63 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 100.76 |

### Все пайплайны

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Welch t-test | confirmatory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 0.94 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 67.77 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 1.0 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | -0.1646 |  |  |  |  | 0.0 | 0.0 | 100.0 | 2.74 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 97.14 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 31.83 |
| HypEx CUPED | integration_parity | смещение выше допуска | -0.1269 |  |  |  |  | 100.0 | 0.0 | 0.0 | 158.17 |
| CUPED по контролю -> HypEx A/B | confirmatory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 77.03 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.1299 |  |  |  |  | 100.0 | 0.0 | 0.0 | 283.62 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 93.14 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 79.61 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 98.0 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 100.06 |
| Регрессионная корректировка | confirmatory | смещение выше допуска | -0.1314 |  |  |  |  | 100.0 | 0.0 | 0.0 | 4.82 |
| ML-RATE | exploratory | смещение выше допуска | -0.1328 |  |  |  |  | 100.0 | 0.0 | 0.0 | 26.55 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1337 |  |  |  |  | 0.0 | 0.0 | 0.0 | 12.46 |
| Matching HypEx | integration_parity | смещение выше допуска | -0.1006 |  |  |  |  | 0.0 | 0.0 | 100.0 | 751.87 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1111 |  |  |  |  | 0.0 | 0.0 | 0.0 | 348.06 |
| welch + raw | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 0.92 |
| welch + cuped_control | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 2.84 |
| welch + cupac_linear_A | exploratory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 7.21 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 11.77 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 10.59 |
| welch + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 6.37 |
| hypex_ab + raw | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 70.8 |
| hypex_ab + cuped_control | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 79.95 |
| hypex_ab + cupac_linear_A | exploratory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 90.04 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 92.47 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 86.29 |
| hypex_ab + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 100.74 |
| permutation + raw | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 175.52 |
| permutation + cuped_control | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 173.23 |
| permutation + cupac_linear_A | exploratory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 205.84 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 217.03 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 199.07 |
| permutation + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 187.99 |
| bootstrap + raw | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 64.45 |
| bootstrap + cuped_control | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 69.33 |
| bootstrap + cupac_linear_A | exploratory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 80.28 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 86.44 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 112.18 |
| bootstrap + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 89.12 |
| regression_adjustment + raw | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 100.0 | 0.0 | 0.0 | 6.92 |
| regression_adjustment + cuped_control | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 100.0 | 0.0 | 0.0 | 9.4 |
| regression_adjustment + cupac_linear_A | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 100.0 | 0.0 | 0.0 | 17.77 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1212 |  |  |  |  | 100.0 | 0.0 | 0.0 | 25.74 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1212 |  |  |  |  | 100.0 | 0.0 | 0.0 | 24.24 |
| regression_adjustment + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1282 |  |  |  |  | 100.0 | 0.0 | 0.0 | 12.72 |
| ml_rate + raw | exploratory | смещение выше допуска | -0.1328 |  |  |  |  | 100.0 | 0.0 | 0.0 | 23.43 |
| ml_rate + cuped_control | exploratory | смещение выше допуска | -0.1323 |  |  |  |  | 100.0 | 0.0 | 0.0 | 30.48 |
| ml_rate + cupac_linear_A | exploratory | смещение выше допуска | -0.1333 |  |  |  |  | 100.0 | 0.0 | 0.0 | 46.2 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1232 |  |  |  |  | 100.0 | 0.0 | 0.0 | 39.68 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1232 |  |  |  |  | 100.0 | 0.0 | 0.0 | 39.62 |
| ml_rate + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1301 |  |  |  |  | 100.0 | 0.0 | 0.0 | 32.87 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1337 |  |  |  |  | 0.0 | 0.0 | 0.0 | 13.27 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.133 |  |  |  |  | 0.0 | 0.0 | 0.0 | 16.98 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1342 |  |  |  |  | 0.0 | 0.0 | 0.0 | 24.76 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1373 |  |  |  |  | 0.0 | 0.0 | 0.0 | 42.63 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1373 |  |  |  |  | 0.0 | 0.0 | 0.0 | 34.51 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1319 |  |  |  |  | 0.0 | 0.0 | 0.0 | 22.31 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1111 |  |  |  |  | 0.0 | 0.0 | 0.0 | 428.97 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1388 |  |  |  |  | 0.0 | 0.0 | 0.0 | 419.34 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1303 |  |  |  |  | 0.0 | 0.0 | 0.0 | 367.87 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1352 |  |  |  |  | 0.0 | 0.0 | 0.0 | 405.34 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1352 |  |  |  |  | 0.0 | 0.0 | 0.0 | 437.27 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1291 |  |  |  |  | 0.0 | 0.0 | 0.0 | 401.67 |
| GA-страты -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.1382 |  |  |  |  | 100.0 | 0.0 | 0.0 | 198.01 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.1329 |  |  |  |  | 100.0 | 0.0 | 0.0 | 202.3 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.1348 |  |  |  |  | 100.0 | 0.0 | 0.0 | 207.59 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | -0.1245 |  |  |  |  | 100.0 | 0.0 | 0.0 | 210.71 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | -0.1245 |  |  |  |  | 100.0 | 0.0 | 0.0 | 212.34 |
| Welch t-test | confirmatory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 1.67 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 96.9 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 1.61 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | -0.1646 |  |  |  |  | 0.0 | 0.0 | 100.0 | 2.68 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 93.93 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 35.82 |
| HypEx CUPED | integration_parity | смещение выше допуска | -0.1269 |  |  |  |  | 100.0 | 0.0 | 0.0 | 158.98 |
| CUPED по контролю -> HypEx A/B | confirmatory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 90.53 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | -0.1299 |  |  |  |  | 100.0 | 0.0 | 0.0 | 295.5 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 84.62 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 106.63 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 100.76 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 86.91 |
| Регрессионная корректировка | confirmatory | смещение выше допуска | -0.1314 |  |  |  |  | 100.0 | 0.0 | 0.0 | 5.54 |
| ML-RATE | exploratory | смещение выше допуска | -0.1328 |  |  |  |  | 100.0 | 0.0 | 0.0 | 21.74 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1337 |  |  |  |  | 0.0 | 0.0 | 0.0 | 8.33 |
| Matching HypEx | integration_parity | смещение выше допуска | -0.1006 |  |  |  |  | 0.0 | 0.0 | 100.0 | 704.69 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1111 |  |  |  |  | 0.0 | 0.0 | 0.0 | 350.83 |
| Panel-prior cupac_linear_A -> HypEx A/B | confirmatory | смещение выше допуска | -0.1594 |  |  |  |  | 100.0 | 0.0 | 0.0 | 79.4 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 79.36 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 79.55 |
| welch + raw | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 1.43 |
| welch + cuped_control | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 3.32 |
| welch + cupac_linear_A | exploratory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 12.76 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 15.01 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 16.82 |
| welch + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 8.06 |
| welch + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1594 |  |  |  |  | 100.0 | 0.0 | 0.0 | 17.7 |
| welch + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 24.55 |
| welch + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 30.56 |
| hypex_ab + raw | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 101.37 |
| hypex_ab + cuped_control | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 82.86 |
| hypex_ab + cupac_linear_A | exploratory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 83.89 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 106.72 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 116.74 |
| hypex_ab + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 97.33 |
| hypex_ab + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1594 |  |  |  |  | 100.0 | 0.0 | 0.0 | 168.88 |
| hypex_ab + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 116.96 |
| hypex_ab + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 100.39 |
| permutation + raw | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 184.39 |
| permutation + cuped_control | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 173.54 |
| permutation + cupac_linear_A | exploratory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 200.58 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 197.06 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 197.43 |
| permutation + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 182.71 |
| permutation + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1594 |  |  |  |  | 100.0 | 0.0 | 0.0 | 197.45 |
| permutation + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 205.53 |
| permutation + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 223.77 |
| bootstrap + raw | exploratory | смещение выше допуска | -0.2334 |  |  |  |  | 100.0 | 0.0 | 0.0 | 61.9 |
| bootstrap + cuped_control | exploratory | смещение выше допуска | -0.124 |  |  |  |  | 100.0 | 0.0 | 0.0 | 69.61 |
| bootstrap + cupac_linear_A | exploratory | смещение выше допуска | -0.1579 |  |  |  |  | 100.0 | 0.0 | 0.0 | 64.8 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 72.6 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1476 |  |  |  |  | 100.0 | 0.0 | 0.0 | 74.29 |
| bootstrap + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1547 |  |  |  |  | 100.0 | 0.0 | 0.0 | 54.54 |
| bootstrap + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1594 |  |  |  |  | 100.0 | 0.0 | 0.0 | 74.26 |
| bootstrap + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 72.58 |
| bootstrap + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1491 |  |  |  |  | 100.0 | 0.0 | 0.0 | 65.01 |
| regression_adjustment + raw | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 100.0 | 0.0 | 0.0 | 5.2 |
| regression_adjustment + cuped_control | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 100.0 | 0.0 | 0.0 | 6.97 |
| regression_adjustment + cupac_linear_A | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 100.0 | 0.0 | 0.0 | 14.45 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1212 |  |  |  |  | 100.0 | 0.0 | 0.0 | 14.93 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1212 |  |  |  |  | 100.0 | 0.0 | 0.0 | 20.69 |
| regression_adjustment + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1282 |  |  |  |  | 100.0 | 0.0 | 0.0 | 12.14 |
| regression_adjustment + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 100.0 | 0.0 | 0.0 | 15.9 |
| regression_adjustment + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1213 |  |  |  |  | 100.0 | 0.0 | 0.0 | 20.64 |
| regression_adjustment + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1213 |  |  |  |  | 100.0 | 0.0 | 0.0 | 23.89 |
| ml_rate + raw | exploratory | смещение выше допуска | -0.1328 |  |  |  |  | 100.0 | 0.0 | 0.0 | 23.5 |
| ml_rate + cuped_control | exploratory | смещение выше допуска | -0.1323 |  |  |  |  | 100.0 | 0.0 | 0.0 | 25.18 |
| ml_rate + cupac_linear_A | exploratory | смещение выше допуска | -0.1333 |  |  |  |  | 100.0 | 0.0 | 0.0 | 31.31 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1232 |  |  |  |  | 100.0 | 0.0 | 0.0 | 33.84 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1232 |  |  |  |  | 100.0 | 0.0 | 0.0 | 40.33 |
| ml_rate + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1301 |  |  |  |  | 100.0 | 0.0 | 0.0 | 31.41 |
| ml_rate + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1332 |  |  |  |  | 100.0 | 0.0 | 0.0 | 43.78 |
| ml_rate + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1232 |  |  |  |  | 100.0 | 0.0 | 0.0 | 51.56 |
| ml_rate + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1232 |  |  |  |  | 100.0 | 0.0 | 0.0 | 50.22 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1337 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.57 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.133 |  |  |  |  | 0.0 | 0.0 | 0.0 | 16.86 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1342 |  |  |  |  | 0.0 | 0.0 | 0.0 | 24.0 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1373 |  |  |  |  | 0.0 | 0.0 | 0.0 | 24.69 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1373 |  |  |  |  | 0.0 | 0.0 | 0.0 | 33.84 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1319 |  |  |  |  | 0.0 | 0.0 | 0.0 | 18.42 |
| nearest_neighbor_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1341 |  |  |  |  | 0.0 | 0.0 | 0.0 | 33.25 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1372 |  |  |  |  | 0.0 | 0.0 | 0.0 | 33.66 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1372 |  |  |  |  | 0.0 | 0.0 | 0.0 | 33.34 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1111 |  |  |  |  | 0.0 | 0.0 | 0.0 | 354.19 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1388 |  |  |  |  | 0.0 | 0.0 | 0.0 | 383.47 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1303 |  |  |  |  | 0.0 | 0.0 | 0.0 | 367.69 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1352 |  |  |  |  | 0.0 | 0.0 | 0.0 | 334.31 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1352 |  |  |  |  | 0.0 | 0.0 | 0.0 | 335.88 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1291 |  |  |  |  | 0.0 | 0.0 | 0.0 | 308.38 |
| optimal_transport_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1299 |  |  |  |  | 0.0 | 0.0 | 0.0 | 328.94 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1349 |  |  |  |  | 0.0 | 0.0 | 0.0 | 325.21 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1349 |  |  |  |  | 0.0 | 0.0 | 0.0 | 341.51 |
| GA-страты -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.1382 |  |  |  |  | 100.0 | 0.0 | 0.0 | 181.85 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.1329 |  |  |  |  | 100.0 | 0.0 | 0.0 | 185.15 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | смещение выше допуска | -0.1348 |  |  |  |  | 100.0 | 0.0 | 0.0 | 191.73 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | -0.1245 |  |  |  |  | 100.0 | 0.0 | 0.0 | 193.64 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | -0.1245 |  |  |  |  | 100.0 | 0.0 | 0.0 | 197.46 |

## Ошибочно безопасный признак зоны B

Отрицательный контроль: экспертная ошибка должна проявиться смещением.

Ни один кандидат не прошёл заданный диагностический допуск.

### Цепочка A/B/C

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 70.37 |
| HypEx temporal CUPAC | integration_parity | SE/SD вне допуска | 0.0015 |  |  |  |  | 0.0 | 100.0 | 100.0 | 229.34 |
| CUPAC A -> HypEx A/B | confirmatory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 94.23 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 79.21 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 85.47 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 79.82 |
| HypEx temporal CUPAC | integration_parity | SE/SD вне допуска | 0.0015 |  |  |  |  | 0.0 | 100.0 | 100.0 | 176.52 |
| CUPAC A -> HypEx A/B | confirmatory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 65.41 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 62.43 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 58.26 |

### Все пайплайны

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Welch t-test | confirmatory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.17 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 70.37 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.25 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | -0.1205 |  |  |  |  | 0.0 | 0.0 | 100.0 | 2.05 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 89.26 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 21.92 |
| HypEx CUPED | integration_parity | SE/SD вне допуска | -0.0035 |  |  |  |  | 0.0 | 100.0 | 100.0 | 142.31 |
| CUPED по контролю -> HypEx A/B | confirmatory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 79.08 |
| HypEx temporal CUPAC | integration_parity | SE/SD вне допуска | 0.0015 |  |  |  |  | 0.0 | 100.0 | 100.0 | 229.34 |
| CUPAC A -> HypEx A/B | confirmatory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 94.23 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 79.21 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 85.47 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 70.51 |
| Регрессионная корректировка | confirmatory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 4.59 |
| ML-RATE | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 16.42 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1299 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.07 |
| Matching HypEx | integration_parity | смещение выше допуска | -0.0273 |  |  |  |  | 0.0 | 0.0 | 100.0 | 682.71 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1446 |  |  |  |  | 0.0 | 0.0 | 0.0 | 283.12 |
| welch + raw | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.95 |
| welch + cuped_control | exploratory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 3.95 |
| welch + cupac_linear_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 10.27 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 13.77 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 18.5 |
| welch + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 9.14 |
| hypex_ab + raw | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 99.2 |
| hypex_ab + cuped_control | exploratory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 102.71 |
| hypex_ab + cupac_linear_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 72.71 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 69.97 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 75.57 |
| hypex_ab + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 71.72 |
| permutation + raw | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 177.9 |
| permutation + cuped_control | exploratory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 166.95 |
| permutation + cupac_linear_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 167.54 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 159.27 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 145.99 |
| permutation + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 141.3 |
| bootstrap + raw | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 46.28 |
| bootstrap + cuped_control | exploratory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 46.45 |
| bootstrap + cupac_linear_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 60.79 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 51.6 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 53.63 |
| bootstrap + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 45.92 |
| regression_adjustment + raw | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 3.63 |
| regression_adjustment + cuped_control | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 4.71 |
| regression_adjustment + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 12.09 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0891 |  |  |  |  | 100.0 | 0.0 | 0.0 | 15.72 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0853 |  |  |  |  | 100.0 | 0.0 | 0.0 | 17.24 |
| regression_adjustment + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0023 |  |  |  |  | 0.0 | 100.0 | 100.0 | 8.46 |
| ml_rate + raw | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 20.88 |
| ml_rate + cuped_control | exploratory | SE/SD вне допуска | 0.0028 |  |  |  |  | 0.0 | 100.0 | 100.0 | 12.62 |
| ml_rate + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0028 |  |  |  |  | 0.0 | 100.0 | 100.0 | 28.69 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0893 |  |  |  |  | 100.0 | 0.0 | 0.0 | 32.29 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0859 |  |  |  |  | 100.0 | 0.0 | 0.0 | 32.44 |
| ml_rate + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0022 |  |  |  |  | 0.0 | 100.0 | 100.0 | 28.27 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1299 |  |  |  |  | 0.0 | 0.0 | 0.0 | 8.09 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.1316 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.78 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1305 |  |  |  |  | 0.0 | 0.0 | 0.0 | 15.82 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 0.0 | 0.0 | 0.0 | 19.32 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1385 |  |  |  |  | 0.0 | 0.0 | 0.0 | 19.89 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.131 |  |  |  |  | 0.0 | 0.0 | 0.0 | 12.7 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1446 |  |  |  |  | 0.0 | 0.0 | 0.0 | 249.49 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1577 |  |  |  |  | 0.0 | 0.0 | 0.0 | 252.75 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1541 |  |  |  |  | 0.0 | 0.0 | 0.0 | 256.74 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1567 |  |  |  |  | 0.0 | 0.0 | 0.0 | 268.46 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1559 |  |  |  |  | 0.0 | 0.0 | 0.0 | 258.59 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1541 |  |  |  |  | 0.0 | 0.0 | 0.0 | 259.53 |
| GA-страты -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0105 |  |  |  |  | 0.0 | 100.0 | 100.0 | 116.04 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0024 |  |  |  |  | 0.0 | 100.0 | 100.0 | 116.93 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0066 |  |  |  |  | 0.0 | 100.0 | 100.0 | 121.53 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0849 |  |  |  |  | 100.0 | 0.0 | 0.0 | 127.1 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0812 |  |  |  |  | 100.0 | 0.0 | 0.0 | 124.99 |
| Welch t-test | confirmatory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.74 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 79.82 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.03 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | -0.1205 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.73 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 62.41 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 19.92 |
| HypEx CUPED | integration_parity | SE/SD вне допуска | -0.0035 |  |  |  |  | 0.0 | 100.0 | 100.0 | 112.77 |
| CUPED по контролю -> HypEx A/B | confirmatory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 59.03 |
| HypEx temporal CUPAC | integration_parity | SE/SD вне допуска | 0.0015 |  |  |  |  | 0.0 | 100.0 | 100.0 | 176.52 |
| CUPAC A -> HypEx A/B | confirmatory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 65.41 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 62.43 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 58.26 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 39.27 |
| Регрессионная корректировка | confirmatory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 1.92 |
| ML-RATE | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 8.58 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.1299 |  |  |  |  | 0.0 | 0.0 | 0.0 | 3.49 |
| Matching HypEx | integration_parity | смещение выше допуска | -0.0273 |  |  |  |  | 0.0 | 0.0 | 100.0 | 537.92 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1446 |  |  |  |  | 0.0 | 0.0 | 0.0 | 223.77 |
| Panel-prior cupac_linear_A -> HypEx A/B | confirmatory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 57.66 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 69.91 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 61.07 |
| welch + raw | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 0.47 |
| welch + cuped_control | exploratory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 2.52 |
| welch + cupac_linear_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 6.51 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 12.51 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 10.29 |
| welch + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 4.64 |
| welch + panel_prior_cupac_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 11.97 |
| welch + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 13.0 |
| welch + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 13.33 |
| hypex_ab + raw | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 56.46 |
| hypex_ab + cuped_control | exploratory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 53.67 |
| hypex_ab + cupac_linear_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 59.97 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 75.37 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 72.29 |
| hypex_ab + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 65.27 |
| hypex_ab + panel_prior_cupac_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 80.5 |
| hypex_ab + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 69.3 |
| hypex_ab + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 60.25 |
| permutation + raw | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 115.99 |
| permutation + cuped_control | exploratory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 116.57 |
| permutation + cupac_linear_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 142.23 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 154.03 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 130.55 |
| permutation + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 126.25 |
| permutation + panel_prior_cupac_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 147.72 |
| permutation + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 179.51 |
| permutation + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 139.0 |
| bootstrap + raw | exploratory | смещение выше допуска | -0.0603 |  |  |  |  | 0.0 | 0.0 | 100.0 | 38.25 |
| bootstrap + cuped_control | exploratory | SE/SD вне допуска | -0.0027 |  |  |  |  | 0.0 | 100.0 | 100.0 | 38.22 |
| bootstrap + cupac_linear_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 44.2 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 49.99 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 54.49 |
| bootstrap + cupac_quadratic_A | exploratory | SE/SD вне допуска | -0.0166 |  |  |  |  | 0.0 | 100.0 | 100.0 | 38.39 |
| bootstrap + panel_prior_cupac_A | exploratory | SE/SD вне допуска | -0.016 |  |  |  |  | 0.0 | 100.0 | 100.0 | 57.88 |
| bootstrap + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1079 |  |  |  |  | 100.0 | 0.0 | 0.0 | 44.6 |
| bootstrap + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.098 |  |  |  |  | 100.0 | 0.0 | 0.0 | 68.28 |
| regression_adjustment + raw | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 3.61 |
| regression_adjustment + cuped_control | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 8.03 |
| regression_adjustment + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 8.68 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0891 |  |  |  |  | 100.0 | 0.0 | 0.0 | 9.8 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0853 |  |  |  |  | 100.0 | 0.0 | 0.0 | 20.22 |
| regression_adjustment + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0023 |  |  |  |  | 0.0 | 100.0 | 100.0 | 11.34 |
| regression_adjustment + panel_prior_cupac_A | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 21.42 |
| regression_adjustment + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0891 |  |  |  |  | 100.0 | 0.0 | 0.0 | 18.01 |
| regression_adjustment + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0853 |  |  |  |  | 100.0 | 0.0 | 0.0 | 19.36 |
| ml_rate + raw | exploratory | SE/SD вне допуска | 0.0029 |  |  |  |  | 0.0 | 100.0 | 100.0 | 12.89 |
| ml_rate + cuped_control | exploratory | SE/SD вне допуска | 0.0028 |  |  |  |  | 0.0 | 100.0 | 100.0 | 18.02 |
| ml_rate + cupac_linear_A | exploratory | SE/SD вне допуска | 0.0028 |  |  |  |  | 0.0 | 100.0 | 100.0 | 31.29 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.0893 |  |  |  |  | 100.0 | 0.0 | 0.0 | 27.55 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0859 |  |  |  |  | 100.0 | 0.0 | 0.0 | 22.78 |
| ml_rate + cupac_quadratic_A | exploratory | SE/SD вне допуска | 0.0022 |  |  |  |  | 0.0 | 100.0 | 100.0 | 16.53 |
| ml_rate + panel_prior_cupac_A | exploratory | SE/SD вне допуска | 0.0028 |  |  |  |  | 0.0 | 100.0 | 100.0 | 25.69 |
| ml_rate + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.0893 |  |  |  |  | 100.0 | 0.0 | 0.0 | 21.11 |
| ml_rate + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.0859 |  |  |  |  | 100.0 | 0.0 | 0.0 | 27.19 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.1299 |  |  |  |  | 0.0 | 0.0 | 0.0 | 5.68 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.1316 |  |  |  |  | 0.0 | 0.0 | 0.0 | 8.87 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1305 |  |  |  |  | 0.0 | 0.0 | 0.0 | 8.06 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 0.0 | 0.0 | 0.0 | 12.74 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1385 |  |  |  |  | 0.0 | 0.0 | 0.0 | 19.94 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.131 |  |  |  |  | 0.0 | 0.0 | 0.0 | 14.14 |
| nearest_neighbor_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1305 |  |  |  |  | 0.0 | 0.0 | 0.0 | 22.87 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1314 |  |  |  |  | 0.0 | 0.0 | 0.0 | 17.7 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1385 |  |  |  |  | 0.0 | 0.0 | 0.0 | 27.91 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1446 |  |  |  |  | 0.0 | 0.0 | 0.0 | 258.73 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1577 |  |  |  |  | 0.0 | 0.0 | 0.0 | 225.9 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1541 |  |  |  |  | 0.0 | 0.0 | 0.0 | 220.24 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1567 |  |  |  |  | 0.0 | 0.0 | 0.0 | 259.52 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1559 |  |  |  |  | 0.0 | 0.0 | 0.0 | 223.98 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1541 |  |  |  |  | 0.0 | 0.0 | 0.0 | 219.26 |
| optimal_transport_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1541 |  |  |  |  | 0.0 | 0.0 | 0.0 | 277.96 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1567 |  |  |  |  | 0.0 | 0.0 | 0.0 | 212.48 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.1559 |  |  |  |  | 0.0 | 0.0 | 0.0 | 206.58 |
| GA-страты -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0105 |  |  |  |  | 0.0 | 100.0 | 100.0 | 63.51 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0024 |  |  |  |  | 0.0 | 100.0 | 100.0 | 68.58 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | SE/SD вне допуска | 0.0066 |  |  |  |  | 0.0 | 100.0 | 100.0 | 71.04 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0849 |  |  |  |  | 100.0 | 0.0 | 0.0 | 77.72 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | -0.0812 |  |  |  |  | 100.0 | 0.0 | 0.0 | 82.73 |

## Нерепрезентативная выборка и известная целевая популяция

Отдельная честная гонка методов, использующих известные доли страт и population moments.

Ни один кандидат не прошёл заданный диагностический допуск.

### Цепочка A/B/C

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 64.78 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | 0.3226 |  |  |  |  | 100.0 | 100.0 | 0.0 | 189.31 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 70.13 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 83.88 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 61.23 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 62.74 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | 0.3226 |  |  |  |  | 100.0 | 100.0 | 0.0 | 243.78 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 86.85 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 79.75 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 81.29 |

### Все пайплайны

| Пайплайн | Контур | Диагностический статус | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Welch t-test | confirmatory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 0.6 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 64.78 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 1.59 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | 0.0791 |  |  |  |  | 0.0 | 100.0 | 100.0 | 1.3 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 56.39 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 16.09 |
| HypEx CUPED | integration_parity | смещение выше допуска | 0.3136 |  |  |  |  | 100.0 | 100.0 | 0.0 | 85.79 |
| CUPED по контролю -> HypEx A/B | confirmatory | смещение выше допуска | 0.3118 |  |  |  |  | 100.0 | 100.0 | 0.0 | 50.28 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | 0.3226 |  |  |  |  | 100.0 | 100.0 | 0.0 | 189.31 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 70.13 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 83.88 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 61.23 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 49.78 |
| Регрессионная корректировка | confirmatory | смещение выше допуска | 0.3252 |  |  |  |  | 100.0 | 100.0 | 0.0 | 3.87 |
| ML-RATE | exploratory | смещение выше допуска | 0.3239 |  |  |  |  | 100.0 | 100.0 | 0.0 | 13.98 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.2026 |  |  |  |  | 0.0 | 0.0 | 0.0 | 11.58 |
| Matching HypEx | integration_parity | смещение выше допуска | 0.3397 |  |  |  |  | 0.0 | 100.0 | 0.0 | 501.13 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1804 |  |  |  |  | 0.0 | 0.0 | 0.0 | 216.03 |
| welch + raw | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 0.78 |
| welch + cuped_control | exploratory | смещение выше допуска | 0.3118 |  |  |  |  | 100.0 | 100.0 | 0.0 | 3.12 |
| welch + cupac_linear_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 7.39 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 12.41 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 23.32 |
| welch + cupac_quadratic_A | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 5.54 |
| hypex_ab + raw | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 74.31 |
| hypex_ab + cuped_control | exploratory | смещение выше допуска | 0.3118 |  |  |  |  | 100.0 | 100.0 | 0.0 | 59.96 |
| hypex_ab + cupac_linear_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 66.22 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 80.13 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 69.7 |
| hypex_ab + cupac_quadratic_A | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 58.46 |
| permutation + raw | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 138.05 |
| permutation + cuped_control | exploratory | смещение выше допуска | 0.3118 |  |  |  |  | 0.0 | 100.0 | 0.0 | 158.4 |
| permutation + cupac_linear_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 133.67 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 148.88 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 148.59 |
| permutation + cupac_quadratic_A | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 165.23 |
| bootstrap + raw | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 44.55 |
| bootstrap + cuped_control | exploratory | смещение выше допуска | 0.3118 |  |  |  |  | 0.0 | 100.0 | 0.0 | 42.85 |
| bootstrap + cupac_linear_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 58.77 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 51.44 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 64.23 |
| bootstrap + cupac_quadratic_A | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 43.87 |
| regression_adjustment + raw | exploratory | смещение выше допуска | 0.3252 |  |  |  |  | 100.0 | 100.0 | 0.0 | 3.86 |
| regression_adjustment + cuped_control | exploratory | смещение выше допуска | 0.3252 |  |  |  |  | 100.0 | 100.0 | 0.0 | 4.39 |
| regression_adjustment + cupac_linear_A | exploratory | смещение выше допуска | 0.3252 |  |  |  |  | 100.0 | 100.0 | 0.0 | 4.92 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.3036 |  |  |  |  | 0.0 | 100.0 | 0.0 | 10.48 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2932 |  |  |  |  | 0.0 | 100.0 | 0.0 | 14.48 |
| regression_adjustment + cupac_quadratic_A | exploratory | смещение выше допуска | 0.3282 |  |  |  |  | 100.0 | 100.0 | 0.0 | 3.95 |
| ml_rate + raw | exploratory | смещение выше допуска | 0.3239 |  |  |  |  | 100.0 | 100.0 | 0.0 | 13.37 |
| ml_rate + cuped_control | exploratory | смещение выше допуска | 0.3233 |  |  |  |  | 100.0 | 100.0 | 0.0 | 17.1 |
| ml_rate + cupac_linear_A | exploratory | смещение выше допуска | 0.3236 |  |  |  |  | 100.0 | 100.0 | 0.0 | 17.16 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.3017 |  |  |  |  | 0.0 | 100.0 | 0.0 | 29.61 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2912 |  |  |  |  | 0.0 | 100.0 | 0.0 | 23.19 |
| ml_rate + cupac_quadratic_A | exploratory | смещение выше допуска | 0.3266 |  |  |  |  | 100.0 | 100.0 | 0.0 | 17.97 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.2026 |  |  |  |  | 0.0 | 0.0 | 0.0 | 6.72 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.204 |  |  |  |  | 0.0 | 0.0 | 0.0 | 8.48 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.2028 |  |  |  |  | 0.0 | 0.0 | 0.0 | 15.79 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.2033 |  |  |  |  | 0.0 | 0.0 | 0.0 | 18.43 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.2015 |  |  |  |  | 0.0 | 0.0 | 0.0 | 19.33 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.2014 |  |  |  |  | 0.0 | 0.0 | 0.0 | 10.62 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1804 |  |  |  |  | 0.0 | 0.0 | 0.0 | 258.97 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1586 |  |  |  |  | 0.0 | 0.0 | 0.0 | 220.36 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1623 |  |  |  |  | 0.0 | 0.0 | 0.0 | 198.74 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1661 |  |  |  |  | 0.0 | 0.0 | 0.0 | 193.82 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.166 |  |  |  |  | 0.0 | 0.0 | 0.0 | 229.89 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1598 |  |  |  |  | 0.0 | 0.0 | 0.0 | 225.97 |
| GA-страты -> стратифицированная оценка | confirmatory | смещение выше допуска | 0.3085 |  |  |  |  | 0.0 | 100.0 | 0.0 | 69.9 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | смещение выше допуска | 0.3139 |  |  |  |  | 100.0 | 100.0 | 0.0 | 71.53 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | смещение выше допуска | 0.3177 |  |  |  |  | 100.0 | 100.0 | 0.0 | 78.77 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | 0.2966 |  |  |  |  | 0.0 | 100.0 | 0.0 | 78.93 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | 0.2872 |  |  |  |  | 0.0 | 100.0 | 0.0 | 85.44 |
| Welch t-test | confirmatory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 0.94 |
| HypEx A/B (Welch) | integration_parity | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 62.74 |
| Разность средних с общей дисперсией | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 0.48 |
| Воспроизводимая взвешенная разность средних | exploratory | смещение выше допуска | 0.0791 |  |  |  |  | 0.0 | 100.0 | 100.0 | 2.84 |
| Перестановочная оценка разности средних | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 70.9 |
| Bootstrap-оценка разности средних | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 19.64 |
| HypEx CUPED | integration_parity | смещение выше допуска | 0.3136 |  |  |  |  | 100.0 | 100.0 | 0.0 | 171.18 |
| CUPED по контролю -> HypEx A/B | confirmatory | смещение выше допуска | 0.3118 |  |  |  |  | 100.0 | 100.0 | 0.0 | 120.32 |
| HypEx temporal CUPAC | integration_parity | смещение выше допуска | 0.3226 |  |  |  |  | 100.0 | 100.0 | 0.0 | 243.78 |
| CUPAC A -> HypEx A/B | confirmatory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 86.85 |
| CUPAC A+B -> HypEx A/B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 79.75 |
| CUPAC A+B+C -> HypEx A/B | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 81.29 |
| Квадратичный CUPAC A -> HypEx A/B | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 61.01 |
| Регрессионная корректировка | confirmatory | смещение выше допуска | 0.3252 |  |  |  |  | 100.0 | 100.0 | 0.0 | 4.63 |
| ML-RATE | exploratory | смещение выше допуска | 0.3239 |  |  |  |  | 100.0 | 100.0 | 0.0 | 28.89 |
| Matching ближайшего соседа | exploratory | смещение выше допуска | -0.2026 |  |  |  |  | 0.0 | 0.0 | 0.0 | 15.22 |
| Matching HypEx | integration_parity | смещение выше допуска | 0.3397 |  |  |  |  | 0.0 | 100.0 | 0.0 | 584.54 |
| Глобальный matching оптимальным транспортом | exploratory | смещение выше допуска | -0.1804 |  |  |  |  | 0.0 | 0.0 | 0.0 | 249.58 |
| Panel-prior cupac_linear_A -> HypEx A/B | confirmatory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 72.14 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | exploratory | смещение выше допуска | 0.275 |  |  |  |  | 0.0 | 100.0 | 0.0 | 79.56 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 93.87 |
| welch + raw | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 1.13 |
| welch + cuped_control | exploratory | смещение выше допуска | 0.3118 |  |  |  |  | 100.0 | 100.0 | 0.0 | 3.24 |
| welch + cupac_linear_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 8.57 |
| welch + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 10.85 |
| welch + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 13.37 |
| welch + cupac_quadratic_A | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 4.92 |
| welch + panel_prior_cupac_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 11.7 |
| welch + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | 0.275 |  |  |  |  | 0.0 | 100.0 | 0.0 | 14.26 |
| welch + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 19.48 |
| hypex_ab + raw | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 61.56 |
| hypex_ab + cuped_control | exploratory | смещение выше допуска | 0.3118 |  |  |  |  | 100.0 | 100.0 | 0.0 | 80.14 |
| hypex_ab + cupac_linear_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 70.56 |
| hypex_ab + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 128.06 |
| hypex_ab + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 79.41 |
| hypex_ab + cupac_quadratic_A | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 60.55 |
| hypex_ab + panel_prior_cupac_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 96.93 |
| hypex_ab + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | 0.275 |  |  |  |  | 0.0 | 100.0 | 0.0 | 83.93 |
| hypex_ab + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 131.08 |
| permutation + raw | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 218.32 |
| permutation + cuped_control | exploratory | смещение выше допуска | 0.3118 |  |  |  |  | 0.0 | 100.0 | 0.0 | 204.23 |
| permutation + cupac_linear_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 233.39 |
| permutation + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 211.77 |
| permutation + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 228.68 |
| permutation + cupac_quadratic_A | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 193.28 |
| permutation + panel_prior_cupac_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 197.96 |
| permutation + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | 0.275 |  |  |  |  | 0.0 | 100.0 | 0.0 | 238.5 |
| permutation + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 204.97 |
| bootstrap + raw | exploratory | смещение выше допуска | 0.2318 |  |  |  |  | 0.0 | 100.0 | 0.0 | 51.76 |
| bootstrap + cuped_control | exploratory | смещение выше допуска | 0.3118 |  |  |  |  | 0.0 | 100.0 | 0.0 | 56.59 |
| bootstrap + cupac_linear_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 58.99 |
| bootstrap + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.2749 |  |  |  |  | 0.0 | 100.0 | 0.0 | 65.33 |
| bootstrap + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 60.82 |
| bootstrap + cupac_quadratic_A | exploratory | смещение выше допуска | 0.2995 |  |  |  |  | 0.0 | 100.0 | 0.0 | 49.93 |
| bootstrap + panel_prior_cupac_A | exploratory | смещение выше допуска | 0.2965 |  |  |  |  | 0.0 | 100.0 | 0.0 | 62.32 |
| bootstrap + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | 0.275 |  |  |  |  | 0.0 | 100.0 | 0.0 | 79.1 |
| bootstrap + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2738 |  |  |  |  | 0.0 | 100.0 | 0.0 | 86.85 |
| regression_adjustment + raw | exploratory | смещение выше допуска | 0.3252 |  |  |  |  | 100.0 | 100.0 | 0.0 | 5.18 |
| regression_adjustment + cuped_control | exploratory | смещение выше допуска | 0.3252 |  |  |  |  | 100.0 | 100.0 | 0.0 | 6.12 |
| regression_adjustment + cupac_linear_A | exploratory | смещение выше допуска | 0.3252 |  |  |  |  | 100.0 | 100.0 | 0.0 | 18.99 |
| regression_adjustment + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.3036 |  |  |  |  | 0.0 | 100.0 | 0.0 | 16.83 |
| regression_adjustment + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2932 |  |  |  |  | 0.0 | 100.0 | 0.0 | 21.91 |
| regression_adjustment + cupac_quadratic_A | exploratory | смещение выше допуска | 0.3282 |  |  |  |  | 100.0 | 100.0 | 0.0 | 6.37 |
| regression_adjustment + panel_prior_cupac_A | exploratory | смещение выше допуска | 0.3252 |  |  |  |  | 100.0 | 100.0 | 0.0 | 21.44 |
| regression_adjustment + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | 0.3036 |  |  |  |  | 0.0 | 100.0 | 0.0 | 34.5 |
| regression_adjustment + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2932 |  |  |  |  | 0.0 | 100.0 | 0.0 | 31.26 |
| ml_rate + raw | exploratory | смещение выше допуска | 0.3239 |  |  |  |  | 100.0 | 100.0 | 0.0 | 20.92 |
| ml_rate + cuped_control | exploratory | смещение выше допуска | 0.3233 |  |  |  |  | 100.0 | 100.0 | 0.0 | 22.87 |
| ml_rate + cupac_linear_A | exploratory | смещение выше допуска | 0.3236 |  |  |  |  | 100.0 | 100.0 | 0.0 | 34.71 |
| ml_rate + cupac_linear_A_plus_B | exploratory | смещение выше допуска | 0.3017 |  |  |  |  | 0.0 | 100.0 | 0.0 | 31.38 |
| ml_rate + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2912 |  |  |  |  | 0.0 | 100.0 | 0.0 | 49.31 |
| ml_rate + cupac_quadratic_A | exploratory | смещение выше допуска | 0.3266 |  |  |  |  | 100.0 | 100.0 | 0.0 | 28.6 |
| ml_rate + panel_prior_cupac_A | exploratory | смещение выше допуска | 0.3236 |  |  |  |  | 100.0 | 100.0 | 0.0 | 34.91 |
| ml_rate + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | 0.3017 |  |  |  |  | 0.0 | 100.0 | 0.0 | 38.29 |
| ml_rate + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | 0.2912 |  |  |  |  | 0.0 | 100.0 | 0.0 | 38.3 |
| nearest_neighbor_matching + raw | exploratory | смещение выше допуска | -0.2026 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.01 |
| nearest_neighbor_matching + cuped_control | exploratory | смещение выше допуска | -0.204 |  |  |  |  | 0.0 | 0.0 | 0.0 | 9.64 |
| nearest_neighbor_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.2028 |  |  |  |  | 0.0 | 0.0 | 0.0 | 16.05 |
| nearest_neighbor_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.2033 |  |  |  |  | 0.0 | 0.0 | 0.0 | 18.09 |
| nearest_neighbor_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.2015 |  |  |  |  | 0.0 | 0.0 | 0.0 | 27.21 |
| nearest_neighbor_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.2014 |  |  |  |  | 0.0 | 0.0 | 0.0 | 23.11 |
| nearest_neighbor_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.2028 |  |  |  |  | 0.0 | 0.0 | 0.0 | 24.94 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.2033 |  |  |  |  | 0.0 | 0.0 | 0.0 | 27.93 |
| nearest_neighbor_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.2015 |  |  |  |  | 0.0 | 0.0 | 0.0 | 31.31 |
| optimal_transport_matching + raw | exploratory | смещение выше допуска | -0.1804 |  |  |  |  | 0.0 | 0.0 | 0.0 | 296.21 |
| optimal_transport_matching + cuped_control | exploratory | смещение выше допуска | -0.1586 |  |  |  |  | 0.0 | 0.0 | 0.0 | 319.63 |
| optimal_transport_matching + cupac_linear_A | exploratory | смещение выше допуска | -0.1623 |  |  |  |  | 0.0 | 0.0 | 0.0 | 274.56 |
| optimal_transport_matching + cupac_linear_A_plus_B | exploratory | смещение выше допуска | -0.1661 |  |  |  |  | 0.0 | 0.0 | 0.0 | 275.53 |
| optimal_transport_matching + cupac_linear_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.166 |  |  |  |  | 0.0 | 0.0 | 0.0 | 300.9 |
| optimal_transport_matching + cupac_quadratic_A | exploratory | смещение выше допуска | -0.1598 |  |  |  |  | 0.0 | 0.0 | 0.0 | 254.6 |
| optimal_transport_matching + panel_prior_cupac_A | exploratory | смещение выше допуска | -0.1623 |  |  |  |  | 0.0 | 0.0 | 0.0 | 285.93 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B | exploratory | смещение выше допуска | -0.1661 |  |  |  |  | 0.0 | 0.0 | 0.0 | 290.25 |
| optimal_transport_matching + panel_prior_cupac_A_plus_B_plus_C | exploratory | смещение выше допуска | -0.166 |  |  |  |  | 0.0 | 0.0 | 0.0 | 300.66 |
| GA-страты -> стратифицированная оценка | confirmatory | смещение выше допуска | 0.3085 |  |  |  |  | 0.0 | 100.0 | 0.0 | 52.88 |
| GA-страты -> CUPED -> стратифицированная оценка | confirmatory | смещение выше допуска | 0.3139 |  |  |  |  | 100.0 | 100.0 | 0.0 | 55.24 |
| GA-страты -> CUPAC A -> стратифицированная оценка | confirmatory | смещение выше допуска | 0.3177 |  |  |  |  | 100.0 | 100.0 | 0.0 | 61.4 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | exploratory | смещение выше допуска | 0.2966 |  |  |  |  | 0.0 | 100.0 | 0.0 | 63.55 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | exploratory | смещение выше допуска | 0.2872 |  |  |  |  | 0.0 | 100.0 | 0.0 | 70.32 |
| Стратификация с долями экспериментальной выборки | confirmatory | смещение выше допуска | 0.2276 |  |  |  |  | 0.0 | 100.0 | 0.0 | 2.81 |
| Постстратификация по известной популяции | confirmatory | другой estimand | 0.0104 |  |  |  |  | 0.0 | 0.0 | 100.0 | 1.65 |
| Entropy calibration по известным population moments | confirmatory | другой estimand | -0.0013 |  |  |  |  | 0.0 | 0.0 | 100.0 | 14.76 |

## Проверки архитектуры

- `HypEx A/B` — интеграционный parity baseline. Его расхождение с локальным Welch должно быть только вычислительным.
- `HypEx temporal CUPAC` не включён в цепочку A/B/C: это библиотечная проверка, а не очередной этап локальной схемы.
- `A+B` и `A+B+C` являются отдельными примитивами политики данных и отдельными полными пайплайнами.
- GA не считается estimator: runtime preprocessing учитывается в end-to-end времени соответствующих пайплайнов.

## Оптимальный транспорт

OT-тест однородности распределений и estimator эффекта через OT-matching являются разными методами. Поэтому они не смешаны с этой гонкой ATE. Для OT-matching нужен отдельный наблюдательный контур с явно выбранными ATT/ATC, правилами сопоставления массы и статистического вывода.

## Ограничения

- Monte Carlo интервалы всё ещё конечной ширины; близкие методы нельзя объявлять различными без отдельного paired сравнения.
- C-gate проверяет среднее, пропуски и одномерное распределение, но сценарий загрязнения пока содержит только легко обнаружимый постоянный сдвиг. Нужны отдельные проверки условных и слабых сдвигов.
- B/C оцениваются по текущему контролю, а обычный Welch считает преобразованный target фиксированным. До cross-fitting или специальной поправки дисперсии эти этапы остаются исследовательскими.
- Пик памяти основан на `tracemalloc` и не включает native буферы NumPy/BLAS/HypEx.
- Синтетический победитель является кандидатом для следующего этапа, а не универсальным production default.
