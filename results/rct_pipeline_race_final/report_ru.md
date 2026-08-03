# Гонка аналитических пайплайнов VarWar v2

## Короткий вывод

Сравниваются полные аналитические пайплайны, а не изолированные примитивы. Сначала проверяются смещение, FPR и покрытие; только после этого рассматриваются дисперсия, мощность и стоимость.

- **Линейный сигнал, нормальный шум.** Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.
- **Линейный сигнал, тяжёлые хвосты.** Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.
- **Нелинейный сигнал, нормальный шум.** Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.
- **Treatment влияет на признак зоны C.** Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.
- **Ошибочно безопасный признак зоны B.** Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.
- **Нерепрезентативная выборка и известная целевая популяция.** Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.

## Постановка

- Monte Carlo повторов на H0 и H1: `5`.
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
| stratified_mean_diff | Стратификация с долями экспериментальной выборки | stratified_mean_diff | Sample ATE | confirmatory | welch |
| population_weighted_stratified | Постстратификация по известной популяции | population_weighted_stratified | Population ATE | confirmatory | stratified_mean_diff |
| calibration_reweighting | Entropy calibration по известным population moments | calibration_reweighting | Population ATE | confirmatory | stratified_mean_diff |

## Линейный сигнал, нормальный шум

Благоприятный базовый сценарий.

Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.

### Отбор по сопоставимым контурам

| data_scenario | race_information_mode | temporal_mode | estimand_id | candidate_count | eligible_count | selected_pipeline | selected_pipeline_name_ru | selected_empirical_sd_h1 | selected_variance_reduction_vs_welch | selected_median_runtime_ms | selection_status | selection_criterion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_gaussian | panel_prior | post_treatment_allowed | sample_ate | 6 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| linear_gaussian | experiment_only | post_treatment_allowed | sample_ate | 4 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| linear_gaussian | experiment_only | pre_treatment_only | sample_ate | 15 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| linear_gaussian | panel_prior | pre_treatment_only | sample_ate | 16 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |

### Цепочка A/B/C

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | -0.0157 | 0.0534 | 0.82 |  |  | 0.0 | 60.0 | 100.0 | 64.17 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | -0.0157 | 0.0534 | 0.82 |  |  | 0.0 | 60.0 | 100.0 | 79.72 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0156 | 0.0593 | 0.712 |  | -23.2 | 0.0 | 80.0 | 100.0 | 78.51 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0156 | 0.0593 | 0.712 |  | -23.2 | 0.0 | 80.0 | 100.0 | 78.01 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0131 | 0.0601 | 0.782 | 37.1 | 37.1 | 0.0 | 60.0 | 100.0 | 73.2 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0131 | 0.0601 | 0.782 | 37.1 | 37.1 | 0.0 | 60.0 | 100.0 | 68.2 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | SE/SD вне допуска | False | -0.0064 | 0.0602 | 0.742 | 36.7 | 36.7 | 0.0 | 80.0 | 100.0 | 202.02 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | SE/SD вне допуска | False | -0.0064 | 0.0602 | 0.742 | 36.7 | 36.7 | 0.0 | 80.0 | 100.0 | 261.59 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 50.32 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 69.15 |

### Все пайплайны

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Глобальный matching оптимальным транспортом | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1602 | 0.0213 | 1.486 | 92.1 |  | 0.0 | 0.0 | 0.0 | 300.97 |
| Глобальный matching оптимальным транспортом | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1602 | 0.0213 | 1.486 | 92.1 |  | 0.0 | 0.0 | 0.0 | 222.58 |
| Matching ближайшего соседа | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1611 | 0.0289 | 1.089 | 85.4 |  | 0.0 | 0.0 | 0.0 | 5.69 |
| Matching ближайшего соседа | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1611 | 0.0289 | 1.089 | 85.4 |  | 0.0 | 0.0 | 0.0 | 7.63 |
| Matching HypEx | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | SE/SD вне допуска | False | -0.0002 | 0.0428 | 1.675 | 68.1 |  | 0.0 | 80.0 | 100.0 | 610.8 |
| Matching HypEx | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | SE/SD вне допуска | False | -0.0002 | 0.0428 | 1.675 | 68.1 |  | 0.0 | 80.0 | 100.0 | 571.04 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | -0.016 | 0.0532 | 0.825 |  |  | 0.0 | 60.0 | 100.0 | 79.67 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | -0.0157 | 0.0534 | 0.82 |  |  | 0.0 | 60.0 | 100.0 | 79.72 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | -0.0157 | 0.0534 | 0.82 |  |  | 0.0 | 60.0 | 100.0 | 64.17 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0087 | 0.0561 | 0.747 |  |  | 20.0 | 80.0 | 80.0 | 82.33 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0087 | 0.0561 | 0.747 |  |  | 20.0 | 80.0 | 80.0 | 113.08 |
| GA-страты -> CUPED -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | -0.0069 | 0.0564 | 0.812 | 44.6 | 19.7 | 0.0 | 80.0 | 100.0 | 97.58 |
| GA-страты -> CUPED -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | -0.0069 | 0.0564 | 0.812 | 44.6 | 19.7 | 0.0 | 80.0 | 100.0 | 67.09 |
| CUPED по контролю -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | -0.0073 | 0.0567 | 0.81 | 44.0 | 44.0 | 0.0 | 80.0 | 100.0 | 60.05 |
| CUPED по контролю -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | -0.0073 | 0.0567 | 0.81 | 44.0 | 44.0 | 0.0 | 80.0 | 100.0 | 73.04 |
| HypEx CUPED | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.0071 | 0.0569 | 0.806 | 43.6 | 43.6 | 0.0 | 80.0 | 100.0 | 101.76 |
| HypEx CUPED | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.0071 | 0.0569 | 0.806 | 43.6 | 43.6 | 0.0 | 80.0 | 100.0 | 128.3 |
| ML-RATE | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0048 | 0.0589 | 0.76 | 39.6 | 39.6 | 0.0 | 80.0 | 100.0 | 12.34 |
| ML-RATE | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0048 | 0.0589 | 0.76 | 39.6 | 39.6 | 0.0 | 80.0 | 100.0 | 12.78 |
| Регрессионная корректировка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0046 | 0.0591 | 0.756 | 39.1 | 39.1 | 0.0 | 80.0 | 100.0 | 3.6 |
| Регрессионная корректировка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0046 | 0.0591 | 0.756 | 39.1 | 39.1 | 0.0 | 80.0 | 100.0 | 2.59 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0158 | 0.0592 | 0.715 |  |  | 0.0 | 80.0 | 100.0 | 96.53 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0156 | 0.0593 | 0.712 |  | -23.2 | 0.0 | 80.0 | 100.0 | 78.51 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0156 | 0.0593 | 0.712 |  | -23.2 | 0.0 | 80.0 | 100.0 | 78.01 |
| Panel-prior cupac_linear_A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0134 | 0.0599 | 0.785 |  |  | 0.0 | 60.0 | 100.0 | 83.93 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0131 | 0.0601 | 0.782 | 37.1 | 37.1 | 0.0 | 60.0 | 100.0 | 73.2 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0131 | 0.0601 | 0.782 | 37.1 | 37.1 | 0.0 | 60.0 | 100.0 | 68.2 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | SE/SD вне допуска | False | -0.0064 | 0.0602 | 0.742 | 36.7 | 36.7 | 0.0 | 80.0 | 100.0 | 202.02 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | SE/SD вне допуска | False | -0.0064 | 0.0602 | 0.742 | 36.7 | 36.7 | 0.0 | 80.0 | 100.0 | 261.59 |
| GA-страты -> CUPAC A -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0059 | 0.0611 | 0.739 | 35.0 | 5.8 | 0.0 | 80.0 | 100.0 | 72.45 |
| GA-страты -> CUPAC A -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0059 | 0.0611 | 0.739 | 35.0 | 5.8 | 0.0 | 80.0 | 100.0 | 105.91 |
| Квадратичный CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0151 | 0.0617 | 0.763 | 33.7 | -5.4 | 0.0 | 60.0 | 100.0 | 58.48 |
| Квадратичный CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0151 | 0.0617 | 0.763 | 33.7 | -5.4 | 0.0 | 60.0 | 100.0 | 65.62 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0115 | 0.0621 | 0.668 |  | -22.6 | 20.0 | 80.0 | 80.0 | 83.54 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0115 | 0.0621 | 0.668 |  | -22.6 | 20.0 | 80.0 | 80.0 | 127.99 |
| GA-страты -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0029 | 0.0629 | 0.786 | 31.0 | 31.0 | 0.0 | 60.0 | 100.0 | 97.26 |
| GA-страты -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0029 | 0.0629 | 0.786 | 31.0 | 31.0 | 0.0 | 60.0 | 100.0 | 65.07 |
| Welch t-test | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 |  | 20.0 | 60.0 | 80.0 | 0.84 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 50.32 |
| Разность средних с общей дисперсией | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 0.4 |
| Перестановочная оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 39.48 |
| Bootstrap-оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 20.08 |
| Welch t-test | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 |  | 20.0 | 60.0 | 80.0 | 1.07 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 69.15 |
| Разность средних с общей дисперсией | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 0.76 |
| Перестановочная оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 85.63 |
| Bootstrap-оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0306 | 0.0757 | 0.881 | 0.0 | 0.0 | 20.0 | 60.0 | 80.0 | 24.75 |
| Воспроизводимая взвешенная разность средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0909 | 0.157 | 0.64 | -329.6 | -329.6 | 20.0 | 20.0 | 80.0 | 1.55 |
| Воспроизводимая взвешенная разность средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0909 | 0.157 | 0.64 | -329.6 | -329.6 | 20.0 | 20.0 | 80.0 | 1.6 |

## Линейный сигнал, тяжёлые хвосты

Проверка устойчивости SE и выигрыша по дисперсии.

Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.

### Отбор по сопоставимым контурам

| data_scenario | race_information_mode | temporal_mode | estimand_id | candidate_count | eligible_count | selected_pipeline | selected_pipeline_name_ru | selected_empirical_sd_h1 | selected_variance_reduction_vs_welch | selected_median_runtime_ms | selection_status | selection_criterion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_heavy_tail | experiment_only | post_treatment_allowed | sample_ate | 4 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| linear_heavy_tail | panel_prior | post_treatment_allowed | sample_ate | 6 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| linear_heavy_tail | experiment_only | pre_treatment_only | sample_ate | 15 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| linear_heavy_tail | panel_prior | pre_treatment_only | sample_ate | 16 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |

### Цепочка A/B/C

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.008 | 0.0404 | 1.009 |  | 27.8 | 0.0 | 100.0 | 100.0 | 108.03 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.008 | 0.0404 | 1.009 |  | 27.8 | 0.0 | 100.0 | 100.0 | 106.66 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.007 | 0.0475 | 0.892 |  |  | 0.0 | 100.0 | 100.0 | 102.27 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.007 | 0.0475 | 0.892 |  |  | 0.0 | 100.0 | 100.0 | 97.02 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.0021 | 0.048 | 0.896 | 42.8 | 42.8 | 0.0 | 80.0 | 100.0 | 286.28 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.0021 | 0.048 | 0.896 | 42.8 | 42.8 | 0.0 | 80.0 | 100.0 | 310.37 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0006 | 0.0499 | 0.913 | 38.2 | 38.2 | 0.0 | 80.0 | 100.0 | 95.36 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0006 | 0.0499 | 0.913 | 38.2 | 38.2 | 0.0 | 80.0 | 100.0 | 97.41 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 85.63 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 83.66 |

### Все пайплайны

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Глобальный matching оптимальным транспортом | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1415 | 0.0137 | 2.228 | 95.4 |  | 0.0 | 0.0 | 0.0 | 324.96 |
| Глобальный matching оптимальным транспортом | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1415 | 0.0137 | 2.228 | 95.4 |  | 0.0 | 0.0 | 0.0 | 355.79 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.011 | 0.0363 | 1.092 |  | 20.7 | 0.0 | 100.0 | 100.0 | 186.0 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.011 | 0.0363 | 1.092 |  | 20.7 | 0.0 | 100.0 | 100.0 | 175.95 |
| GA-страты -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | 0.0163 | 0.0378 | 1.271 | 64.6 | 64.6 | 0.0 | 100.0 | 100.0 | 153.51 |
| GA-страты -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | 0.0163 | 0.0378 | 1.271 | 64.6 | 64.6 | 0.0 | 100.0 | 100.0 | 165.38 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.008 | 0.0399 | 1.02 |  |  | 0.0 | 100.0 | 100.0 | 118.24 |
| Matching ближайшего соседа | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1407 | 0.0402 | 0.756 | 59.9 |  | 0.0 | 0.0 | 0.0 | 11.58 |
| Matching ближайшего соседа | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1407 | 0.0402 | 0.756 | 59.9 |  | 0.0 | 0.0 | 0.0 | 10.98 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.008 | 0.0404 | 1.009 |  | 27.8 | 0.0 | 100.0 | 100.0 | 108.03 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.008 | 0.0404 | 1.009 |  | 27.8 | 0.0 | 100.0 | 100.0 | 106.66 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0105 | 0.0408 | 0.984 |  |  | 0.0 | 100.0 | 100.0 | 165.94 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0105 | 0.0408 | 0.984 |  |  | 0.0 | 100.0 | 100.0 | 182.0 |
| GA-страты -> CUPAC A -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0038 | 0.0437 | 0.993 | 52.6 | -33.9 | 0.0 | 100.0 | 100.0 | 163.64 |
| GA-страты -> CUPAC A -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0038 | 0.0437 | 0.993 | 52.6 | -33.9 | 0.0 | 100.0 | 100.0 | 176.31 |
| Регрессионная корректировка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | -0.0017 | 0.0449 | 0.956 | 50.0 | 50.0 | 0.0 | 80.0 | 100.0 | 4.71 |
| Регрессионная корректировка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | -0.0017 | 0.0449 | 0.956 | 50.0 | 50.0 | 0.0 | 80.0 | 100.0 | 4.83 |
| ML-RATE | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | -0.0014 | 0.045 | 0.956 | 49.9 | 49.9 | 0.0 | 80.0 | 100.0 | 23.61 |
| ML-RATE | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | -0.0014 | 0.045 | 0.956 | 49.9 | 49.9 | 0.0 | 80.0 | 100.0 | 23.59 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0069 | 0.0471 | 0.901 |  |  | 0.0 | 100.0 | 100.0 | 107.9 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.007 | 0.0475 | 0.892 |  |  | 0.0 | 100.0 | 100.0 | 97.02 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.007 | 0.0475 | 0.892 |  |  | 0.0 | 100.0 | 100.0 | 102.27 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.0021 | 0.048 | 0.896 | 42.8 | 42.8 | 0.0 | 80.0 | 100.0 | 310.37 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.0021 | 0.048 | 0.896 | 42.8 | 42.8 | 0.0 | 80.0 | 100.0 | 286.28 |
| Panel-prior cupac_linear_A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0006 | 0.0492 | 0.926 |  |  | 0.0 | 80.0 | 100.0 | 105.93 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0006 | 0.0499 | 0.913 | 38.2 | 38.2 | 0.0 | 80.0 | 100.0 | 95.36 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0006 | 0.0499 | 0.913 | 38.2 | 38.2 | 0.0 | 80.0 | 100.0 | 97.41 |
| Квадратичный CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0009 | 0.0509 | 0.895 | 35.7 | -4.1 | 0.0 | 80.0 | 100.0 | 88.54 |
| Квадратичный CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0009 | 0.0509 | 0.895 | 35.7 | -4.1 | 0.0 | 80.0 | 100.0 | 92.48 |
| GA-страты -> CUPED -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0014 | 0.0547 | 0.804 | 25.9 | -109.4 | 0.0 | 80.0 | 100.0 | 156.27 |
| GA-страты -> CUPED -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0014 | 0.0547 | 0.804 | 25.9 | -109.4 | 0.0 | 80.0 | 100.0 | 167.89 |
| HypEx CUPED | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | FPR выше допуска | False | -0.0002 | 0.0554 | 0.801 | 24.0 | 24.0 | 20.0 | 80.0 | 80.0 | 171.97 |
| HypEx CUPED | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | FPR выше допуска | False | -0.0002 | 0.0554 | 0.801 | 24.0 | 24.0 | 20.0 | 80.0 | 80.0 | 179.03 |
| CUPED по контролю -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0003 | 0.0558 | 0.795 | 22.8 | 22.8 | 20.0 | 80.0 | 80.0 | 90.84 |
| CUPED по контролю -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0003 | 0.0558 | 0.795 | 22.8 | 22.8 | 20.0 | 80.0 | 80.0 | 84.06 |
| Matching HypEx | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.0068 | 0.0618 | 1.133 | 5.4 |  | 0.0 | 60.0 | 100.0 | 776.09 |
| Matching HypEx | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.0068 | 0.0618 | 1.133 | 5.4 |  | 0.0 | 60.0 | 100.0 | 779.07 |
| Welch t-test | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 |  | 0.0 | 60.0 | 100.0 | 1.37 |
| Разность средних с общей дисперсией | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 0.91 |
| Перестановочная оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 96.96 |
| Bootstrap-оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 30.7 |
| Перестановочная оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 104.03 |
| Разность средних с общей дисперсией | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 1.04 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 83.66 |
| Welch t-test | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 |  | 0.0 | 60.0 | 100.0 | 1.21 |
| Bootstrap-оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 27.11 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | 0.0088 | 0.0635 | 1.051 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 85.63 |
| Воспроизводимая взвешенная разность средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | -0.0079 | 0.0934 | 1.074 | -116.4 | -116.4 | 0.0 | 40.0 | 100.0 | 2.64 |
| Воспроизводимая взвешенная разность средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | -0.0079 | 0.0934 | 1.074 | -116.4 | -116.4 | 0.0 | 40.0 | 100.0 | 2.75 |

## Нелинейный сигнал, нормальный шум

Проверка линейных и квадратичных корректировок.

Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.

### Отбор по сопоставимым контурам

| data_scenario | race_information_mode | temporal_mode | estimand_id | candidate_count | eligible_count | selected_pipeline | selected_pipeline_name_ru | selected_empirical_sd_h1 | selected_variance_reduction_vs_welch | selected_median_runtime_ms | selection_status | selection_criterion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nonlinear_gaussian | experiment_only | pre_treatment_only | sample_ate | 15 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| nonlinear_gaussian | panel_prior | pre_treatment_only | sample_ate | 16 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| nonlinear_gaussian | experiment_only | post_treatment_allowed | sample_ate | 4 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| nonlinear_gaussian | panel_prior | post_treatment_allowed | sample_ate | 6 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |

### Цепочка A/B/C

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.065 | 0.0463 | 1.297 | 83.8 | 83.8 | 20.0 | 40.0 | 80.0 | 345.81 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.065 | 0.0463 | 1.297 | 83.8 | 83.8 | 20.0 | 40.0 | 80.0 | 340.54 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0546 | 0.0525 | 1.225 |  | 50.8 | 0.0 | 40.0 | 100.0 | 116.8 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0546 | 0.0525 | 1.225 |  | 50.8 | 0.0 | 40.0 | 100.0 | 119.24 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0727 | 0.07 | 0.997 | 63.0 | 63.0 | 20.0 | 40.0 | 80.0 | 107.16 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0727 | 0.07 | 0.997 | 63.0 | 63.0 | 20.0 | 40.0 | 80.0 | 103.06 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0653 | 0.0748 | 0.909 |  |  | 20.0 | 40.0 | 80.0 | 116.81 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0653 | 0.0748 | 0.909 |  |  | 20.0 | 40.0 | 80.0 | 108.6 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 96.7 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 104.13 |

### Все пайплайны

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matching ближайшего соседа | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1078 | 0.0304 | 1.103 | 93.0 |  | 20.0 | 20.0 | 20.0 | 11.27 |
| Matching ближайшего соседа | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1078 | 0.0304 | 1.103 | 93.0 |  | 20.0 | 20.0 | 20.0 | 11.32 |
| Matching HypEx | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0325 | 0.033 | 2.274 | 91.8 |  | 0.0 | 20.0 | 100.0 | 867.84 |
| Matching HypEx | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0325 | 0.033 | 2.274 | 91.8 |  | 0.0 | 20.0 | 100.0 | 835.5 |
| Глобальный matching оптимальным транспортом | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1252 | 0.0357 | 0.947 | 90.4 |  | 0.0 | 0.0 | 0.0 | 241.47 |
| Глобальный matching оптимальным транспортом | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1252 | 0.0357 | 0.947 | 90.4 |  | 0.0 | 0.0 | 0.0 | 241.58 |
| Квадратичный CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0497 | 0.0389 | 1.303 | 88.5 | 69.0 | 20.0 | 80.0 | 80.0 | 100.92 |
| Квадратичный CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0497 | 0.0389 | 1.303 | 88.5 | 69.0 | 20.0 | 80.0 | 80.0 | 99.01 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0436 | 0.0396 | 1.477 |  | 52.5 | 0.0 | 40.0 | 100.0 | 166.5 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0436 | 0.0396 | 1.477 |  | 52.5 | 0.0 | 40.0 | 100.0 | 171.05 |
| GA-страты -> CUPED -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0598 | 0.0416 | 1.373 | 86.9 | 68.8 | 0.0 | 60.0 | 100.0 | 142.72 |
| GA-страты -> CUPED -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0598 | 0.0416 | 1.373 | 86.9 | 68.8 | 0.0 | 60.0 | 100.0 | 143.67 |
| HypEx CUPED | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0679 | 0.0446 | 1.366 | 84.9 | 84.9 | 20.0 | 0.0 | 80.0 | 203.73 |
| HypEx CUPED | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0679 | 0.0446 | 1.366 | 84.9 | 84.9 | 20.0 | 0.0 | 80.0 | 187.83 |
| Регрессионная корректировка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0649 | 0.0458 | 1.32 | 84.1 | 84.1 | 20.0 | 40.0 | 80.0 | 5.47 |
| Регрессионная корректировка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0649 | 0.0458 | 1.32 | 84.1 | 84.1 | 20.0 | 40.0 | 80.0 | 5.14 |
| ML-RATE | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0646 | 0.046 | 1.319 | 84.0 | 84.0 | 20.0 | 40.0 | 80.0 | 25.31 |
| ML-RATE | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0646 | 0.046 | 1.319 | 84.0 | 84.0 | 20.0 | 40.0 | 80.0 | 23.24 |
| CUPED по контролю -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0665 | 0.0463 | 1.319 | 83.8 | 83.8 | 20.0 | 20.0 | 80.0 | 100.98 |
| CUPED по контролю -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0665 | 0.0463 | 1.319 | 83.8 | 83.8 | 20.0 | 20.0 | 80.0 | 91.28 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.065 | 0.0463 | 1.297 | 83.8 | 83.8 | 20.0 | 40.0 | 80.0 | 340.54 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.065 | 0.0463 | 1.297 | 83.8 | 83.8 | 20.0 | 40.0 | 80.0 | 345.81 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0547 | 0.0522 | 1.227 |  |  | 0.0 | 40.0 | 100.0 | 121.17 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0546 | 0.0525 | 1.225 |  | 50.8 | 0.0 | 40.0 | 100.0 | 119.24 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0546 | 0.0525 | 1.225 |  | 50.8 | 0.0 | 40.0 | 100.0 | 116.8 |
| GA-страты -> CUPAC A -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0594 | 0.0572 | 1.072 | 75.2 | 41.0 | 20.0 | 40.0 | 80.0 | 151.81 |
| GA-страты -> CUPAC A -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0594 | 0.0572 | 1.072 | 75.2 | 41.0 | 20.0 | 40.0 | 80.0 | 151.3 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0522 | 0.0575 | 1.032 |  |  | 0.0 | 40.0 | 100.0 | 156.84 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0522 | 0.0575 | 1.032 |  |  | 0.0 | 40.0 | 100.0 | 154.22 |
| Panel-prior cupac_linear_A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0728 | 0.0695 | 1.001 |  |  | 20.0 | 40.0 | 80.0 | 117.88 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0727 | 0.07 | 0.997 | 63.0 | 63.0 | 20.0 | 40.0 | 80.0 | 107.16 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0727 | 0.07 | 0.997 | 63.0 | 63.0 | 20.0 | 40.0 | 80.0 | 103.06 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0654 | 0.0743 | 0.911 |  |  | 20.0 | 40.0 | 80.0 | 110.26 |
| GA-страты -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0606 | 0.0745 | 0.944 | 58.0 | 58.0 | 0.0 | 40.0 | 100.0 | 141.3 |
| GA-страты -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0606 | 0.0745 | 0.944 | 58.0 | 58.0 | 0.0 | 40.0 | 100.0 | 140.63 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0653 | 0.0748 | 0.909 |  |  | 20.0 | 40.0 | 80.0 | 108.6 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0653 | 0.0748 | 0.909 |  |  | 20.0 | 40.0 | 80.0 | 116.81 |
| Welch t-test | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 |  | 20.0 | 20.0 | 80.0 | 1.41 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 96.7 |
| Разность средних с общей дисперсией | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 1.18 |
| Перестановочная оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 105.2 |
| Bootstrap-оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 31.96 |
| Welch t-test | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 |  | 20.0 | 20.0 | 80.0 | 1.4 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 104.13 |
| Разность средних с общей дисперсией | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 1.21 |
| Перестановочная оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 103.2 |
| Bootstrap-оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0808 | 0.115 | 0.826 | 0.0 | 0.0 | 20.0 | 20.0 | 80.0 | 29.78 |
| Воспроизводимая взвешенная разность средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0724 | 0.1386 | 1.029 | -45.3 | -45.3 | 0.0 | 20.0 | 100.0 | 2.89 |
| Воспроизводимая взвешенная разность средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0724 | 0.1386 | 1.029 | -45.3 | -45.3 | 0.0 | 20.0 | 100.0 | 2.86 |

## Treatment влияет на признак зоны C

Проверка, что C-gate блокирует treatment-dependent признак.

Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.

### Отбор по сопоставимым контурам

| data_scenario | race_information_mode | temporal_mode | estimand_id | candidate_count | eligible_count | selected_pipeline | selected_pipeline_name_ru | selected_empirical_sd_h1 | selected_variance_reduction_vs_welch | selected_median_runtime_ms | selection_status | selection_criterion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contaminated_c | experiment_only | pre_treatment_only | sample_ate | 15 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| contaminated_c | panel_prior | pre_treatment_only | sample_ate | 16 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| contaminated_c | experiment_only | post_treatment_allowed | sample_ate | 4 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| contaminated_c | panel_prior | post_treatment_allowed | sample_ate | 6 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |

### Цепочка A/B/C

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0291 | 0.0608 | 0.719 | 63.5 | 63.5 | 20.0 | 80.0 | 80.0 | 336.07 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0291 | 0.0608 | 0.719 | 63.5 | 63.5 | 20.0 | 80.0 | 80.0 | 336.82 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0411 | 0.0649 | 0.673 |  |  | 20.0 | 80.0 | 80.0 | 109.17 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0411 | 0.0649 | 0.673 |  | 0.0 | 20.0 | 80.0 | 80.0 | 107.78 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0411 | 0.0649 | 0.673 |  |  | 20.0 | 80.0 | 80.0 | 114.23 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0411 | 0.0649 | 0.673 |  | 0.0 | 20.0 | 80.0 | 80.0 | 115.22 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.045 | 0.0705 | 0.659 | 50.8 | 50.8 | 20.0 | 60.0 | 80.0 | 99.04 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.045 | 0.0705 | 0.659 | 50.8 | 50.8 | 20.0 | 60.0 | 80.0 | 107.89 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 94.49 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 92.47 |

### Все пайплайны

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Глобальный matching оптимальным транспортом | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1478 | 0.0257 | 1.202 | 93.5 |  | 0.0 | 0.0 | 0.0 | 373.36 |
| Глобальный matching оптимальным транспортом | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1478 | 0.0257 | 1.202 | 93.5 |  | 0.0 | 0.0 | 0.0 | 377.1 |
| Matching ближайшего соседа | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1638 | 0.0326 | 0.945 | 89.5 |  | 0.0 | 0.0 | 0.0 | 12.02 |
| Matching ближайшего соседа | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1638 | 0.0326 | 0.945 | 89.5 |  | 0.0 | 0.0 | 0.0 | 11.61 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0292 | 0.055 | 0.751 |  |  | 20.0 | 80.0 | 80.0 | 166.23 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0292 | 0.055 | 0.751 |  | 0.0 | 20.0 | 80.0 | 80.0 | 170.76 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0292 | 0.055 | 0.751 |  |  | 20.0 | 80.0 | 80.0 | 165.44 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0292 | 0.055 | 0.751 |  | 0.0 | 20.0 | 80.0 | 80.0 | 169.69 |
| Matching HypEx | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | SE/SD вне допуска | False | -0.0086 | 0.0564 | 1.247 | 68.6 |  | 0.0 | 60.0 | 100.0 | 825.01 |
| Matching HypEx | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | SE/SD вне допуска | False | -0.0086 | 0.0564 | 1.247 | 68.6 |  | 0.0 | 60.0 | 100.0 | 826.07 |
| GA-страты -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0391 | 0.0589 | 0.83 | 65.7 | 65.7 | 20.0 | 80.0 | 80.0 | 152.04 |
| GA-страты -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0391 | 0.0589 | 0.83 | 65.7 | 65.7 | 20.0 | 80.0 | 80.0 | 150.27 |
| Регрессионная корректировка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0306 | 0.0604 | 0.722 | 63.9 | 63.9 | 20.0 | 80.0 | 80.0 | 5.23 |
| Регрессионная корректировка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0306 | 0.0604 | 0.722 | 63.9 | 63.9 | 20.0 | 80.0 | 80.0 | 5.38 |
| CUPED по контролю -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0235 | 0.0605 | 0.742 | 63.8 | 63.8 | 20.0 | 80.0 | 80.0 | 98.46 |
| CUPED по контролю -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0235 | 0.0605 | 0.742 | 63.8 | 63.8 | 20.0 | 80.0 | 80.0 | 101.25 |
| GA-страты -> CUPAC A -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0331 | 0.0606 | 0.731 | 63.7 | -5.9 | 20.0 | 80.0 | 80.0 | 165.39 |
| GA-страты -> CUPAC A -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0331 | 0.0606 | 0.731 | 63.7 | -5.9 | 20.0 | 80.0 | 80.0 | 161.87 |
| ML-RATE | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0315 | 0.0607 | 0.72 | 63.6 | 63.6 | 20.0 | 80.0 | 80.0 | 25.13 |
| ML-RATE | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0315 | 0.0607 | 0.72 | 63.6 | 63.6 | 20.0 | 80.0 | 80.0 | 24.91 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0291 | 0.0608 | 0.719 | 63.5 | 63.5 | 20.0 | 80.0 | 80.0 | 336.07 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0291 | 0.0608 | 0.719 | 63.5 | 63.5 | 20.0 | 80.0 | 80.0 | 336.82 |
| HypEx CUPED | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0244 | 0.0615 | 0.73 | 62.6 | 62.6 | 20.0 | 80.0 | 80.0 | 189.59 |
| HypEx CUPED | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0244 | 0.0615 | 0.73 | 62.6 | 62.6 | 20.0 | 80.0 | 80.0 | 187.79 |
| GA-страты -> CUPED -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0271 | 0.0627 | 0.713 | 61.1 | -13.3 | 20.0 | 80.0 | 80.0 | 151.91 |
| GA-страты -> CUPED -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0271 | 0.0627 | 0.713 | 61.1 | -13.3 | 20.0 | 80.0 | 80.0 | 154.27 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0411 | 0.0649 | 0.673 |  |  | 20.0 | 80.0 | 80.0 | 109.17 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0411 | 0.0649 | 0.673 |  | 0.0 | 20.0 | 80.0 | 80.0 | 107.78 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0411 | 0.0649 | 0.673 |  |  | 20.0 | 80.0 | 80.0 | 114.23 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0411 | 0.0649 | 0.673 |  | 0.0 | 20.0 | 80.0 | 80.0 | 115.22 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0415 | 0.0654 | 0.669 |  |  | 20.0 | 80.0 | 80.0 | 113.78 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0415 | 0.0654 | 0.669 |  |  | 20.0 | 80.0 | 80.0 | 120.43 |
| Квадратичный CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.045 | 0.0693 | 0.671 | 52.5 | 3.5 | 20.0 | 60.0 | 80.0 | 107.61 |
| Квадратичный CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.045 | 0.0693 | 0.671 | 52.5 | 3.5 | 20.0 | 60.0 | 80.0 | 102.46 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.045 | 0.0705 | 0.659 | 50.8 | 50.8 | 20.0 | 60.0 | 80.0 | 99.04 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.045 | 0.0705 | 0.659 | 50.8 | 50.8 | 20.0 | 60.0 | 80.0 | 107.89 |
| Panel-prior cupac_linear_A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0454 | 0.071 | 0.655 |  |  | 20.0 | 60.0 | 80.0 | 114.85 |
| Bootstrap-оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 31.48 |
| Перестановочная оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 105.64 |
| Разность средних с общей дисперсией | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0805 | 0.1005 | 0.667 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 1.24 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 92.47 |
| Welch t-test | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 |  | 20.0 | 40.0 | 80.0 | 1.5 |
| Welch t-test | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 |  | 20.0 | 40.0 | 80.0 | 1.37 |
| Перестановочная оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 103.24 |
| Разность средних с общей дисперсией | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0805 | 0.1005 | 0.667 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 1.15 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 94.49 |
| Bootstrap-оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0805 | 0.1005 | 0.668 | 0.0 | 0.0 | 20.0 | 40.0 | 80.0 | 30.43 |
| Воспроизводимая взвешенная разность средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0109 | 0.1597 | 0.632 | -152.1 | -152.1 | 20.0 | 60.0 | 80.0 | 2.82 |
| Воспроизводимая взвешенная разность средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0109 | 0.1597 | 0.632 | -152.1 | -152.1 | 20.0 | 60.0 | 80.0 | 2.92 |

## Ошибочно безопасный признак зоны B

Отрицательный контроль: экспертная ошибка должна проявиться смещением.

Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.

### Отбор по сопоставимым контурам

| data_scenario | race_information_mode | temporal_mode | estimand_id | candidate_count | eligible_count | selected_pipeline | selected_pipeline_name_ru | selected_empirical_sd_h1 | selected_variance_reduction_vs_welch | selected_median_runtime_ms | selection_status | selection_criterion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| misclassified_b | experiment_only | pre_treatment_only | sample_ate | 15 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| misclassified_b | panel_prior | pre_treatment_only | sample_ate | 16 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| misclassified_b | panel_prior | post_treatment_allowed | sample_ate | 6 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| misclassified_b | experiment_only | post_treatment_allowed | sample_ate | 4 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |

### Цепочка A/B/C

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0924 | 0.053 | 0.783 |  | 6.6 | 60.0 | 40.0 | 40.0 | 118.71 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0924 | 0.053 | 0.783 |  | 6.6 | 60.0 | 40.0 | 40.0 | 131.54 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0884 | 0.0548 | 0.786 |  |  | 60.0 | 40.0 | 40.0 | 112.92 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0884 | 0.0548 | 0.786 |  |  | 60.0 | 40.0 | 40.0 | 118.25 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0013 | 0.0619 | 0.742 | 33.7 | 33.7 | 0.0 | 80.0 | 100.0 | 115.51 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0013 | 0.0619 | 0.742 | 33.7 | 33.7 | 0.0 | 80.0 | 100.0 | 111.1 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | FPR выше допуска | False | -0.0045 | 0.0649 | 0.672 | 27.2 | 27.2 | 20.0 | 80.0 | 80.0 | 348.22 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | FPR выше допуска | False | -0.0045 | 0.0649 | 0.672 | 27.2 | 27.2 | 20.0 | 80.0 | 80.0 | 334.72 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 102.79 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 106.66 |

### Все пайплайны

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Глобальный matching оптимальным транспортом | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1506 | 0.0063 | 4.901 | 99.3 |  | 0.0 | 0.0 | 0.0 | 326.62 |
| Глобальный matching оптимальным транспортом | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1506 | 0.0063 | 4.901 | 99.3 |  | 0.0 | 0.0 | 0.0 | 302.28 |
| Matching ближайшего соседа | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1599 | 0.02 | 1.569 | 93.1 |  | 0.0 | 0.0 | 0.0 | 12.77 |
| Matching ближайшего соседа | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1599 | 0.02 | 1.569 | 93.1 |  | 0.0 | 0.0 | 0.0 | 11.75 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0921 | 0.0526 | 0.789 |  |  | 60.0 | 40.0 | 40.0 | 133.25 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0924 | 0.053 | 0.783 |  | 6.6 | 60.0 | 40.0 | 40.0 | 131.54 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0924 | 0.053 | 0.783 |  | 6.6 | 60.0 | 40.0 | 40.0 | 118.71 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0881 | 0.0545 | 0.791 |  |  | 60.0 | 40.0 | 40.0 | 125.35 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0884 | 0.0548 | 0.786 |  |  | 60.0 | 40.0 | 40.0 | 118.25 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0884 | 0.0548 | 0.786 |  |  | 60.0 | 40.0 | 40.0 | 112.92 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0913 | 0.0587 | 0.693 |  | 9.1 | 60.0 | 40.0 | 40.0 | 177.43 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0913 | 0.0587 | 0.693 |  | 9.1 | 60.0 | 40.0 | 40.0 | 170.74 |
| Panel-prior cupac_linear_A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0009 | 0.0615 | 0.748 |  |  | 0.0 | 80.0 | 100.0 | 113.59 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0872 | 0.0615 | 0.668 |  |  | 60.0 | 40.0 | 40.0 | 161.56 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.0872 | 0.0615 | 0.668 |  |  | 60.0 | 40.0 | 40.0 | 172.27 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0013 | 0.0619 | 0.742 | 33.7 | 33.7 | 0.0 | 80.0 | 100.0 | 115.51 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | SE/SD вне допуска | False | -0.0013 | 0.0619 | 0.742 | 33.7 | 33.7 | 0.0 | 80.0 | 100.0 | 111.1 |
| Квадратичный CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0013 | 0.0629 | 0.732 | 31.7 | -3.1 | 0.0 | 80.0 | 100.0 | 104.81 |
| Квадратичный CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | SE/SD вне допуска | False | -0.0013 | 0.0629 | 0.732 | 31.7 | -3.1 | 0.0 | 80.0 | 100.0 | 107.17 |
| Регрессионная корректировка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0012 | 0.0635 | 0.686 | 30.3 | 30.3 | 20.0 | 80.0 | 80.0 | 4.87 |
| Регрессионная корректировка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0012 | 0.0635 | 0.686 | 30.3 | 30.3 | 20.0 | 80.0 | 80.0 | 4.99 |
| ML-RATE | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0011 | 0.0639 | 0.681 | 29.3 | 29.3 | 20.0 | 80.0 | 80.0 | 25.48 |
| ML-RATE | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0011 | 0.0639 | 0.681 | 29.3 | 29.3 | 20.0 | 80.0 | 80.0 | 25.05 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | FPR выше допуска | False | -0.0045 | 0.0649 | 0.672 | 27.2 | 27.2 | 20.0 | 80.0 | 80.0 | 348.22 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | FPR выше допуска | False | -0.0045 | 0.0649 | 0.672 | 27.2 | 27.2 | 20.0 | 80.0 | 80.0 | 334.72 |
| GA-страты -> CUPAC A -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0004 | 0.0687 | 0.642 | 18.3 | 27.2 | 20.0 | 80.0 | 80.0 | 157.13 |
| GA-страты -> CUPAC A -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0004 | 0.0687 | 0.642 | 18.3 | 27.2 | 20.0 | 80.0 | 80.0 | 168.31 |
| Matching HypEx | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.002 | 0.0704 | 1.013 | 14.4 |  | 0.0 | 60.0 | 100.0 | 862.43 |
| Matching HypEx | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | -0.002 | 0.0704 | 1.013 | 14.4 |  | 0.0 | 60.0 | 100.0 | 856.86 |
| GA-страты -> CUPED -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0068 | 0.0719 | 0.621 | 10.7 | 20.4 | 20.0 | 80.0 | 80.0 | 157.42 |
| GA-страты -> CUPED -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0068 | 0.0719 | 0.621 | 10.7 | 20.4 | 20.0 | 80.0 | 80.0 | 146.94 |
| HypEx CUPED | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | FPR выше допуска | False | -0.009 | 0.0735 | 0.609 | 6.5 | 6.5 | 20.0 | 80.0 | 80.0 | 204.76 |
| HypEx CUPED | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | FPR выше допуска | False | -0.009 | 0.0735 | 0.609 | 6.5 | 6.5 | 20.0 | 80.0 | 80.0 | 204.36 |
| CUPED по контролю -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0092 | 0.0738 | 0.607 | 5.9 | 5.9 | 20.0 | 80.0 | 80.0 | 97.27 |
| CUPED по контролю -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | -0.0092 | 0.0738 | 0.607 | 5.9 | 5.9 | 20.0 | 80.0 | 80.0 | 102.06 |
| Welch t-test | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 |  | 0.0 | 60.0 | 100.0 | 1.47 |
| Bootstrap-оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 32.52 |
| Разность средних с общей дисперсией | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 1.28 |
| Перестановочная оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 113.28 |
| Bootstrap-оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 32.87 |
| Welch t-test | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 |  | 0.0 | 60.0 | 100.0 | 1.35 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 106.66 |
| Разность средних с общей дисперсией | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 1.33 |
| Перестановочная оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 108.32 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | прошёл диагностический допуск | False | 0.0059 | 0.0761 | 0.878 | 0.0 | 0.0 | 0.0 | 60.0 | 100.0 | 102.79 |
| GA-страты -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | 0.007 | 0.0806 | 0.604 | -12.2 | -12.2 | 20.0 | 80.0 | 80.0 | 145.11 |
| GA-страты -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | FPR выше допуска | False | 0.007 | 0.0806 | 0.604 | -12.2 | -12.2 | 20.0 | 80.0 | 80.0 | 154.91 |
| Воспроизводимая взвешенная разность средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0099 | 0.1689 | 0.594 | -393.4 | -393.4 | 20.0 | 20.0 | 80.0 | 3.11 |
| Воспроизводимая взвешенная разность средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | FPR выше допуска | False | -0.0099 | 0.1689 | 0.594 | -393.4 | -393.4 | 20.0 | 20.0 | 80.0 | 3.16 |

## Нерепрезентативная выборка и известная целевая популяция

Отдельная честная гонка методов, использующих известные доли страт и population moments.

Отбор не выполнен: недостаточно повторений для научного рейтинга. Результаты оставлены диагностическими.

### Отбор по сопоставимым контурам

| data_scenario | race_information_mode | temporal_mode | estimand_id | candidate_count | eligible_count | selected_pipeline | selected_pipeline_name_ru | selected_empirical_sd_h1 | selected_variance_reduction_vs_welch | selected_median_runtime_ms | selection_status | selection_criterion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_shifted_strata | experiment_only | pre_treatment_only | sample_ate | 15 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| population_shifted_strata | panel_prior | pre_treatment_only | sample_ate | 17 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| population_shifted_strata | panel_prior | post_treatment_allowed | sample_ate | 6 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| population_shifted_strata | experiment_only | post_treatment_allowed | sample_ate | 4 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |
| population_shifted_strata | panel_prior | pre_treatment_only | population_ate | 2 | 0 |  |  |  |  |  | недостаточно повторений для рейтинга | diagnostic screen + scientific sample size; min empirical SD, then runtime, then pipeline ID |

### Цепочка A/B/C

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 103.59 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 97.6 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1981 | 0.0509 | 0.976 |  |  | 0.0 | 100.0 | 0.0 | 112.44 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1981 | 0.0509 | 0.976 |  |  | 0.0 | 100.0 | 0.0 | 115.45 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2148 | 0.0527 | 0.989 | -22.8 | -22.8 | 0.0 | 100.0 | 0.0 | 109.56 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2148 | 0.0527 | 0.989 | -22.8 | -22.8 | 0.0 | 100.0 | 0.0 | 111.09 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1914 | 0.0532 | 0.91 |  | -9.0 | 0.0 | 100.0 | 0.0 | 123.07 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1914 | 0.0532 | 0.91 |  | -9.0 | 0.0 | 100.0 | 0.0 | 120.7 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.2318 | 0.0579 | 0.866 | -48.2 | -48.2 | 20.0 | 100.0 | 0.0 | 336.1 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.2318 | 0.0579 | 0.866 | -48.2 | -48.2 | 20.0 | 100.0 | 0.0 | 369.76 |

### Все пайплайны

| Пайплайн | Режим информации | Временной доступ | Estimand | Повторов | Контур | Диагностический статус | Допущен в рейтинг | Bias | SD | SE/SD | VR к Welch, % | Инкремент, % | FPR, % | Power, % | Coverage, % | Время, мс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Глобальный matching оптимальным транспортом | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1548 | 0.0305 | 1.167 | 58.9 |  | 0.0 | 0.0 | 0.0 | 378.41 |
| Глобальный matching оптимальным транспортом | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1548 | 0.0305 | 1.167 | 58.9 |  | 0.0 | 0.0 | 0.0 | 351.65 |
| Matching ближайшего соседа | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1549 | 0.0378 | 0.936 | 36.8 |  | 0.0 | 0.0 | 0.0 | 12.36 |
| Matching ближайшего соседа | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | -0.1549 | 0.0378 | 0.936 | 36.8 |  | 0.0 | 0.0 | 0.0 | 11.5 |
| Стратификация с долями экспериментальной выборки | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.1818 | 0.0399 | 1.662 | 29.7 | 29.7 | 0.0 | 100.0 | 0.0 | 3.81 |
| Bootstrap-оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 32.42 |
| Перестановочная оценка разности средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 106.05 |
| Разность средних с общей дисперсией | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1711 | 0.0476 | 1.478 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 1.09 |
| Welch t-test | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 |  | 0.0 | 100.0 | 20.0 | 1.4 |
| HypEx A/B (Welch) | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 97.6 |
| Перестановочная оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 110.79 |
| Разность средних с общей дисперсией | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1711 | 0.0476 | 1.478 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 1.33 |
| HypEx A/B (Welch) | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 103.59 |
| Welch t-test | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 |  | 0.0 | 100.0 | 20.0 | 1.53 |
| Bootstrap-оценка разности средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1711 | 0.0476 | 1.48 | 0.0 | 0.0 | 0.0 | 100.0 | 20.0 | 30.99 |
| Panel-prior cupac_linear_A_plus_B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1983 | 0.0508 | 0.977 |  |  | 0.0 | 100.0 | 0.0 | 126.48 |
| CUPAC A+B -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1981 | 0.0509 | 0.976 |  |  | 0.0 | 100.0 | 0.0 | 112.44 |
| CUPAC A+B -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1981 | 0.0509 | 0.976 |  |  | 0.0 | 100.0 | 0.0 | 115.45 |
| Panel-prior cupac_linear_A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.215 | 0.0527 | 0.99 |  |  | 0.0 | 100.0 | 0.0 | 120.37 |
| CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2148 | 0.0527 | 0.989 | -22.8 | -22.8 | 0.0 | 100.0 | 0.0 | 109.56 |
| CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2148 | 0.0527 | 0.989 | -22.8 | -22.8 | 0.0 | 100.0 | 0.0 | 111.09 |
| Panel-prior cupac_linear_A_plus_B_plus_C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1916 | 0.0531 | 0.911 |  |  | 0.0 | 100.0 | 0.0 | 124.43 |
| CUPAC A+B+C -> HypEx A/B | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1914 | 0.0532 | 0.91 |  | -9.0 | 0.0 | 100.0 | 0.0 | 123.07 |
| CUPAC A+B+C -> HypEx A/B | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1914 | 0.0532 | 0.91 |  | -9.0 | 0.0 | 100.0 | 0.0 | 120.7 |
| Квадратичный CUPAC A -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.2156 | 0.0536 | 0.973 | -26.9 | -3.4 | 0.0 | 100.0 | 0.0 | 112.74 |
| Квадратичный CUPAC A -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.2156 | 0.0536 | 0.973 | -26.9 | -3.4 | 0.0 | 100.0 | 0.0 | 110.92 |
| CUPED по контролю -> HypEx A/B | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2259 | 0.0562 | 0.916 | -39.4 | -39.4 | 20.0 | 100.0 | 0.0 | 102.13 |
| CUPED по контролю -> HypEx A/B | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2259 | 0.0562 | 0.916 | -39.4 | -39.4 | 20.0 | 100.0 | 0.0 | 101.94 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1995 | 0.0569 | 0.837 |  | 1.6 | 0.0 | 100.0 | 0.0 | 135.54 |
| GA-страты -> CUPAC A+B+C -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1995 | 0.0569 | 0.837 |  | 1.6 | 0.0 | 100.0 | 0.0 | 144.02 |
| ML-RATE | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.235 | 0.0572 | 0.877 | -44.3 | -44.3 | 20.0 | 100.0 | 0.0 | 24.77 |
| ML-RATE | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.235 | 0.0572 | 0.877 | -44.3 | -44.3 | 20.0 | 100.0 | 0.0 | 26.76 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | experiment_only | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.2113 | 0.0573 | 0.837 |  |  | 0.0 | 100.0 | 0.0 | 130.54 |
| GA-страты -> CUPAC A+B -> стратифицированная оценка | panel_prior | post_treatment_allowed | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.2113 | 0.0573 | 0.837 |  |  | 0.0 | 100.0 | 0.0 | 126.82 |
| HypEx CUPED | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.2254 | 0.0575 | 0.895 | -45.8 | -45.8 | 20.0 | 100.0 | 0.0 | 192.92 |
| HypEx CUPED | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.2254 | 0.0575 | 0.895 | -45.8 | -45.8 | 20.0 | 100.0 | 0.0 | 195.52 |
| HypEx temporal CUPAC | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.2318 | 0.0579 | 0.866 | -48.2 | -48.2 | 20.0 | 100.0 | 0.0 | 369.76 |
| HypEx temporal CUPAC | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.2318 | 0.0579 | 0.866 | -48.2 | -48.2 | 20.0 | 100.0 | 0.0 | 336.1 |
| Регрессионная корректировка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2343 | 0.0581 | 0.862 | -49.1 | -49.1 | 20.0 | 100.0 | 0.0 | 5.86 |
| Регрессионная корректировка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2343 | 0.0581 | 0.862 | -49.1 | -49.1 | 20.0 | 100.0 | 0.0 | 5.88 |
| GA-страты -> CUPED -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2254 | 0.0583 | 0.878 | -49.9 | 6.4 | 20.0 | 100.0 | 0.0 | 116.96 |
| GA-страты -> CUPED -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2254 | 0.0583 | 0.878 | -49.9 | 6.4 | 20.0 | 100.0 | 0.0 | 112.01 |
| GA-страты -> CUPAC A -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2271 | 0.0584 | 0.867 | -50.4 | 6.1 | 20.0 | 100.0 | 0.0 | 128.07 |
| GA-страты -> CUPAC A -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2271 | 0.0584 | 0.867 | -50.4 | 6.1 | 20.0 | 100.0 | 0.0 | 121.68 |
| GA-страты -> стратифицированная оценка | experiment_only | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2203 | 0.0602 | 0.907 | -60.2 | -60.2 | 0.0 | 100.0 | 0.0 | 114.63 |
| GA-страты -> стратифицированная оценка | panel_prior | pre_treatment_only | sample_ate | 5 | confirmatory | смещение выше допуска | False | 0.2203 | 0.0602 | 0.907 | -60.2 | -60.2 | 0.0 | 100.0 | 0.0 | 109.74 |
| Воспроизводимая взвешенная разность средних | experiment_only | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1717 | 0.0656 | 1.613 | -89.8 | -89.8 | 0.0 | 100.0 | 60.0 | 3.53 |
| Воспроизводимая взвешенная разность средних | panel_prior | pre_treatment_only | sample_ate | 5 | exploratory | смещение выше допуска | False | 0.1717 | 0.0656 | 1.613 | -89.8 | -89.8 | 0.0 | 100.0 | 60.0 | 3.24 |
| Matching HypEx | experiment_only | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.2338 | 0.0664 | 1.221 | -94.6 |  | 0.0 | 100.0 | 0.0 | 856.08 |
| Matching HypEx | panel_prior | pre_treatment_only | sample_ate | 5 | integration_parity | смещение выше допуска | False | 0.2338 | 0.0664 | 1.221 | -94.6 |  | 0.0 | 100.0 | 0.0 | 871.27 |
| Entropy calibration по известным population moments | panel_prior | pre_treatment_only | population_ate | 5 | confirmatory | другой estimand | False | -0.072 | 0.0745 | 1.279 |  |  | 0.0 | 0.0 | 80.0 | 6.91 |
| Постстратификация по известной популяции | panel_prior | pre_treatment_only | population_ate | 5 | confirmatory | другой estimand | False | -0.0628 | 0.0755 | 1.253 |  |  | 20.0 | 0.0 | 80.0 | 3.71 |

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
