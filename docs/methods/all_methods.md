# Полный реестр методов VarWar

Ниже перечислены все публичные R&D-компоненты и именованные участники гонок.
Внутренние функции агрегации, валидации и сериализации в список не входят.
Машиночитаемый источник тех же ID, категорий и статусов находится в
`varwar.method_catalog`.

## Как читать статус

- **Реализован для R&D**: контракт реализован, код покрыт автоматическими тестами.
  Статус не означает завершённую production-валидацию на внутренних данных.
- **Интеграционный адаптер реализован**: локальный контракт и fallback готовы,
  основной расчёт связан с PyPI HypEx.
- **Заглушка: кандидат R&D**: метод отобран как потенциально конкурентный, но
  исполняемого алгоритма пока нет.
- **Заглушка: нужна формализация**: названия недостаточно; до реализации нужно
  зафиксировать estimand, алгоритм и inference policy.

## Estimator

| ID | Статус | Назначение | Участие |
| --- | --- | --- | --- |
| `welch_mean_diff` | Реализован для R&D | Разность средних и Welch inference; канонический baseline без variance reduction | estimator race |
| `hypex_ab` | Интеграционный адаптер реализован | A/B baseline HypEx с явным локальным fallback | estimator race |
| `pooled_mean_diff` | Реализован для R&D | Разность средних при предположении равных дисперсий | estimator race |
| `ks_test` | Реализован для R&D | Точный двухвыборочный тест Колмогорова-Смирнова | estimator race |
| `approx_ks_fast` | Реализован для R&D | Быстрое histogram-приближение KS | estimator race |
| `energy_test` | Реализован для R&D | Двухвыборочный energy-distance test с permutation p-value | estimator race |
| `randomized_t_test` | Реализован для R&D | Рандомизированная «ёжиковая» t-статистика | estimator race |
| `permutation_mean_test` | Реализован для R&D | Monte Carlo randomization test разности средних | estimator race |
| `regression_adjustment` | Реализован для R&D | OLS-коэффициент treatment с HC1 standard error | estimator race |
| `ml_rate` | Реализован для R&D | Cross-fitted ridge nuisance prediction и robust adjustment | estimator race |
| `nearest_neighbor_matching` | Реализован для R&D | Простой nearest-neighbor ATT baseline | estimator race |
| `ratio_delta` | Реализован для R&D | Разность ratio-метрик с delta-method inference | estimator race |
| `stratified_mean_diff` | Реализован для R&D | Population-weighted среднее эффектов внутри страт | обе гонки |
| `variance_weighted_stratified` | Реализован для R&D | Inverse-variance комбинация эффектов внутри страт | estimator race |
| `trimmed_mean_diff` | Реализован для R&D | Разность trimmed means для тяжёлых хвостов | estimator race |
| `winsorized_mean_diff` | Реализован для R&D | Разность winsorized means | estimator race |
| `wald_late` | Реализован для R&D | Wald ratio для LATE при instrumented take-up | estimator race |
| `anti_nuclear_ks_test` | Заглушка: нужна формализация | Точная статистика по названию из таблицы не определена | не запускается |
| `hypex_matcher` | Заглушка: нужна формализация | Требуются matching policy, estimand и inference contract | не запускается |
| `calibration_reweighting` | Заглушка: нужна формализация | Требуются calibration totals, constraints и variance estimator | не запускается |
| `optimal_transport_check` | Заглушка: нужна формализация | Пока это идея диагностики распределений, а не определённый effect estimator | не запускается |
| `optimal_transport_matching` | Заглушка: нужна формализация | Требуются distance, regularization и inference policy | не запускается |

## TargetTransformer

| ID | Статус | Назначение | Данные для fit |
| --- | --- | --- | --- |
| `cuped` | Реализован для R&D | Линейная коррекция target по одной pre-experiment covariate | historical/reference или явно допустимая схема fit |
| `cupac_linear` | Реализован для R&D | Линейный/ridge prognostic adjustment по набору признаков | historical/reference controls |
| `sklearn_cupac_A` | Реализован для R&D | CUPAC только на pre-treatment features | historical/reference controls |
| `sklearn_cupac_A_plus_B` | Реализован для R&D | CUPAC на A и экспертно безопасных in-time features B | historical/reference controls |
| `sklearn_cupac_A_plus_B_plus_C` | Реализован для R&D | CUPAC на A+B и прошедших balance gate features C | historical/reference controls |
| `winsorized` | Реализован для R&D | Ограничение target по reference quantiles | reference data |
| `bayesian_cupac` | Заглушка: нужна формализация | Bayesian prognostic adjustment без зафиксированных model/posterior contracts | не запускается |
| `nonlinear_cupac` | Заглушка: кандидат R&D | Нелинейная prognostic-модель с центрированным control-variate adjustment | только независимые historical/reference outcomes |

## FeatureAdapter

| ID | Статус | Назначение | Участие |
| --- | --- | --- | --- |
| `numeric_feature_binner` | Реализован для R&D | Фиксированные quantile, uniform или custom bins числовых признаков | перед design preprocessing |

## DesignPreprocessor

| ID | Статус | Назначение | Участие |
| --- | --- | --- | --- |
| `atomic_strata` | Реализован для R&D | Baseline без укрупнения атомарных страт | stratification race |
| `random_reduction` | Реализован для R&D | Случайное сбалансированное укрупнение | stratification race |
| `mean_sorted_reduction` | Реализован для R&D | Детерминированное укрупнение по historical means | stratification race |
| `genetic_strata_reducer` | Реализован для R&D | GA-укрупнение атомарных страт до фиксированного числа макрострат | stratification race |
| `ab_proportional` | Реализован для R&D | Objective для естественного пропорционального A/B-трафика | GA и diagnostics |
| `neyman` | Реализован для R&D | Objective и точная integer Neyman allocation | GA и stratification race |
| `genetic_strata_ab` | Реализован для R&D | Конфигурация GA с `ab_proportional` | stratification race |
| `genetic_strata_neyman` | Реализован для R&D | Конфигурация GA с `neyman` | stratification race |
| `prognostic_score_block_randomization` | Заглушка: кандидат R&D | Рандомизация treatment внутри блоков близкого pre-treatment prognostic score | только prospective experiment design |

## InferenceCorrection

| ID | Статус | Назначение | Участие |
| --- | --- | --- | --- |
| `bonferroni` | Реализован для R&D | Одношаговый FWER control | после inference |
| `holm_bonferroni` | Реализован для R&D | Step-down FWER control | после inference |
| `romano_wolf_stepdown` | Реализован для R&D | Step-down max-statistic correction | после caller-defined bootstrap |

## CompositePipeline

| ID | Статус | Состав | Участие |
| --- | --- | --- | --- |
| `cuped+hypex_ab` | Интеграционный адаптер реализован | CUPED → HypEx A/B | обе гонки |
| `hypex_cupac` | Интеграционный адаптер реализован | PyPI HypEx CUPAC parity slot с локальным fallback | обе гонки |
| `sklearn_cupac_A+hypex_ab` | Интеграционный адаптер реализован | CUPAC A → HypEx A/B | обе гонки |
| `sklearn_cupac_A_plus_B+hypex_ab` | Интеграционный адаптер реализован | CUPAC A+B → HypEx A/B | обе гонки |
| `sklearn_cupac_A_plus_B_plus_C+hypex_ab` | Интеграционный адаптер реализован | CUPAC A+B+C → HypEx A/B | обе гонки |
| `winsorized+hypex_ab` | Интеграционный адаптер реализован | Winsorization → HypEx A/B | estimator race |

Stratification race дополнительно строит композиции вида
`preprocessor+target_transformer+stratified_mean_diff`. Это производные
pipeline-конфигурации, а не новые статистические методы, поэтому каждая возможная
комбинация отдельной строкой в реестре не дублируется.

## Отбор кандидатов из дополнительного обзора

Из [обзора variance-reduction techniques](https://j-sephb-lt-n.github.io/exploring_statistics/cuped_cupac_and_other_variance_reduction_techniques.html)
в backlog добавлены только два метода:

- `nonlinear_cupac`: может улучшить объяснённую долю variance при нелинейной
  зависимости outcome от pre-treatment features;
- `prognostic_score_block_randomization`: уменьшает исходный дисбаланс ещё на
  стадии назначения групп и потому потенциально дополняет post-randomization
  estimators.

Второй кандидат формулируется как **рандомизация внутри блоков**, а не как
детерминированное чередование treatment: это сохраняет валидный randomization
design и требует соответствующего inference.

Не добавлены отдельные дубликаты simple difference, linear CUPAC, regression
adjustment, outcome stratification и nearest-neighbor matching: эти семейства уже
есть в VarWar. Propensity-score matching/weighting не выделены в приоритет для
обычного рандомизированного A/B, где assignment probability известна по дизайну.
CCA/PCA control-variate варианты не добавлены как отдельные методы: для
одномерного target их полезная линейная часть покрывается `cupac_linear` и
`regression_adjustment`. Вариант, обучающий transformation на outcome текущего
эксперимента без честного разделения данных, исключён из-за leakage.
