# Полный нумерованный реестр VarWar

Этот файл перечисляет все публичные методы из `varwar.method_catalog`.
В основной R&D-гонке сравниваются полные пайплайны с одинаковым estimand.
Примитивы сравниваются между собой только в ablation-анализе.

Статусы:

- **реализован** — есть исполняемый код и тесты;
- **интеграция** — адаптер к внешнему пакету с контролируемым fallback;
- **legacy** — код сохранён, но исключён из активной гонки;
- **кандидат R&D** — постановка понятна, реализация ещё не допущена;
- **нужна спецификация** — не зафиксированы estimand или inference;
- **исключён** — старое имя не обозначает самостоятельный метод.

## Estimator

| № | ID | Русское название | Статус | Назначение |
| --- | --- | --- | --- | --- |
| E01 | `welch_mean_diff` | Разность средних с тестом Уэлча | реализован | основной ATE baseline |
| E02 | `hypex_ab` | A/B-тест HypEx, по умолчанию Welch | интеграция | сверка с PyPI HypEx |
| E03 | `regression_adjustment` | Регрессионная корректировка | реализован | ATE с HC1 inference |
| E04 | `ml_rate` | ML-RATE | реализован | cross-fitted ковариатная корректировка |
| E05 | `stratified_mean_diff` | Стратифицированная разность средних | реализован | итоговый estimator после подготовки страт |
| E06 | `variance_weighted_stratified` | Обратное дисперсионное взвешивание страт | реализован | отдельный precision-weighted estimand |
| E07 | `nearest_neighbor_matching` | Matching ближайшего соседа | реализован | наблюдательный ATT baseline |
| E08 | `ratio_delta` | Delta-оценка ratio-of-means | реализован | отдельный контур ratio-метрик |
| E09 | `wald_late` | Оценка Уолда для LATE | реализован | отдельный IV-контур |
| E10 | `calibration_reweighting` | Калибровочное перевзвешивание | нужна спецификация | нужны population totals |
| E11 | `optimal_transport_matching` | Matching оптимальным транспортом | нужна спецификация | отдельный наблюдательный контур ATT/ATC |
| E12 | `pooled_mean_diff` | Разность средних с общей дисперсией | legacy | диагностика допущения равных дисперсий |
| E13 | `trimmed_mean_diff` | Разность усечённых средних | legacy | заменяется композицией трансформации |
| E14 | `winsorized_mean_diff` | Разность винзоризированных средних | legacy | заменяется композицией трансформации |

## HypothesisTest

| № | ID | Русское название | Статус |
| --- | --- | --- | --- |
| T01 | `ks_test` | Двухвыборочный тест Колмогорова-Смирнова | реализован |
| T02 | `approx_ks_fast` | Быстрый приближённый KS-тест | реализован |
| T03 | `energy_test` | Тест энергетического расстояния | реализован |
| T04 | `randomized_t_test` | Рандомизированный t-тест «ёжик» | реализован |
| T05 | `permutation_mean_test` | Перестановочный тест разности средних | реализован |
| T06 | `anti_nuclear_ks_test` | Tie-aware KS с контролируемым размыканием совпадений | кандидат R&D |
| T07 | `optimal_transport_homogeneity_test` | OT-тест однородности распределений | кандидат R&D |
| X01 | `optimal_transport_check` | Старое неопределённое OT-имя | исключён |

OT-тест однородности и OT-matching решают разные задачи. Первый проверяет
различие распределений, второй должен оценивать причинный эффект в
наблюдательном контуре, поэтому T07 и E11 разделены.

## TargetTransformer

| № | ID | Русское название | Статус | Контракт |
| --- | --- | --- | --- | --- |
| V01 | `cuped` | CUPED с явной политикой fit-данных | реализован | общий движок |
| V02 | `cuped_control` | CUPED по текущему контролю | реализован | theta по control, применение к обеим группам |
| V03 | `cupac_linear` | Temporal linear CUPAC | реализован | lag2 features → lag1 pre-target |
| V04 | `linear_intime_adjustment` | Линейная in-time корректировка | реализован | second-stage fit по текущему контролю |
| V05 | `cupac_linear_A` | CUPAC A | реализован | только pre-treatment temporal CUPAC |
| V06 | `cupac_linear_A_plus_B` | CUPAC A+B | реализован | CUPAC A → отдельный linear B |
| V07 | `cupac_linear_A_plus_B_plus_C` | CUPAC A+B+C | реализован | A → B → gated linear C |
| V08 | `cupac_quadratic_A` | Квадратичный temporal CUPAC A | реализован | линейные, квадратные и парные признаки |
| V09 | `nonlinear_cupac` | Общий нелинейный CUPAC | кандидат R&D | шире ограниченной V08 модели |
| V10 | `robust_cuped` | Робастный CUPED | кандидат R&D | тяжёлые хвосты |
| V11 | `winsorized` | Винзоризация target | реализован | меняет целевую метрику |
| V12 | `trimming` | Усечение хвостов target | кандидат R&D | нужен row-filter/inference контракт |
| V13 | `bayesian_cupac` | Байесовский CUPAC | нужна спецификация | нужен posterior estimand |

### Точное значение зон A/B/C

- **A** — признаки, измеренные до assignment/exposure/trigger. Они используются
  temporal CUPAC-моделью `lag2 → lag1`.
- **B** — экспертно очевидные safe in-time признаки. Они не добавляются в
  CUPAC A, а образуют отдельную линейную second-stage корректировку.
- **C** — предварительно очищенные от outcome, leakage и mediator-risk in-time
  признаки. Они образуют следующий линейный этап только после проверки SMD,
  разности missing-rate и KS-расстояния.
- Balance gate не является доказательством причинной безопасности.
- D (DAG-required), E (mediator-risk) и F (leakage/outcome-derived) не
  допускаются в основной estimator.

## MetricTransformer

| № | ID | Русское название | Статус |
| --- | --- | --- | --- |
| M01 | `ratio_linearization` | Линеаризация ratio-метрики | кандидат R&D |

## FeatureAdapter и DesignPreprocessor

| № | ID | Категория | Русское название | Статус |
| --- | --- | --- | --- | --- |
| F01 | `numeric_feature_binner` | FeatureAdapter | Биннинг числовых признаков | реализован |
| D01 | `atomic_strata` | DesignPreprocessor | Полные атомарные страты | реализован |
| D02 | `random_reduction` | DesignPreprocessor | Случайное укрупнение страт | реализован |
| D03 | `mean_sorted_reduction` | DesignPreprocessor | Укрупнение по pre-period среднему | реализован |
| D04 | `genetic_strata_reducer` | DesignPreprocessor | Генетическое укрупнение страт | реализован |
| D05 | `prognostic_score_block_randomization` | DesignPreprocessor | Блочная рандомизация | кандидат R&D |
| O01 | `ab_proportional` | OptimizationObjective | Критерий GA для естественного A/B-трафика | реализован |

GA обучается на pre-experiment target доступного датасета и не является
самостоятельным estimator.

## InferenceCorrection и ConfigurationProfile

| № | ID | Категория | Русское название | Статус |
| --- | --- | --- | --- | --- |
| C01 | `bonferroni` | InferenceCorrection | Поправка Бонферрони | реализован |
| C02 | `holm_bonferroni` | InferenceCorrection | Поправка Холма-Бонферрони | реализован |
| C03 | `romano_wolf_stepdown` | InferenceCorrection | Step-down Романо-Вольфа | реализован |
| Q01 | `hypex_matcher` | ConfigurationProfile | Профиль интеграции HypEx Matching | нужна спецификация |

## CompositePipeline

| № | ID | Состав | Контур |
| --- | --- | --- | --- |
| P00 | `welch` | `welch_mean_diff` без преобразований | подтверждающий baseline |
| P01 | `hypex_cuped` | native HypEx CUPED → HypEx Welch | parity |
| P02 | `cuped_control__hypex_ab` | CUPED по control → HypEx Welch | подтверждающий |
| P03 | `hypex_cupac_temporal` | native temporal CUPAC → HypEx Welch | parity |
| P04 | `cupac_linear_A__hypex_ab` | temporal CUPAC A → HypEx Welch | подтверждающий |
| P05 | `cupac_linear_A_plus_B__hypex_ab` | CUPAC A → linear B → HypEx Welch | исследовательский |
| P06 | `cupac_linear_A_plus_B_plus_C__hypex_ab` | A → B → gated C → HypEx Welch | исследовательский |
| P07 | `cupac_quadratic_A__hypex_ab` | quadratic temporal CUPAC A → HypEx Welch | исследовательский |
| P08 | `winsorized__hypex_ab` | control-winsorization → HypEx Welch | другой estimand |
| P09 | `ga_strata__stratified` | GA → стратифицированная оценка | подтверждающий |
| P10 | `ga_strata__cuped_control__stratified` | GA → CUPED по control → стратифицированная оценка | подтверждающий |
| P11 | `ga_strata__cupac_linear_A__stratified` | GA → CUPAC A → стратифицированная оценка | подтверждающий |
| P12 | `ga_strata__cupac_linear_A_plus_B__stratified` | GA → CUPAC A → linear B → стратифицированная оценка | исследовательский |
| P13 | `ga_strata__cupac_linear_A_plus_B_plus_C__stratified` | GA → A → B → gated C → стратифицированная оценка | исследовательский |

Эти строки генерируются функциями из `varwar.pipelines`, а их точный состав
сохраняется вместе с каждым запуском гонки.
