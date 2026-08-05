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
| E03 | `randomized_weighted_mean_diff` | Разность нормированных ёжиковых средних | реализован | взвешенный ATE с seed и согласованным inference |
| E04 | `permutation_mean_diff` | Перестановочная оценка разности средних | реализован | permutation p-value и Welch CI |
| E05 | `bootstrap_mean_diff` | Bootstrap-оценка разности средних | реализован | bootstrap SE и percentile CI |
| E06 | `regression_adjustment` | Регрессионная корректировка | реализован | ATE с HC1 inference |
| E07 | `ml_rate` | ML-RATE | реализован | cross-fitted ковариатная корректировка |
| E08 | `stratified_mean_diff` | Стратифицированная разность средних | реализован | итоговый estimator после подготовки страт |
| E09 | `variance_weighted_stratified` | Обратное дисперсионное взвешивание страт | реализован | отдельный precision-weighted estimand |
| E10 | `nearest_neighbor_matching` | Matching ближайшего соседа | реализован | наблюдательный ATT baseline |
| E11 | `ratio_delta` | Delta-оценка ratio-of-means | реализован | отдельный контур ratio-метрик |
| E12 | `wald_late` | Оценка Уолда для LATE | реализован | отдельный IV-контур |
| E13 | `calibration_reweighting` | Калибровочное перевзвешивание | нужна спецификация | нужны population totals |
| E14 | `optimal_transport_matching` | Matching оптимальным транспортом | нужна спецификация | отдельный наблюдательный контур ATT/ATC |
| E15 | `pooled_mean_diff` | Разность средних с общей дисперсией | legacy | диагностика допущения равных дисперсий |
| E16 | `trimmed_mean_diff` | Разность усечённых средних | legacy | заменяется композицией трансформации |
| E17 | `winsorized_mean_diff` | Разность винзоризированных средних | legacy | заменяется композицией трансформации |

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

## FeatureAdapter и RandomizationStructureBuilder

| № | ID | Категория | Русское название | Статус |
| --- | --- | --- | --- | --- |
| F01 | `numeric_feature_binner` | FeatureAdapter | Биннинг числовых признаков | реализован |
| D01 | `atomic_strata` | RandomizationStructureBuilder | Полные атомарные страты | реализован |
| D02 | `random_reduction` | RandomizationStructureBuilder | Случайное укрупнение страт | реализован |
| D03 | `mean_sorted_reduction` | RandomizationStructureBuilder | Укрупнение по pre-period среднему | реализован |
| D04 | `genetic_strata_reducer` | RandomizationStructureBuilder | Генетическое укрупнение страт | реализован |
| D05 | `prognostic_blocks` | RandomizationStructureBuilder | Блоки по pre-treatment prognostic score | реализован |
| D06 | `prognostic_score_block_randomization` | ConfigurationProfile | Старое имя профиля; заменено на `prognostic_blocks` + assignment | исключён |
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
## Дополнительные записи текущего каталога

| ID | Категория | Название | Статус |
| --- | --- | --- | --- |
| `population_weighted_stratified` | PopulationGeneralizationEstimator | Стратифицированная оценка с известными долями целевой популяции | реализован |
| `hypex_matching` | Estimator | HypEx Matching с извлечением сырой таблицы MatchingAnalyzer | интеграция |
| `isolation_forest_trimming` | TargetTransformer | Treatment-blind отсечка аномалий Isolation Forest | реализован, изменённая метрика |
| `isolation_forest_winsorization` | TargetTransformer | Treatment-blind ограничение аномалий Isolation Forest | реализован, изменённая метрика |
| `panel_prior_cupac_A` | TargetTransformer | CUPAC A с prior из статистик исторической панели | реализован |
| `panel_prior_cupac_A_plus_B` | TargetTransformer | Panel-prior CUPAC A с отдельной стадией B | реализован |
| `panel_prior_cupac_A_plus_B_plus_C` | TargetTransformer | Panel-prior CUPAC A, B и balance-gated C | реализован |
| `panel_prior_cupac_A__hypex_ab` | CompositePipeline | Panel-prior CUPAC A -> HypEx Welch | интеграция |
| `panel_prior_cupac_A_plus_B__hypex_ab` | CompositePipeline | Panel-prior CUPAC A+B -> HypEx Welch | интеграция |
| `panel_prior_cupac_A_plus_B_plus_C__hypex_ab` | CompositePipeline | Panel-prior CUPAC A+B+C -> HypEx Welch | интеграция |

## Актуальная матрица активной RCT-гонки

Следующие записи имеют исполняемый маршрут в текущей синтетической гонке.
Статус `реализован` означает наличие кода и unit/smoke-проверок; он не означает
автоматически production-ready статистическую гарантию.

| Группа | Пайплайны | `experiment_only` | `panel_prior` | Временной режим |
| --- | --- | --- | --- | --- |
| Базовые оценки | `welch`, `hypex_ab`, `pooled_mean_diff`, `randomized_weighted_mean_diff` | да | да, без использования prior | pre-only |
| Рандомизационный вывод | `permutation_mean_diff`, `bootstrap_mean_diff` | да | да, без использования prior | pre-only |
| Коррекция target | `hypex_cuped`, `cuped_control__hypex_ab`, `hypex_cupac_temporal`, `cupac_linear_A__hypex_ab` | да | да, без использования prior | pre-only |
| In-time-коррекция | `cupac_linear_A_plus_B__hypex_ab`, `cupac_linear_A_plus_B_plus_C__hypex_ab` | да | да, без использования prior | post-allowed |
| Регрессия и matching | `regression_adjustment`, `ml_rate`, `nearest_neighbor_matching`, `hypex_matching`, `optimal_transport_matching` | да | да, без использования prior | pre-only |
| Panel-prior CUPAC | `panel_prior_cupac_A__hypex_ab` | нет | да | pre-only |
| Panel-prior CUPAC + in-time | `panel_prior_cupac_A_plus_B__hypex_ab`, `panel_prior_cupac_A_plus_B_plus_C__hypex_ab` | нет | да | post-allowed |
| Стратификация | `ga_strata__*__stratified` | да | да, без использования prior | по компонентам |
| Population generalization | `population_weighted_stratified`, `calibration_reweighting` | нет | только сценарий с заданными population totals | pre-only |

В отчёте `race_information_mode` показывает фактически выбранный режим
гонки, а `information_mode` в каталоге показывает минимальный контракт
пайплайна. Поэтому обычный Welch может появиться и в строке `panel_prior`:
он не использует prior, но участвует в честной паре сравнений с методами,
которым prior разрешён.

Полное произведение совместимых компонентов формирует канонические ID вида
`adapt.<id>__structure.<id>__assign.<id>__target.<id>__analysis.<id>`.
Пайплайны не являются отдельными методами: их матрица хранится в Parquet и на
листе `Пайплайны` итоговой книги.

## Канонический реестр R&D v3

Пайплайны больше не считаются самостоятельными методами. Их ID строится из
компонентов, а ниже перечислены новые исполняемые примитивы и неделимые пары
`point estimator + inference`. Слова `A_plus_B` и `A_plus_B_plus_C` обозначают
отдельные последовательные стадии внутри каждого семейства CUPAC.

| Категория | ID |
| --- | --- |
| Механизм назначения | `complete_randomization`, `stratified_randomization`, `rerandomization_mahalanobis`, `rerandomization_max_smd`, `rerandomization_mixed_balance`, `hypex_aa_best_of_k` |
| Структура рандомизации | `none`, `atomic_strata`, `prognostic_blocks`, `genetic_strata_reducer` |
| CUPED | `cuped_historical`, `cuped_huber_historical` |
| CUPAC Auto-ML | `cupac_automl_A`, `cupac_automl_A_plus_B`, `cupac_automl_A_plus_B_plus_C` |
| CUPAC linear | `cupac_linear_A`, `cupac_linear_A_plus_B`, `cupac_linear_A_plus_B_plus_C` |
| CUPAC quadratic | `cupac_quadratic_A`, `cupac_quadratic_A_plus_B`, `cupac_quadratic_A_plus_B_plus_C` |
| CUPAC panel prior | `panel_prior_cupac_A`, `panel_prior_cupac_A_plus_B`, `panel_prior_cupac_A_plus_B_plus_C` |
| CUPAC HypEx-compatible | `cupac_hypex_compatible_A`, `cupac_hypex_compatible_A_plus_B`, `cupac_hypex_compatible_A_plus_B_plus_C` |
| Недоступный target-only HypEx API | `cupac_hypex_official_target_A`, `cupac_hypex_official_target_A_plus_B`, `cupac_hypex_official_target_A_plus_B_plus_C` |
| Бейзлайн target | `identity` |
| Разность средних | `diff_means__welch`, `diff_means__student_pooled`, `diff_means__design_based_normal`, `diff_means__welch_studentized_randomization`, `diff_means__bootstrap_t`, `diff_means__sharp_null_randomization` |
| Регрессионная корректировка | `lin_regression__hc2`, `lin_regression__hc3`, `ml_rate__crossfit_hc2` |
| HypEx parity | `hypex_ab__integration`, `hypex_cuped__integration`, `hypex_cupac_A__integration`, `hypex_matching_ate__integration` |
| Стратификация | `stratified_ate__native`, `common_effect_inverse_variance_stratified__native` |
| Обобщение на популяцию | `population_weighted_stratified_ate__native`, `calibration_weighted_population_ate__native` |
| Matching ATE | `nearest_neighbor_matching_ate__reuse_aware`, `optimal_transport_matching_ate__point`, `optimal_transport_matching_ate__sharp_null_ri` |
| Ёжиковые веса | `hedgehog_weighted_ate__native` |
| Устойчивые native effects | `yuen_trimmed_effect`, `winsorized_effect_bootstrap`, `trimmed_mean__yuen`, `winsorized_mean__full_refit_bootstrap`, `iforest_trimmed_mean__full_refit`, `iforest_winsorized_mean__full_refit` |
| Устойчивые кандидаты исходного среднего | `median_of_means_ate`, `catoni_mean_ate`, `median_of_means_ate__bootstrap`, `catoni_mean_ate__bootstrap` |
