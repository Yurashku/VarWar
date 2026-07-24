# Полный нумерованный реестр VarWar

Номер — удобная ссылка в документах и Excel, а `ID` — стабильный программный
идентификатор. В основной гонке сравниваются только итоговые конвейеры с
одинаковой целевой величиной.

Статусы:

- **реализован** — есть исполняемый код и тесты;
- **интеграция** — адаптер к внешнему пакету с контролируемым fallback;
- **кандидат R&D** — постановка понятна, но исполняемого кода ещё нет;
- **нужна спецификация** — перед реализацией надо зафиксировать estimand или
  статистический контракт;
- **legacy** — код сохранён, но исключён из активной гонки;
- **исключён** — отдельный метод не обоснован.

## Оценивание эффекта

| № | ID | Русское название | Статус | Роль в R&D |
| --- | --- | --- | --- | --- |
| E01 | `welch_mean_diff` | Разность средних с тестом Уэлча | реализован | основной baseline ATE |
| E02 | `hypex_ab` | A/B-тест HypEx, по умолчанию Welch | интеграция | проверка паритета с PyPI HypEx |
| E03 | `regression_adjustment` | Регрессионная корректировка | реализован | основной конкурент ATE |
| E04 | `ml_rate` | ML-RATE с cross-fitting | реализован | основной конкурент ATE |
| E05 | `stratified_mean_diff` | Стратифицированная разность средних | реализован | итоговый estimator для стратификации |
| E06 | `ratio_delta` | Delta-оценка разности ratio-of-means | реализован | отдельная гонка ratio-метрик |
| E07 | `nearest_neighbor_matching` | Matching ближайшего соседа | реализован | специализированный ATT baseline |
| E08 | `variance_weighted_stratified` | Страты с весами обратной дисперсии | реализован | исследовательский precision-weighted estimand |
| E09 | `wald_late` | Оценка Уолда для LATE | реализован | отдельная IV-задача |
| E10 | `calibration_reweighting` | Калибровочное перевзвешивание | нужна спецификация | перенос на популяцию с известными totals |
| E11 | `optimal_transport_matching` | Matching оптимальным транспортом | нужна спецификация | наблюдательные данные |
| E12 | `pooled_mean_diff` | Разность средних с общей дисперсией Стьюдента | legacy | диагностика допущения равных дисперсий |
| E13 | `trimmed_mean_diff` | Разность усечённых средних | legacy | заменяется композицией трансформации |
| E14 | `winsorized_mean_diff` | Разность винзоризированных средних | legacy | заменяется `winsorized` → estimator |

## Статистические тесты

| № | ID | Русское название | Статус | Роль в R&D |
| --- | --- | --- | --- | --- |
| T01 | `ks_test` | Двухвыборочный тест Колмогорова-Смирнова | реализован | различие распределений |
| T02 | `approx_ks_fast` | Быстрый приближённый KS-тест | реализован | benchmark точность/скорость |
| T03 | `energy_test` | Energy distance test | реализован | различие распределений |
| T04 | `randomized_t_test` | Рандомизированный t-тест «ёжик» | реализован | small-sample R&D |
| T05 | `permutation_mean_test` | Перестановочный тест разности средних | реализован | randomization inference |
| T06 | `anti_nuclear_ks_test` | KS-тест с контролируемым размыканием совпадений | кандидат R&D | tie-aware проверка с permutation calibration |
| X01 | `optimal_transport_check` | Неопределённая transport-диагностика | исключён | отдельная статистика не задана |

## Трансформации target

| № | ID | Русское название | Статус | Роль в R&D |
| --- | --- | --- | --- | --- |
| V01 | `cuped` | CUPED по одной pre-treatment ковариате | реализован | базовое снижение дисперсии |
| V02 | `cupac_linear` | Линейный CUPAC, общий движок | реализован | параметризуемая реализация семейства |
| V03 | `sklearn_cupac_A` | Линейный CUPAC, зона A | реализован | отдельный примитив: только pre-treatment |
| V04 | `sklearn_cupac_A_plus_B` | Линейный CUPAC, зоны A+B | реализован | отдельный примитив: плюс safe in-time |
| V05 | `sklearn_cupac_A_plus_B_plus_C` | Линейный CUPAC, зоны A+B+C | реализован | отдельный примитив: плюс balance-gated |
| V06 | `nonlinear_cupac` | Нелинейный CUPAC | кандидат R&D | исторически обученная нелинейная модель |
| V07 | `robust_cuped` | Робастный CUPED | кандидат R&D | устойчивый коэффициент для тяжёлых хвостов |
| V08 | `winsorized` | Винзоризация target | реализован | меняет целевую метрику |
| V09 | `trimming` | Усечение хвостов target | кандидат R&D | нужен контракт удаления строк |
| V10 | `bayesian_cupac` | Байесовская CUPAC-корректировка | нужна спецификация | низкоприоритетная гипотеза |

## Преобразование исходной метрики

| № | ID | Русское название | Статус | Роль в R&D |
| --- | --- | --- | --- | --- |
| M01 | `ratio_linearization` | Линеаризация ratio-метрики | кандидат R&D | будущая композиция с CUPED/CUPAC |

## Признаки и дизайн

| № | ID | Категория | Русское название | Статус |
| --- | --- | --- | --- | --- |
| F01 | `numeric_feature_binner` | `FeatureAdapter` | Биннинг числовых признаков | реализован |
| D01 | `atomic_strata` | `DesignPreprocessor` | Полные атомарные страты | реализован baseline |
| D02 | `random_reduction` | `DesignPreprocessor` | Случайное укрупнение страт | реализован baseline |
| D03 | `mean_sorted_reduction` | `DesignPreprocessor` | Укрупнение по historical mean | реализован baseline |
| D04 | `genetic_strata_reducer` | `DesignPreprocessor` | Генетическое укрупнение страт | реализован |
| D05 | `prognostic_score_block_randomization` | `DesignPreprocessor` | Блочная рандомизация по prognostic score | кандидат R&D |
| O01 | `ab_proportional` | `OptimizationObjective` | Дисперсионная objective для пропорционального A/B | реализована, внутренняя |

## Коррекции вывода

| № | ID | Русское название | Статус |
| --- | --- | --- | --- |
| C01 | `bonferroni` | Поправка Бонферрони | реализован |
| C02 | `holm_bonferroni` | Пошаговая поправка Холма-Бонферрони | реализован |
| C03 | `romano_wolf_stepdown` | Пошаговая поправка Романо-Вольфа | реализован |

## Интеграционные профили

| № | ID | Русское название | Статус |
| --- | --- | --- | --- |
| Q01 | `hypex_matcher` | Профиль интеграции HypEx Matching | нужна спецификация; дублирует семейство matching |

## Готовые конвейеры

| № | ID | Состав | Статус |
| --- | --- | --- | --- |
| P01 | `cuped+hypex_ab` | CUPED VarWar → HypEx Welch A/B | интеграция |
| P02 | `hypex_cupac` | temporal linear CUPAC HypEx → HypEx Welch A/B | интеграция |
| P03 | `sklearn_cupac_A+hypex_ab` | CUPAC A → HypEx Welch A/B | интеграция |
| P04 | `sklearn_cupac_A_plus_B+hypex_ab` | CUPAC A+B → HypEx Welch A/B | интеграция |
| P05 | `sklearn_cupac_A_plus_B_plus_C+hypex_ab` | CUPAC A+B+C → HypEx Welch A/B | интеграция |
| P06 | `winsorized+hypex_ab` | Winsorization → HypEx Welch A/B | интеграция; другой estimand |
| P07 | `genetic_strata_ab` | GA reducer с `ab_proportional` → стратифицированный анализ | реализован профиль |

## Покрытие старой таблицы и приложенных заметок

- t-test и Welch покрыты E01/E02; pooled-вариант сохранён только как E12.
- robust estimation конкретизирован как V07, V08 и V09, а не оставлен одним
  неясным названием.
- stratified estimation покрыт E05; генетический алгоритм — D04, а не estimator.
- variance-weighted estimator покрыт E08 с явным предупреждением об estimand.
- MLRATE покрыт E04.
- HypEx matcher осознанно оставлен Q01 до фиксации отличий от E07.
- optimal transport matching сохранён E11; безымянный transport check исключён X01.
- CUPED и все три зоны CUPAC сохранены как V01 и V03–V05.
- Bayesian CUPAC сохранён V10, но не считается готовым методом.
- идеи Robust CUPED, Stratified CUPED и CUPAC with CV из приложенных заметок
  покрываются соответственно V07, композициями на листе конвейеров и обычным
  честным model-selection внутри CUPAC; некорректные демонстрационные реализации
  не переносятся.
