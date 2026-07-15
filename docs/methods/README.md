# Каталог методов VarWar

Этот каталог разделяет статистические методы по роли в экспериментальном
pipeline. Одинаковая терминология используется в коде, документации и R&D
таблицах.

| Категория | Что делает | Контракт | Примеры |
| --- | --- | --- | --- |
| `Estimator` | Оценивает эффект или проверяет гипотезу | `estimate(data, spec)` | HypEx AB, Welch, stratified mean |
| `TargetTransformer` | Меняет target перед оценкой | `fit_transform(data, spec, auxiliary_data)` | CUPED, CUPAC, winsorization |
| `DesignPreprocessor` | Готовит дизайн или страты | `fit(reference); transform(experiment)` | `genetic_strata_reducer` |
| `FeatureAdapter` | Меняет представление признаков без target | `fit(reference); transform(data)` | `numeric_feature_binner` |
| `InferenceCorrection` | Корректирует множественный вывод | функция над p-values/statistics | Bonferroni, Romano-Wolf |
| `CompositePipeline` | Соединяет совместимые компоненты | adapter → preprocessor → transformer → estimator | binning → GA → CUPED → stratified mean |

## Навигация

- [Полный реестр методов и статусов](all_methods.md)
- [Эстиматоры](estimators.md)
- [Трансформации target](target_transformers.md)
- [Предобработка дизайна](design_preprocessors.md)
- [Генетическое укрупнение страт](genetic_strata_reducer.md)
- [Адаптеры признаков](feature_adapters.md)
- [Коррекции статистического вывода](inference_corrections.md)
- [Композиция pipeline](pipelines.md)

## Правило безопасности данных

`Estimator` видит экспериментальный outcome. `TargetTransformer` может
использовать outcome в рамках явно определённой схемы fit. `DesignPreprocessor`
и `FeatureAdapter` обучаются только на historical/pre-experiment данных и не
имеют права подбирать параметры по экспериментальному outcome.

`run_estimator_race` сравнивает эстиматоры и target transformations.
`run_stratification_race` отдельно сравнивает design preprocessing pipelines,
чтобы время обучения и качество разбиения не смешивались со временем inference.
Точкой отсчёта без variance reduction является `welch_mean_diff`; HypEx строки
остаются обязательными integration/parity baselines.
