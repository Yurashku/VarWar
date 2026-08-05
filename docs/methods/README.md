# Каталог методов VarWar

Сравниваемой единицей является полная экспериментальная процедура, а не
изолированный алгоритм. Компоненты процедуры имеют разные роли.

| Категория | Роль | Бейзлайн |
| --- | --- | --- |
| `FeatureAdapter` | Меняет представление признаков, не используя target | `identity` |
| `RandomizationStructureBuilder` | Строит страты, блоки или prognostic score до назначения | `none` |
| `AssignmentMechanism` | Назначает treatment; критерий баланса является его внутренней частью | `complete_randomization` |
| `TargetTransformer` | Корректирует наблюдаемый target перед анализом | `identity` |
| `AnalysisMethod` | Неделимая пара точечной оценки и допустимого статистического вывода | `diff_means__welch` |
| `InferenceCorrection` | Корректирует семейство статистических выводов | без коррекции |

`Estimator` и `HypothesisTest` сохранены для совместимости со старым API.
Каноническая гонка v3 использует `AnalysisMethod`: свободно переставлять способы
построения CI и p-value между оценщиками нельзя.

## CUPED и CUPAC

- Historical CUPED оценивает коэффициент только по переходу `lag2 -> lag1` и
  не использует outcome эксперимента для подбора коррекции.
- Trial CUPED представлен как Lin regression adjustment с HC2/HC3, а не как
  target-only обёртка с наивным t-test.
- CUPAC A обучает прогностическую модель на pre-period данных.
- A+B добавляет отдельную cross-fitted стадию B на разрешённых in-time
  признаках.
- A+B+C добавляет отдельную стадию C. Она запускается в `full_timeline`, но
  получает явную маркировку повышенного риска утечки или медиаторного смещения.
- Linear, quadratic, Auto-ML, panel-prior и HypEx-compatible семейства имеют
  все три варианта зон. Auto-ML выбирает модель только по pre-period проверке.

## Как читать результаты

1. Сначала сравните только строки одного `estimand`, информационного режима и
   временного доступа.
2. Проверьте `scientific_sample_size_ok` и `inference_quality`.
3. Затем смотрите bias, FPR, super-uniformity p-value, coverage и SE/SD.
4. Только прошедшие этот экран сравнивайте по empirical SD, RMSE, power, MDE,
   длине интервала, времени и памяти.
5. Smoke хранит все числа, но имеет предварительный статус и не выбирает
   лидеров.

FPR около 0.05 желателен. Сильно больший FPR означает антиконсервативный вывод;
слишком малый FPR обычно означает консервативность и потерю мощности. Для FPR и
coverage используются интервалы Wilson, поэтому пять прогонов не трактуются
так же, как тысяча.

## Документы

- [Полный нумерованный реестр](all_methods.md)
- [Научный дизайн R&D](scientific_rnd_design.md)
- [Статистическая валидация по ground truth](statistical_validation.md)
- [Оценивание эффекта](estimators.md)
- [CUPED, CUPAC и устойчивые преобразования](target_transformers.md)
- [Построение структуры рандомизации](design_preprocessors.md)
- [Генетическое укрупнение страт](genetic_strata_reducer.md)
- [Адаптеры признаков](feature_adapters.md)
- [Panel prior](panel_prior.md)
- [Допустимые композиции](pipelines.md)

Ratio, IV/LATE, тесты распределений и observational-сценарии остаются вне
активного рейтинга ATE. Anti-nuclear KS используется только внутри mixed
rerandomization как исследовательская tie-диагностика.
