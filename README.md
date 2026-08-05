# VarWar

VarWar - исследовательский стенд для сравнения **полных процедур анализа
рандомизированных экспериментов** на синтетике с известной истиной. Старые
прототипы сохранены в `legacy/`; активный код разделяет построение структуры
рандомизации, назначение воздействия, преобразование target и согласованную
пару «точечная оценка + статистический вывод».

## Что реализовано

- Механизмы назначения: полная и стратифицированная рандомизация,
  rerandomization по Mahalanobis, max-SMD и mixed balance, а также интеграция
  с `HypEx.AATest`.
- Структуры рандомизации: атомарные страты, prognostic-score blocks и
  генетическое укрупнение страт.
- Анализ среднего эффекта: Welch и pooled Student, design-based normal,
  studentized permutation, bootstrap-t, Lin HC2/HC3, ML-RATE, стратифицированные
  оценки, calibration weighting, nearest-neighbor и optimal-transport ATE.
- Преобразования: historical и Huber CUPED; linear, quadratic, Auto-ML,
  panel-prior и HypEx-compatible CUPAC в вариантах A, A+B и A+B+C.
- Устойчивые дорожки: Yuen trimmed effect, winsorized effect, median-of-means,
  Catoni и чувствительность к Isolation Forest.
- Интеграционные бейзлайны PyPI `hypex==1.0.6`: AB, CUPED, CUPAC и Matching.
- Численная проверка по известной истине: bias, RMSE, empirical SD, SE/SD,
  FPR с интервалом Wilson, super-uniformity p-value, coverage, power, ошибка
  второго рода, MDE, время и RSS-память.

## Быстрый запуск

```bash
python -m pip install -e ".[dev,ml]"
python -m unittest discover -s tests -v
python examples/run_rnd_v3.py --stage smoke
powershell -File examples/build_rnd_v3_workbook.ps1
```

Smoke проверяет исполнение и не создаёт научный рейтинг. Screening требует не
менее 200 общих реализаций, confirmation - новых seed и 1000 реализаций для
финалистов. Полные trial-level результаты сохраняются в Parquet; Excel содержит
компактные агрегаты, диагностику и фильтруемый дашборд.

## Архитектура гонки

```text
FeatureAdapter
-> RandomizationStructureBuilder
-> AssignmentMechanism
-> TargetTransformer
-> AnalysisMethod(PointEstimator + InferenceProcedure + contract)
```

Канонический ID процедуры:

```text
adapt.<id>__structure.<id>__assign.<id>__target.<id>__analysis.<id>
```

Гонка разделена по estimand и по двум независимым измерениям доступа:

- `experiment_only` или `panel_prior`;
- `pre_treatment_only` или `full_timeline` (`pre + in-time + post`).

Методы A+B и A+B+C запускаются в общей матрице, но явно отмечаются как
`conditional` или `post_treatment_risk`. Несопоставимые estimand никогда не
ранжируются вместе. Для trimming/winsorization отдельно показываются качество
оценки собственного функционала и смещение относительно исходного ATE.

## Навигация

- [Каталог и терминология](docs/methods/README.md)
- [Полный реестр методов](docs/methods/all_methods.md)
- [Научный дизайн R&D](docs/methods/scientific_rnd_design.md)
- [Численная проверка статистического вывода](docs/methods/statistical_validation.md)
- [CUPED/CUPAC и временные зоны](docs/methods/target_transformers.md)
- [Генетическое укрупнение страт](docs/methods/genetic_strata_reducer.md)
- [Panel prior](docs/methods/panel_prior.md)

Основной пакет находится в `src/varwar/`, пользовательские команды - в
`examples/`, тесты - в `tests/`, прежние материалы - в `legacy/`.

## Канонические результаты

- `results/rnd_v3/smoke/` - техническая проверка 28 маршрутов, без рейтинга.
- `results/rnd_v3/screening/` - 200 реализаций утверждённого ядра на первом
  Gaussian-мире.
- `results/rnd_v3/confirmation/` - независимые 1000 реализаций пяти
  финалистов; подробный русский отчёт находится в `report_ru.md`.
- `outputs/rnd-v3/VarWar_RnD_v3.xlsx` - фильтруемый Excel по confirmation.

Текущий confirmation подтверждает только мир `gaussian_r2_04_n200`. Он не
заменяет будущую confirmation по тяжёлым хвостам, нелинейности, temporal leakage,
population shift и prior mismatch.
