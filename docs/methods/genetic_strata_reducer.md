# Генетическое укрупнение страт

| Поле | Значение |
| --- | --- |
| ID / категория | `genetic_strata_reducer` / `DesignPreprocessor` |
| Статус | Реализован: A/B objective, Neyman objective, fit/transform, holdout diagnostics |
| Вход | Категориальные ключи и historical/pre-experiment target |
| Выход | Mapping, macro summaries, fitness, convergence history и diagnostics |
| Estimand | Отсутствует; reducer готовит design для следующего estimator |

## Задача

Пусть несколько категориальных колонок образуют `H` наблюдаемых комбинаций.
Каждая комбинация является атомарной стратой: её можно целиком перенести в
макрострату, но нельзя разделить. Требуется получить ровно
`U = min(n_strata, H)` непустых макрострат.

`U` трактуется как точное целевое число. Условие «не более U» без отдельного
штрафа вырождено: чистый variance objective почти всегда выбирает максимально
допустимое число страт.

Fit использует historical/pre-experiment target. Mapping после обучения зависит
только от категориальных ключей и поэтому может быть применён до чтения outcome
эксперимента.

## Достаточные статистики

Для атома `j` сохраняются только:

\[
N_j,\qquad \mu_j,\qquad M_{2j}=\sum_i(Y_i-\mu_j)^2.
\]

Для объединения `g`:

\[
N_g=\sum_{j\in g}N_j,\qquad
\mu_g=\frac{\sum_{j\in g}N_j\mu_j}{N_g},
\]

\[
M_{2g}=\sum_{j\in g}\left[M_{2j}+N_j(\mu_j-\mu_g)^2\right],
\qquad S_g^2=\frac{M_{2g}}{N_g-1}.
\]

Исходные строки и списки ID после агрегации не хранятся.

## Objective для естественного A/B-трафика

\[
J_{AB}=\sum_g\frac{N_g}{N}S_g^2.
\]

При плановом sample size `n` и treatment share `p` приближённая variance
стратифицированной оценки эффекта:

\[
Var(\hat\tau)\approx\frac{J_{AB}}{n p(1-p)}.
\]

Это основной режим `ABProportionalVarianceObjective`.

## Objective Неймана

При возможности выбирать число наблюдений из каждой макростраты непрерывный
оптимум имеет вид:

\[
n_g\propto N_gS_g.
\]

`NeymanVarianceObjective` оценивает фактическую integer allocation. Она имеет
точную сумму `sample_size`, нижнюю границу `min_sample_per_stratum` и cap
`n_g <= N_g`. Сначала решается bounded continuous allocation, затем остаток
распределяется по максимальному marginal variance decrease. Стоимость
`O(U log U)`, а не `O(sample_size)`.

При finite population correction fitness равен:

\[
J_N=\frac{1}{N^2}\sum_g N_g^2
\left(1-\frac{n_g}{N_g}\right)\frac{S_g^2}{n_g}.
\]

В diagnostics сохраняется continuous lower bound, но GA оптимизирует integer
fitness.

В `run_stratification_race` строки с Neyman objective применяют рассчитанные
квоты к экспериментальным данным случайной выборкой только по stratum. Target
и treatment при выборе строк не читаются. Если наблюдаемая страта не может
заполнить квоту, pipeline явно завершается ошибкой вместо изменения design.
Итоговый stratified estimator использует population weights `N_g/N` из
reference summary, а не диспропорциональные sample shares `n_g/n`.

## Grouping genetic algorithm

```text
1. Построить атомарные count/mean/M2.
2. Создать warm starts и сбалансированные random partitions.
3. Посчитать fitness каждого уникального partition.
4. Перенести elite в следующее поколение.
5. Выбрать родителей tournament selection.
6. Внедрить целые группы первого родителя во второго.
7. Применить move или merge-split mutation.
8. Repair: убрать пустые/малые группы и восстановить ровно U групп.
9. Повторять до max_generations или patience без улучшения.
```

Метки канонизируются по первому появлению. Поэтому одинаковое разбиение с
метками `[0, 0, 1]` и `[7, 7, 3]` имеет один cache key.

## API

```python
from varwar import ABProportionalVarianceObjective, GeneticStrataReducer

reducer = GeneticStrataReducer(
    category_cols=("region", "device", "age__bin"),
    reference_target_col="y_pre",
    n_strata=12,
    objective=ABProportionalVarianceObjective(),
    random_state=42,
)
reducer.fit(history, validation_data=history_holdout)
experiment = reducer.augment(experiment)
spec = spec.with_stratum("stratum")
```

Если атомарные статистики уже рассчитаны, их можно переиспользовать между
несколькими objective и seed без повторного прохода по исходным строкам:

```python
from varwar import AtomicStrataSummary

summary = AtomicStrataSummary.from_frame(history, category_cols, "y_pre")
reducer.fit_summary(summary)
```

`mapping_` можно сохранить как Parquet. `macro_summary_` содержит count, mean,
variance и, в режиме Неймана, allocation. `result_` содержит историю
сходимости, train/validation score и число фактических fitness evaluations.

## Неизвестные комбинации

Default `unknown_policy="error"` запрещает незаметно назначать новые
комбинации. Режим `largest` отправляет их в крупнейшую reference-макрострату.
R&D race использует этот fallback явно, потому что редкие комбинации могут
отсутствовать в history даже при неизменном data-generating process.

## Ограничения

- GA не гарантирует global optimum на больших задачах.
- Historical target должен быть информативен для будущего outcome.
- Малые или дрейфующие макростраты могут быть непригодны для inference даже при
  хорошем historical fitness.
- Несколько target metrics пока не объединяются в один multi-objective score.
