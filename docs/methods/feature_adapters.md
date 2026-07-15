# Адаптеры признаков

## NumericFeatureBinner

| Поле | Значение |
| --- | --- |
| ID / категория | `numeric_feature_binner` / `FeatureAdapter` |
| Вход / выход | Числовые колонки → новые `pandas.Categorical` колонки с suffix `__bin` |
| Цель | Стабильное дискретное представление для design preprocessing |
| Период данных | `fit` только на reference; target не используется |
| Сложность | Quantile `O(N log N)`, uniform/custom `O(N)`; transform `O(N log B)` |
| Статус | Реализован |
| Ограничения | Одномерные независимые bins; совместные зависимости обрабатывает следующий preprocessor |

```python
from varwar import NumericFeatureBinner

binner = NumericFeatureBinner(
    columns=("age", "income"),
    strategy="quantile",
    n_bins={"age": 8, "income": 12},
)
binner.fit(history)
history_with_bins = binner.augment(history)
experiment_with_bins = binner.augment(experiment)
```

Стратегии:

- `quantile`: интервалы содержат близкое число reference-наблюдений;
- `uniform`: равная ширина между reference minimum и maximum;
- `custom`: пользователь передаёт строго возрастающие внутренние границы.

Границы после `fit` не меняются. Будущие экстремальные значения попадают в
крайние интервалы `(-inf, ...)` и `(..., +inf)`. `NaN`, строки, которые нельзя
перевести в число, и бесконечности получают категорию `missing`. Повторяющиеся
quantile boundaries удаляются; фактическое число интервалов доступно в
`diagnostics()`.

Adapter не изменяет исходные числовые колонки. Его output можно передать в
`GeneticStrataReducer.category_cols` вместе с исходными категориями.
