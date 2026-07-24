# Предобработка дизайна

`DesignPreprocessor` подготавливает эксперимент до итогового оценивания. Он
может строить выборку, блоки, страты, mapping или назначение treatment, но не
обязан выполнять все эти действия сразу.

| ID | Что делает | Использует target | Статус |
| --- | --- | --- | --- |
| `atomic_strata` | Оставляет все комбинации категорий отдельными стратами | только historical target для диагностики | baseline |
| `random_reduction` | Случайно объединяет атомарные страты | нет при transform | baseline |
| `mean_sorted_reduction` | Объединяет атомарные страты, отсортированные по historical mean | только historical target | baseline |
| `genetic_strata_reducer` | Оптимизирует объединение атомарных страт | только historical/reference target | реализован |
| `prognostic_score_block_randomization` | Формирует блоки по честному прогнозу и рандомизирует внутри блоков | только historical/reference target | кандидат |

`ab_proportional` — не метод выбора A/B-групп. Это внутренняя целевая функция
генетического алгоритма для естественного пропорционального трафика:

\[
J=\sum_g\frac{N_g}{N}S_g^2.
\]

`genetic_strata_ab` — имя готового профиля гонки
`genetic_strata_reducer(objective=ab_proportional)`, а не ещё один алгоритм.
Режим Неймана перемещён в `legacy/neyman/` и не входит в активные R&D-прогоны.
