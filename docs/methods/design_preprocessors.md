# Предобработка дизайна

`DesignPreprocessor` формирует структуру эксперимента до статистического
вывода. Он не возвращает effect, standard error или p-value.

## Genetic strata reducer

| Поле | Значение |
| --- | --- |
| ID / категория | `genetic_strata_reducer` / `DesignPreprocessor` |
| Вход / выход | Категории + historical target → frozen atom-to-macrostratum mapping |
| Цель | Минимизировать `ab_proportional` или `neyman` variance objective |
| Estimand | Отсутствует; его задаёт следующий estimator |
| Период данных | Fit на historical/pre-experiment; transform не читает outcome |
| Сложность | `O(N)` aggregation + примерно `O(PGH)` search |
| Статус | Оба objective и integer Neyman allocation реализованы и протестированы |
| Ограничения | GA не гарантирует global optimum; неизвестные комбинации требуют явной policy |

`fit_summary()` переиспользует готовые атомарные статистики; после агрегации
runtime и память поиска зависят от `H`, а не от числа исходных строк `N`.

Полное описание: [genetic_strata_reducer.md](genetic_strata_reducer.md).

## Prognostic-score block randomization: кандидат

| Поле | Значение |
| --- | --- |
| ID / категория | `prognostic_score_block_randomization` / `DesignPreprocessor` |
| Вход / выход | Historical model + pre-treatment features экспериментальных единиц → randomized assignment |
| Цель | Снизить случайный дисбаланс по прогнозу baseline outcome до наблюдения экспериментального outcome |
| Период данных | Model fit только на historical/pre-experiment outcomes |
| Статус | Заглушка: отобран кандидат R&D, исполняемого алгоритма нет |
| Ограничения | Только prospective design; для уже назначенных групп неприменим |

Предполагаемая постановка: обучить prognostic-модель на истории, получить score
для будущих единиц, сформировать малые блоки близких score и **случайно** назначить
treatment внутри каждого блока с заданной allocation ratio. Детерминированное
чередование не используется, потому что метод должен сохранять известный
randomization mechanism.

До реализации нужно определить размер и остаточную политику блоков, поддержку
несимметричных allocation ratios и clusters, seed contract, balance diagnostics
и randomization-aware inference. Кандидат должен участвовать в отдельной гонке
design pipelines, а не в `run_estimator_race`.

## Почему это не estimator

Один reducer можно соединить со stratified mean, CUPED + stratified mean,
CUPAC + stratified mean или другим estimator. Если записать GA как estimator,
эта композиция потеряется, а runtime поиска будет ошибочно сравниваться с
runtime статистической формулы.
