# Эстиматоры и статистические тесты

Все строки имеют категорию `Estimator` и возвращают `EstimateResult`. Колонка
«Данные» явно отмечает допустимые дополнительные признаки; экспериментальные
`treatment` и `target` требуются всегда.

| ID | Цель или estimand | Данные | Доп. период | Сложность | Статус и главное ограничение |
| --- | --- | --- | --- | --- | --- |
| `welch_mean_diff` | ATE, unequal variance | Не менее 2 строк на arm | Не нужен | `O(N)` | Канонический baseline; closed-form Welch inference |
| `hypex_ab` | Разность средних | Две группы | Не нужен | `O(N)` | Integration/parity baseline через PyPI HypEx; fallback явно отмечается в diagnostics |
| `pooled_mean_diff` | Разность средних | Не менее 2 строк на arm | Не нужен | `O(N)` | Реализован; требует равных дисперсий |
| `regression_adjustment` | Treatment coefficient | Числовые covariates | Только pre-treatment | `O(Nd²+d³)` | Реализован с HC1; линейная спецификация |
| `ml_rate` | Cross-fitted adjusted ATE | Числовые covariates | Только pre-treatment | Зависит от folds и ridge fit | Реализован; нужен достаточный размер folds |
| `stratified_mean_diff` | Взвешенный ATE внутри страт | Stratum column и optional population weights | Страты строятся до теста | `O(N+U)` | Реализован; Neyman design требует `N_g/N`, sparse strata дают ошибку, opt-in `drop_renormalize` перенормирует веса |
| `variance_weighted_stratified` | Inverse-variance summary | Готовая stratum column | Страты строятся до теста | `O(N+U)` | Реализован; estimand зависит от precision weights |
| `ratio_delta` | Разность ratio metrics | Numerator, denominator | Не нужен | `O(N)` | Реализован; delta approximation нестабилен около нулевого denominator mean |
| `wald_late` | LATE для compliers | Instrument, received treatment | Не нужен | `O(N)` | Реализован; требует ненулевую first stage |
| `nearest_neighbor_matching` | ATT | Числовые covariates | Только pre-treatment | `O(N_tN_cd)` | Реализован как baseline; без replacement/transport inference |
| `trimmed_mean_diff` | Разность trimmed means | Числовой target | Не нужен | `O(N log N)` | Реализован; меняет estimand |
| `winsorized_mean_diff` | Разность winsorized means | Числовой target | Не нужен | `O(N log N)` | Реализован; меняет estimand |
| `ks_test` | Расстояние CDF и p-value | Две независимые выборки | Не нужен | `O(N log N)` | Реализован через SciPy; не является ATE estimator |
| `approx_ks_fast` | Histogram approximation KS | `n_bins` | Не нужен | `O(N+B)` | Реализован; accuracy зависит от bins |
| `energy_test` | Energy distance | Resample count и seed | Не нужен | Зависит от resamples | Реализован; permutation runtime задаёт пользователь |
| `randomized_t_test` | Hedgehog-weighted t statistic | Две числовые выборки | Не нужен | `O(N)` | Реализован; p-value использует Welch degrees of freedom |
| `permutation_mean_test` | Randomization test mean difference | Resample count и seed | Не нужен | `O(RN)` | Реализован memory-bounded; Monte Carlo error зависит от `R` |

Методы, математическая постановка которых пока не идентифицирована, остаются
явными `PlaceholderEstimator` и не имитируют готовую реализацию. Полный список
готовых методов и заглушек находится в [общем реестре](all_methods.md).
