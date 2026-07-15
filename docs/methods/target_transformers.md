# Трансформации target

`TargetTransformer` не является самостоятельным estimator. Он создаёт новый
target и передаёт его базовому estimator через `WrappedEstimator`.

| ID | Вход и выход | Fit-период | Сложность | Статус и ограничение |
| --- | --- | --- | --- | --- |
| `cuped` | Target + одна covariate → adjusted target | Historical или явно заданный reference | `O(N)` | Реализован; covariate должна быть pre-treatment |
| `cupac_linear` | Target + feature matrix → adjusted target | Historical/reference controls | `O(Nd²+d³)` | Реализован; линейная/ridge модель |
| `sklearn_cupac_A` | Pre-treatment feature set A → adjusted target | Historical/reference controls | Зависит от модели | Реализован |
| `sklearn_cupac_A_plus_B` | A + safe in-time B → adjusted target | Historical/reference controls | Зависит от модели | Реализован; B задаётся экспертной политикой |
| `sklearn_cupac_A_plus_B_plus_C` | A+B + balance-gated C → adjusted target | Historical/reference controls | Зависит от модели | Реализован; C требует заранее заданного gate |
| `winsorized` | Target → clipped target | Reference quantiles | `O(N log N)` | Реализован; меняет estimand |
| `nonlinear_cupac` | Target + features → adjusted target | Только независимые historical/reference outcomes | Зависит от модели | Заглушка: отобран кандидат R&D |
| `bayesian_cupac` | Target + posterior prediction → adjusted target | Требует отдельного контракта | Зависит от модели | Заглушка: нужна формализация |

## CUPED

ID: `cuped`. Категория: `TargetTransformer`.

Использует одну pre-experiment covariate:

\[
Y^*=Y-\theta(X-E[X]),\qquad \theta=\frac{Cov(Y,X)}{Var(X)}.
\]

При корректной pre-treatment covariate математическое ожидание эффекта не
меняется, а variance может уменьшиться. Нулевая variance covariate приводит к
`theta=0`, а не к делению на ноль.

## CUPAC

ID: `cupac_linear` и RnD6-варианты `sklearn_cupac_A`,
`sklearn_cupac_A_plus_B`, `sklearn_cupac_A_plus_B_plus_C`.

CUPAC строит прогноз target по набору признаков и центрированно вычитает его.
Feature sets имеют явную политику безопасности:

- A: только pre-treatment признаки;
- A+B: A и экспертно безопасные in-time признаки;
- A+B+C: дополнительно признаки, прошедшие balance gate;
- unsafe demo: только для демонстрации leakage, opt-in.

Референсный `hypex_cupac` использует временной контракт PyPI HypEx. Для target
и каждого cofounder нужны два согласованных pre-experiment среза: `lag=2`
служит обучающей выборкой для прогноза `lag=1`, а предсказание на `lag=1`
корректирует target текущего эксперимента. В синтетическом генераторе такие
колонки имеют суффикс `__lag2`. Текущий экспериментальный outcome не участвует
в обучении prognostic-модели. Если временной контракт недоступен, адаптер явно
переходит к локальному CUPAC, обученному на переданной historical-выборке, и
записывает причину fallback в diagnostics.

## Нелинейный CUPAC: кандидат

ID: `nonlinear_cupac`. Категория: `TargetTransformer`. Статус: заглушка,
исполняемого алгоритма пока нет.

Предполагаемый метод обучает нелинейную prognostic-модель
`m(X) ≈ E[Y|X]` только на независимых historical/reference outcomes. В
эксперименте он применяет центрированную control-variate коррекцию по прогнозу
`m(X)`. Кандидат полезен, когда связь pre-treatment features с outcome заметно
нелинейна и `cupac_linear` недообъясняет шум.

Перед реализацией необходимо зафиксировать model API, модель по умолчанию,
ограничения памяти и runtime, tuning protocol, обработку missing values и схему
честной оценки. Подбор модели или гиперпараметров по outcome анализируемого
эксперимента запрещён. В обычную гонку метод не включён до появления реализации
и тестов на bias, FPR и power.

## Winsorization

ID: `winsorized`. Значения target ограничиваются quantiles, обученными на
reference data. Метод меняет estimand для тяжёлых хвостов, поэтому результат
следует интерпретировать как эффект на winsorized metric.
