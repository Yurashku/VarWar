# CUPED, CUPAC и преобразования target

`TargetTransformer` сам не оценивает эффект. Он формирует скорректированный
target, который передаётся совместимому `AnalysisMethod`. Если статистический
вывод использует resampling, вся outcome-adaptive цепочка переобучается внутри
каждого resample.

## CUPED

Для заранее измеренной ковариаты X классическая коррекция имеет вид:

$$
Y^* = Y - \theta(X - \bar X),
\qquad
\theta = \frac{Cov(Y, X)}{Var(X)}.
$$

### `cuped_historical`

Коэффициент фиксируется по историческому переходу `lag2 -> lag1`, то есть до
назначения воздействия. Затем одна и та же центрированная коррекция применяется
к обеим группам. Это основной причинно безопасный target-only вариант.

### `cuped_huber_historical`

Та же историческая схема, но наклон оценивается Huber-регрессией. Метод защищает
коэффициент от отдельных тяжёлых исторических выбросов; он не превращает сам
ATE в «робастный estimand» и потому всё равно проверяется по исходному ATE.

Предупреждение: коэффициент `cuped_historical` — это AR-наклон
pre-периода, а не `Cov(Y, X)/Var(X)` для текущего outcome. Гарантии
эффективности у него нет. Если связь `lag2 -> lag1` не переносится на `Y`,
коррекция увеличивает дисперсию: в `strong_prior_mismatch`
`Var(Y*)/Var(Y)` ≈ 1.8 (медиана 300 реализаций).

### `cuped_pooled_theta`

Классический CUPED (Deng et al., 2013): `theta = Cov(Y, X)/Var(X)` по всем
строкам текущего эксперимента, обе группы вместе, `X = pre_target_lag1`.
Реализован как `CupedTransformer(fit_data="experiment_all")`.

- Асимптотически не увеличивает дисперсию: это ANCOVA с одним общим наклоном.
  В `strong_prior_mismatch` `Var(Y*)/Var(Y)` ≈ 0.96.
- Outcome текущего эксперимента используется для theta, поэтому метод помечен
  `conditional`. Welch по скорректированному target не учитывает оценку theta;
  это поправка порядка `O(1/n)`.
- **Где проигрывает.** Общий наклон неверно взвешивает наклоны групп при
  неравном разбиении, если эффект коррелирован с ковариатой (Freedman, 2008).
  В `freedman_p20_panel` MSE ≈ 1.45 × MSE Welch, в `freedman_p30_panel` ≈ 1.17.
  Lin с раздельными наклонами там не хуже DIM (0.98 и 1.00).

### Trial CUPED в форме регрессии

Lin-style regression adjustment с центрированными ковариатами,
взаимодействиями treatment и HC2/HC3 зарегистрирован как `AnalysisMethod`, а не
как target transformer.

### Диагностика дисперсии

Каждый CUPED/CUPAC-трансформер пишет `variance_ratio = Var(Y*)/Var(Y)` по всем
строкам и `variance_ratio_within_arm` по pooled within-arm дисперсиям.
Значение больше единицы означает, что коррекция добавила шум. Это маргинальная
диагностика. Freedman-проигрыш pooled-наклона она не видит: within-arm
дисперсия падает, а дисперсия разности средних растёт.

Источники: [Deng et al. (2013)](https://doi.org/10.1145/2433396.2433413),
[Lin (2013)](https://doi.org/10.1214/12-AOAS583).

## CUPAC

CUPAC использует прогноз m(X) как control variate. CUPED обычно использует
готовый pre-target и линейный коэффициент; CUPAC обучает прогностическую модель
по набору признаков. Нелинейность определяется внутренней моделью, а не самим
названием CUPAC.

### Временные зоны

- **A**: модель обучается только на историческом переходе `lag2 -> lag1`.
- **B**: отдельная cross-fitted стадия на разрешённых in-time признаках.
- **C**: ещё одна отдельная cross-fitted стадия на in-time/post признаках с
  диагностикой баланса и явной маркировкой риска утечки или медиаторства.

B и C не добавляются в модель A. Для каждого семейства существуют отдельные
ID: `*_A`, `*_A_plus_B`, `*_A_plus_B_plus_C`. Варианты B/C запускаются в общей
матрице, но требуют `full_timeline` и не могут автоматически стать
`causal_safe` лидерами.

## Семейства CUPAC

| Семейство | Внутренняя модель A | Назначение |
| --- | --- | --- |
| `cupac_linear_*` | ridge-регрессия | прозрачный основной вариант |
| `cupac_quadratic_*` | квадраты и взаимодействия + ridge | гладкая нелинейность при умеренном числе признаков |
| `cupac_automl_*` | ridge, quadratic ridge, random forest, extra trees, histogram gradient boosting | автоматический выбор по pre-period cross-validation |
| `panel_prior_cupac_*` | Gaussian ridge prior из агрегатов исторической панели | использование большого пула без передачи строк и ID |
| `cupac_hypex_compatible_*` | локальная реализация контракта, сопоставимого с HypEx | интеграционная сверка |
| `cupac_linear_pooled_theta_A` | ridge `lag2 -> lag1` с автоподбором, затем `theta = Cov(Y, g)/Var(g)` по эксперименту | стандартный CUPAC без раздувания дисперсии |

`cupac_linear_A` вычитает прогноз с фиксированным коэффициентом 1. Если
pre-period модель не переносится на текущий outcome по масштабу или знаку,
дисперсия растёт: в `strong_prior_mismatch` `Var(Y*)/Var(Y)` ≈ 2.2. Вариант
`cupac_linear_pooled_theta_A` использует ту же модель A, но оценивает theta
по текущему outcome обеих групп, как CUPAC DoorDash и HypEx 1.0.7. Поэтому он
асимптотически не хуже Welch (в том же мире отношение ≈ 0.95), но помечен
`conditional`. Модель A по-прежнему не видит текущий outcome. Оба варианта
сохранены, чтобы гонка показывала цену фиксированного коэффициента.

В панельных мирах `*_panel` outcome — следующий период того же ряда. Там
CUPAC A и pooled-theta CUPAC дают ту же дисперсию, что и Lin
(`gaussian_n2000_panel`: MSE/MSE Welch ≈ 0.55 у всех трёх).

Auto-ML никогда не выбирает модель по экспериментальному outcome. Сетка
panel-prior обязательно содержит нулевую силу prior: при несовместимом пуле
историческая информация может быть полностью отброшена.

Официальный HypEx CUPAC (версия берётся из `importlib.metadata`, сейчас
1.0.7) участвует как неделимый
`hypex_cupac_A__integration`. Отдельного стабильного публичного target-only API
для всех зон нет, поэтому `cupac_hypex_official_target_*` честно помечены как
недоступные комбинации, а не подменяются локальным кодом.

Источник идеи CUPAC: [DoorDash Engineering](https://careersatdoordash.com/blog/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/).

## Крупные значения

Winsorization и trimming не являются безобидными обёртками ATE. Они имеют
собственные estimand и собственный статистический вывод, поэтому находятся в
слое `AnalysisMethod`:

| ID | Что оценивает | Вывод |
| --- | --- | --- |
| `trimmed_mean__yuen` | разность усечённых средних | тест и интервал Юэна |
| `winsorized_mean__full_refit_bootstrap` | разность винзоризированных средних | bootstrap с повторной оценкой границ |
| `iforest_trimmed_mean__full_refit` | функционал после удаления Isolation Forest anomalies | исследовательский full-refit bootstrap |
| `iforest_winsorized_mean__full_refit` | функционал после ограничения Isolation Forest anomalies | исследовательский full-refit bootstrap |
| `median_of_means_ate__bootstrap` | исходное среднее | исследовательский bootstrap |
| `catoni_mean_ate__bootstrap` | исходное среднее | исследовательский bootstrap |

Для первых четырёх одновременно показываются смещение относительно native
estimand и `ATE_displacement`. Поэтому хороший результат по trimmed effect не
выдаётся за несмещённую оценку исходного ATE.

## Вне активной ATE-гонки

Ratio-линеаризация меняет определение метрики и требует отдельного R&D-контура.
IV/LATE и observational-методы также не участвуют в текущем рейтинге
рандомизированных экспериментов.
