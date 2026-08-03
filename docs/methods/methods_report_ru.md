# Методы VarWar: экспортный справочник

Документ описывает только методы, для которых в VarWar зафиксированы
математическая цель, входные данные и правило вывода. Реализованные методы
можно запускать; интеграционные методы используют внешний пакет HypEx;
legacy-методы сохранены для диагностики и не входят в основную гонку.

Сравнивать следует полные пайплайны с одинаковым estimand. Преобразователь
target, адаптер признаков, подготовка дизайна и поправка вывода не являются
самостоятельными ATE-оценками.

## 1. Оценка эффекта

### `welch_mean_diff` — разность средних с тестом Уэлча

**Категория:** Estimator. **Статус:** реализован. **Estimand:** ATE.

Для независимых treatment и control вычисляется

$$
\hat\tau=\bar Y_T-\bar Y_C.
$$

Стандартная ошибка:

$$
SE^2=\frac{s_T^2}{n_T}+\frac{s_C^2}{n_C}.
$$

Степени свободы рассчитываются по формуле Уэлча--Саттертуэйта. Равенство
дисперсий не требуется. Это канонический нескорректированный baseline VarWar.
Теоретическая основа: [Welch, 1947](https://doi.org/10.1093/biomet/34.1-2.28);
реализация аналогичного теста: [SciPy `ttest_ind`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html).

### `hypex_ab` — A/B-анализ через HypEx

**Категория:** Estimator / интеграция. **Статус:** интеграция. **Estimand:** ATE.

Публичный адаптер вызывает PyPI HypEx. В текущем parity-контракте результат
сопоставляется с локальным Welch baseline; это интеграционная строка, а не
новый estimand. Источник API: [PyPI HypEx](https://pypi.org/project/hypex/).

### `pooled_mean_diff` — разность средних с общей дисперсией

**Категория:** Estimator. **Статус:** legacy. **Estimand:** ATE при допущении
равенства дисперсий.

Точечная оценка совпадает с Welch, но стандартная ошибка использует

$$
s_p^2=\frac{(n_T-1)s_T^2+(n_C-1)s_C^2}{n_T+n_C-2},\qquad
SE^2=s_p^2\left(\frac1{n_T}+\frac1{n_C}\right).
$$

Метод нужен как диагностический контраст, а не как production-baseline.

### `randomized_weighted_mean_diff` — воспроизводимая взвешенная разность средних

**Категория:** Estimator. **Статус:** реализован. **Estimand:** средняя разность
при заданном механизме весов.

После контролируемого перемешивания внутри каждой группы строятся коэффициенты
«ёжиковой» статистики и нормируются:

$$
w_{g,i}=\frac{\theta_{g,i}}{\sum_j\theta_{g,j}},\qquad
\tilde\mu_g=\sum_i w_{g,i}Y_{g,i}.
$$

Итоговая оценка и inference используют одну и ту же величину:

$$
\tilde\tau=\tilde\mu_T-\tilde\mu_C,qquad
SE^2=s_T^2\sum_iw_{T,i}^2+s_C^2\sum_iw_{C,i}^2.
$$

При весах, независимых от target, оценка несмещена относительно соответствующей
взвешенной средней. Это собственная реализация VarWar; внешняя статья не
используется как нормативный источник. `seed` и стабильный порядок по
`unit_col` делают результат воспроизводимым и независимым от порядка строк.

### `permutation_mean_diff` — перестановочная оценка разности средних

**Категория:** Estimator. **Статус:** реализован. **Estimand:** ATE.

Точечная оценка сохраняет обычную разность средних:

$$
\hat\tau=\bar Y_T-\bar Y_C.
$$

Перестановки treatment-меток формируют randomization-распределение. p-value
получается сравнением наблюдаемой разности с его хвостами, а доверительный
интервал строится инверсией того же двустороннего теста:

$$
CI_{1-\alpha}=\left[
\hat\tau-q_{1-\alpha/2}^{perm},\;
\hat\tau-q_{\alpha/2}^{perm}
\right].
$$

Это interval inversion для sharp null и постоянного аддитивного эффекта, а не
обычный Welch или bootstrap-интервал. Поэтому метод требует обменности под
нулевой гипотезой; при неодинаковых распределениях и неоднородном эффекте
покрытие нужно проверять отдельно на синтетике. Это отдельный estimator и его
не следует путать с `permutation_mean_test`, который является только тестом.

### `bootstrap_mean_diff` — bootstrap-оценка разности средних

**Категория:** Estimator. **Статус:** реализован. **Estimand:** ATE.

В каждой группе независимо выполняется выборка с возвращением. Для каждой
bootstrap-реплики $b$ вычисляется $\hat\tau^{(b)}$, после чего

$$
\widehat{SE}_{boot}=sd\left(\hat\tau^{(1)},\ldots,\hat\tau^{(B)}\right).
$$

Интервал строится по эмпирическим квантилям bootstrap-распределения
(percentile interval), а p-value — нормальным приближением по bootstrap SE.
Метод воспроизводим при фиксированном seed и не требует параметрической формы
распределения, но расходует $O(B)$ памяти на реплики и имеет Monte Carlo-ошибку.

### `regression_adjustment` — регрессионная корректировка

**Категория:** Estimator. **Статус:** реализован. **Estimand:** ATE.

Оценивается модель

$$
Y_i=\alpha+\tau T_i+\beta^\mathsf{T}X_i+\varepsilon_i.
$$

Коэффициент $\hat\tau$ оценивает эффект в рандомизированном эксперименте,
а стандартная ошибка использует HC1-коррекцию. Для связи точности с качеством
прогноза см. [Bloniarz et al., 2016](https://doi.org/10.1073/pnas.1614732113)
и [Lin, 2013](https://doi.org/10.1111/j.1468-0262.2013.01256.x).

### `ml_rate` — cross-fitted прогнозная корректировка

**Категория:** Estimator. **Статус:** реализован. **Estimand:** ATE.

Модель прогноза строит out-of-fold значение $\hat m(X)$, после чего оценивается

$$
Y_i=\alpha+\tau T_i+\gamma\hat m(X_i)
      +\delta T_i\hat m(X_i)+\varepsilon_i.
$$

Cross-fitting не позволяет использовать прогноз, обученный на той же строке.
В текущем коде nuisance-модель — Ridge, поэтому `ml_rate` не следует называть
универсальным ML-оценивателем. Теоретический контекст cross-fitting и
ортогональных оценочных уравнений: [Chernozhukov et al., 2018](https://doi.org/10.1111/ectj.12097).

### `stratified_mean_diff` — стратифицированная разность средних

**Категория:** Estimator. **Статус:** реализован. **Estimand:** ATE при заданных
весах страт.

Внутри страты $h$ вычисляется $\hat\tau_h=\bar Y_{T,h}-\bar Y_{C,h}$, затем

$$
\hat\tau_{strat}=\sum_h w_h\hat\tau_h,qquad
SE^2=\sum_h w_h^2\left(\frac{s^2_{T,h}}{n_{T,h}}+
\frac{s^2_{C,h}}{n_{C,h}}\right).
$$

При reference-весах целевая популяция задаётся reference-распределением.
Пропуск малых страт может изменить estimand.

### `variance_weighted_stratified` — обратное дисперсионное взвешивание страт

**Категория:** Estimator. **Статус:** реализован. **Estimand:** precision-weighted
контраст, не обязательно общий ATE.

Для локальной дисперсии $v_h$ используются веса

$$
w_h=\frac{1/v_h}{\sum_j1/v_j},\qquad
\hat\tau=\sum_hw_h\hat\tau_h.
$$

Метод повышает вес точных страт, поэтому при неоднородном эффекте отвечает на
другой вопрос, чем population-weighted ATE. Классическая мотивация
Это стандартная схема precision-weighted aggregation; в VarWar она используется
только как явно обозначенный альтернативный estimand.

### `nearest_neighbor_matching` — сопоставление ближайшего соседа

**Категория:** Estimator. **Статус:** реализован. **Estimand:** ATT при
наблюдательной интерпретации.

Для каждого treatment-наблюдения выбирается ближайший control по
стандартизированным числовым признакам, с возвращением:

$$
\hat\tau_{ATT}=\frac1{n_T}\sum_{i:T_i=1}(Y_i-Y_{m(i)}).
$$

Это не тот же объект, что HypEx matching: локальная реализация фиксирует
собственную метрику, правило replacement и упрощённый inference. Теория
matching-оценок: [Abadie and Imbens, 2006](https://www.hks.harvard.edu/publications/large-sample-properties-matching-estimators-average-treatment-effects).

### `ratio_delta` — delta-оценка разности отношений средних

**Категория:** Estimator для ratio-метрики. **Статус:** реализован.

Для $R=E[N]/E[D]$ оценивается

$$
\hat\tau=\frac{\bar N_T}{\bar D_T}-\frac{\bar N_C}{\bar D_C}.
$$

Градиент функции отношения применяется к ковариации числителя и знаменателя.
Это не конкурент обычному mean-difference estimator: сначала задаётся
специализированный ratio estimand. Основа приближения: [Oehlert, 1992](https://doi.org/10.1080/00031305.1992.10475842).

### `wald_late` — оценка Вальда для LATE

**Категория:** Estimator / instrumental variables. **Статус:** реализован.
**Estimand:** LATE для compliers.

При бинарном инструменте $Z$ и фактическом treatment $D$:

$$
\hat\tau_{LATE}=\frac{E[Y\mid Z=1]-E[Y\mid Z=0]}
{E[D\mid Z=1]-E[D\mid Z=0]}.
$$

Нужны relevance, exclusion restriction и monotonicity. Это не общий ATE.
Первоисточник: [Imbens and Angrist, 1994](https://doi.org/10.2307/2951620).

### `trimmed_mean_diff` и `winsorized_mean_diff`

**Категория:** исторические Estimator-обёртки. **Статус:** legacy.

Оба метода меняют целевую метрику: trimming удаляет хвосты, winsorization
заменяет их квантильными границами. В активной архитектуре эту идею следует
выражать как target transformer плюс обычный estimator. Для стандартного
усечённого t-теста см. [SciPy `ttest_ind`, параметр `trim`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html).

## 2. Проверка гипотез

### `ks_test` — двухвыборочный критерий Колмогорова--Смирнова

**Категория:** HypothesisTest. **Статус:** реализован. **Цель:**
$H_0:F_T=F_C$.

Статистика — максимум разности эмпирических функций распределения. Тест не
оценивает ATE и реагирует на форму распределения, хвосты и разброс. Реализация:
[SciPy `ks_2samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).

### `approx_ks_fast` — приближённый KS по гистограмме

**Категория:** HypothesisTest. **Статус:** реализован.

Эмпирические функции строятся по фиксированным bins. Это инженерная
аппроксимация KS: стоимость зависит от числа bins, но точность зависит от их
границ. Метод проверяет распределения, а не средние.

### `energy_test` — тест энергетического расстояния

**Категория:** HypothesisTest. **Статус:** реализован.

Для одномерных данных энергетическое расстояние имеет вид

$$
\mathcal E=2E|X-Y|-E|X-X'|-E|Y-Y'|.
$$

p-value в VarWar строится перестановками меток. Теория energy statistics:
[Székely and Rizzo, 2013](https://doi.org/10.1016/j.jspi.2013.03.018).

### `randomized_t_test` — тест однородности с ёжик-статистикой

**Категория:** HypothesisTest. **Статус:** реализован.

Используется специальная статистика

$$
t_{rand}=\frac{\theta_T^\mathsf{T}Y_T-\theta_C^\mathsf{T}Y_C}
{\sqrt{s_T^2\|\theta_T\|^2+s_C^2\|\theta_C\|^2}}.
$$

Это тестовая статистика для исследовательской проверки однородности, а не
согласованный ATE-оцениватель. Поля эффекта и CI общего API для неё не следует
интерпретировать как доверительный интервал ATE. Внешний источник для этой
конкретной VarWar-статистики не заявляется.

### `permutation_mean_test` — перестановочный тест разности средних

**Категория:** HypothesisTest. **Статус:** реализован.

Наблюдаемая статистика — $\bar Y_T-\bar Y_C$; нулевое распределение строится
перемешиванием treatment-меток. Перестановочная калибровка уменьшает зависимость
от параметрической формы распределения, но имеет вычислительную стоимость и
Monte Carlo-ошибку. Общая реализация: [SciPy `permutation_test`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
В отличие от `permutation_mean_diff`, этот объект не заявляет согласованный
confidence interval для ATE: его основной результат — p-value проверки
рандомизации.

## 3. Преобразование target

### `cuped` и `cuped_control` — линейная CUPED-коррекция

**Категория:** TargetTransformer. **Статус:** реализован.

Для pre-experiment ковариаты $X$:

$$
Y'=Y-\theta(X-\bar X),\qquad
\theta=\frac{Cov(Y,X)}{Var(X)}.
$$

`cuped_control` оценивает $\theta$ по текущему control и применяет его к обеим
группам. CUPED — линейная корректировка по ковариате, а не обязательное
обучение ML-модели. Первоисточник: [Deng et al., 2013](https://exp-platform.com/cuped/).

### `cupac_linear` и `cupac_linear_A` — temporal CUPAC зоны A

**Категория:** TargetTransformer. **Статус:** реализован.

В текущем контракте A строит линейный прогноз lag1 по lag2:

$$
\widehat Y_{pre,lag1}=f_{linear}(X_{lag2}).
$$

Прогноз используется как baseline для корректировки текущего target. В отличие
от CUPED здесь есть обучаемая модель, но это не означает произвольную
нелинейность: текущая реализация линейная. Общий современный контекст сочетания
pre- и in-experiment признаков: [Lin and Crespo, 2026](https://proceedings.mlr.press/v323/lin26a.html).

### `cupac_linear_A_plus_B` и `cupac_linear_A_plus_B_plus_C`

**Категория:** TargetTransformer. **Статус:** реализован.

Это разные компоненты, определяющие разные пайплайны:

- A использует только pre-treatment temporal-признаки;
- A+B добавляет отдельную control-fitted линейную стадию по разрешённым
  current/in-time признакам;
- A+B+C добавляет gated-стадию C после диагностических проверок.

Стадии B и C не расширяют модель A. Их можно использовать только при заранее
зафиксированном контракте доступности признаков и проверке, что treatment не
попал в обучение. В текущей парадигме VarWar отдельная историческая выборка не
обязательна: `auxiliary_data` передаётся только явно.

### `linear_intime_adjustment` — линейная in-time-корректировка

**Категория:** TargetTransformer. **Статус:** реализован.

Отдельная линейная модель по разрешённым текущим признакам обучается по control,
а её прогноз вычитается из target. Это отдельная стадия после A и не синоним
`cupac_linear_A`.

### `cupac_quadratic_A` — квадратичный CUPAC зоны A

**Категория:** TargetTransformer. **Статус:** реализован.

К линейным признакам lag2 добавляются квадраты и попарные произведения; прогноз
lag1 строится ridge/OLS-моделью. Это ограниченное расширение A с нелинейными
признаками, а не общий нелинейный CUPAC.

### `winsorized` — винзоризация target

**Категория:** TargetTransformer. **Статус:** реализован.

По control-defined квантилям target преобразуется как

$$
Y'=\min(\max(Y,q_{low}),q_{high}).
$$

Изменяется estimand: эффект относится к винзоризированной метрике. Поэтому
корректное сравнение выполняется только внутри пайплайнов с одинаковой
политикой target.

## 4. Изменение представления признаков

### `numeric_feature_binner` — разбиение числовых признаков на категории

**Категория:** FeatureAdapter. **Статус:** реализован.

На reference data фиксируются quantile, uniform или custom-границы. Далее
значения переводятся в memory-efficient `pandas.Categorical` с категориями
`bin_000`, `bin_001` и так далее. Границы не переобучаются на experiment data;
NaN получает отдельную категорию. Документация представления категорий:
[pandas Categorical](https://pandas.pydata.org/docs/user_guide/categorical.html).

## 5. Подготовка дизайна

### `atomic_strata` — все исходные комбинации категорий

**Категория:** DesignPreprocessor. **Статус:** реализован.

Каждая уникальная комбинация категориальных признаков остаётся отдельной
стратой. Это baseline перед укрупнением.

### `random_reduction` — случайное укрупнение страт

**Категория:** DesignPreprocessor. **Статус:** реализован.

Атомарные страты распределяются по заданному числу макрострат с контролируемым
seed. Это случайный baseline для проверки, даёт ли оптимизация дополнительный
выигрыш.

### `mean_sorted_reduction` — укрупнение по историческому среднему

**Категория:** DesignPreprocessor. **Статус:** реализован.

Атомарные страты сортируются по reference-среднему и последовательно
группируются. Метод быстрый и детерминированный, но оптимизирует только
одномерный порядок средних.

### `genetic_strata_reducer` — генетическое укрупнение страт

**Категория:** DesignPreprocessor. **Статус:** реализован.

Атомарные страты не разделяются. Для макростраты $g$ агрегируются размер,
среднее и центральная сумма квадратов:

$$
N_g=\sum_{j\in g}N_j,\qquad
\mu_g=\frac{\sum_{j\in g}N_j\mu_j}{N_g},
$$

$$
M_{2g}=\sum_{j\in g}\left[M_{2j}+N_j(\mu_j-\mu_g)^2\right],qquad
S_g^2=\frac{M_{2g}}{N_g-1}.
$$

Для естественного A/B-трафика минимизируется

$$
J_{AB}=\sum_g\frac{N_g}{N}S_g^2.
$$

После агрегации исходные строки не нужны: генетический поиск работает по
атомарным статистикам. Используются целочисленные хромосомы, tournament
selection, crossover, mutation, repair, кэш повторов и остановка по patience.
Это собственная реализация VarWar по зафиксированной постановке; внешний код
не является частью алгоритма.

## 6. Целевая функция укрупнения

### `ab_proportional` — objective для естественного A/B-трафика

**Категория:** OptimizationObjective. **Статус:** реализован.

Это внутренняя цель `genetic_strata_reducer`, а не estimator и не способ
распределить пользователей по группам. Она оценивает ожидаемую
внутристратовую дисперсию при естественных долях трафика по формуле $J_{AB}$.

## 7. Поправки множественного вывода

### `bonferroni` — поправка Бонферрони

**Категория:** InferenceCorrection. **Статус:** реализован.

Для $m$ проверок $p_i^{adj}=\min(mp_i,1)$. Поправка контролирует FWER, но может
быть консервативной.

### `holm_bonferroni` — пошаговая поправка Холма

**Категория:** InferenceCorrection. **Статус:** реализован.

Отсортированные p-value сравниваются с порогами $\alpha/(m-k+1)$, после чего
решения возвращаются в исходном порядке. Первоисточник: [Holm, 1979](https://doi.org/10.2307/4615733).

### `romano_wolf_stepdown` — пошаговая поправка Romano--Wolf

**Категория:** InferenceCorrection. **Статус:** реализован.

Метод использует совместное bootstrap-распределение статистик и step-down
max-statistic, учитывая зависимость между проверками. Первоисточники:
[Romano and Wolf, 2005, JASA](https://doi.org/10.1198/016214504000000539) и
[Romano and Wolf, 2005, Econometrica](https://doi.org/10.1111/j.1468-0262.2005.00615.x).

## 8. Готовые композиции

Пайплайн — это не новый estimator, а фиксированный порядок компонентов. В
основной гонке сравниваются следующие композиции:

| ID | Состав | Роль |
| --- | --- | --- |
| `welch` | `welch_mean_diff` | нескорректированный baseline |
| `hypex_cuped` | HypEx CUPED → HypEx A/B | интеграционная parity-проверка |
| `pooled_mean_diff` | `pooled_mean_diff` | диагностический legacy-контур |
| `randomized_weighted_mean_diff` | `randomized_weighted_mean_diff` | взвешенный estimand |
| `permutation_mean_diff` | `permutation_mean_diff` | permutation p-value + interval inversion |
| `bootstrap_mean_diff` | `bootstrap_mean_diff` | percentile bootstrap CI |
| `cuped_control__hypex_ab` | `cuped_control` → `hypex_ab` | локальный CUPED |
| `hypex_cupac_temporal` | HypEx temporal CUPAC → HypEx A/B | интеграционная parity-проверка |
| `cupac_linear_A__hypex_ab` | A → HypEx A/B | pre-treatment CUPAC |
| `cupac_linear_A_plus_B__hypex_ab` | A → B → HypEx A/B | exploratory in-time-корректировка |
| `cupac_linear_A_plus_B_plus_C__hypex_ab` | A → B → C → HypEx A/B | gated exploratory-вариант |
| `cupac_quadratic_A__hypex_ab` | quadratic A → HypEx A/B | ограниченная нелинейность |
| `winsorized__hypex_ab` | winsorization → HypEx A/B | изменённый target |
| `ga_strata__stratified` | GA-страты → `stratified_mean_diff` | дизайн плюс estimator |
| `ga_strata__cuped_control__stratified` | GA → CUPED → stratified | дизайн плюс target-коррекция |
| `ga_strata__cupac_linear_A__stratified` | GA → CUPAC A → stratified | pre-treatment CUPAC |
| `ga_strata__cupac_linear_A_plus_B__stratified` | GA → A+B → stratified | exploratory-вариант |
| `ga_strata__cupac_linear_A_plus_B_plus_C__stratified` | GA → A+B+C → stratified | gated exploratory-вариант |
| `variance_weighted_stratified` | `variance_weighted_stratified` | precision-weighted strata |
| `nearest_neighbor_matching` | `nearest_neighbor_matching` | observational ATT |
| `trimmed_mean_diff` | `trimmed_mean_diff` | изменённый estimand |
| `winsorized_mean_diff` | `winsorized_mean_diff` | изменённый estimand |
| `ratio_delta` | `ratio_delta` | только при наличии numerator/denominator |
| `wald_late` | `wald_late` | только при наличии instrument/received treatment |

## 9. Как интерпретировать результаты

Для ATE-пайплайнов сравниваются effect, bias, empirical SD, средний SE,
coverage и power при одинаковом estimand. Для KS, energy и randomized t
сравнивается поведение тестовой статистики и FPR, а не величина ATE. Для CUPED,
CUPAC, winsorization и GA нельзя строить отдельный рейтинг только по p-value:
оценивается вся композиция вместе с её контрактом данных и правилом inference.

## Источники

1. Welch, B. L. (1947). [The generalisation of student's problems when several different population variances are involved](https://doi.org/10.1093/biomet/34.1-2.28).
2. Deng, A. et al. (2013). [Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/cuped/).
3. Lin, W. (2013). [Agnostic Notes on Regression Adjustments to Experimental Data](https://doi.org/10.1214/12-AOAS583).
4. Bloniarz, A. et al. (2016). [Lasso adjustments of treatment effect estimates in randomized experiments](https://doi.org/10.1073/pnas.1510506113).
5. Chernozhukov, V. et al. (2018). [Double/debiased machine learning for treatment and structural parameters](https://doi.org/10.1111/ectj.12097).
6. Abadie, A. and Imbens, G. (2006). [Large Sample Properties of Matching Estimators for Average Treatment Effects](https://www.hks.harvard.edu/publications/large-sample-properties-matching-estimators-average-treatment-effects).
7. Oehlert, G. (1992). [A Note on the Delta Method](https://doi.org/10.1080/00031305.1992.10475842).
8. Imbens, G. and Angrist, J. (1994). [Identification and Estimation of Local Average Treatment Effects](https://doi.org/10.2307/2951620).
9. Székely, G. and Rizzo, M. (2013). [Energy statistics: A class of statistics based on distances](https://doi.org/10.1016/j.jspi.2013.03.018).
10. Lin, Z. and Crespo, P. (2026). [Variance reduction combining pre-experiment and in-experiment data](https://proceedings.mlr.press/v323/lin26a.html).
11. Holm, S. (1979). [A simple sequentially rejective multiple test procedure](https://doi.org/10.2307/4615733).
12. Romano, J. P. and Wolf, M. (2005). [Exact and Approximate Stepdown Methods for Multiple Hypothesis Testing](https://doi.org/10.1198/016214504000000539).
13. Rigdon, J. and Hudgens, M. (2015). [Randomization inference for treatment effects on a binary outcome](https://pmc.ncbi.nlm.nih.gov/articles/PMC4459717/).
14. SciPy. [`ttest_ind`, `ks_2samp` and `permutation_test`](https://docs.scipy.org/doc/scipy/reference/stats.html).
15. pandas. [Categorical data](https://pandas.pydata.org/docs/user_guide/categorical.html).

## Актуальные уточнения реализации

Некоторые строки выше сохранены как исторический отчёт и могут описывать
старую версию кода. В текущей RCT-гонке:

- `bootstrap_mean_diff` использует studentized bootstrap-t: интервал строится
  по распределению bootstrap-статистики `(effect - effect_boot) / se_boot`, а
  не как простой percentile-интервал;
- `regression_adjustment` и `ml_rate` используют HC2-оценку ковариации;
- `ml_rate` выполняет cross-fitting, а силу ridge выбирает внутри каждого
  обучающего фолда. Outcome из отложенного фолда не участвует в его прогнозе;
- temporal CUPAC выбирает ridge по holdout-хвосту исторической пары
  `lag2 -> lag1`, затем переобучает модель на всей разрешённой исторической
  зоне. Текущий post-treatment target не используется для выбора модели;
- panel-prior CUPAC отдельно выбирает силу внешнего prior по временной
  проверке и допускает полное игнорирование prior при несовпадении панели.
