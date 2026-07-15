# Коррекции статистического вывода

Все строки имеют категорию `InferenceCorrection`, не меняют estimand и
применяются после расчёта исходных статистик.

| ID | Вход → выход | Сложность | Статус и ограничение |
| --- | --- | --- | --- |
| `bonferroni` | p-values → `min(m·p, 1)` | `O(m)` | Реализован; консервативный FWER baseline |
| `holm_bonferroni` | p-values → step-down adjusted p-values | `O(m log m)` | Реализован; не слабее обычного Bonferroni |
| `romano_wolf_stepdown` | Observed + bootstrap statistics → adjusted p-values | `O(Bm log m)` | Реализован helper; resampling design задаёт caller |

Romano-Wolf helper не генерирует bootstrap самостоятельно. Caller отвечает за
корректный resampling design, центрирование null distribution и одинаковый
порядок гипотез в observed и bootstrap arrays.
