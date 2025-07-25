import time
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Any

@dataclass
class EstimationResult:
    """Контейнер для результатов оценки"""
    method: str
    estimate: float
    stderr: float
    ci: tuple
    t_stat: float
    p_value: float
    time: float

    def __getitem__(self, key: str) -> Any:
        """Доступ к элементам через квадратные скобки"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Invalid key: '{key}'.")

    def summary(self):
        """Форматированный вывод результатов"""
        return (
            f"Method: {self.method}\n"
            f"Estimate: {self.estimate:.4f}\n"
            f"Std Error: {self.stderr:.4f}\n"
            f"95% CI: ({self.ci[0]:.4f}, {self.ci[1]:.4f})\n"
            f"t-statistic: {self.t_stat:.4f}\n"
            f"p-value: {self.p_value:.6f}\n"
            f"Time: {self.time:.4f} seconds"
        )

class WaldEstimator:
    def __init__(self, df, y_col='y', z_col='z', d_col='d'):
        """
        Инициализация LATE Estimator
        
        :param df: DataFrame с данными
        :param y_col: Название колонки с outcome
        :param z_col: Название колонки с treatment assignment
        :param d_col: Название колонки с accept
        """
        self.df = df
        self.y_col = y_col
        self.z_col = z_col
        self.d_col = d_col
        self.results = None
        
        # Валидация данных
        self._validate_data()
        
    def _validate_data(self):
        """Проверка наличия необходимых колонок"""
        required_cols = {self.y_col, self.z_col, self.d_col}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
            
    def estimate(self, method='base'):
        """
        Запуск оценки указанным методом
        
        :param method: Метод оценки ('base', 'iv')
        :return: Экземпляр EstimationResult
        """
        start_time = time.time()
        
        # Выбор метода оценки
        if method == 'base':
            point_estimate = self._base_wald_estimate()
        else:
            raise NotImplementedError(f"Method '{method}' not implemented")
        
        # Расчет статистик
        stderr = self._calculate_stderr()
        t_stat = point_estimate / stderr
        ci = self._calculate_ci(point_estimate, stderr)
        p_value = self._calculate_p_value(t_stat)
        
        # Создание и сохранение результата
        self.results = EstimationResult(
            method=method,
            estimate=point_estimate,
            stderr=stderr,
            ci=ci,
            t_stat=t_stat,
            p_value=p_value,
            time=time.time() - start_time
        )
        
        return self.results

    def _base_wald_estimate(self):
        """Базовый Метод Вальда для оценки LATE"""
        itt_y = self._calculate_itt(self.y_col)
        itt_d = self._calculate_itt(self.d_col)
        
        if abs(itt_d) < 1e-10:
            raise ZeroDivisionError("ITT for accept is close to zero - cannot compute LATE")
            
        return itt_y / itt_d
    
    def _calculate_itt(self, variable):
        """Расчет ITT эффекта для указанной переменной"""
        treatment_mean = self.df[self.df[self.z_col] == 1][variable].mean()
        control_mean = self.df[self.df[self.z_col] == 0][variable].mean()
        return treatment_mean - control_mean

    def _calculate_stderr(self):
        """Расчет стандартной ошибки для разности средних"""
        control = self.df[self.df[self.z_col] == 0][self.y_col]
        treatment = self.df[self.df[self.z_col] == 1][self.y_col]
        
        var_control = control.var(ddof=1)
        var_treatment = treatment.var(ddof=1)
        n_control = len(control)
        n_treatment = len(treatment)
        
        return np.sqrt(var_control/n_control + var_treatment/n_treatment)
    
    def _calculate_ci(self, estimate, stderr, alpha=0.05):
        """Расчет доверительного интервала"""
        z_score = stats.norm.ppf(1 - alpha/2)
        lower = estimate - z_score * stderr
        upper = estimate + z_score * stderr
        return (lower, upper)
    
    def _calculate_p_value(self, t_stat):
        """Расчет p-value"""
        return 2 * stats.t.sf(np.abs(t_stat), df=len(self.df) - 1)
    
    def summary(self):
        """Краткое текстовое представление результатов"""
        return self.results.summary()