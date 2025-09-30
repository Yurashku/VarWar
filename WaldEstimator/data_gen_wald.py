
import numpy as np
import pandas as pd
from scipy import stats


class DataGenerator:
    """
    Усовершенствованный генератор данных с поддержкой двух лагов для Y
    и контролем корреляционной структуры.
    """

    def __init__(
        self,
        n_samples=2000,
        distributions=None,
        time_correlations=None,
        effect_size=5.0,
        seed=None,
    ):
        self.n_samples = n_samples
        self.distributions = distributions or {
            "X1": {"type": "normal", "mean": 1, "std": 2},
            "X2": {"type": "bernoulli", "p": 0.4},
            "y0": {"type": "normal", "mean": 10, "std": 3},
        }
        self.time_correlations = time_correlations or {"X1": 0.7, "X2": 0.6, "y0": 0.8}
        self.effect_size = effect_size
        self.seed = seed
        np.random.seed(seed)

    def _generate_bernoulli_pair(self, p, rho):
        """Точная генерация коррелированных бинарных переменных"""
        rho_max = min(p / (1 - p), (1 - p) / p)
        if abs(rho) > rho_max:
            raise ValueError(f"Невозможная корреляция {rho} для p={p}")

        p11 = p * p + rho * p * (1 - p)
        p10 = p * (1 - p) - rho * p * (1 - p)
        p01 = (1 - p) * p - rho * p * (1 - p)
        p00 = (1 - p) * (1 - p) + rho * p * (1 - p)

        states = np.random.choice(4, size=self.n_samples, p=[p00, p01, p10, p11])
        lag = (states == 1) | (states == 3)
        current = (states == 2) | (states == 3)
        return current.astype(int), lag.astype(int)

    def _generate_correlated_pair(self, dist_type, params, rho):
        """Генератор пар с корреляцией"""
        if dist_type == "normal":
            cov = [
                [params["std"] ** 2, rho * params["std"] ** 2],
                [rho * params["std"] ** 2, params["std"] ** 2],
            ]
            return np.random.multivariate_normal(
                [params["mean"], params["mean"]], cov, self.n_samples
            ).T

        elif dist_type == "bernoulli":
            return self._generate_bernoulli_pair(params["p"], rho)

        elif dist_type == "gamma":
            Z = np.random.multivariate_normal(
                [0, 0], [[1, rho], [rho, 1]], self.n_samples
            )
            U = stats.norm.cdf(Z)
            current = stats.gamma.ppf(U[:, 0], a=params["shape"], scale=params["scale"])
            lag = stats.gamma.ppf(U[:, 1], a=params["shape"], scale=params["scale"])
            return current, lag

        else:
            raise ValueError(f"Неподдерживаемое распределение: {dist_type}")

    def _generate_correlated_chain(self, params, rho, n_points):
        """Генерация цепочки коррелированных значений для нормального распределения"""
        mean = params["mean"]
        std = params["std"]

        # Создаем ковариационную матрицу для AR(1) процесса
        cov = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                cov[i, j] = (std**2) * (rho ** abs(i - j))

        return np.random.multivariate_normal([mean] * n_points, cov, self.n_samples).T

    def generate(self):
        """Генерация полного датасета с двумя лагами для Y"""
        data = {}

        # Генерация базовых переменных
        for var in ["X1", "X2"]:
            current, lag = self._generate_correlated_pair(
                self.distributions[var]["type"],
                self.distributions[var],
                self.time_correlations[var],
            )
            data[var] = current
            data[f"{var}_lag"] = lag

        # Специальная обработка для Y с двумя лагами
        y_params = self.distributions["y0"]
        y_rho = self.time_correlations["y0"]

        if y_params["type"] == "normal":
            # Генерация цепочки из трех значений (y_lag_2, y_lag_1, y0)
            y_chain = self._generate_correlated_chain(y_params, y_rho, 3)
            data["y0"] = y_chain[2]
            data["y0_lag_1"] = y_chain[1]
            data["y0_lag_2"] = y_chain[0]
        else:
            # Фолбэк для других распределений (генерируем простые лаги)
            current, lag1 = self._generate_correlated_pair(
                y_params["type"], y_params, y_rho
            )
            lag2, _ = self._generate_correlated_pair(y_params["type"], y_params, y_rho)
            data["y0"] = current
            data["y0_lag_1"] = lag1
            data["y0_lag_2"] = lag2

        # Инструмент и скрытый конфандер
        data["z"] = np.random.binomial(1, 0.5, self.n_samples)
        data["U"] = np.random.normal(0, 1, self.n_samples)

        # Механика акцепта
        D_propensity = 0.3 + 0.4 * data["z"] + 0.3 * data["U"]
        data["D"] = np.random.binomial(1, np.clip(D_propensity, 0, 1))
        data["d"] = data["D"] * data["z"]

        # Потенциальные исходы
        data["y1"] = (
            data["y0"]
            + self.effect_size
            #+ np.random.normal(0, 2, self.n_samples) 
            #+ 1.5 * data["U"]
        )
        
        data["y"] = np.where(data["d"] == 1, data["y1"], data["y0"])

        return pd.DataFrame(data)
