
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from typing import Optional, Dict, List, Union


class CUPACTransformer:
    """
    Улучшенный CUPAC трансформер с расширенной отчетностью
    """

    def __init__(
        self,
        target_col: str,
        lag_suffix: str = "_lag",
        target_counterfactual_suffix="0",
        models: Optional[Dict] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        self.target_col = target_col
        self.target_counterfactual_suffix = target_counterfactual_suffix
        self.lag_suffix = lag_suffix
        self.n_folds = n_folds
        self.random_state = random_state

        # Инициализация моделей
        self.models = models or {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=0.5),
            "Lasso": Lasso(alpha=0.01, max_iter=10000),
            "CatBoost": CatBoostRegressor(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                silent=True,
                random_state=random_state,
                allow_writing_files=False,
            ),
        }

        # Состояние модели
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.variance_reduction = None
        self.lag_features = None
        self.current_features = None
        self.is_fitted = False
        self.model_results_ = {}
        self.feature_importances_ = None

    def _prepare_train_data(self, df: pd.DataFrame) -> tuple:
        """Подготовка данных для обучения"""
        target_counterfactual_name = (
            f"{self.target_col}{self.target_counterfactual_suffix}{self.lag_suffix}"
        )

        self.lag_features = [
            col
            for col in df.columns
            if col.endswith(self.lag_suffix)
            and col != f"{self.target_col}{self.lag_suffix}"
        ]

        if not self.lag_features:
            raise ValueError("Не найдены лаговые признаки для обучения")

        self.current_features = [
            col.replace(self.lag_suffix, "") for col in self.lag_features
        ]

        self.lag_features.append(f"{target_counterfactual_name}_2")
        self.current_features.append(f"{target_counterfactual_name}_1")

        return df[self.lag_features], df[f"{target_counterfactual_name}_1"]

    def _prepare_inference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка данных для применения"""
        if not self.current_features:
            raise RuntimeError("Сначала обучите модель (fit())")

        missing = [col for col in self.current_features if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют признаки: {missing}")

        return df[self.current_features].rename(
            columns=dict(zip(self.current_features, self.lag_features))
        )

    def _calculate_variance_reduction(self, y: pd.Series, pred: pd.Series) -> float:
        """Расчет снижения дисперсии"""
        pred_centered = pred - pred.mean()
        if pred_centered.var() < 1e-10:
            return 0.0

        theta = np.cov(y, pred_centered)[0, 1] / pred_centered.var()
        y_adj = y - theta * pred_centered
        return max(0, (1 - y_adj.var() / y.var()) * 100)

    def fit(self, df: pd.DataFrame) -> "CUPACTransformer":
        """Обучение модели на исторических данных"""
        X, y = self._prepare_train_data(df)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        results = {}

        for name, model in self.models.items():
            fold_scores = []
            fold_var_reductions = []
            status = "success"

            try:
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    if name == "CatBoost":
                        m = CatBoostRegressor(**model.get_params())
                        m.fit(X_train, y_train, verbose=False)
                    else:
                        m = model.__class__(**model.get_params())
                        m.fit(X_train, y_train)

                    pred = m.predict(X_val)
                    fold_scores.append(r2_score(y_val, pred))
                    fold_var_reductions.append(
                        self._calculate_variance_reduction(y_val, pred)
                    )

                results[name] = {
                    "r2": np.nanmean(fold_scores),
                    "var_reduction": np.nanmean(fold_var_reductions),
                    "status": status,
                }

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                results[name] = {
                    "r2": None,
                    "var_reduction": None,
                    "status": f"failed: {error_msg}",
                }
                print(f"Ошибка в {name}: {error_msg}")

        self.model_results_ = results

        # Выбор лучшей модели из успешных
        successful_models = {
            k: v for k, v in results.items() if v["status"] == "success"
        }
        if not successful_models:
            raise RuntimeError("Все модели завершились с ошибкой")

        self.best_model_name = max(
            successful_models, key=lambda x: successful_models[x]["var_reduction"]
        )
        self.best_score = successful_models[self.best_model_name]["r2"]
        self.variance_reduction = successful_models[self.best_model_name][
            "var_reduction"
        ]

        # Финальное обучение и feature importance
        X, y = self._prepare_train_data(df)
        best_model_params = self.models[self.best_model_name].get_params()

        if self.best_model_name == "CatBoost":
            self.best_model = CatBoostRegressor(**best_model_params)
            self.best_model.fit(X, y, verbose=False)
            self.feature_importances_ = dict(
                zip(X.columns, self.best_model.get_feature_importance())
            )
        else:
            self.best_model = self.models[self.best_model_name].__class__(
                **best_model_params
            )
            self.best_model.fit(X, y)
            if hasattr(self.best_model, "coef_"):
                self.feature_importances_ = dict(zip(X.columns, self.best_model.coef_))
            else:
                self.feature_importances_ = None

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Применение модели к новым данным"""
        if not self.is_fitted:
            raise RuntimeError("Сначала вызовите fit()")

        X = self._prepare_inference_data(df)
        y = df[self.target_col]
        pred = self.best_model.predict(X)

        pred_centered = pred - pred.mean()
        theta = np.cov(y, pred_centered)[0, 1] / pred_centered.var()
        y_adj = y - theta * pred_centered

        if inplace:
            df[f"{self.target_col}_cupac"] = y_adj
            return df
        return df.assign(**{f"{self.target_col}_cupac": y_adj})

    def get_report(self) -> str:
        """Генерация расширенного отчета"""
        if not self.is_fitted:
            return "Модель не обучена. Сначала вызовите fit()."

        # Сортировка фич по важности
        sorted_features = (
            sorted(
                self.feature_importances_.items(), key=lambda x: abs(x[1]), reverse=True
            )[:10]
            if self.feature_importances_
            else []
        )

        # Форматирование сравнения моделей
        model_comparison = []
        for name, data in self.model_results_.items():
            if data["status"] != "success":
                line = f"{name}: {data['status']}"
            else:
                line = (
                    f"{name}: R²={data['r2']:.3f}, "
                    f"Var.Red.={data['var_reduction']:.1f}%"
                )
            model_comparison.append(line)

        # Форматирование фич
        feature_analysis = []
        if sorted_features:
            max_coef = max(abs(v) for _, v in sorted_features)
            for feat, coef in sorted_features:
                rel_impact = abs(coef) / max_coef if max_coef != 0 else 0
                feature_analysis.append(
                    f"- {feat:<25} {coef:>7.3f} {'▇'*int(10*rel_impact)}"
                )

        report = [
            "Расширенный CUPAC Report",
            "=" * 40,
            "Сравнение моделей:",
            *model_comparison,
            "",
            f"Лучшая модель: {self.best_model_name}",
            f"Снижение дисперсии: {self.variance_reduction:.1f}%",
            f"Качество предсказания (R²): {self.best_score:.3f}",
            "",
            "Топ-10 значимых признаков:",
            *(
                feature_analysis
                if feature_analysis
                else ["Нет данных о важности признаков"]
            ),
            "",
            "Интерпретация:",
            "▇▇▇▇▇▇▇▇▇▇ - максимальное влияние",
            "Коэффициенты > 0: положительная связь с целевой переменной",
            "Коэффициенты < 0: отрицательная связь",
        ]
        return "\n".join(report)

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        df_apply: Optional[pd.DataFrame] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        self.fit(df_train)
        df_apply = df_train if df_apply is None else df_apply
        return self.transform(df_apply, inplace=inplace)

    def get_feature_mapping(self) -> Dict[str, str]:
        return dict(zip(self.lag_features, self.current_features))
