from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

from src.feature_engineering import build_preprocessor


@dataclass
class EvalMetrics:
    rmse: float
    mae: float
    r2: float


class SpreadPredictor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_estimator_ = None
        self.residual_std_ = None

    def fit(self, X, y):
        preprocessor = build_preprocessor()
        xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=4
        )

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", xgb),
        ])

        param_dist = {
            "model__n_estimators": [200, 500, 800],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=20,
            scoring="neg_root_mean_squared_error",
            cv=3,
            verbose=1,
            random_state=self.random_state,
            n_jobs=2,
        )
        search.fit(X, y)

        self.model = search
        self.best_estimator_ = search.best_estimator_

        preds = self.best_estimator_.predict(X)
        residuals = y - preds
        self.residual_std_ = float(np.std(residuals))
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def evaluate(self, X, y) -> EvalMetrics:
        preds = self.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        r2 = float(r2_score(y, preds))
        return EvalMetrics(rmse=rmse, mae=mae, r2=r2)

    def predict_with_confidence(self, X, confidence: float = 0.90):
        preds = self.predict(X)
        z = 1.645 if confidence == 0.90 else 1.96
        std = self.residual_std_ if self.residual_std_ is not None else 50.0
        lower = preds - z * std
        upper = preds + z * std
        return preds, lower, upper

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)