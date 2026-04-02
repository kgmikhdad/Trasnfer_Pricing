from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["debt_tenor_interaction"] = out["debt_to_equity"] * out["tenor_years"]
    out["coverage_base_interaction"] = out["interest_coverage"] * out["base_rate_pct"]
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "debt_to_equity",
        "interest_coverage",
        "ebitda_margin",
        "log_total_assets",
        "tenor_years",
        "base_rate_pct",
        "debt_tenor_interaction",
        "coverage_base_interaction",
    ]
    categorical_features = ["sector", "credit_rating"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def split_features_target(df: pd.DataFrame):
    work = add_interactions(df)
    X = work.drop(columns=["spread_bps", "yield_pct"])
    y = work["spread_bps"]
    return X, y