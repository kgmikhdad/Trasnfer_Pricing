from __future__ import annotations
import os
from sklearn.model_selection import train_test_split
from src.data_collection import generate_synthetic_bond_data
from src.feature_engineering import split_features_target
from src.spread_model import SpreadPredictor


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = generate_synthetic_bond_data(n_samples=6000, random_state=42)
    df.to_csv("data/processed/synthetic_bonds.csv", index=False)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SpreadPredictor(random_state=42)
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print("=== Spread Model Metrics ===")
    print(f"RMSE: {metrics.rmse:.2f}")
    print(f"MAE : {metrics.mae:.2f}")
    print(f"R²  : {metrics.r2:.3f}")

    model.save("models/spread_model.joblib")
    print("Saved model to models/spread_model.joblib")


if __name__ == "__main__":
    main()