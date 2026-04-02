from __future__ import annotations
import numpy as np
import pandas as pd
from src.feature_engineering import add_interactions
from src.data_collection import SPREAD_TABLE, RATINGS


class ArmLengthRateEstimator:
    def __init__(self, spread_model):
        self.spread_model = spread_model

    def _approx_feature_contrib(self, row_df: pd.DataFrame) -> dict:
        # Lightweight SHAP-style approximation using finite differences
        base_pred = float(self.spread_model.predict(row_df)[0])
        contrib = {}
        numeric_cols = [
            "debt_to_equity", "interest_coverage", "ebitda_margin",
            "log_total_assets", "tenor_years", "base_rate_pct"
        ]
        for c in numeric_cols:
            perturbed = row_df.copy()
            step = 0.05 * (abs(float(perturbed[c].iloc[0])) + 1e-6)
            perturbed[c] = perturbed[c] + step
            perturbed = add_interactions(perturbed)
            new_pred = float(self.spread_model.predict(perturbed)[0])
            contrib[c] = new_pred - base_pred

        sorted_items = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        return dict(sorted_items)

    def estimate(self, borrower_profile: dict) -> dict:
        rating = borrower_profile.get("known_credit_rating", "BBB")
        if rating in [None, "", "Unknown"]:
            rating = "BBB"  # MVP fallback

        total_assets_m = float(borrower_profile["total_assets_m"])
        log_assets = np.log(max(total_assets_m, 1e-3) * 1_000_000)

        row = pd.DataFrame([{
            "credit_rating": rating,
            "debt_to_equity": float(borrower_profile["debt_to_equity"]),
            "interest_coverage": float(borrower_profile["interest_coverage"]),
            "ebitda_margin": float(borrower_profile["ebitda_margin"]),
            "log_total_assets": log_assets,
            "sector": borrower_profile["sector"],
            "tenor_years": int(borrower_profile["tenor_years"]),
            "base_rate_pct": float(borrower_profile["base_rate_pct"]),
        }])

        row = add_interactions(row)

        pred, low, high = self.spread_model.predict_with_confidence(row, confidence=0.90)
        pred_spread = float(pred[0])
        ci = (float(low[0]), float(high[0]))
        implied_yield = float(borrower_profile["base_rate_pct"]) + pred_spread / 100.0

        tenor = int(borrower_profile["tenor_years"])
        comparable = {
            r: SPREAD_TABLE[r][tenor] for r in RATINGS if tenor in SPREAD_TABLE[r]
        }

        return {
            "predicted_rating": rating,
            "predicted_spread_bps": round(pred_spread, 1),
            "confidence_interval_bps_90": (round(ci[0], 1), round(ci[1], 1)),
            "estimated_yield_pct": round(implied_yield, 3),
            "comparable_rating_spreads_bps": comparable,
            "top_feature_contributions": self._approx_feature_contrib(row),
        }
