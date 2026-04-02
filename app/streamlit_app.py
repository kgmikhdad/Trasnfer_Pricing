from __future__ import annotations
import streamlit as st
import pandas as pd
from src.spread_model import SpreadPredictor
from src.estimator import ArmLengthRateEstimator


st.set_page_config(page_title="Arm's Length Interest Rate Estimator", layout="wide")
st.title("Arm's Length Interest Rate Estimator")
st.caption("Transfer Pricing Benchmarking Tool (MVP)")

@st.cache_resource
def load_estimator():
    model = SpreadPredictor.load("models/spread_model.joblib")
    return ArmLengthRateEstimator(model)

estimator = load_estimator()

with st.sidebar:
    st.header("Borrower Inputs")
    debt_to_equity = st.slider("Debt-to-Equity", 0.0, 6.0, 1.2, 0.1)
    interest_coverage = st.slider("Interest Coverage (EBIT / Interest)", 0.0, 30.0, 6.0, 0.5)
    ebitda_margin = st.slider("EBITDA Margin", -0.2, 0.6, 0.18, 0.01)
    total_assets_m = st.number_input("Total Assets (USD millions)", min_value=1.0, value=1200.0, step=10.0)
    sector = st.selectbox("Sector", [
        "Technology", "Healthcare", "Energy", "Financials",
        "Industrials", "Consumer", "Utilities", "Materials"
    ])
    tenor_years = st.selectbox("Loan Tenor (years)", [1,2,3,5,7,10,15,20], index=3)
    currency = st.selectbox("Currency", ["USD", "EUR", "GBP"], index=0)
    known_credit_rating = st.selectbox("Known Credit Rating", ["Unknown", "AAA", "AA", "A", "BBB", "BB", "B", "CCC"], index=0)
    base_rate_pct = st.number_input("Base Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

run = st.button("Estimate Arm's Length Rate", type="primary")

if run:
    profile = {
        "debt_to_equity": debt_to_equity,
        "interest_coverage": interest_coverage,
        "ebitda_margin": ebitda_margin,
        "total_assets_m": total_assets_m,
        "sector": sector,
        "tenor_years": tenor_years,
        "currency": currency,
        "known_credit_rating": known_credit_rating,
        "base_rate_pct": base_rate_pct,
    }

    result = estimator.estimate(profile)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Rating", result["predicted_rating"])
    c2.metric("Predicted Spread (bps)", result["predicted_spread_bps"])
    c3.metric("Estimated Yield (%)", result["estimated_yield_pct"])

    st.subheader("90% Confidence Interval (Spread bps)")
    st.write(f"{result['confidence_interval_bps_90'][0]} to {result['confidence_interval_bps_90'][1]}")

    st.subheader("Comparable Market Spreads by Rating (same tenor)")
    comp_df = pd.DataFrame(
        [{"rating": k, "spread_bps": v} for k, v in result["comparable_rating_spreads_bps"].items()]
    ).sort_values("spread_bps")
    st.dataframe(comp_df, use_container_width=True)

    st.subheader("Top Feature Contributions (approx.)")
    contrib_df = pd.DataFrame(
        [{"feature": k, "delta_bps_if_increased": round(v, 2)}
         for k, v in result["top_feature_contributions"].items()]
    )
    st.bar_chart(contrib_df.set_index("feature"))
    st.dataframe(contrib_df, use_container_width=True)

else:
    st.info("Set inputs and click **Estimate Arm's Length Rate**.")
