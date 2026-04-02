# Intercompany Loan Interest Rate Estimator (MVP)

A machine learning MVP for estimating an arm's length intercompany loan spread and implied yield for transfer pricing screening.

## What this MVP includes
- Synthetic corporate bond dataset generation
- Spread prediction model (XGBoost + preprocessing pipeline)
- 90% prediction interval (residual-based)
- Streamlit UI for interactive estimates
- Comparable spread table by rating/tenor

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python train.py
streamlit run app/streamlit_app.py
```

## Project structure
- `src/data_collection.py` — synthetic data generator + spread tables
- `src/feature_engineering.py` — interaction features + preprocessing
- `src/spread_model.py` — model training/evaluation/serialization
- `src/estimator.py` — end-to-end estimate logic
- `app/streamlit_app.py` — UI
- `train.py` — training entrypoint

## Notes
- This MVP is trained on synthetic data calibrated to stylized market spread curves.
- It is intended for demonstration/screening, not as a substitute for formal benchmarking documentation.