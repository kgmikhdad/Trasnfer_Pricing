from __future__ import annotations
import numpy as np
import pandas as pd


RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_PROBS = [0.02, 0.05, 0.12, 0.35, 0.25, 0.15, 0.06]
SECTORS = [
    "Technology", "Healthcare", "Energy", "Financials",
    "Industrials", "Consumer", "Utilities", "Materials"
]
TENORS = [1, 2, 3, 5, 7, 10, 15, 20]

# Mean spread table (bps): rating -> tenor -> mean spread
SPREAD_TABLE = {
    "AAA": {1: 40, 2: 45, 3: 50, 5: 60, 7: 70, 10: 80, 15: 95, 20: 110},
    "AA":  {1: 55, 2: 60, 3: 70, 5: 85, 7: 100, 10: 120, 15: 145, 20: 165},
    "A":   {1: 75, 2: 85, 3: 95, 5: 120, 7: 145, 10: 170, 15: 200, 20: 230},
    "BBB": {1: 95, 2: 110, 3: 125, 5: 150, 7: 180, 10: 220, 15: 260, 20: 300},
    "BB":  {1: 160, 2: 185, 3: 210, 5: 260, 7: 310, 10: 360, 15: 430, 20: 500},
    "B":   {1: 240, 2: 280, 3: 320, 5: 390, 7: 450, 10: 520, 15: 610, 20: 700},
    "CCC": {1: 420, 2: 480, 3: 540, 5: 650, 7: 760, 10: 880, 15: 1020, 20: 1150},
}

NOISE_STD_BY_RATING = {
    "AAA": 12, "AA": 15, "A": 20, "BBB": 30, "BB": 50, "B": 80, "CCC": 120
}


def _sample_financials_by_rating(rng: np.random.Generator, rating: str) -> dict:
    # Simple ranges by credit quality
    ranges = {
        "AAA": ((0.10, 0.50), (10, 30), (0.20, 0.45), (10.5, 12.5)),
        "AA":  ((0.20, 0.70), (8, 22),  (0.18, 0.40), (10.2, 12.2)),
        "A":   ((0.30, 1.00), (5, 18),  (0.14, 0.35), (9.8, 11.8)),
        "BBB": ((0.50, 1.80), (3, 12),  (0.10, 0.30), (9.2, 11.5)),
        "BB":  ((1.00, 2.80), (1.5, 8), (0.06, 0.22), (8.8, 11.0)),
        "B":   ((1.80, 4.00), (0.8, 5), (0.02, 0.18), (8.2, 10.5)),
        "CCC": ((2.50, 6.00), (0.2, 3), (-0.05, 0.12), (7.5, 10.0)),
    }
    (de_lo, de_hi), (ic_lo, ic_hi), (em_lo, em_hi), (la_lo, la_hi) = ranges[rating]
    return {
        "debt_to_equity": rng.uniform(de_lo, de_hi),
        "interest_coverage": rng.uniform(ic_lo, ic_hi),
        "ebitda_margin": rng.uniform(em_lo, em_hi),
        "log_total_assets": rng.uniform(la_lo, la_hi),
    }


def generate_synthetic_bond_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []

    for _ in range(n_samples):
        rating = rng.choice(RATINGS, p=RATING_PROBS)
        tenor = int(rng.choice(TENORS))
        sector = rng.choice(SECTORS)
        base_rate = float(np.clip(rng.normal(loc=3.0, scale=1.5), 0.0, 8.0))

        fin = _sample_financials_by_rating(rng, rating)
        spread_mean = SPREAD_TABLE[rating][tenor]
        spread_noise = rng.normal(0, NOISE_STD_BY_RATING[rating])
        spread_bps = max(20.0, spread_mean + spread_noise)

        # yield in % = base rate + spread in %
        yield_pct = base_rate + spread_bps / 100.0

        rows.append({
            "credit_rating": rating,
            "debt_to_equity": fin["debt_to_equity"],
            "interest_coverage": fin["interest_coverage"],
            "ebitda_margin": fin["ebitda_margin"],
            "log_total_assets": fin["log_total_assets"],
            "sector": sector,
            "tenor_years": tenor,
            "base_rate_pct": base_rate,
            "spread_bps": spread_bps,
            "yield_pct": yield_pct,
        })

    return pd.DataFrame(rows)