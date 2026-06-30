import pandas as pd
import numpy as np
import joblib, json

"""
Precompute REAL per-month closure probabilities for the two highway segments shown in
the dashboard, so the live app reads genuine model output instead of inventing random
numbers. Writes app/precomputed_probs.csv (month, Zoji_La, Rohtang_Pass).

For each (segment, month) we take the mean predicted closure probability the trained
model assigns across all historical days for that segment+month — a real, defensible
monthly risk profile. The deployed app only reads this small CSV (no xgboost needed).
"""

FEATURE_COLS = json.load(open("models/feature_names.json"))
AVAL_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]
SEGMENTS = ["Zoji_La", "Rohtang_Pass"]


def precompute():
    df = pd.read_csv("data/processed/features.csv")
    aval = joblib.load("models/xgb_avalanche.pkl")["model"]
    land = joblib.load("models/xgb_landslide.pkl")["model"]

    rows = []
    for month in range(1, 13):
        model = aval if month in AVAL_MONTHS else land
        row = {"month": month}
        for seg in SEGMENTS:
            sub = df[(df["segment"] == seg) & (df["month"] == month)]
            if len(sub) == 0:
                row[seg] = 0.0
            else:
                prob = model.predict_proba(sub[FEATURE_COLS])[:, 1].mean()
                row[seg] = round(float(prob), 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv("app/precomputed_probs.csv", index=False)
    print("Saved: app/precomputed_probs.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    precompute()
