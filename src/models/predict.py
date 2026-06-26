import joblib, json
import pandas as pd
import numpy as np

FEATURE_COLS = json.load(open("models/feature_names.json"))

def predict_closure_probability(
    weather_7day: pd.DataFrame,
    segment: str,
    month: int
) -> pd.DataFrame:
    """
    Input:
        weather_7day: DataFrame, 7 rows, columns: date, temp_max, temp_min,
                      snowfall_sum, precip_sum, windspeed_max
        segment:      one of Leh_city | Zoji_La | Rohtang_Pass | Khardung_La | Manali
        month:        integer 1-12

    Output: DataFrame with columns: date, closure_prob, hazard_type, segment
    """
    terrain = pd.read_csv("data/terrain_features.csv")
    seg = terrain[terrain["segment"] == segment].iloc[0]

    df = weather_7day.copy().sort_values("date").reset_index(drop=True)
    df["snowfall_3d"] = df["snowfall_sum"].rolling(3, min_periods=1).sum()
    df["precip_3d"]   = df["precip_sum"].rolling(3, min_periods=1).sum()
    df["temp_min_3d"] = df["temp_min"].rolling(3, min_periods=1).min()
    df["month"]       = month
    doy = pd.to_datetime(df["date"]).dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["elevation_m"] = seg["elevation_m"]
    df["slope_deg"]   = seg["slope_deg"]
    df["aspect_deg"]  = seg["aspect_deg"]

    model_path = "models/xgb_avalanche.pkl" if month in [10,11,12,1,2,3,4,5] \
                 else "models/xgb_landslide.pkl"
    hazard = "avalanche" if month in [10,11,12,1,2,3,4,5] else "landslide"

    bundle = joblib.load(model_path)
    probs  = bundle["model"].predict_proba(df[FEATURE_COLS])[:, 1]

    return pd.DataFrame({
        "date":         df["date"].values,
        "closure_prob": probs,
        "hazard_type":  hazard,
        "segment":      segment
    })
