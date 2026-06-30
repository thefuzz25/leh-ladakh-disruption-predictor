import pandas as pd
import numpy as np

FEATURE_COLS = [
    "temp_max", "temp_min", "snowfall_sum", "precip_sum", "windspeed_max",
    "snowfall_3d", "precip_3d", "temp_min_3d",
    "month", "doy_sin", "doy_cos",
    "elevation_m", "slope_deg", "aspect_deg"
]

def build_features(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ML input matrix (14 features).

    METEOROLOGICAL:    temp_max, temp_min, snowfall_sum, precip_sum, windspeed_max
    ROLLING WINDOWS:   snowfall_3d (avalanche buildup), precip_3d (soil saturation),
                       temp_min_3d (freeze-thaw cycles)
    TEMPORAL:          month, doy_sin, doy_cos
                       (sin/cos encoding so Dec and Jan are numerically adjacent)
    TERRAIN (static):  elevation_m, slope_deg, aspect_deg — joined from terrain_features.csv
    """
    df = labels_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["snowfall_3d"] = df["snowfall_sum"].rolling(3, min_periods=1).sum()
    df["precip_3d"]   = df["precip_sum"].rolling(3, min_periods=1).sum()
    df["temp_min_3d"] = df["temp_min"].rolling(3, min_periods=1).min()

    df["month"] = pd.to_datetime(df["date"]).dt.month
    doy = pd.to_datetime(df["date"]).dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # Cross-join: each weather day × each road segment
    terrain = pd.read_csv("data/terrain_features.csv")
    df["_key"] = 1
    terrain["_key"] = 1
    df = df.merge(terrain, on="_key").drop("_key", axis=1)

    df = df.dropna().reset_index(drop=True)
    df.to_csv("data/processed/features.csv", index=False)
    print(f"Feature matrix: {df.shape[0]} rows x {df.shape[1]} cols")
    assert df[FEATURE_COLS].isnull().sum().sum() == 0, "Nulls in feature matrix"
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/labels.csv")
    build_features(df)
