import pandas as pd
import numpy as np

def generate_labels(weather_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic binary closure labels for two separate hazard types.

    Design decision: two hazard types mirror how BRO actually categorises closures.
    Thresholds grounded in Tiwari et al. (2021) field data for Leh-Manali highway.

    AVALANCHE (Oct-May, snowfall-driven):
      P = 0.80 if snowfall > 20cm AND temp_min < -12C
      P = 0.45 if snowfall > 20cm only
      P = 0.05 otherwise (within season)

    LANDSLIDE (Jun-Sep, rain-driven):
      P = 0.70 if precip > 40mm AND windspeed > 50km/h
      P = 0.35 if precip > 40mm only
      P = 0.03 otherwise (within season)

    12% random label noise simulates real-world labelling uncertainty.
    """
    np.random.seed(seed)
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    aval_season = df["month"].isin([10, 11, 12, 1, 2, 3, 4, 5])
    land_season = df["month"].isin([6, 7, 8, 9])

    p_aval = np.where(
        aval_season & (df["snowfall_sum"] > 20) & (df["temp_min"] < -12), 0.80,
        np.where(aval_season & (df["snowfall_sum"] > 20), 0.45,
        np.where(aval_season, 0.05, 0.01))
    )

    p_land = np.where(
        land_season & (df["precip_sum"] > 40) & (df["windspeed_max"] > 50), 0.70,
        np.where(land_season & (df["precip_sum"] > 40), 0.35,
        np.where(land_season, 0.03, 0.01))
    )

    noise = np.random.binomial(1, 0.12, len(df))
    df["avalanche_closure"] = np.where(
        noise == 1,
        np.random.binomial(1, 0.5, len(df)),
        np.random.binomial(1, p_aval)
    )
    df["landslide_closure"] = np.where(
        noise == 1,
        np.random.binomial(1, 0.5, len(df)),
        np.random.binomial(1, p_land)
    )

    df.to_csv("data/processed/synthetic_labels.csv", index=False)
    print(f"Avalanche closure rate: {df['avalanche_closure'].mean():.2%}")
    print(f"Landslide closure rate: {df['landslide_closure'].mean():.2%}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/weather_leh.csv")
    generate_labels(df)
