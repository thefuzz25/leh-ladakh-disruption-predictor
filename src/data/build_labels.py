import pandas as pd
import numpy as np

"""
Build REAL-ANCHORED closure labels (replaces the old fully-synthetic generator).

Provenance hierarchy (highest authority first):
  1. DOCUMENTED  — real reported closure events (data/documented_closures.csv).
                   Deterministic label = 1 on those date ranges. Real ground truth.
  2. CALENDAR    — real BRO seasonal core-winter hard-closure windows
                   (data/closure_calendar.csv). Deep winter is known-shut; we set a
                   high closure probability inside these windows. Real, deterministic-ish.
  3. WEATHER     — for the remaining (shoulder / open-season) days, closure probability
                   is derived from the REAL Open-Meteo weather via published physical
                   thresholds (Tiwari et al. 2021 style). Logical, weather-driven.

KEY HONESTY CHANGE vs the old version: the 12% random 50/50 coin-flip relabelling is
GONE. Stochasticity now only enters as a Bernoulli draw from a weather/calendar-derived
probability — a principled generative process, not invented noise. The deep-winter
seasonal closure is the real, known part; the shoulder-season event layer is where a
predictive model actually adds value.
"""

AVAL_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]
LAND_MONTHS = [6, 7, 8, 9]


def _in_recurring_window(dates: pd.Series, start_md: str, end_md: str) -> np.ndarray:
    """True where the month-day falls in [start_md, end_md], handling year-wrap."""
    md = dates.dt.strftime("%m-%d")
    if start_md <= end_md:
        return (md >= start_md) & (md <= end_md)
    # wraps over new year (e.g. 12-15 .. 03-15)
    return (md >= start_md) | (md <= end_md)


def build_labels(weather_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    aval_season = df["month"].isin(AVAL_MONTHS)
    land_season = df["month"].isin(LAND_MONTHS)

    # ── 3. WEATHER-DERIVED baseline probabilities (real weather, published thresholds) ──
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

    # provenance tags (default: weather-modelled within season, else off-season)
    src_aval = np.where(aval_season, "weather", "offseason").astype(object)
    src_land = np.where(land_season, "weather", "offseason").astype(object)

    # ── 2. CALENDAR hard-closure windows raise the avalanche probability ──
    cal = pd.read_csv("data/closure_calendar.csv")
    cal_mask_total = np.zeros(len(df), dtype=bool)
    for _, c in cal.iterrows():
        if c["hazard"] != "avalanche":
            continue
        win = _in_recurring_window(df["date"], c["start_md"], c["end_md"])
        p_aval = np.where(win, np.maximum(p_aval, 0.75), p_aval)
        src_aval = np.where(win, "calendar", src_aval)
        cal_mask_total |= win.values

    # ── draw stochastic labels from the principled probabilities ──
    df["avalanche_closure"] = np.random.binomial(1, p_aval)
    df["landslide_closure"] = np.random.binomial(1, p_land)
    df["aval_label_source"] = src_aval
    df["land_label_source"] = src_land

    # ── 1. DOCUMENTED real events override deterministically with label = 1 ──
    doc = pd.read_csv("data/documented_closures.csv", comment="#", parse_dates=["start_date", "end_date"])
    doc_aval_days = 0
    doc_land_days = 0
    for _, e in doc.iterrows():
        span = (df["date"] >= e["start_date"]) & (df["date"] <= e["end_date"])
        if e["hazard"] == "avalanche":
            df.loc[span, "avalanche_closure"] = 1
            df.loc[span, "aval_label_source"] = "documented"
            doc_aval_days += int(span.sum())
        elif e["hazard"] == "landslide":
            df.loc[span, "landslide_closure"] = 1
            df.loc[span, "land_label_source"] = "documented"
            doc_land_days += int(span.sum())

    df.to_csv("data/processed/labels.csv", index=False)

    # ── provenance report ──
    n = len(df)
    print(f"Total days: {n}")
    print(f"Avalanche closure rate: {df['avalanche_closure'].mean():.2%}")
    print(f"Landslide closure rate: {df['landslide_closure'].mean():.2%}")
    print("\nLabel provenance (avalanche track):")
    print(f"  documented (REAL events):        {doc_aval_days} days")
    print(f"  calendar (REAL seasonal window): {int(cal_mask_total.sum())} days")
    print(f"  weather-modelled:                {int((src_aval == 'weather').sum())} days")
    print(f"Landslide documented (REAL) days:  {doc_land_days}")
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/weather_leh.csv")
    build_labels(df)
