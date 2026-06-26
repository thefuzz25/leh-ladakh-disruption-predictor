import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd

def fetch_weather_leh() -> pd.DataFrame:
    """
    Fetch daily weather for Leh (34.15N, 77.58E), 2019-01-01 to 2024-12-31.
    Open-Meteo archive API — free, no key required.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": 34.15,
        "longitude": 77.58,
        "start_date": "2019-01-01",
        "end_date": "2024-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "snowfall_sum",
            "precipitation_sum",
            "windspeed_10m_max"
        ],
        "timezone": "Asia/Kolkata"
    }

    responses = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    r = responses[0]
    daily = r.Daily()

    df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temp_max":      daily.Variables(0).ValuesAsNumpy(),
        "temp_min":      daily.Variables(1).ValuesAsNumpy(),
        "snowfall_sum":  daily.Variables(2).ValuesAsNumpy(),
        "precip_sum":    daily.Variables(3).ValuesAsNumpy(),
        "windspeed_max": daily.Variables(4).ValuesAsNumpy(),
    })
    df["date"] = df["date"].dt.date
    df.to_csv("data/raw/weather_leh.csv", index=False)
    print(f"Fetched {len(df)} days of weather data")
    return df

if __name__ == "__main__":
    fetch_weather_leh()
