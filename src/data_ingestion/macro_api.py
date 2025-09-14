import os
import requests
import pandas as pd
from datetime import datetime

BASE_PATH = "data/raw/news/macro_api"
os.makedirs(BASE_PATH, exist_ok=True)

FRED_API_KEY = os.getenv("FRED_API_KEY")  # free key from https://fred.stlouisfed.org/

def fetch_fred_series(series_id, name):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    resp = requests.get(url, params=params)
    data = resp.json()["observations"]

    df = pd.DataFrame(data)
    df.to_parquet(os.path.join(BASE_PATH, f"{name}_{datetime.today().strftime('%Y-%m-%d')}.parquet"))
    print(f"✅ Saved macro series {name} -> {BASE_PATH}")

def run_macro_api():
    fetch_fred_series("CPIAUCSL", "us_cpi")
    fetch_fred_series("GDP", "us_gdp")
    fetch_fred_series("FEDFUNDS", "fed_funds_rate")
    fetch_fred_series("UNRATE", "us_unemployment")
