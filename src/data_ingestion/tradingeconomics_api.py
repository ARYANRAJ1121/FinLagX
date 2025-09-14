# src/data_ingestion/tradingeconomics_api.py
import os
import requests
import pandas as pd
from datetime import datetime

BASE_PATH = "data/raw/news/macro_api"
os.makedirs(BASE_PATH, exist_ok=True)

def fetch_te_news():
    url = "https://api.tradingeconomics.com/news?c=guest:guest&f=json"
    resp = requests.get(url)
    data = resp.json()

    df = pd.DataFrame(data)
    file_path = os.path.join(BASE_PATH, f"tradingeconomics_news_{datetime.today().strftime('%Y-%m-%d')}.parquet")
    df.to_parquet(file_path, index=False)
    print(f"✅ Saved TradingEconomics news -> {file_path}")
