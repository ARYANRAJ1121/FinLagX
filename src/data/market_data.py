import os
import pandas as pd
from datetime import datetime
import yaml
import yfinance as yf

CONFIG_PATH = "src/data/config_market.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

START_DATE = config["start_date"]
BASE_DATA_DIR = "data/raw/market"

def download_asset(ticker: str, name: str, category: str, start: str, end: str):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            print(f"⚠️ No data for {name} ({ticker})")
            return None
        
        # Create subfolder for category
        category_dir = os.path.join(BASE_DATA_DIR, category)
        os.makedirs(category_dir, exist_ok=True)

        file_path = os.path.join(category_dir, f"{name}.csv")
        df.to_csv(file_path, index=True) # THIS HAS BEEN CORRECTED
        print(f"✅ Saved {name} -> {file_path}")
        return df
    except Exception as e:
        print(f"❌ Failed {name} ({ticker}): {e}")
        return None

def download_all_assets():
    end_date = datetime.today().strftime("%Y-%m-%d")
    for category, assets in config.items():
        if category == "start_date":
            continue
        print(f"\n📈 Category: {category.upper()}")
        for name, ticker in assets.items():
            download_asset(ticker, name, category, START_DATE, end_date)

if __name__ == "__main__":
    download_all_assets()