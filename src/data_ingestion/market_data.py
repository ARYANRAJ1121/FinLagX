import os
import pandas as pd
from datetime import datetime
import yaml
import yfinance as yf
from sqlalchemy import create_engine
from src.data_storage.database_setup import get_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/config_market.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

START_DATE = config["start_date"]

def download_asset_to_db(ticker: str, name: str, category: str, start: str, end: str, engine):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            logger.warning(f"No data for {name} ({ticker})")
            return None
        df = df.reset_index()
        df['symbol'] = name
        df['category'] = category
        df = df.rename(columns={
            'Date': 'time',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        })
        df = df[['time', 'symbol', 'category', 'open_price', 'high_price',
                'low_price', 'close_price', 'adj_close', 'volume']]
        df.to_sql('market_data', engine, if_exists='append', index=False, method='multi')
        logger.info(f"✅ Saved {len(df)} rows for {name} to database")
        return df
    except Exception as e:
        logger.error(f"❌ Failed {name} ({ticker}): {e}")
        return None

def download_all_assets():
    engine = get_engine()
    end_date = datetime.today().strftime("%Y-%m-%d")
    for category, assets in config.items():
        if category == "start_date":
            continue
        logger.info(f"\n📈 Category: {category.upper()}")
        for name, ticker in assets.items():
            download_asset_to_db(ticker, name, category, START_DATE, end_date, engine)

def get_latest_data(symbol=None, category=None, limit=100):
    engine = get_engine()
    query = """
    SELECT * FROM market_data 
    WHERE 1=1
    """
    params = {}
    
    if symbol:
        query += " AND symbol = %(symbol)s"
        params['symbol'] = symbol
    
    if category:
        query += " AND category = %(category)s"
        params['category'] = category
    
    query += " ORDER BY time DESC LIMIT %(limit)s"
    params['limit'] = limit
    
    return pd.read_sql(query, engine, params=params)


def get_price_data_range(symbol, start_date, end_date):
    engine = get_engine()
    query = """
    SELECT time, symbol, open_price, high_price, low_price, close_price, volume
    FROM market_data 
    WHERE symbol = %(symbol)s 
    AND time BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY time
    """
    return pd.read_sql(query, engine, params={
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date
    })

if __name__ == "__main__":
    logger.info("🚀 Starting Market Data Pipeline with TimescaleDB...\n")
    download_all_assets()
    logger.info("\n📊 Testing data retrieval...")
    recent_data = get_latest_data(limit=5)
    logger.info(f"Recent data shape: {recent_data.shape}")
    if not recent_data.empty:
        print(recent_data[['time', 'symbol', 'close_price']].head())
    logger.info("\n✅ Market data pipeline completed!")