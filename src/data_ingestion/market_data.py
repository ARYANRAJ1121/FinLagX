import os
import pandas as pd
from datetime import datetime, date
import yaml
import yfinance as yf
from sqlalchemy import text
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
        # Download data
        logger.info(f"Downloading {name} ({ticker})...")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        
        if df.empty:
            logger.warning(f"No data for {name} ({ticker})")
            return None
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index
        df = df.reset_index()
        
        # Add metadata
        df['symbol'] = name
        df['category'] = category
        
        # Rename columns
        column_mapping = {
            'Date': 'time',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'volume'
        }
        
        if 'Adj Close' in df.columns:
            column_mapping['Adj Close'] = 'adj_close'
        else:
            df['adj_close'] = df['Close']
        
        df = df.rename(columns=column_mapping)
        
        # Ensure correct column order
        df = df[['time', 'symbol', 'category', 'open_price', 'high_price', 
                 'low_price', 'close_price', 'adj_close', 'volume']]
        
        # Convert data types explicitly
        df['time'] = pd.to_datetime(df['time'])
        df['symbol'] = df['symbol'].astype(str)
        df['category'] = df['category'].astype(str)
        
        # Convert numeric columns
        numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'adj_close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in critical columns
        df = df.dropna(subset=['time', 'close_price'])
        
        # Fill remaining NaN
        df = df.fillna(method='ffill').fillna(0)
        
        if df.empty:
            logger.warning(f"No valid data after cleaning for {name}")
            return None
        
        # Insert using raw SQL with ON CONFLICT to handle duplicates properly
        inserted_count = 0
        updated_count = 0
        
        with engine.connect() as conn:
            for _, row in df.iterrows():
                try:
                    # Use INSERT ... ON CONFLICT UPDATE to handle duplicates
                    sql = text("""
                        INSERT INTO market_data 
                        (time, symbol, category, open_price, high_price, low_price, 
                         close_price, adj_close, volume)
                        VALUES 
                        (:time, :symbol, :category, :open_price, :high_price, :low_price,
                         :close_price, :adj_close, :volume)
                        ON CONFLICT (time, symbol) 
                        DO UPDATE SET
                            open_price = EXCLUDED.open_price,
                            high_price = EXCLUDED.high_price,
                            low_price = EXCLUDED.low_price,
                            close_price = EXCLUDED.close_price,
                            adj_close = EXCLUDED.adj_close,
                            volume = EXCLUDED.volume
                    """)
                    
                    conn.execute(sql, {
                        'time': row['time'],
                        'symbol': row['symbol'],
                        'category': row['category'],
                        'open_price': float(row['open_price']),
                        'high_price': float(row['high_price']),
                        'low_price': float(row['low_price']),
                        'close_price': float(row['close_price']),
                        'adj_close': float(row['adj_close']),
                        'volume': int(row['volume'])
                    })
                    inserted_count += 1
                    
                except Exception as e:
                    logger.error(f"Error inserting row for {name} at {row['time']}: {e}")
                    continue
            
            conn.commit()
        
        logger.info(f"  Processed {inserted_count} rows for {name}")
        return df
        
    except Exception as e:
        logger.error(f"  Failed {name} ({ticker}): {e}")
        import traceback
        traceback.print_exc()
        return None

def download_all_assets():
    engine = get_engine()
    end_date = date.today().isoformat()
    
    logger.info(f"📅 Downloading from {START_DATE} to {end_date}\n")
    
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
    logger.info("  Starting Market Data Pipeline...\n")
    download_all_assets()
    
    logger.info("\n📊 Testing data retrieval...")
    recent_data = get_latest_data(limit=5)
    logger.info(f"Recent data shape: {recent_data.shape}")
    
    if not recent_data.empty:
        print("\n" + "="*80)
        print("LATEST DATA SAMPLE:")
        print("="*80)
        print(recent_data[['time', 'symbol', 'close_price', 'volume']].head())
        print("="*80)
    else:
        logger.warning("  No data retrieved from database!")
    
    logger.info("\n  Market data pipeline completed!")