import os
import pandas as pd
from datetime import datetime
import yaml
import pandas_datareader.data as web
from sqlalchemy import text
from src.data_storage.database_setup import get_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/config_macro.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

START_DATE = config["start_date"]
MACRO = config["macro"]

def download_macro_indicator_to_db(code: str, name: str, start: str, end: str, engine):
    try:
        df = web.DataReader(code, "fred", start, end)
        if df.empty:
            logger.warning(f"No data for {name} ({code})")
            return None
        df = df.reset_index()
        df = df.rename(columns={'DATE': 'time', code: 'value'})
        df['indicator'] = name
        df = df[['time', 'indicator', 'value']]
        df = df.dropna()
        df.to_sql('macro_data', engine, if_exists='append', index=False, method='multi')
        logger.info(f"  Saved {len(df)} rows for {name} to database")
        return df
    except Exception as e:
        logger.error(f"  Failed {name} ({code}): {e}")
        return None

def download_all_macro():
    engine = get_engine()
    end_date = datetime.today().strftime("%Y-%m-%d")
    logger.info("📊 Starting Macro Data Download...")
    for name, code in MACRO.items():
        logger.info(f"\n📊 Downloading {name} ({code})...")
        download_macro_indicator_to_db(code, name, START_DATE, end_date, engine)

def get_macro_data(indicator=None, start_date=None, end_date=None):
    engine = get_engine()
    query = """
    SELECT * FROM macro_data 
    WHERE 1=1
    """
    params = {}
    if indicator:
        query += " AND indicator = %(indicator)s"
        params['indicator'] = indicator
    if start_date:
        query += " AND time >= %(start_date)s"
        params['start_date'] = start_date
    if end_date:
        query += " AND time <= %(end_date)s"
        params['end_date'] = end_date
    query += " ORDER BY time DESC"
    
    return pd.read_sql(query, engine, params=params)


def get_latest_macro_values():
    engine = get_engine()
    query = """
    SELECT DISTINCT ON (indicator) 
        indicator, time, value
    FROM macro_data 
    ORDER BY indicator, time DESC
    """
    return pd.read_sql(query, engine)

def get_macro_correlation_data(indicators, start_date, end_date):
    engine = get_engine()
    query = """
    SELECT time, indicator, value
    FROM macro_data 
    WHERE indicator = ANY(%(indicators)s)
    AND time BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY time, indicator
    """
    df = pd.read_sql(query, engine, params={
        'indicators': indicators,
        'start_date': start_date,
        'end_date': end_date
    })
    return df.pivot(index='time', columns='indicator', values='value')

if __name__ == "__main__":
    logger.info("  Starting Macro Data Pipeline with TimescaleDB...\n")
    download_all_macro()
    logger.info("\n📊 Testing data retrieval...")
    latest = get_latest_macro_values()
    logger.info(f"Latest macro values: {latest.shape}")
    if not latest.empty:
        print(latest)
    cpi_data = get_macro_data(indicator='CPI', start_date='2023-01-01')
    logger.info(f"CPI data points since 2023: {len(cpi_data)}")
    logger.info("\n  Macro data pipeline completed!")