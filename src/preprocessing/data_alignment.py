"""
Data Alignment Module - LEAN VERSION
Simple alignment of market + news data, no heavy features
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

from src.preprocessing.market_preprocessing import get_processed_market_data
from src.preprocessing.news_preprocessing import get_processed_news_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAligner:
    """Lean data aligner - simple market + news alignment"""
    
    def __init__(self):
        pass
    
    def aggregate_news_by_day(self, news_df):
        """Aggregate news by day - just counts per category"""
        logger.info("📅 Aggregating news by day...")
        
        if news_df.empty:
            return pd.DataFrame()
        
        news_df = news_df.copy()
        news_df['date'] = pd.to_datetime(news_df['timestamp']).dt.date
        
        # Simple count by date and category
        daily_news = news_df.groupby(['date', 'category']).size().reset_index(name='news_count')
        daily_news['date'] = pd.to_datetime(daily_news['date'])
        
        logger.info(f"  Aggregated to {len(daily_news)} day-category pairs")
        
        return daily_news
    
    def align_market_news(self, market_df, news_df):
        """
        Simple alignment: just add news counts to market data
        No lags, no complex features
        """
        logger.info(f"🔗 Aligning market and news data...")
        
        if market_df.empty:
            logger.error("  Empty market data")
            return pd.DataFrame()
        
        market_df = market_df.copy()
        market_df['date'] = pd.to_datetime(market_df['time']).dt.date
        market_df['date'] = pd.to_datetime(market_df['date'])
        
        if news_df.empty:
            logger.warning("⚠️ No news data, continuing with market only")
            # Add empty news columns
            market_df['news_count'] = 0
            return market_df
        
        # Aggregate news
        daily_news = self.aggregate_news_by_day(news_df)
        
        # Pivot: categories as columns
        news_pivot = daily_news.pivot_table(
            index='date',
            columns='category',
            values='news_count',
            fill_value=0
        ).reset_index()
        
        # Rename columns
        news_pivot.columns = ['date'] + [f'news_{col}' for col in news_pivot.columns[1:]]
        
        # Simple merge
        merged = market_df.merge(news_pivot, on='date', how='left')
        
        # Fill missing news counts with 0
        news_cols = [col for col in merged.columns if 'news_' in col]
        merged[news_cols] = merged[news_cols].fillna(0)
        
        logger.info(f"  Aligned: {len(merged)} rows, {len(merged.columns)} columns")
        
        return merged
    
    def create_complete_dataset(self, symbols=None, categories=None, 
                                start_date=None, end_date=None):
        """
        Create simple aligned dataset
        Just: time, symbol, category, close_price, volume, returns, volatility, sma, news_counts
        """
        logger.info("  Creating lean aligned dataset...\n")
        
        # Get data
        market_df = get_processed_market_data(symbols, start_date, end_date)
        
        if market_df.empty:
            logger.error("  No market data")
            return pd.DataFrame()
        
        news_df = get_processed_news_data(categories, start_date, end_date)
        
        # Align
        aligned_df = self.align_market_news(market_df, news_df)
        
        # Sort
        aligned_df = aligned_df.sort_values(['symbol', 'time'])
        
        # Drop the 'date' helper column
        if 'date' in aligned_df.columns:
            aligned_df = aligned_df.drop('date', axis=1)
        
        logger.info("\n  Lean dataset created!")
        logger.info(f"   Shape: {aligned_df.shape}")
        logger.info(f"   Symbols: {aligned_df['symbol'].nunique()}")
        logger.info(f"   Date range: {aligned_df['time'].min()} to {aligned_df['time'].max()}")
        logger.info(f"   Memory: {aligned_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return aligned_df
    
    def save_aligned_data(self, df, filename='aligned_dataset.parquet'):
        """Save to parquet with compression"""
        import os
        
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        # Use compression to reduce file size
        df.to_parquet(filepath, index=False, compression='gzip')
        
        # Check file size
        file_size = os.path.getsize(filepath) / 1024**2
        logger.info(f"💾 Saved to {filepath} ({file_size:.2f} MB)")

def load_aligned_data(filename='aligned_dataset.parquet'):
    """Load aligned dataset"""
    import os
    filepath = os.path.join('data/processed', filename)
    
    if not os.path.exists(filepath):
        logger.error(f"  File not found: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_parquet(filepath)
    logger.info(f"  Loaded from {filepath}")
    logger.info(f"   Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    aligner = DataAligner()
    
    # Create lean dataset
    aligned_df = aligner.create_complete_dataset()
    
    if not aligned_df.empty:
        aligner.save_aligned_data(aligned_df)
        
        print("\n" + "="*80)
        print("LEAN ALIGNED DATASET")
        print("="*80)
        print(aligned_df.head(10))
        print(f"\nColumns: {aligned_df.columns.tolist()}")
        print(f"Shape: {aligned_df.shape}")
        print(f"Memory: {aligned_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")