from src.data_ingestion.market_data import download_all_assets
from src.data_ingestion.macro_data import download_all_macro
from src.data_ingestion.news_data import download_all_news
from src.data_storage.database_setup import test_connection, setup_timescaledb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """Run the complete data pipeline"""
    logger.info("🚀 Starting FinLagX Data Pipeline with TimescaleDB...\n")
    
    # Test database connection first
    logger.info("🔗 Testing database connection...")
    test_connection()
    
    try:
        # 1. Market Data
        logger.info("\n📈 Starting Market Data Collection...")
        download_all_assets()
        
        # 2. Macro Economic Data
        logger.info("\n📊 Starting Macro Data Collection...")
        download_all_macro()
        
        # 3. News Data
        logger.info("\n📰 Starting News Data Collection...")
        download_all_news()
        
        logger.info("\n✅ Complete data pipeline finished successfully!")
        
        # Show summary
        show_pipeline_summary()
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

def show_pipeline_summary():
    """Show summary of data collected"""
    try:
        from src.data_ingestion.market_data import get_latest_data
        from src.data_ingestion.macro_data import get_latest_macro_values
        from src.data_ingestion.news_data import get_news_stats
        
        logger.info("\n📋 PIPELINE SUMMARY:")
        logger.info("=" * 50)
        
        # Market data summary (TimescaleDB)
        market_data = get_latest_data(limit=1)
        if not market_data.empty:
            latest_market = market_data.iloc[0]['time']
            logger.info(f"📈 Market Data (TimescaleDB): Latest entry from {latest_market}")
        
        # Macro data summary (TimescaleDB)
        macro_data = get_latest_macro_values()
        if not macro_data.empty:
            logger.info(f"📊 Macro Data (TimescaleDB): {len(macro_data)} indicators updated")
        
        # News data summary (MongoDB)
        news_stats = get_news_stats()
        if not news_stats.empty:
            total_articles = news_stats['article_count'].sum()
            logger.info(f"📰 News Data (MongoDB): {total_articles} articles across {len(news_stats)} categories")
            
    except Exception as e:
        logger.warning(f"Could not generate summary: {e}")

def run_market_only():
    """Run only market data pipeline"""
    logger.info("📈 Running Market Data Only...")
    test_connection()
    download_all_assets()

def run_macro_only():
    """Run only macro data pipeline"""
    logger.info("📊 Running Macro Data Only...")
    test_connection()
    download_all_macro()

def run_news_only():
    """Run only news data pipeline"""
    logger.info("📰 Running News Data Only...")
    test_connection()
    download_all_news()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "market":
            run_market_only()
        elif mode == "macro":
            run_macro_only()
        elif mode == "news":
            run_news_only()
        else:
            logger.error("Usage: python run_data_pipeline.py [market|macro|news]")
    else:
        run_full_pipeline()