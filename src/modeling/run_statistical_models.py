#!/usr/bin/env python3
"""
Run Complete Statistical Analysis Pipeline
Executes both Granger Causality and VAR analysis
"""
import sys
import logging
import argparse
from datetime import datetime, timedelta

# Add parent directory to path to import modules
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling.granger_causality import GrangerCausalityAnalyzer
from src.modeling.var_analysis import VARAnalyzer
from src.data_storage.database_setup import check_tables

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_data_exists():
    """
    Verify that market_features table has data
    """
    logger.info("🔍 Verifying data availability...")
    
    from src.data_storage.database_setup import get_engine
    from sqlalchemy import text
    
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM market_features"))
            count = result.fetchone()[0]
            
            if count == 0:
                logger.error("  market_features table is empty!")
                logger.info("   Run preprocessing first:")
                logger.info("   python run_complete_pipeline.py")
                return False
            
            # Get date range
            result = conn.execute(text("""
                SELECT 
                    MIN(time) as min_date,
                    MAX(time) as max_date,
                    COUNT(DISTINCT symbol) as num_symbols
                FROM market_features
            """))
            
            row = result.fetchone()
            logger.info(f"  Data available:")
            logger.info(f"   Rows: {count:,}")
            logger.info(f"   Symbols: {row[2]}")
            logger.info(f"   Date range: {row[0]} to {row[1]}")
            
            return True
            
    except Exception as e:
        logger.error(f"  Error checking data: {e}")
        return False


def run_granger_analysis(symbols=None, start_date='2015-01-01', end_date=None, feature='returns'):
    """
    Run Granger Causality Analysis
    
    Args:
        symbols: List of symbols to analyze
        start_date: Start date (default: 2015-01-01 for ~2500-3000 rows)
        end_date: End date
        feature: Feature to analyze ('returns', 'return_5d', 'volatility_20')
    """
    logger.info("\n" + "  "*20)
    logger.info("STEP 1: GRANGER CAUSALITY ANALYSIS")
    logger.info(f"Feature: {feature} | From: {start_date}")
    logger.info("  "*20 + "\n")
    
    try:
        analyzer = GrangerCausalityAnalyzer(
            max_lag=10,
            significance_level=0.05
        )
        
        results = analyzer.run_full_analysis(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            save=True,
            top_n=20,
            feature=feature
        )
        
        if results is not None and not results.empty:
            logger.info("  Granger analysis completed")
            return True
        else:
            logger.error("  Granger analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"  Granger analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_var_analysis(symbols=None, start_date=None, end_date=None):
    """
    Run VAR Model Analysis
    """
    logger.info("\n" + "  "*20)
    logger.info("STEP 2: VAR MODEL ANALYSIS")
    logger.info("  "*20 + "\n")
    
    try:
        analyzer = VARAnalyzer(max_lags=10)
        
        results = analyzer.run_full_analysis(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            save=True,
            lag_order=None
        )
        
        if results is not None and not results.empty:
            logger.info("  VAR analysis completed")
            return True
        else:
            logger.error("  VAR analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"  VAR analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_results():
    """
    Verify that results were saved to database
    """
    logger.info("\n🔍 Verifying saved results...")
    
    from src.data_storage.database_setup import get_engine
    from sqlalchemy import text
    
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            # Check Granger results
            result = conn.execute(text("SELECT COUNT(*) FROM granger_results"))
            granger_count = result.fetchone()[0]
            
            # Check VAR features
            result = conn.execute(text("SELECT COUNT(*) FROM var_features"))
            var_count = result.fetchone()[0]
            
            logger.info(f"  Results verification:")
            logger.info(f"   granger_results: {granger_count:,} rows")
            logger.info(f"   var_features: {var_count:,} rows")
            
            if granger_count > 0 and var_count > 0:
                return True
            else:
                logger.warning("  Some results tables are empty")
                return False
                
    except Exception as e:
        logger.error(f"  Error verifying results: {e}")
        return False


def show_next_steps():
    """
    Show what to do next
    """
    logger.info("\n" + "="*80)
    logger.info("  NEXT STEPS")
    logger.info("="*80)
    logger.info("\n1. View results in database:")
    logger.info("   • Granger results: SELECT * FROM granger_results ORDER BY granger_score DESC;")
    logger.info("   • VAR features: SELECT * FROM var_features LIMIT 100;")
    logger.info("\n2. Visualize results:")
    logger.info("   python -m src.visualization.plot_granger_network")
    logger.info("\n3. Run deep learning models:")
    logger.info("   python -m src.modeling.lstm_model")
    logger.info("\n4. View in PgAdmin:")
    logger.info("   http://localhost:8080")
    logger.info("="*80 + "\n")


def main():
    """
    Main execution with command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Run Statistical Models (Granger + VAR)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='List of symbols to analyze (default: all)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--granger-only',
        action='store_true',
        help='Run only Granger causality analysis'
    )
    
    parser.add_argument(
        '--var-only',
        action='store_true',
        help='Run only VAR model analysis'
    )
    
    args = parser.parse_args()
    
    # Banner
    logger.info("\n" + "="*80)
    logger.info("FINLAGX STATISTICAL MODELING PIPELINE")
    logger.info("="*80 + "\n")
    
    start_time = datetime.now()
    
    # Step 1: Verify data
    if not verify_data_exists():
        logger.error("  Cannot proceed without data. Exiting.")
        sys.exit(1)
    
    # Step 2: Show tables
    check_tables()
    
    # Step 3: Run analyses
    results = {}
    
    if not args.var_only:
        logger.info("\n>>> Running Granger Causality Analysis...")
        results['granger'] = run_granger_analysis(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date
        )
    
    if not args.granger_only:
        logger.info("\n>>> Running VAR Model Analysis...")
        results['var'] = run_var_analysis(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date
        )
    
    # Step 4: Verify results
    verify_results()
    
    # Summary
    duration = datetime.now() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*80)
    
    for model, success in results.items():
        status = "  SUCCESS" if success else "  FAILED"
        logger.info(f"   {model.upper()}: {status}")
    
    logger.info(f"\n  Total Duration: {duration}")
    logger.info("="*80)
    
    if all(results.values()):
        logger.info("\n  Statistical modeling completed successfully!")
        show_next_steps()
        return True
    else:
        logger.error("\n  Pipeline completed with errors")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)