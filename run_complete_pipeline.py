#!/usr/bin/env python3
"""
Complete FinLagX Pipeline - Market & Macro Data Only
Runs: Data Ingestion → Preprocessing → Feature Store → Ready for Modeling
"""
import sys
import logging
import argparse
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_data_ingestion(clean_first=False):
    """Run data ingestion pipeline for market and macro data"""
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA INGESTION (Market + Macro)")
    logger.info("="*80)
    
    from src.data_ingestion.market_data import download_all_assets
    from src.data_ingestion.macro_data import download_all_macro
    from src.data_storage.database_setup import clean_raw_data
    
    try:
        if clean_first:
            logger.info("🧹 Cleaning existing data...")
            clean_raw_data()
        
        # Download market data
        logger.info("\n📈 Downloading market data...")
        download_all_assets()
        
        # Download macro data
        logger.info("\n📊 Downloading macro data...")
        download_all_macro()
        
        return True
    except Exception as e:
        logger.error(f"  Data ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_market_preprocessing():
    """Run market data preprocessing"""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: MARKET DATA PREPROCESSING")
    logger.info("="*80)
    
    from src.preprocessing.market_preprocessing import MarketDataPreprocessor
    
    try:
        preprocessor = MarketDataPreprocessor()
        df = preprocessor.run_full_preprocessing(save=False)
        
        if df.empty:
            logger.error("  No data after preprocessing")
            return False
        
        logger.info(f"  Market preprocessing completed: {df.shape}")
        
        # Save to feature store
        logger.info("\n💾 Saving to Feature Store...")
        from src.feature_store import FeatureStore
        fs = FeatureStore()
        fs.save_base_features(df)
        
        return True
    except Exception as e:
        logger.error(f"  Market preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_statistical_modeling():
    """Run Granger Causality and VAR analysis"""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: STATISTICAL MODELING (Granger + VAR)")
    logger.info("="*80)
    
    try:
        # This will be implemented in modeling phase
        logger.info("⏳ Statistical models will be run in modeling phase")
        logger.info("   → Granger Causality Analysis")
        logger.info("   → VAR Model")
        return True
    except Exception as e:
        logger.error(f"  Statistical modeling failed: {e}")
        return False

def verify_feature_store():
    """Verify feature store has data"""
    logger.info("\n" + "="*80)
    logger.info("STEP 4: VERIFY FEATURE STORE")
    logger.info("="*80)
    
    try:
        from src.feature_store import FeatureStore
        from src.data_storage.database_setup import check_tables
        
        fs = FeatureStore()
        
        # Check if we have data
        features = fs.get_base_features()
        
        if features.empty:
            logger.warning("⚠️ No features in feature store yet")
            return False
        
        logger.info(f"  Feature store contains {len(features)} feature rows")
        logger.info(f"   Symbols: {features['symbol'].nunique()}")
        logger.info(f"   Date range: {features['time'].min()} to {features['time'].max()}")
        
        # Show table status
        check_tables()
        
        return True
    except Exception as e:
        logger.error(f"  Feature store verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_complete_pipeline(skip_ingestion=False, clean_first=False):
    """
    Run the complete pipeline
    
    Args:
        skip_ingestion: Skip data ingestion step (use existing data)
        clean_first: Clean database before ingesting new data
    """
    logger.info("\n" + "  "*20)
    logger.info("FINLAGX COMPLETE PIPELINE")
    logger.info("  "*20 + "\n")
    
    start_time = datetime.now()
    
    steps = []
    
    if not skip_ingestion:
        steps.append(("Data Ingestion", lambda: run_data_ingestion(clean_first)))
    
    steps.extend([
        ("Market Preprocessing", run_market_preprocessing),
        ("Statistical Modeling Setup", run_statistical_modeling),
        ("Verify Feature Store", verify_feature_store)
    ])
    
    # Run all steps
    results = {}
    for step_name, step_func in steps:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {step_name}")
        logger.info(f"{'='*80}")
        
        success = step_func()
        results[step_name] = success
        
        if not success and step_name != "Statistical Modeling Setup":
            logger.error(f"  Pipeline failed at: {step_name}")
            return False
    
    # Summary
    duration = datetime.now() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*80)
    
    for step_name, success in results.items():
        status = "  SUCCESS" if success else "  FAILED"
        logger.info(f"   {step_name}: {status}")
    
    logger.info(f"\n  Total Duration: {duration}")
    logger.info("="*80)
    
    if all(results.values()):
        logger.info("\n  Complete pipeline finished successfully!")
        logger.info("\n  Next Steps:")
        logger.info("   1. Run Granger Causality: python -m src.modeling.granger_causality")
        logger.info("   2. Run VAR Model: python -m src.modeling.var_model")
        logger.info("   3. Run Deep Learning: python -m src.modeling.lstm_model")
        logger.info("   4. View MLflow UI: http://localhost:5000")
        return True
    else:
        logger.error("\n  Pipeline completed with errors")
        return False

def run_individual_step(step):
    """Run individual pipeline step"""
    steps = {
        'ingest': run_data_ingestion,
        'preprocess': run_market_preprocessing,
        'verify': verify_feature_store
    }
    
    if step not in steps:
        logger.error(f"  Unknown step: {step}")
        logger.info(f"Available steps: {list(steps.keys())}")
        return False
    
    logger.info(f"\n  Running step: {step}")
    return steps[step]()

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='FinLagX Complete Pipeline')
    
    parser.add_argument(
        '--skip-ingestion',
        action='store_true',
        help='Skip data ingestion (use existing data)'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean database before ingesting new data'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['ingest', 'preprocess', 'verify'],
        help='Run only a specific step'
    )
    
    args = parser.parse_args()
    
    if args.step:
        success = run_individual_step(args.step)
    else:
        success = run_complete_pipeline(
            skip_ingestion=args.skip_ingestion,
            clean_first=args.clean
        )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()