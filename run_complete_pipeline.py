#!/usr/bin/env python3
"""
Complete FinLagX Pipeline Orchestrator
Runs data ingestion, preprocessing, and prepares data for modeling
"""
import sys
import logging
import argparse
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_data_ingestion(clean_first=False):
    """Run data ingestion pipeline"""
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("="*80)
    
    from src.data_ingestion.run_data_pipeline import run_full_pipeline
    
    try:
        run_full_pipeline(clean_first=clean_first)
        return True
    except Exception as e:
        logger.error(f"❌ Data ingestion failed: {e}")
        return False

def run_market_preprocessing():
    """Run market data preprocessing"""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: MARKET DATA PREPROCESSING")
    logger.info("="*80)
    
    from src.preprocessing import MarketDataPreprocessor
    
    try:
        preprocessor = MarketDataPreprocessor()
        df = preprocessor.run_full_preprocessing(save=True)
        
        logger.info(f"✅ Market preprocessing completed: {df.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Market preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_news_preprocessing():
    """Run news data preprocessing"""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: NEWS DATA PREPROCESSING")
    logger.info("="*80)
    
    from src.preprocessing import NewsDataPreprocessor
    
    try:
        preprocessor = NewsDataPreprocessor()
        df = preprocessor.run_full_preprocessing(save=True)
        
        if not df.empty:
            logger.info(f"✅ News preprocessing completed: {df.shape}")
        else:
            logger.warning("⚠️ No news data to preprocess")
        return True
    except Exception as e:
        logger.error(f"❌ News preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_data_alignment():
    """Run data alignment"""
    logger.info("\n" + "="*80)
    logger.info("STEP 4: DATA ALIGNMENT")
    logger.info("="*80)
    
    from src.preprocessing import DataAligner
    
    try:
        aligner = DataAligner()
        df = aligner.create_complete_dataset()
        
        if not df.empty:
            aligner.save_aligned_data(df)
            logger.info(f"✅ Data alignment completed: {df.shape}")
            return True
        else:
            logger.warning("⚠️ No aligned data created")
            return False
    except Exception as e:
        logger.error(f"❌ Data alignment failed: {e}")
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
    logger.info("\n" + "🚀 "*20)
    logger.info("FINLAGX COMPLETE PIPELINE")
    logger.info("🚀 "*20 + "\n")
    
    start_time = datetime.now()
    
    steps = []
    
    if not skip_ingestion:
        steps.append(("Data Ingestion", lambda: run_data_ingestion(clean_first)))
    
    steps.extend([
        ("Market Preprocessing", run_market_preprocessing),
        ("News Preprocessing", run_news_preprocessing),
        ("Data Alignment", run_data_alignment)
    ])
    
    # Run all steps
    results = {}
    for step_name, step_func in steps:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {step_name}")
        logger.info(f"{'='*80}")
        
        success = step_func()
        results[step_name] = success
        
        if not success:
            logger.error(f"❌ Pipeline failed at: {step_name}")
            logger.info("\n⚠️ You can try running individual steps:")
            logger.info("   python run_complete_pipeline.py --step market")
            logger.info("   python run_complete_pipeline.py --step news")
            logger.info("   python run_complete_pipeline.py --step align")
            return False
    
    # Summary
    duration = datetime.now() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*80)
    
    for step_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"   {step_name}: {status}")
    
    logger.info(f"\n⏱️ Total Duration: {duration}")
    logger.info("="*80)
    
    if all(results.values()):
        logger.info("\n🎉 Complete pipeline finished successfully!")
        logger.info("\n📋 Next Steps:")
        logger.info("   1. Check aligned dataset: data/processed/aligned_dataset.parquet")
        logger.info("   2. Start modeling: python -m src.modeling.granger_causality")
        logger.info("   3. View in notebooks: jupyter notebook notebooks/")
        return True
    else:
        logger.error("\n❌ Pipeline completed with errors")
        return False

def run_individual_step(step):
    """Run individual pipeline step"""
    steps = {
        'ingest': run_data_ingestion,
        'market': run_market_preprocessing,
        'news': run_news_preprocessing,
        'align': run_data_alignment
    }
    
    if step not in steps:
        logger.error(f"❌ Unknown step: {step}")
        logger.info(f"Available steps: {list(steps.keys())}")
        return False
    
    logger.info(f"\n🚀 Running step: {step}")
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
        choices=['ingest', 'market', 'news', 'align'],
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