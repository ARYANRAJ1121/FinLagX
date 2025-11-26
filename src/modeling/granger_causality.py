"""
Granger Causality Analysis for FinLagX
Analyzes lead-lag relationships between assets
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import combinations
import logging
from datetime import datetime
from sqlalchemy import text
from src.feature_store import FeatureStore
from src.data_storage.database_setup import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrangerCausalityAnalyzer:
    """
    Performs Granger Causality tests to identify lead-lag relationships
    """
    
    def __init__(self, max_lag=10, significance_level=0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.feature_store = FeatureStore()
        self.engine = get_engine()
    
    def fetch_market_features(self, symbols=None, start_date=None, end_date=None):
        """
        Fetch processed features from market_features table
        """
        logger.info("📊 Fetching market features from database...")
        
        query = """
        SELECT 
            time,
            symbol,
            returns,
            return_5d,
            return_10d,
            volatility_20,
            sma_20,
            sma_50
        FROM market_features
        WHERE returns IS NOT NULL
        AND volatility_20 IS NOT NULL
        """
        
        params = {}
        
        if symbols:
            query += " AND symbol = ANY(:symbols)"
            params['symbols'] = symbols
        
        if start_date:
            query += " AND time >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND time <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY symbol, time"
        
        df = pd.read_sql(text(query), self.engine, params=params)
        
        if df.empty:
            logger.error("  No data found in market_features table")
            return None
        
        logger.info(f"  Fetched {len(df)} rows for {df['symbol'].nunique()} symbols")
        logger.info(f"   Date range: {df['time'].min()} to {df['time'].max()}")
        
        return df
    
    def prepare_data_for_granger(self, df, variable='returns'):
        """
        Prepare data: pivot to wide format with each symbol as column
        """
        logger.info("🔄 Preparing data for Granger tests...")
        
        if variable not in df.columns:
            raise ValueError(f"Requested feature '{variable}' not found in dataframe")
        
        # Pivot selected feature
        returns_pivot = df.pivot(
            index='time',
            columns='symbol',
            values=variable
        )
        
        # Drop NaN values
        returns_pivot = returns_pivot.dropna()
        
        logger.info(f"  Prepared data shape: {returns_pivot.shape}")
        logger.info(f"   Symbols: {list(returns_pivot.columns)}")
        
        return returns_pivot
    
    def test_granger_causality(self, data, asset_x, asset_y):
        """
        Test if asset_x Granger-causes asset_y
        
        Returns:
            dict with optimal lag, p-value, f-statistic, granger_score
        """
        try:
            # Create dataframe with both series
            test_data = data[[asset_y, asset_x]].dropna()
            
            if len(test_data) < 50:  # Need sufficient data
                logger.warning(f"⚠️ Insufficient data for {asset_x} -> {asset_y}")
                return None
            
            # Run Granger test for multiple lags
            test_result = grangercausalitytests(
                test_data,
                maxlag=self.max_lag,
                verbose=False
            )
            
            # Extract results for each lag
            results = []
            for lag in range(1, self.max_lag + 1):
                # Get F-test results
                f_test = test_result[lag][0]['ssr_ftest']
                p_value = f_test[1]
                f_stat = f_test[0]
                
                results.append({
                    'lag': lag,
                    'p_value': p_value,
                    'f_statistic': f_stat
                })
            
            # Find optimal lag (lowest p-value)
            results_df = pd.DataFrame(results)
            optimal = results_df.loc[results_df['p_value'].idxmin()]
            
            # Calculate Granger score (inverse of p-value, capped)
            granger_score = min(-np.log10(optimal['p_value'] + 1e-10), 10)
            
            return {
                'asset_x': asset_x,
                'asset_y': asset_y,
                'optimal_lag': int(optimal['lag']),
                'p_value': float(optimal['p_value']),
                'f_statistic': float(optimal['f_statistic']),
                'granger_score': float(granger_score),
                'is_significant': optimal['p_value'] < self.significance_level
            }
            
        except Exception as e:
            logger.error(f"  Error testing {asset_x} -> {asset_y}: {e}")
            return None
    
    def run_all_granger_tests(self, data):
        """
        Run Granger causality tests for all asset pairs
        """
        logger.info(f"🔍 Running Granger causality tests...")
        logger.info(f"   Testing all pairs with max_lag={self.max_lag}")
        
        symbols = list(data.columns)
        n_pairs = len(symbols) * (len(symbols) - 1)
        
        logger.info(f"   Total tests to run: {n_pairs}")
        
        results = []
        tested = 0
        
        # Test all directed pairs (A->B and B->A are different)
        for asset_x in symbols:
            for asset_y in symbols:
                if asset_x == asset_y:
                    continue
                
                tested += 1
                if tested % 10 == 0:
                    logger.info(f"   Progress: {tested}/{n_pairs} tests completed")
                
                result = self.test_granger_causality(data, asset_x, asset_y)
                
                if result:
                    results.append(result)
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            significant = results_df['is_significant'].sum()
            logger.info(f"  Completed {len(results_df)} tests")
            logger.info(f"   Significant relationships: {significant}/{len(results_df)}")
        else:
            logger.warning("⚠️ No valid test results")
        
        return results_df
    
    def save_results_to_database(self, results_df, computed_date=None):
        """
        Save Granger results to granger_results table
        """
        if computed_date is None:
            computed_date = datetime.now().date()
        
        logger.info(f"💾 Saving {len(results_df)} Granger results to database...")
        
        self.feature_store.save_granger_results(results_df, computed_date)
        
        logger.info("  Results saved to granger_results table")
    
    def get_top_relationships(self, results_df, top_n=20):
        """
        Get top N most significant relationships
        """
        if results_df.empty:
            return pd.DataFrame()
        
        # Filter significant and sort by score
        significant = results_df[results_df['is_significant']].copy()
        significant = significant.sort_values('granger_score', ascending=False)
        
        return significant.head(top_n)
    
    def run_full_analysis(self, symbols=None, start_date=None, end_date=None, 
                         save=True, top_n=20, feature='returns'):
        """
        Run complete Granger causality analysis pipeline
        """
        logger.info("\n" + "="*80)
        logger.info("GRANGER CAUSALITY ANALYSIS")
        logger.info("="*80 + "\n")
        
        # 1. Fetch data
        df = self.fetch_market_features(symbols, start_date, end_date)
        
        if df is None or df.empty:
            logger.error("  No data available for analysis")
            return None
        
        # 2. Prepare data
        data_pivot = self.prepare_data_for_granger(df, variable=feature)
        
        # 3. Run tests
        results_df = self.run_all_granger_tests(data_pivot)
        
        if results_df.empty:
            logger.warning("⚠️ No valid results from Granger tests")
            return results_df
        
        # 4. Save to database
        if save:
            self.save_results_to_database(results_df)
        
        # 5. Show top relationships
        logger.info("\n" + "="*80)
        logger.info(f"TOP {top_n} LEAD-LAG RELATIONSHIPS")
        logger.info("="*80)
        
        top_relationships = self.get_top_relationships(results_df, top_n)
        
        if not top_relationships.empty:
            print("\n")
            print(top_relationships[['asset_x', 'asset_y', 'optimal_lag', 
                                   'p_value', 'granger_score']].to_string(index=False))
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"   Total tests: {len(results_df)}")
        logger.info(f"   Significant: {results_df['is_significant'].sum()}")
        logger.info(f"   Stored in: granger_results table")
        
        return results_df


def main():
    """
    Main execution function
    """
    # Initialize analyzer
    analyzer = GrangerCausalityAnalyzer(
        max_lag=10,
        significance_level=0.05
    )
    
    # Run full analysis
    # You can specify symbols, dates here if needed
    results = analyzer.run_full_analysis(
        symbols=None,  # None = all symbols
        start_date='2020-01-01',  # Adjust as needed
        end_date=None,  # None = up to latest
        save=True,
        top_n=20
    )
    
    if results is not None and not results.empty:
        logger.info("\n  Granger causality analysis completed successfully!")
    else:
        logger.error("\n  Analysis failed or returned no results")


if __name__ == "__main__":
    main()