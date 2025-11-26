"""
Vector Autoregression (VAR) Model for FinLagX
Models multivariate time series relationships
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import logging
from sqlalchemy import text
from src.feature_store import FeatureStore
from src.data_storage.database_setup import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VARAnalyzer:
    """
    Vector Autoregression model for analyzing asset relationships
    """
    
    def __init__(self, max_lags=10):
        self.max_lags = max_lags
        self.feature_store = FeatureStore()
        self.engine = get_engine()
        self.model = None
        self.model_result = None
    
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
        AND return_5d IS NOT NULL
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
    
    def prepare_data_for_var(self, df, variable='returns'):
        """
        Prepare data: pivot to wide format
        """
        logger.info(f"🔄 Preparing data for VAR (using {variable})...")
        
        # Pivot data
        data_pivot = df.pivot(
            index='time',
            columns='symbol',
            values=variable
        )
        
        # Drop NaN values
        data_pivot = data_pivot.dropna()
        
        logger.info(f"  Prepared data shape: {data_pivot.shape}")
        logger.info(f"   Symbols: {list(data_pivot.columns)}")
        logger.info(f"   Observations: {len(data_pivot)}")
        
        return data_pivot
    
    def check_stationarity(self, data):
        """
        Check if series are stationary using ADF test
        """
        logger.info("🔍 Checking stationarity (ADF test)...")
        
        results = {}
        
        for col in data.columns:
            try:
                adf_result = adfuller(data[col].dropna(), autolag='AIC')
                p_value = adf_result[1]
                is_stationary = p_value < 0.05
                
                results[col] = {
                    'adf_statistic': adf_result[0],
                    'p_value': p_value,
                    'is_stationary': is_stationary
                }
                
                status = "  Stationary" if is_stationary else "⚠️ Non-stationary"
                logger.info(f"   {col}: {status} (p={p_value:.4f})")
                
            except Exception as e:
                logger.error(f"   {col}: Error in ADF test: {e}")
                results[col] = {'error': str(e)}
        
        stationary_count = sum(r.get('is_stationary', False) for r in results.values())
        logger.info(f"\n   Stationary series: {stationary_count}/{len(results)}")
        
        return results
    
    def select_optimal_lag(self, data):
        """
        Select optimal lag using information criteria
        """
        logger.info("📏 Selecting optimal lag order...")
        
        try:
            model = VAR(data)
            lag_order = model.select_order(maxlags=self.max_lags)
            
            logger.info("\n   Information Criteria:")
            logger.info(f"   AIC:\n{lag_order.aic}")
            logger.info(f"   BIC:\n{lag_order.bic}")
            logger.info(f"   FPE:\n{lag_order.fpe}")
            logger.info(f"   HQIC:\n{lag_order.hqic}")
            
            # Use BIC for more conservative estimate
            optimal_lag = lag_order.selected_orders.get('bic')
            if optimal_lag is None:
                raise ValueError("BIC did not return a selected lag order")
            
            optimal_lag = int(optimal_lag)
            if optimal_lag < 1:
                logger.warning("   BIC suggested lag < 1; defaulting to lag=1")
                optimal_lag = 1
            
            logger.info(f"\n  Selected lag order (BIC): {optimal_lag}")
            
            return optimal_lag
            
        except Exception as e:
            logger.error(f"  Error in lag selection: {e}")
            logger.info("   Using default lag=2")
            return 2
    
    def fit_var_model(self, data, lag_order=None):
        """
        Fit VAR model
        """
        logger.info("🔧 Fitting VAR model...")
        
        if lag_order is None:
            lag_order = self.select_optimal_lag(data)
        
        try:
            self.model = VAR(data)
            self.model_result = self.model.fit(lag_order)
            
            logger.info(f"  VAR model fitted with lag order {self.model_result.k_ar}")
            
            return self.model_result
            
        except Exception as e:
            logger.error(f"  Error fitting VAR model: {e}")
            return None
    
    def extract_var_features(self, data):
        """
        Extract VAR model features: fitted values, residuals
        """
        if self.model_result is None:
            logger.error("  Model not fitted yet")
            return None
        
        logger.info("📊 Extracting VAR features...")
        
        try:
            # Get fitted values
            fitted = self.model_result.fittedvalues
            
            # Get residuals
            residuals = self.model_result.resid
            
            # Create features dataframe
            features_list = []
            
            for symbol in data.columns:
                symbol_data = pd.DataFrame({
                    'time': fitted.index,
                    'symbol': symbol,
                    'var_fitted_value': fitted[symbol].values,
                    'var_residual': residuals[symbol].values
                })
                
                features_list.append(symbol_data)
            
            features_df = pd.concat(features_list, ignore_index=True)
            
            logger.info(f"  Extracted VAR features: {features_df.shape}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"  Error extracting features: {e}")
            return None
    
    def compute_impulse_responses(self, periods=10):
        """
        Compute impulse response functions
        """
        if self.model_result is None:
            logger.error("  Model not fitted yet")
            return None
        
        logger.info(f"📈 Computing impulse responses ({periods} periods)...")
        
        try:
            irf = self.model_result.irf(periods)
            
            # Extract IRF values at final period
            irf_final = pd.DataFrame(
                irf.irfs[-1],
                index=self.model_result.names,
                columns=self.model_result.names
            )
            
            logger.info("  Impulse responses computed")
            
            return irf_final
            
        except Exception as e:
            logger.error(f"  Error computing IRF: {e}")
            return None
    
    def save_results_to_database(self, features_df):
        """
        Save VAR features to var_features table
        """
        logger.info(f"💾 Saving {len(features_df)} VAR features to database...")
        
        # Add placeholder for impulse_response (can be computed separately)
        if 'impulse_response' not in features_df.columns:
            features_df['impulse_response'] = 0.0
        
        self.feature_store.save_var_features(features_df)
        
        logger.info("  Results saved to var_features table")
    
    def print_model_summary(self):
        """
        Print VAR model summary
        """
        if self.model_result is None:
            logger.error("  Model not fitted yet")
            return
        
        logger.info("\n" + "="*80)
        logger.info("VAR MODEL SUMMARY")
        logger.info("="*80)
        
        print(self.model_result.summary())
        
        logger.info("\n" + "="*80)
    
    def run_full_analysis(self, symbols=None, start_date=None, end_date=None,
                         save=True, lag_order=None):
        """
        Run complete VAR analysis pipeline
        """
        logger.info("\n" + "="*80)
        logger.info("VAR MODEL ANALYSIS")
        logger.info("="*80 + "\n")
        
        # 1. Fetch data
        df = self.fetch_market_features(symbols, start_date, end_date)
        
        if df is None or df.empty:
            logger.error("  No data available for analysis")
            return None
        
        # 2. Prepare data (using returns)
        data_pivot = self.prepare_data_for_var(df, variable='returns')
        
        # 3. Check stationarity
        stationarity = self.check_stationarity(data_pivot)
        
        # 4. Fit VAR model
        model_result = self.fit_var_model(data_pivot, lag_order)
        
        if model_result is None:
            logger.error("  VAR model fitting failed")
            return None
        
        # 5. Extract features
        features_df = self.extract_var_features(data_pivot)
        
        if features_df is None or features_df.empty:
            logger.error("  Feature extraction failed")
            return None
        
        # 6. Save to database
        if save:
            self.save_results_to_database(features_df)
        
        # 7. Print summary
        self.print_model_summary()
        
        # 8. Compute impulse responses
        irf = self.compute_impulse_responses(periods=10)
        
        if irf is not None:
            logger.info("\n" + "="*80)
            logger.info("IMPULSE RESPONSE FUNCTIONS (Period 10)")
            logger.info("="*80)
            print("\n", irf)
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"   Model lag order: {model_result.k_ar}")
        logger.info(f"   Features saved: {len(features_df)} rows")
        logger.info(f"   Stored in: var_features table")
        
        return features_df


def main():
    """
    Main execution function
    """
    # Initialize analyzer
    analyzer = VARAnalyzer(max_lags=10)
    
    # Run full analysis
    results = analyzer.run_full_analysis(
        symbols=None,  # None = all symbols
        start_date='2020-01-01',  # Adjust as needed
        end_date=None,  # None = up to latest
        save=True,
        lag_order=None  # None = auto-select
    )
    
    if results is not None and not results.empty:
        logger.info("\n  VAR analysis completed successfully!")
    else:
        logger.error("\n  Analysis failed or returned no results")


if __name__ == "__main__":
    main()