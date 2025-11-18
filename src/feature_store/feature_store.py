"""
FinLagX Feature Store
Central repository for all features and model outputs
"""
import pandas as pd
import numpy as np
from sqlalchemy import text
from src.data_storage.database_setup import get_engine
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Feature Store for FinLagX
    Manages all features, model outputs, and provides versioned access
    """
    
    def __init__(self):
        self.engine = get_engine()
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def initialize_feature_store(self):
        """Create all feature store tables"""
        logger.info("🏗️ Initializing Feature Store...")
        
        with self.engine.connect() as conn:
            # 1. Base features table (from preprocessing)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS market_features (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    returns NUMERIC,
                    volatility_20 NUMERIC,
                    sma_20 NUMERIC,
                    volume_change NUMERIC,
                    news_equities NUMERIC DEFAULT 0,
                    news_commodities NUMERIC DEFAULT 0,
                    news_forex NUMERIC DEFAULT 0,
                    news_crypto NUMERIC DEFAULT 0,
                    news_bonds NUMERIC DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (time, symbol)
                );
            """))
            
            # Make it a hypertable if not already
            try:
                conn.execute(text("SELECT create_hypertable('market_features', 'time', if_not_exists => TRUE);"))
            except:
                pass
            
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_market_features_symbol ON market_features(symbol, time DESC);"))
            
            # 2. Granger causality results
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS granger_results (
                    id SERIAL PRIMARY KEY,
                    computed_date DATE NOT NULL,
                    asset_x VARCHAR(20) NOT NULL,
                    asset_y VARCHAR(20) NOT NULL,
                    optimal_lag INT NOT NULL,
                    p_value NUMERIC,
                    f_statistic NUMERIC,
                    granger_score NUMERIC,
                    is_significant BOOLEAN,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_granger_date ON granger_results(computed_date DESC);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_granger_assets ON granger_results(asset_x, asset_y);"))
            
            # 3. VAR model features
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS var_features (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    var_fitted_value NUMERIC,
                    var_residual NUMERIC,
                    impulse_response NUMERIC,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (time, symbol)
                );
            """))
            
            try:
                conn.execute(text("SELECT create_hypertable('var_features', 'time', if_not_exists => TRUE);"))
            except:
                pass
            
            # 4. LSTM predictions
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS lstm_predictions (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    predicted_return NUMERIC,
                    confidence NUMERIC,
                    lead_lag_indicator NUMERIC,
                    model_version VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (time, symbol, model_version)
                );
            """))
            
            try:
                conn.execute(text("SELECT create_hypertable('lstm_predictions', 'time', if_not_exists => TRUE);"))
            except:
                pass
            
            # 5. Sentiment features
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentiment_features (
                    date DATE NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    avg_sentiment NUMERIC,
                    article_count INT,
                    positive_count INT,
                    negative_count INT,
                    neutral_count INT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (date, category)
                );
            """))
            
            conn.commit()
        
        logger.info("✅ Feature Store initialized!")
    
    # ==================== BASE FEATURES ====================
    
    def save_base_features(self, df):
        """Save base features from preprocessing"""
        logger.info(f"💾 Saving base features...")
        
        try:
            # Select relevant columns
            feature_cols = ['time', 'symbol', 'returns', 'volatility_20', 'sma_20', 'volume_change']
            news_cols = [col for col in df.columns if 'news_' in col]
            
            features_df = df[feature_cols + news_cols].copy()
            
            # Upsert to database
            with self.engine.connect() as conn:
                for _, row in features_df.iterrows():
                    conn.execute(text("""
                        INSERT INTO market_features 
                        (time, symbol, returns, volatility_20, sma_20, volume_change, 
                         news_equities, news_commodities, news_forex, news_crypto, news_bonds)
                        VALUES 
                        (:time, :symbol, :returns, :volatility_20, :sma_20, :volume_change,
                         :news_equities, :news_commodities, :news_forex, :news_crypto, :news_bonds)
                        ON CONFLICT (time, symbol) DO UPDATE SET
                            returns = EXCLUDED.returns,
                            volatility_20 = EXCLUDED.volatility_20,
                            sma_20 = EXCLUDED.sma_20,
                            volume_change = EXCLUDED.volume_change,
                            news_equities = EXCLUDED.news_equities,
                            news_commodities = EXCLUDED.news_commodities,
                            news_forex = EXCLUDED.news_forex,
                            news_crypto = EXCLUDED.news_crypto,
                            news_bonds = EXCLUDED.news_bonds
                    """), {
                        'time': row['time'],
                        'symbol': row['symbol'],
                        'returns': float(row.get('returns', 0)),
                        'volatility_20': float(row.get('volatility_20', 0)),
                        'sma_20': float(row.get('sma_20', 0)),
                        'volume_change': float(row.get('volume_change', 0)),
                        'news_equities': int(row.get('news_equities', 0)),
                        'news_commodities': int(row.get('news_commodities', 0)),
                        'news_forex': int(row.get('news_forex', 0)),
                        'news_crypto': int(row.get('news_crypto', 0)),
                        'news_bonds': int(row.get('news_bonds', 0))
                    })
                
                conn.commit()
            
            logger.info(f"✅ Saved {len(features_df)} base feature rows")
            
        except Exception as e:
            logger.error(f"❌ Error saving base features: {e}")
            raise
    
    def get_base_features(self, symbols=None, start_date=None, end_date=None):
        """Get base features from feature store"""
        query = "SELECT * FROM market_features WHERE 1=1"
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
        
        return pd.read_sql(query, self.engine, params=params)
    
    # ==================== GRANGER CAUSALITY ====================
    
    def save_granger_results(self, results_df, computed_date=None):
        """Save Granger causality test results"""
        logger.info(f"💾 Saving Granger causality results...")
        
        if computed_date is None:
            computed_date = datetime.now().date()
        
        try:
            with self.engine.connect() as conn:
                for _, row in results_df.iterrows():
                    conn.execute(text("""
                        INSERT INTO granger_results 
                        (computed_date, asset_x, asset_y, optimal_lag, p_value, 
                         f_statistic, granger_score, is_significant)
                        VALUES 
                        (:computed_date, :asset_x, :asset_y, :optimal_lag, :p_value,
                         :f_statistic, :granger_score, :is_significant)
                    """), {
                        'computed_date': computed_date,
                        'asset_x': row['asset_x'],
                        'asset_y': row['asset_y'],
                        'optimal_lag': int(row['optimal_lag']),
                        'p_value': float(row['p_value']),
                        'f_statistic': float(row.get('f_statistic', 0)),
                        'granger_score': float(row.get('granger_score', 0)),
                        'is_significant': bool(row['p_value'] < 0.05)
                    })
                
                conn.commit()
            
            logger.info(f"✅ Saved {len(results_df)} Granger results")
            
        except Exception as e:
            logger.error(f"❌ Error saving Granger results: {e}")
            raise
    
    def get_granger_results(self, asset_x=None, asset_y=None, date=None, significant_only=True):
        """Get Granger causality results"""
        query = "SELECT * FROM granger_results WHERE 1=1"
        params = {}
        
        if asset_x:
            query += " AND asset_x = :asset_x"
            params['asset_x'] = asset_x
        
        if asset_y:
            query += " AND asset_y = :asset_y"
            params['asset_y'] = asset_y
        
        if date:
            query += " AND computed_date = :date"
            params['date'] = date
        
        if significant_only:
            query += " AND is_significant = TRUE"
        
        query += " ORDER BY computed_date DESC, granger_score DESC"
        
        return pd.read_sql(query, self.engine, params=params)
    
    def get_latest_granger_network(self):
        """Get most recent Granger causality network"""
        query = """
        SELECT asset_x, asset_y, granger_score, p_value
        FROM granger_results
        WHERE computed_date = (SELECT MAX(computed_date) FROM granger_results)
        AND is_significant = TRUE
        ORDER BY granger_score DESC
        """
        
        return pd.read_sql(query, self.engine)
    
    # ==================== VAR MODEL FEATURES ====================
    
    def save_var_features(self, df):
        """Save VAR model fitted values and residuals"""
        logger.info(f"💾 Saving VAR features...")
        
        try:
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    conn.execute(text("""
                        INSERT INTO var_features 
                        (time, symbol, var_fitted_value, var_residual, impulse_response)
                        VALUES 
                        (:time, :symbol, :var_fitted_value, :var_residual, :impulse_response)
                        ON CONFLICT (time, symbol) DO UPDATE SET
                            var_fitted_value = EXCLUDED.var_fitted_value,
                            var_residual = EXCLUDED.var_residual,
                            impulse_response = EXCLUDED.impulse_response
                    """), {
                        'time': row['time'],
                        'symbol': row['symbol'],
                        'var_fitted_value': float(row.get('var_fitted_value', 0)),
                        'var_residual': float(row.get('var_residual', 0)),
                        'impulse_response': float(row.get('impulse_response', 0))
                    })
                
                conn.commit()
            
            logger.info(f"✅ Saved {len(df)} VAR feature rows")
            
        except Exception as e:
            logger.error(f"❌ Error saving VAR features: {e}")
            raise
    
    # ==================== LSTM PREDICTIONS ====================
    
    def save_lstm_predictions(self, df, model_version=None):
        """Save LSTM predictions"""
        logger.info(f"💾 Saving LSTM predictions...")
        
        if model_version is None:
            model_version = self.version
        
        try:
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    conn.execute(text("""
                        INSERT INTO lstm_predictions 
                        (time, symbol, predicted_return, confidence, lead_lag_indicator, model_version)
                        VALUES 
                        (:time, :symbol, :predicted_return, :confidence, :lead_lag_indicator, :model_version)
                        ON CONFLICT (time, symbol, model_version) DO UPDATE SET
                            predicted_return = EXCLUDED.predicted_return,
                            confidence = EXCLUDED.confidence,
                            lead_lag_indicator = EXCLUDED.lead_lag_indicator
                    """), {
                        'time': row['time'],
                        'symbol': row['symbol'],
                        'predicted_return': float(row['predicted_return']),
                        'confidence': float(row.get('confidence', 0)),
                        'lead_lag_indicator': float(row.get('lead_lag_indicator', 0)),
                        'model_version': model_version
                    })
                
                conn.commit()
            
            logger.info(f"✅ Saved {len(df)} LSTM predictions")
            
        except Exception as e:
            logger.error(f"❌ Error saving LSTM predictions: {e}")
            raise
    
    # ==================== COMBINED FEATURE RETRIEVAL ====================
    
    def get_features_for_training(self, symbols, start_date, end_date,
                                  include_granger=False, include_var=False, 
                                  include_lstm=False):
        """
        Get combined features for model training
        Joins base features with model outputs
        """
        logger.info(f"📊 Building training dataset...")
        
        # Start with base features
        features = self.get_base_features(symbols, start_date, end_date)
        
        if features.empty:
            logger.error("❌ No base features found")
            return pd.DataFrame()
        
        # Add Granger scores
        if include_granger:
            granger_df = self.get_granger_results()
            if not granger_df.empty:
                # Pivot to get scores for each asset pair
                # This is simplified - you'd want to aggregate scores per date
                logger.info("   Added Granger causality features")
        
        # Add VAR features
        if include_var:
            var_query = """
            SELECT time, symbol, var_fitted_value, var_residual
            FROM var_features
            WHERE symbol = ANY(:symbols)
            AND time BETWEEN :start_date AND :end_date
            """
            var_df = pd.read_sql(var_query, self.engine, params={
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date
            })
            
            if not var_df.empty:
                features = features.merge(var_df, on=['time', 'symbol'], how='left')
                logger.info("   Added VAR features")
        
        # Add LSTM predictions
        if include_lstm:
            lstm_query = """
            SELECT time, symbol, predicted_return, confidence
            FROM lstm_predictions
            WHERE symbol = ANY(:symbols)
            AND time BETWEEN :start_date AND :end_date
            """
            lstm_df = pd.read_sql(lstm_query, self.engine, params={
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date
            })
            
            if not lstm_df.empty:
                features = features.merge(lstm_df, on=['time', 'symbol'], how='left')
                logger.info("   Added LSTM predictions")
        
        logger.info(f"✅ Training dataset ready: {features.shape}")
        
        return features
    
    def get_latest_features(self, symbols=None, days=30, include_all=True):
        """Get most recent features for inference"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_features_for_training(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            include_granger=include_all,
            include_var=include_all,
            include_lstm=include_all
        )
    
    def get_current_version(self):
        """Get current feature store version"""
        return self.version

if __name__ == "__main__":
    # Initialize feature store
    fs = FeatureStore()
    fs.initialize_feature_store()
    
    logger.info("\n✅ Feature Store ready!")
    logger.info("📊 Available tables:")
    logger.info("   • market_features - Base features")
    logger.info("   • granger_results - Causality scores")
    logger.info("   • var_features - VAR model outputs")
    logger.info("   • lstm_predictions - LSTM predictions")
    logger.info("   • sentiment_features - Sentiment scores")