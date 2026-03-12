import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from datetime import datetime
import mlflow
import mlflow.lightgbm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROCESSED_DATA_PATH = "data/processed/market/aligned_market_data.parquet"
MLFLOW_EXPERIMENT_NAME = "FinLagX_Hurdle_Model"
TARGET_ASSET = "SP500"

def prepare_hurdle_data(df, target_symbol="SP500"):
    """
    Prepares data for the two-stage Hurdle Model.
    Stage 1: Binary Classification (Direction)
    Stage 2: Regression (Magnitude of absolute returns)
    """
    logger.info(f"🛠️ Preparing Hurdle Model features for {target_symbol}...")
    
    df_asset = df[df['symbol'] == target_symbol].copy()
    df_asset = df_asset.sort_values('time')
    
    # Target 1: Direction (1 if returns > 0, else 0)
    df_asset['target_dir'] = (df_asset['returns'].shift(-1) > 0).astype(int)
    
    # Target 2: Magnitude (Absolute value of next day return)
    df_asset['target_mag'] = df_asset['returns'].shift(-1).abs()
    
    # Features
    essential_cols = [
        'returns', 'volatility_20', 'sma_20', 'sma_50', 
        'bb_upper', 'bb_lower', 'macd', 'macd_signal', 'rsi_14', 'momentum_10'
    ]
    sentiment_cols = [c for c in df_asset.columns if 'sentiment' in c or 'news_count' in c]
    feature_cols = [c for c in essential_cols + sentiment_cols if c in df_asset.columns]
    
    data = df_asset[feature_cols + ['target_dir', 'target_mag']].dropna()
    
    X = data[feature_cols]
    y_dir = data['target_dir']
    y_mag = data['target_mag']
    
    return X, y_dir, y_mag

def train_hurdle_model():
    logger.info("🚀 Starting Two-Stage Hurdle Model Pipeline...")
    
    # 1. Load Data
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error("Dataset not found. Please run build_features.py first.")
        return

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X, y_dir, y_mag = prepare_hurdle_data(df, TARGET_ASSET)
    
    # 2. Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_dir_train, y_dir_test = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]
    y_mag_train, y_mag_test = y_mag.iloc[:split_idx], y_mag.iloc[split_idx:]

    # 3. MLflow Setup
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        # --- STAGE 1: CLASSIFICATION (Direction) ---
        logger.info("⏳ Training Stage 1: Direction Classifier...")
        clf_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'verbose': -1,
            'random_state': 42
        }
        
        train_set_dir = lgb.Dataset(X_train, label=y_dir_train)
        clf_model = lgb.train(clf_params, train_set_dir, num_boost_round=100)
        
        y_dir_pred = (clf_model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_dir_test, y_dir_pred)
        f1 = f1_score(y_dir_test, y_dir_pred)
        
        # --- STAGE 2: REGRESSION (Magnitude) ---
        logger.info("⏳ Training Stage 2: Magnitude Regressor...")
        reg_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'verbose': -1,
            'random_state': 42
        }
        
        train_set_mag = lgb.Dataset(X_train, label=y_mag_train)
        reg_model = lgb.train(reg_params, train_set_mag, num_boost_round=100)
        
        y_mag_pred = reg_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_mag_test, y_mag_pred))
        
        # Log Metrics
        mlflow.log_metric("clf_accuracy", acc)
        mlflow.log_metric("clf_f1", f1)
        mlflow.log_metric("reg_rmse", rmse)
        
        logger.info(f"✅ Hurdle Model Results -> Direction Acc: {acc:.4f}, Mag RMSE: {rmse:.6f}")
        
        # Log Models
        mlflow.lightgbm.log_model(clf_model, "direction_classifier")
        mlflow.lightgbm.log_model(reg_model, "magnitude_regressor")
        
    logger.info("🎉 Hurdle Model training complete and logged to MLflow.")

if __name__ == "__main__":
    train_hurdle_model()
