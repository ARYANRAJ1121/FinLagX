import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROCESSED_DATA_PATH = "data/processed/market/aligned_market_data.parquet"
TARGET_ASSET = "SP500"

def objective(trial):
    # Load dataset
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    df_asset = df[df['symbol'] == TARGET_ASSET].sort_values('time')
    
    # Simple classification features
    essential_cols = ['returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 'momentum_10']
    feature_cols = [c for c in essential_cols if c in df_asset.columns]
    
    df_asset['target'] = (df_asset['returns'].shift(-1) > 0).astype(int)
    data = df_asset[feature_cols + ['target']].dropna()
    
    X = data[feature_cols]
    y = data['target']
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 1. Define Search Space
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100)
    }
    
    # 2. Train
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # 3. Evaluate
    preds = (model.predict(X_test) > 0.5).astype(int)
    score = f1_score(y_test, preds)
    
    return score

def run_tuning():
    logger.info(f"🔍 Starting Optuna tuning for {TARGET_ASSET}...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    logger.info("✅ Tuning complete!")
    logger.info(f"Best Score: {study.best_value}")
    logger.info(f"Best Params: {study.best_params}")
    
    # Log best params to MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("FinLagX_Hyperparameter_Tuning")
    
    with mlflow.start_run():
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1_score", study.best_value)
        logger.info("🚀 Best parameters logged to MLflow.")

if __name__ == "__main__":
    run_tuning()
