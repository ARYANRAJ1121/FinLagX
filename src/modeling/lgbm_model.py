import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
from datetime import datetime
import mlflow
import mlflow.lightgbm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROCESSED_DATA_PATH = "data/processed/market/aligned_market_data.parquet"
MLFLOW_EXPERIMENT_NAME = "FinLagX_LGBM_Benchmark"
TARGET_ASSET = "S&P 500"

def create_tabular_features(df, target_col, lookback_window=[1, 3, 5, 10]):
    """
    Converts Time-Series to Tabular classification format
    Calculates rolling means and standard deviations
    Target: 1 if returns > 0 else 0
    """
    logger.info("🛠️ Creating tabular features for LightGBM...")
    
    # We want to predict tomorrow's direction
    # Target = 1 if tomorrow's return > 0, else 0
    target = (df[target_col].shift(-1) > 0).astype(int)
    
    features = pd.DataFrame(index=df.index)
    
    # Add rolling features for ALL assets
    for col in df.columns:
        features[f"{col}_return"] = df[col]
        
        for w in lookback_window:
            features[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w).mean()
            if w > 1:
                features[f"{col}_roll_std_{w}"] = df[col].rolling(window=w).std()
            
    features['target'] = target
    
    logger.info(f"Features created before dropna: {features.shape}")
    
    # Drop rows that don't have enough history for the max lookback window (10 days)
    features = features.dropna()
    
    logger.info(f"✅ Created {features.shape[1] - 1} features. Data shape after dropna: {features.shape}")
    return features.drop(columns=['target']), features['target']

def train_lightgbm_model():
    logger.info("🚀 Starting FinLagX LightGBM Modeling Pipeline...")
    
    # 1. Load Data
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    
    if TARGET_ASSET not in df.columns:
        logger.error(f"Target '{TARGET_ASSET}' not found in dataset!")
        return
        
    X, y = create_tabular_features(df, TARGET_ASSET)
    
    # 2. Time-Series Train-Test Split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"📊 Training on {len(X_train)} rows, Testing on {len(X_test)} rows")
    
    # 3. Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        logger.info(f"🔬 MLflow Run Started (ID: {run.info.run_id})")
        
        # Hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'feature_fraction': 0.8,
            'verbose': -1,
            'random_state': 42
        }
        
        mlflow.log_params(params)
        
        # 4. Train Model
        logger.info("⏳ Training LightGBM Model...")
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        # 5. Evaluate
        logger.info("📈 Evaluating model...")
        y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Test Accuracy: {acc:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")
        
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("best_iteration", model.best_iteration)
        
        # 6. Feature Importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        top_10 = importance.head(10)
        logger.info("\n🏆 Top 10 Most Important Features:")
        for idx, row in top_10.iterrows():
            logger.info(f"  - {row['feature']} : {row['importance']:.2f}")
            
        # Log importance to MLflow
        importance_csv = "data/results/lgbm_feature_importance.csv"
        os.makedirs(os.path.dirname(importance_csv), exist_ok=True)
        importance.to_csv(importance_csv, index=False)
        mlflow.log_artifact(importance_csv)
        
        # 7. Log Model
        mlflow.lightgbm.log_model(model, "lgbm_model")
        logger.info("✅ Model and metrics securely logged to local MLflow database!")

if __name__ == "__main__":
    train_lightgbm_model()
