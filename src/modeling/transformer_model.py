import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.pytorch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROCESSED_DATA_PATH = "data/processed/market/aligned_market_data.parquet"
TARGET_SYMBOL = "SP500"
SEQ_LENGTH = 10
EPOCHS = 30
HEADS = 4
D_MODEL = 64

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_in = nn.Linear(input_dim, d_model)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.fc_in(x) # [batch, seq_len, d_model]
        x = self.transformer_encoder(x)
        x = x[:, -1, :] # Take the last timestamp
        x = self.fc_out(x)
        return x

def prepare_transformer_data(file_path, target_symbol="SP500", seq_length=10):
    logger.info("🎬 Preparing data for Transformer...")
    df = pd.read_parquet(file_path)
    df_asset = df[df['symbol'] == target_symbol].copy().sort_values('time')
    
    # Selecting the returns and essential indicators as features
    feature_cols = ['returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 'momentum_10']
    cols = [c for c in feature_cols if c in df_asset.columns]
    
    data = df_asset[cols].values.astype(float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length][0]) # Predict next return
        
    split = int(len(X) * 0.8)
    X_train = torch.FloatTensor(np.array(X[:split]))
    y_train = torch.FloatTensor(np.array(y[:split])).view(-1, 1)
    X_test = torch.FloatTensor(np.array(X[split:]))
    y_test = torch.FloatTensor(np.array(y[split:])).view(-1, 1)
    
    return X_train, y_train, X_test, y_test, len(cols)

def train_transformer():
    logger.info("🚀 Starting FinLagX Transformer Modeling Pipeline...")
    X_train, y_train, X_test, y_test, input_dim = prepare_transformer_data(PROCESSED_DATA_PATH, TARGET_SYMBOL, SEQ_LENGTH)
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("FinLagX_Transformer_Research")
    
    with mlflow.start_run():
        model = TimeSeriesTransformer(input_dim=input_dim, d_model=D_MODEL, n_heads=HEADS)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        logger.info(f"⏳ Training Transformer for {EPOCHS} epochs...")
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"   Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.6f}")
            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = criterion(test_preds, y_test)
            logger.info(f"📈 Test MSE: {test_loss.item():.6f}")
            mlflow.log_metric("test_mse", test_loss.item())
            
        mlflow.pytorch.log_model(model, "transformer_model")
        logger.info("✅ Transformer model logged to MLflow.")

if __name__ == "__main__":
    train_transformer()
