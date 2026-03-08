"""
TCN (Temporal Convolutional Network) Lead-Lag Detection
Advanced time series modeling using dilated causal convolutions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text
import logging

from src.feature_store.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """
    Temporal Block: The basic building block of TCN
    Consists of dilated causal convolutions + residual connections
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First dilated causal convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated causal convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if input/output dims differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Main pathway
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Chop the output to match input length (causal convolution produces extra length)
        # This is necessary because causal padding adds to the front
        if out.size(2) != x.size(2):
            out = out[:, :, :x.size(2)]
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for time series prediction
    Uses stacked dilated causal convolutions with exponentially increasing dilation
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size  # Causal padding
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Pass through TCN
        y = self.network(x)
        
        # Take the last timestep
        y = y[:, :, -1]
        
        # Final prediction
        return self.fc(y)


def get_available_symbols(fs):
    """Get all available symbols"""
    with fs.engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT symbol FROM market_features ORDER BY symbol"))
        symbols = [row[0] for row in result]
    return symbols


def get_granger_relationships(fs, symbol):
    """Get Granger causality relationships for a symbol"""
    logger.info(f"   🔗 Fetching Granger relationships for {symbol}...")
    
    with fs.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT asset_x, asset_y, optimal_lag, granger_score, is_significant
            FROM granger_results 
            WHERE asset_y = :symbol 
            AND is_significant = TRUE
            ORDER BY granger_score DESC
        """), {'symbol': symbol})
        
        relationships = result.fetchall()
    
    if relationships:
        logger.info(f"     Found {len(relationships)} significant lead-lag relationships")
        for asset_x, asset_y, lag, score, sig in relationships:
            logger.info(f"      • {asset_x} leads {asset_y} by {lag} days (score: {score:.4f})")
    else:
        logger.info(f"     No significant Granger relationships found")
    
    return relationships


def prepare_data_with_leadlag(fs, symbol, lookback=20):
    """Prepare TCN data with lead-lag features"""
    logger.info(f"\n📊 Preparing data for {symbol}...")
    
    # Get market data
    query = """
        SELECT * FROM market_features 
        WHERE symbol = %(symbol)s 
        ORDER BY time
    """
    df = pd.read_sql(query, fs.engine, params={'symbol': symbol})
    
    if df.empty:
        logger.error(f"  No data found for {symbol}")
        return None
    
    logger.info(f"   📥 Loaded {len(df)} rows ({df['time'].min()} to {df['time'].max()})")
    
    # Base features
    base_features = ['returns', 'volatility_20', 'sma_20', 'sma_50']
    feature_cols = [col for col in base_features if col in df.columns]
    
    # Get Granger relationships
    granger_rels = get_granger_relationships(fs, symbol)
    granger_info = []
    
    for asset_x, asset_y, lag, score, sig in granger_rels:
        col_name = f'{asset_x}_lag{lag}'
        
        if col_name in df.columns or col_name in feature_cols:
            logger.info(f"     Skipping duplicate feature: {col_name}")
            continue
        
        # Get leading asset data
        lead_query = """
            SELECT time, returns 
            FROM market_features 
            WHERE symbol = %(symbol)s 
            ORDER BY time
        """
        lead_df = pd.read_sql(lead_query, fs.engine, params={'symbol': asset_x})
        
        if not lead_df.empty:
            lead_df['time'] = pd.to_datetime(lead_df['time'])
            df['time'] = pd.to_datetime(df['time'])
            
            # Shift by lag
            lead_df[col_name] = lead_df['returns'].shift(lag)
            
            # Merge
            df = df.merge(
                lead_df[['time', col_name]], 
                on='time', 
                how='left'
            )
            
            feature_cols.append(col_name)
            granger_info.append({
                'Leading_Asset': asset_x,
                'Target_Asset': asset_y,
                'Lag_Days': lag,
                'Granger_Score': float(score),
                'Feature_Name': col_name
            })
            
            logger.info(f"     Added feature: {col_name} (Granger score: {score:.4f})")
    
    # Drop NaN
    df = df.dropna(subset=feature_cols + ['returns']).reset_index(drop=True)
    
    # Extract features
    features = df[feature_cols].values
    target = df['returns'].values
    dates = df['time'].values
    
    # Normalize
    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0) + 1e-8
    features = (features - mean) / std
    features = np.nan_to_num(features, 0)
    
    # Create sequences (TCN can handle variable length, but we'll use fixed for consistency)
    X, y, seq_dates = [], [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback])
        y.append(target[i + lookback])
        seq_dates.append(dates[i + lookback])
    
    X = np.array(X).astype(np.float32)
    y = np.array(y).reshape(-1, 1).astype(np.float32)
    
    # Calculate lead-lag indicator
    lead_lag_indicator = np.mean([g['Granger_Score'] for g in granger_info]) if granger_info else 0.0
    
    logger.info(f"     Created {len(X)} sequences (shape: {X.shape})")
    logger.info(f"   📊 Features: {feature_cols}")
    logger.info(f"   📍 Lead-lag indicator: {lead_lag_indicator:.4f}")
    
    return {
        'X': X,
        'y': y,
        'dates': seq_dates,
        'feature_cols': feature_cols,
        'granger_info': granger_info,
        'lead_lag_indicator': lead_lag_indicator,
        'raw_data': df
    }


def train_and_predict(X_train, y_train, X_test, epochs=100):
    """Train TCN and make predictions"""
    logger.info(f"\n  Training TCN ({epochs} epochs)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TCN with 3 levels of dilated convolutions: dilation = 1, 2, 4
    model = TCNModel(
        input_dim=X_train.shape[2],
        num_channels=[32, 32, 32],  # 3 layers with 32 channels each
        kernel_size=3,
        dropout=0.2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_size = 32
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_t[i:i + batch_size]
            batch_y = y_train_t[i:i + batch_size]
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / (len(X_train) // batch_size + 1)
            logger.info(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    # Predict
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_t).cpu().numpy()
    
    logger.info("  Training complete!")
    return predictions


def calculate_metrics(y_true, predictions):
    """Calculate performance metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mse)
    
    # Directional accuracy
    y_dir = (y_true.flatten() > 0).astype(int)
    p_dir = (predictions.flatten() > 0).astype(int)
    directional_acc = (y_dir == p_dir).mean()
    
    # Correlation
    correlation = np.corrcoef(y_true.flatten(), predictions.flatten())[0, 1]
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'Directional_Accuracy_%': directional_acc * 100,
        'Correlation': correlation,
        'Correct_Predictions': (y_dir == p_dir).sum(),
        'Total_Predictions': len(y_dir)
    }
    
    logger.info(f"\n📈 Performance Metrics:")
    logger.info(f"   RMSE: {rmse:.6f}")
    logger.info(f"   MAE: {mae:.6f}")
    logger.info(f"   Directional Accuracy: {directional_acc:.2%}")
    logger.info(f"   Correlation: {correlation:.4f}")
    
    return metrics


def export_to_csv(symbol, data_dict, y_test, predictions, metrics):
    """Export all results to CSV files with tcn_leadlag prefix"""
    
    # Create data folder if it doesn't exist
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_folder = os.path.join(project_root, 'data', 'results')
    os.makedirs(data_folder, exist_ok=True)
    
    symbol_lower = symbol.lower()
    
    logger.info(f"\n💾 Exporting TCN results to CSV files...")
    
    # File 1: TCN Predictions
    pred_file = os.path.join(data_folder, f'tcn_leadlag_{symbol_lower}_predictions.csv')
    pred_df = pd.DataFrame({
        'Date': data_dict['dates'],
        'Actual_Return': y_test.flatten(),
        'Predicted_Return': predictions.flatten(),
        'Prediction_Error': y_test.flatten() - predictions.flatten(),
        'Actual_Direction': ['UP' if x > 0 else 'DOWN' for x in y_test.flatten()],
        'Predicted_Direction': ['UP' if x > 0 else 'DOWN' for x in predictions.flatten()],
        'Correct_Prediction': (y_test.flatten() > 0) == (predictions.flatten() > 0),
        'Lead_Lag_Indicator': data_dict['lead_lag_indicator'],
        'Model': 'TCN'
    })
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"     Saved predictions to {pred_file}")
    
    # File 2: Performance Metrics
    metrics_file = os.path.join(data_folder, f'tcn_leadlag_{symbol_lower}_metrics.csv')
    metrics_df = pd.DataFrame([metrics])
    metrics_df['Model'] = 'TCN'
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"     Saved metrics to {metrics_file}")
    
    # File 3: Lead-Lag Relationships (same as LSTM)
    if data_dict['granger_info']:
        granger_file = os.path.join(data_folder, f'tcn_leadlag_{symbol_lower}_relationships.csv')
        granger_df = pd.DataFrame(data_dict['granger_info'])
        granger_df.to_csv(granger_file, index=False)
        logger.info(f"     Saved relationships to {granger_file}")
    
    # File 4: Summary
    summary_file = os.path.join(data_folder, f'tcn_leadlag_{symbol_lower}_summary.csv')
    summary_df = pd.DataFrame({
        'Metric': [
            'Symbol',
            'Model_Type',
            'Total_Data_Points',
            'Training_Samples',
            'Test_Samples',
            'Lookback_Window',
            'Number_of_Features',
            'Lead_Lag_Relationships_Found',
            'Average_Granger_Score',
            'TCN_Accuracy_%',
            'TCN_Layers',
            'Dilation_Pattern',
            'Generated_At'
        ],
        'Value': [
            symbol,
            'Temporal Convolutional Network',
            len(data_dict['raw_data']),
            len(data_dict['X']) - len(predictions),
            len(predictions),
            20,
            len(data_dict['feature_cols']),
            len(data_dict['granger_info']),
            f"{data_dict['lead_lag_indicator']:.4f}",
            f"{metrics['Directional_Accuracy_%']:.2f}",
            '3 (32-32-32 channels)',
            'Exponential (1, 2, 4)',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    })
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"     Saved summary to {summary_file}")
    
    logger.info(f"\n  All TCN CSV files exported for {symbol}")
    return pred_file


def main():
    """Main execution"""
    logger.info("\n" + "="*70)
    logger.info("📊 TCN Lead-Lag Detection - Temporal Convolutional Network")
    logger.info("="*70)
    
    # Initialize
    fs = FeatureStore()
    
    # Get available symbols
    logger.info("\n🔍 Checking available symbols...")
    symbols = get_available_symbols(fs)
    
    if not symbols:
        logger.error("  No symbols found in market_features table!")
        return
    
    logger.info(f"  Found {len(symbols)} symbols: {symbols}")
    
    # Process each symbol
    for symbol in symbols:
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🎯 Processing {symbol} with TCN")
            logger.info(f"{'='*70}")
            
            # Prepare data
            data_dict = prepare_data_with_leadlag(fs, symbol, lookback=20)
            if data_dict is None:
                continue
            
            # Train/test split
            split_idx = int(len(data_dict['X']) * 0.8)
            X_train = data_dict['X'][:split_idx]
            X_test = data_dict['X'][split_idx:]
            y_train = data_dict['y'][:split_idx]
            y_test = data_dict['y'][split_idx:]
            data_dict['dates'] = data_dict['dates'][split_idx:]
            
            logger.info(f"\n📊 Split: {len(X_train)} train, {len(X_test)} test")
            
            # Train and predict
            predictions = train_and_predict(X_train, y_train, X_test, epochs=100)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, predictions)
            
            # Export to CSV
            filename = export_to_csv(symbol, data_dict, y_test, predictions, metrics)
            
            logger.info(f"\n  Completed {symbol}! TCN results saved to {filename}\n")
            
        except Exception as e:
            logger.error(f"  Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*70)
    logger.info("  ALL DONE! TCN models trained for all assets")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
