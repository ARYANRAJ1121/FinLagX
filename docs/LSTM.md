# LSTM Predictor for FinLagX

## What It Does

Advanced LSTM model that:
1. Fetches data from `market_features` table
2. **Integrates Granger causality lead-lag relationships** (e.g., if GOLD leads BITCOIN by 3 days)
3. **Includes VAR model outputs** (fitted values, residuals)
4. Trains on historical returns with enhanced features
5. Saves predictions to `lstm_predictions` table

## Key Features

### Automatic Feature Engineering
- **Base features**: returns, volatility_20, sma_20, sma_50
- **Granger features**: Automatically adds lagged returns from assets that Granger-cause the target
  - Example: If SP500 → BITCOIN (lag 1), adds `SP500_lag1` feature
  - Uses top 5 most significant lead-lag relationships
- **VAR features**: Includes VAR fitted values and residuals if available

### Smart Integration
- Queries `granger_results` table for significant relationships
- Dynamically creates lag features based on optimal lag periods
- Falls back gracefully if Granger/VAR data not available

## Usage

```bash
python src/modeling/lstm_predictor.py
```

That's it. It will:
- Find all symbols in your database
- Train LSTM for each symbol
- Save predictions to `lstm_predictions` table

## What Gets Saved

Table: `lstm_predictions`

Columns:
- `time` - Prediction date
- `symbol` - Asset symbol (e.g., BTC/USD)
- `predicted_return` - Predicted return (%)
- `confidence` - Confidence score (0-1)
- `lead_lag_indicator` - Set to 0.0 for now
- `model_version` - Model version like `lstm_20241125_193045`

## View Results

```python
from sqlalchemy import text, create_engine
import os
from dotenv import load_dotenv

load_dotenv()
db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(db_url)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT time, symbol, predicted_return, confidence
        FROM lstm_predictions
        ORDER BY time DESC
        LIMIT 20
    """))
    
    for row in result:
        print(f"{row[0]} | {row[1]} | {row[2]:.6f}% | {row[3]:.4f}")
```

Or in SQL:
```sql
SELECT * FROM lstm_predictions ORDER BY time DESC LIMIT 20;
```

## Model Details

- **Architecture**: 1-layer LSTM (32 hidden units) → Dense (16) → Output (1)
- **Input**: 20 days of enhanced features:
  - Base: returns, volatility_20, sma_20, sma_50
  - **Granger lead-lag**: Lagged returns from leading assets (auto-discovered)
  - **VAR outputs**: Fitted values, residuals (if available)
- **Output**: Next-day return prediction
- **Training**: 50 epochs, Adam optimizer, MSE loss
- **Metrics**: RMSE, MAE, Directional Accuracy

### Example Feature Set
For BITCOIN with Granger relationships:
- `returns`, `volatility_20`, `sma_20`, `sma_50` (base)
- `GOLD_lag3` (Gold leads Bitcoin by 3 days)
- `SP500_lag1` (SP500 leads Bitcoin by 1 day)
- `var_fitted_value`, `var_residual` (VAR outputs)

**Total: 8 features instead of just 4!**

## Customization

Edit `lstm_predictor.py`:

**Change lookback period:**
```python
predictor = LSTMPredictor(lookback=30)  # Use 30 days
```

**Change model size:**
```python
predictor = LSTMPredictor(hidden_dim=64)  # Bigger model
```

**Change epochs:**
```python
predictor.run_for_symbol(symbol, epochs=100)  # Train longer
```

## Requirements

- Data in `market_features` table (run data pipeline first)
- PyTorch installed: `pip install torch`
- Feature store initialized

## Troubleshooting

**"No data in market_features"**  
→ Run: `python run_complete_pipeline.py`

**"Module not found: torch"**  
→ Run: `pip install torch`

**Database connection error**  
→ Check: `docker ps` (TimescaleDB should be running)

## That's It

One file. One command. Gets the job done.

Run: `python src/modeling/lstm_predictor.py`
