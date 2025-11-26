# TCN (Temporal Convolutional Network) for Lead-Lag Prediction

## Overview

**TCN** (Temporal Convolutional Network) is an advanced deep learning architecture specifically designed for sequence modeling. Unlike LSTMs, TCNs use **dilated causal convolutions** to capture long-range temporal dependencies while maintaining computational efficiency.

## Architecture Details

### Key Components

1. **Dilated Causal Convolutions**
   - **Causal**: Future information never leaks into past predictions
   - **Dilated**: Exponentially increasing receptive field (dilation: 1, 2, 4)
   - Captures dependencies spanning up to 2^n timesteps

2. **Residual Connections**
   - Skip connections around each temporal block
   - Enables training of very deep networks
   - Prevents vanishing gradients

3. **Network Structure**
   ```
   Input (batch, seq_len, features)
       ↓
   Temporal Block 1 (dilation=1, channels=32)
       ↓
   Temporal Block 2 (dilation=2, channels=32)
       ↓
   Temporal Block 3 (dilation=4, channels=32)
       ↓
   Fully Connected Layer → Prediction
   ```

### Hyperparameters

- **Look-back Window**: 20 days
- **Kernel Size**: 3
- **Number of Channels**: [32, 32, 32] (3 layers)
- **Dilation Pattern**: Exponential (1, 2, 4)
- **Dropout**: 0.2
- **Learning Rate**: 0.001
- **Epochs**: 100
- **Batch Size**: 32

## Advantages over LSTM

| Feature | TCN | LSTM |
|---------|-----|------|
| **Parallelization** | Fully parallelizable | Sequential (slower) |
| **Memory** | Fixed-size context | Unbounded (can forget) |
| **Receptive Field** | Exponential growth | Linear |
| **Training Speed** | Faster (parallel) | Slower |
| **Long Dependencies** | Better capture | Gradient issues |

## Input Features

Same as LSTM implementation:

1. **Base Features**:
   - Returns
   - 20-day volatility
   - 20-day SMA
   - 50-day SMA

2. **Lead-Lag Features** (from Granger causality):
   - Lagged returns from assets that Granger-cause the target
   - Example: `SP500_lag2`, `Gold_lag5`

## Output Files

For each asset, the following CSV files are generated with `tcn_leadlag_` prefix:

1. **`tcn_leadlag_{asset}_predictions.csv`**
   - Date, Actual Return, Predicted Return
   - Prediction Error, Directions
   - Correct Prediction flag
   - Lead-lag indicator score
   - Model type: 'TCN'

2. **`tcn_leadlag_{asset}_metrics.csv`**
   - RMSE, MAE, MSE
   - Directional Accuracy (%)
   - Correlation
   - Model type: 'TCN'

3. **`tcn_leadlag_{asset}_relationships.csv`**
   - Leading assets and their lag days
   - Granger scores
   - Feature names used in model

4. **`tcn_leadlag_{asset}_summary.csv`**
   - Model architecture details
   - Dataset statistics
   - TCN-specific info (layers, dilation pattern)
   - Accuracy metrics

## How to Run

```bash
cd d:\FinLagX
python src/modeling/tcn_leadlag.py
```

This will:
1. Connect to TimescaleDB
2. Load market features for all 15+ assets
3. Incorporate Granger lead-lag features
4. Train TCN models (100 epochs each)
5. Save all results with `tcn_leadlag_` prefix

**Estimated Runtime**: 15-30 minutes for all assets (GPU: 10-15 min)

## Expected Performance

Based on TCN literature and our data:

- **Directional Accuracy**: 60-70% (similar to or better than LSTM)
- **RMSE**: Comparable to LSTM, possibly 5-10% improvement
- **Training Time**: 30-50% faster than LSTM
- **Stability**: More consistent across different seeds

## Theoretical Foundation

### Why TCN Works for Financial Time Series

1. **No Information Leakage**: Causal convolutions ensure strict temporal ordering
2. **Multi-Scale Patterns**: Dilated convolutions capture both short-term (intraday) and long-term (weeks) dependencies
3. **Gradient Flow**: Residual connections enable deep architectures without vanishing gradients
4. **Flexibility**: Receptive field size = 2^(num_layers) × kernel_size

### Citation

```
Bai, S., Kolter, J. Z., & Koltun, V. (2018).
"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
arXiv preprint arXiv:1803.01271
```

## Comparison with LSTM

After running, you can compare:

```python
import pandas as pd

# Load LSTM results
lstm_metrics = pd.read_csv('data/sp500_metrics.csv')

# Load TCN results
tcn_metrics = pd.read_csv('data/tcn_leadlag_sp500_metrics.csv')

# Compare
print("LSTM Accuracy:", lstm_metrics['Directional_Accuracy_%'].values[0])
print("TCN Accuracy: ", tcn_metrics['Directional_Accuracy_%'].values[0])
```

## Future Enhancements

1. **Attention-TCN**: Add self-attention between TCN layers
2. **Wavenet-style**: Use gated activations (tanh/sigmoid)
3. **Multi-Horizon**: Predict multiple future timesteps
4. **Ensemble**: Combine TCN + LSTM predictions

## Troubleshooting

**Out of Memory?**
- Reduce `num_channels` to [16, 16, 16]
- Decrease batch size to 16

**Slower than expected?**
- Ensure PyTorch is using GPU (`torch.cuda.is_available()`)
- Reduce number of epochs

**Poor performance?**
- Increase num_channels to [64, 64, 64]
- Add more temporal blocks (increase dilation levels)
- Tune kernel size (try 5 or 7)

---

**Author**: FinLagX Research Team  
**Date**: November 2025  
**Model Type**: Temporal Convolutional Network (TCN)
