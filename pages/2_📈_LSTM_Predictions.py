"""
LSTM Predictions Page
Per-asset analysis of LSTM predictions and performance metrics
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.dashboard_helpers import (
    ASSETS, ASSET_DISPLAY_NAMES,
    load_asset_predictions,
    load_asset_metrics,
    load_asset_relationships,
    create_prediction_chart,
    create_error_distribution,
    get_available_assets
)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="LSTM Predictions - FinLagX",
    page_icon="📈",
    layout="wide"
)

# ==================== HEADER ====================

st.markdown("# 📈 LSTM Predictions Analysis")
st.markdown("""
Explore LSTM model predictions for individual assets. View performance metrics, prediction accuracy, 
and discover which lead-lag relationships the model leverages for each asset.
""")

st.markdown("---")

# ==================== ASSET SELECTION ====================

available_assets = get_available_assets()

if not available_assets:
    st.error("No asset data available. Please ensure CSV files are in the data folder.")
    st.stop()

# Use session state asset if available, else default to first
default_asset = st.session_state.get('selected_asset', available_assets[0])

selected_asset = st.selectbox(
    "Select Asset",
    options=available_assets,
    format_func=lambda x: ASSET_DISPLAY_NAMES.get(x, x),
    index=available_assets.index(default_asset) if default_asset in available_assets else 0
)

asset_name = ASSET_DISPLAY_NAMES.get(selected_asset, selected_asset)

st.markdown(f"## 🎯 Analyzing: **{asset_name}**")

st.markdown("---")

# ==================== LOAD DATA ====================

predictions_df = load_asset_predictions(selected_asset)
metrics_df = load_asset_metrics(selected_asset)
relationships_df = load_asset_relationships(selected_asset)

# ==================== METRICS DISPLAY ====================

if not metrics_df.empty:
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("RMSE", f"{metrics_df['RMSE'].values[0]:.4f}")
    
    with col2:
        st.metric("MAE", f"{metrics_df['MAE'].values[0]:.4f}")
    
    with col3:
        st.metric("MSE", f"{metrics_df['MSE'].values[0]:.4f}")
    
    with col4:
        accuracy = metrics_df['Directional_Accuracy_%'].values[0]
        st.metric("Directional Accuracy", f"{accuracy:.2f}%")
    
    with col5:
        corr = metrics_df['Correlation'].values[0]
        st.metric("Correlation", f"{corr:.4f}")
    
    st.markdown("---")
else:
    st.warning(f"No metrics available for {asset_name}")

# ==================== PREDICTIONS CHART ====================

if not predictions_df.empty:
    st.markdown("### 📉 Actual vs Predicted Returns")
    
    # Date range filter
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=predictions_df['Date'].min(),
            min_value=predictions_df['Date'].min(),
            max_value=predictions_df['Date'].max()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=predictions_df['Date'].max(),
            min_value=predictions_df['Date'].min(),
            max_value=predictions_df['Date'].max()
        )
    
    # Filter data
    mask = (predictions_df['Date'] >= pd.Timestamp(start_date)) & (predictions_df['Date'] <= pd.Timestamp(end_date))
    filtered_preds = predictions_df[mask]
    
    # Create chart
    if not filtered_preds.empty:
        chart = create_prediction_chart(filtered_preds, asset_name)
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No data in selected date range")
    
    st.markdown("---")
    
    # ==================== ERROR DISTRIBUTION ====================
    
    st.markdown("### 📊 Prediction Error Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        error_chart = create_error_distribution(filtered_preds)
        st.plotly_chart(error_chart, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Understanding Errors:**
        
        - **Centered at 0** = Unbiased predictions
        - **Narrow distribution** = Consistent accuracy
        - **Long tails** = Occasional large misses
        
        The model aims for a tight, zero-centered distribution.
        """)
    
    st.markdown("---")

else:
    st.warning(f"No predictions available for {asset_name}")

# ==================== RECENT PREDICTIONS TABLE ====================

if not predictions_df.empty:
    st.markdown("### 🔍 Recent Predictions")
    
    n_recent = st.slider("Number of recent predictions to show", 5, 50, 20)
    
    recent_preds = predictions_df.tail(n_recent).copy()
    recent_preds = recent_preds[['Date', 'Actual_Return', 'Predicted_Return', 'Prediction_Error', 
                                 'Actual_Direction', 'Predicted_Direction', 'Correct_Prediction']]
    
    st.dataframe(
        recent_preds.style.format({
            'Actual_Return': '{:.4f}',
            'Predicted_Return': '{:.4f}',
            'Prediction_Error': '{:.4f}'
        }).applymap(
            lambda val: 'background-color: #48BB78; color: white' if val == True else 'background-color: #FC8181; color: white',
            subset=['Correct_Prediction']
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="📥 Download All Predictions",
        data=csv,
        file_name=f"{selected_asset}_predictions.csv",
        mime="text/csv"
    )
    
    st.markdown("---")

# ==================== LEAD-LAG RELATIONSHIPS ====================

if not relationships_df.empty:
    st.markdown("### 🔗 Lead-Lag Relationships")
    
    st.info(f"""
    The LSTM model for **{asset_name}** uses {len(relationships_df)} lead-lag relationships 
    based on Granger causality analysis. These features help the model predict future returns.
    """)
    
    st.dataframe(
        relationships_df.style.format({
            'Granger_Score': '{:.4f}',
            'Lag_Days': '{:.0f}'
        }).background_gradient(subset=['Granger_Score'], cmap='YlGn'),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("""
    **Interpretation:**
    - **Leading Asset**: The asset that predicts the target
    - **Lag Days**: How many days in advance the signal appears  
    - **Granger Score**: Strength of the predictive relationship
    - **Feature Name**: The feature added to the LSTM model
    """)
    
    # Download button
    csv = relationships_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Relationships",
        data=csv,
        file_name=f"{selected_asset}_relationships.csv",
        mime="text/csv"
    )
else:
    st.info(f"No lead-lag relationships found for {asset_name}. The model uses only base features.")

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p><strong>Tip:</strong> Assets with more lead-lag relationships often have better prediction accuracy!</p>
</div>
""", unsafe_allow_html=True)
