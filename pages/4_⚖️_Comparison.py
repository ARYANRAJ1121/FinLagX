"""
Multi-Asset Comparison Page
Compare performance, correlations, and relationships across multiple assets
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.dashboard_helpers import (
    ASSETS, ASSET_DISPLAY_NAMES,
    load_asset_metrics,
    load_asset_predictions,
    create_correlation_heatmap,
    create_performance_comparison,
    get_available_assets
)
import plotly.graph_objects as go

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Comparison - FinLagX",
    page_icon="⚖️",
    layout="wide"
)

# ==================== HEADER ====================

st.markdown("# ⚖️ Multi-Asset Comparison")
st.markdown("""
Compare multiple assets side-by-side. Analyze correlations, relative performance, and discover 
cross-asset opportunities for portfolio optimization and risk management.
""")

st.markdown("---")

# ==================== ASSET SELECTION ====================

available_assets = get_available_assets()

if not available_assets:
    st.error("No asset data available.")
    st.stop()

selected_assets = st.multiselect(
    "Select 2-5 Assets to Compare",
    options=available_assets,
    format_func=lambda x: ASSET_DISPLAY_NAMES.get(x, x),
    default=available_assets[:3] if len(available_assets) >= 3 else available_assets
)

if len(selected_assets) < 2:
    st.warning("⚠️ Please select at least 2 assets for comparison")
    st.stop()

if len(selected_assets) > 5:
    st.warning("⚠️ Please select maximum 5 assets for optimal visualization")
    selected_assets = selected_assets[:5]

st.markdown("---")

# ==================== PERFORMANCE METRICS COMPARISON ====================

st.markdown("### 📊 Performance Metrics Comparison")

metrics_list = []
for asset in selected_assets:
    m = load_asset_metrics(asset)
    if not m.empty:
        m['Asset'] = ASSET_DISPLAY_NAMES.get(asset, asset)
        m['asset_id'] = asset
        metrics_list.append(m)

if metrics_list:
    metrics_df = pd.concat(metrics_list, ignore_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Accuracy Comparison")
        perf_chart = create_performance_comparison(selected_assets)
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Metrics Table")
        display_df = metrics_df[['Asset', 'Directional_Accuracy_%', 'RMSE', 'MAE', 'Correlation']]
        st.dataframe(
            display_df.style.format({
                'Directional_Accuracy_%': '{:.2f}%',
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}',
                'Correlation': '{:.4f}'
            }).background_gradient(subset=['Directional_Accuracy_%'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
else:
    st.warning("No metrics available for selected assets")

# ==================== CORRELATION ANALYSIS ====================

st.markdown("### 🔗 Correlation Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    corr_chart = create_correlation_heatmap(selected_assets)
    if corr_chart:
        st.plotly_chart(corr_chart, use_container_width=True)
    else:
        st.info("Need at least 2 assets with data for correlation analysis")

with col2:
    st.markdown("""
    **Understanding Correlations:**
    
    - **+1.0** = Perfect positive correlation
    - **0.0** = No correlation
    - **-1.0** = Perfect negative correlation
    
    **Portfolio Tips:**
    - Low correlation (0 to 0.3) = Good diversification
    - Negative correlation (-0.3 to -1.0) = Natural hedges
    - High correlation (0.7 to 1.0) = Redundant positions
    
    Use this to build balanced portfolios!
    """)

st.markdown("---")

# ==================== COMPARATIVE RETURNS ====================

st.markdown("### 📉 Comparative Returns (Normalized)")

# Load predictions for all selected assets
returns_data = {}
for asset in selected_assets:
    preds = load_asset_predictions(asset)
    if not preds.empty:
        # Use actual returns
        preds = preds.sort_values('Date')
        cumulative = (1 + preds['Actual_Return']).cumprod() - 1  # Cumulative returns
        returns_data[ASSET_DISPLAY_NAMES.get(asset, asset)] = {
            'dates': preds['Date'],
            'returns': cumulative * 100  # Convert to percentage
        }

if returns_data:
    fig = go.Figure()
    
    for asset_name, data in returns_data.items():
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['returns'],
            mode='lines',
            name=asset_name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Cumulative Returns Comparison (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Interpretation:** This chart shows cumulative returns starting from 0%. 
    Higher lines indicate better performance over the period.
    """)
else:
    st.warning("No prediction data available for comparison")

st.markdown("---")

# ==================== RISK-RETURN SCATTER ====================

st.markdown("### 🎯 Risk-Return Profile")

if metrics_list:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate risk (volatility) and return metrics
        risk_return_data = []
        
        for asset in selected_assets:
            preds = load_asset_predictions(asset)
            if not preds.empty:
                avg_return = preds['Actual_Return'].mean() * 100  # Convert to %
                volatility = preds['Actual_Return'].std() * 100  # Convert to %
                risk_return_data.append({
                    'Asset': ASSET_DISPLAY_NAMES.get(asset, asset),
                    'Return': avg_return,
                    'Risk': volatility
                })
        
        if risk_return_data:
            rr_df = pd.DataFrame(risk_return_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rr_df['Risk'],
                y=rr_df['Return'],
                mode='markers+text',
                text=rr_df['Asset'],
                textposition='top center',
                marker=dict(
                    size=15,
                    color=rr_df['Return'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Return (%)")
                ),
                hovertemplate='<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title='Risk-Return Scatter',
                xaxis_title='Risk (Volatility %)',
                yaxis_title='Average Return (%)',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Ideal Positioning:**
        
        - **Top-Left Quadrant** = Low risk, high return (Best!)
        - **Top-Right** = High return, high risk
        - **Bottom-Left** = Low return, low risk (Safe but low reward)
        - **Bottom-Right** = High risk, low return (Worst!)
        
        Look for assets in the top-left for optimal risk-adjusted returns.
        """)

st.markdown("---")

# ==================== BEST PRACTICES ====================

st.markdown("### 💡 Portfolio Recommendations")

if metrics_list:
    best_accuracy = metrics_df.loc[metrics_df['Directional_Accuracy_%'].idxmax()]
    lowest_correlation = None
    
    # Find pair with lowest correlation
    if len(selected_assets) >= 2:
        min_corr = 1.0
        min_pair = None
        
        for i, asset1 in enumerate(selected_assets):
            for asset2 in selected_assets[i+1:]:
                preds1 = load_asset_predictions(asset1)
                preds2 = load_asset_predictions(asset2)
                
                if not preds1.empty and not preds2.empty:
                    # Merge on date
                    merged = pd.merge(
                        preds1[['Date', 'Actual_Return']],
                        preds2[['Date', 'Actual_Return']],
                        on='Date',
                        suffixes=('_1', '_2')
                    )
                    
                    if len(merged) > 10:
                        corr = merged['Actual_Return_1'].corr(merged['Actual_Return_2'])
                        if corr < min_corr:
                            min_corr = corr
                            min_pair = (ASSET_DISPLAY_NAMES.get(asset1, asset1), 
                                      ASSET_DISPLAY_NAMES.get(asset2, asset2))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"""
        **🏆 Best Performer**
        
        {best_accuracy['Asset']}
        - Accuracy: {best_accuracy['Directional_Accuracy_%']:.2f}%
        - RMSE: {best_accuracy['RMSE']:.4f}
        """)
    
    with col2:
        if min_pair:
            st.info(f"""
            **🔀 Best Diversification**
            
            {min_pair[0]} + {min_pair[1]}
            - Correlation: {min_corr:.3f}
            - Good for risk reduction
            """)
    
    with col3:
        st.warning("""
        **⚠️ Risk Management**
        
        - Monitor correlations regularly
        - Rebalance when correlations spike
        - Use lead-lag signals for timing
        """)

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p><strong>Pro Tip:</strong> Combine low-correlated assets with strong individual performance for optimal portfolios!</p>
</div>
""", unsafe_allow_html=True)
