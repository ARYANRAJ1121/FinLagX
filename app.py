"""
FinLagX Analytics Dashboard
Multi-page Streamlit application for visualizing market lead-lag relationships,
LSTM predictions, and Granger causality networks
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.dashboard_helpers import (
    ASSETS, ASSET_DISPLAY_NAMES,
    calculate_summary_stats,
    get_recent_predictions,
    load_all_metrics,
    format_metric,
    get_available_assets
)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="FinLagX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667EEA 0%, #764BA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #667EEA;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .info-box {
        background: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #48BB78;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667EEA/FFFFFF?text=FinLagX", use_container_width=True)
    
    st.markdown("## 🎯 Navigation")
    st.markdown("""
    Welcome to the **FinLagX Analytics Dashboard**!
    
    Navigate through the pages using the menu on the left:
    - 🏠 **Home** - Overview and key metrics
    - 🌐 **Network Analysis** - Granger causality networks
    - 📈 **LSTM Predictions** - Per-asset analysis
    - 📊 **Market Data** - Data explorer
    - ⚖️ **Comparison** - Multi-asset comparison
    """)
    
    st.markdown("---")
   
    
    available_assets = get_available_assets()
    default_asset = available_assets[0] if available_assets else None
    
    
    
    # Store in session state for access across pages
  
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
        <p><strong>FinLagX v1.0</strong></p>
        <p>Advanced Lead-Lag Analysis</p>
        <p>© 2025</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN PAGE ====================

st.markdown("<h1 class='main-header'>📊 FinLagX Analytics Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""
Welcome to the **FinLagX Analytics Dashboard** - your comprehensive platform for exploring lead-lag relationships, 
LSTM predictions, and market intelligence across global financial assets.
""")

# ==================== SUMMARY METRICS ====================

st.markdown("## 📈 Overview")

stats = calculate_summary_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Assets",
        value=stats['total_assets'],
        delta=None
    )

with col2:
    st.metric(
        label="Average Accuracy",
        value=f"{stats['avg_accuracy']:.2f}%",
        delta=None
    )

with col3:
    st.metric(
        label="Total Predictions",
        value=f"{int(stats['total_predictions']):,}",
        delta=None
    )

with col4:
    st.metric(
        label="Average RMSE",
        value=f"{stats['avg_rmse']:.4f}",
        delta=None
    )

st.markdown("---")

# ==================== PERFORMANCE LEADERBOARD ====================

st.markdown("## 🏆 Model Performance Leaderboard")

all_metrics = load_all_metrics()

if not all_metrics.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 5 Performers")
        top_5 = all_metrics.nlargest(5, 'Directional_Accuracy_%')[['Asset', 'Directional_Accuracy_%', 'RMSE', 'Correlation']]
        st.dataframe(
            top_5.style.format({
                'Directional_Accuracy_%': '{:.2f}%',
                'RMSE': '{:.4f}',
                'Correlation': '{:.4f}'
            }).background_gradient(subset=['Directional_Accuracy_%'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("### Bottom 5 Performers")
        bottom_5 = all_metrics.nsmallest(5, 'Directional_Accuracy_%')[['Asset', 'Directional_Accuracy_%', 'RMSE', 'Correlation']]
        st.dataframe(
            bottom_5.style.format({
                'Directional_Accuracy_%': '{:.2f}%',
                'RMSE': '{:.4f}',
                'Correlation': '{:.4f}'
            }).background_gradient(subset=['Directional_Accuracy_%'], cmap='Reds'),
            use_container_width=True,
            hide_index=True
        )
else:
    st.warning("No metrics data available. Please ensure CSV files are present in the data folder.")

st.markdown("---")

# ==================== RECENT PREDICTIONS ====================

st.markdown("## 🔮 Recent Predictions")

recent = get_recent_predictions(n=10)

if not recent.empty:
    display_cols = ['Date', 'Asset', 'Actual_Return', 'Predicted_Return', 'Prediction_Error', 'Correct_Prediction']
    
    st.dataframe(
        recent[display_cols].style.format({
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
else:
    st.info("No recent predictions available.")

st.markdown("---")

# ==================== QUICK LINKS ====================

st.markdown("##   Quick Access")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='info-box'>
        <h3>🌐 Network Analysis</h3>
        <p>Explore Granger causality networks and discover which assets lead the market.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='info-box'>
        <h3>📈 LSTM Predictions</h3>
        <p>Deep dive into asset-specific predictions and model performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='info-box'>
        <h3>⚖️ Comparison</h3>
        <p>Compare multiple assets and analyze correlations and relative performance.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== NETWORK PREVIEW ====================

st.markdown("## 🌐 Granger Causality Network Preview")

col1, col2 = st.columns([2, 1])

with col1:
    network_path = Path("data/network_dark_premium.png")
    if network_path.exists():
        st.image(str(network_path), caption="Granger Causality Network - Top Market Relationships")
    else:
        st.warning("Network visualization not found. Please run the premium PNG generator first.")

with col2:
    st.markdown("""
    ### Understanding the Network
    
    This network visualization shows the **lead-lag relationships** between different financial assets:
    
    - **Nodes** represent different assets
    - **Edges** (arrows) show predictive relationships
    - **Arrow direction** indicates which asset leads
    - **Edge thickness** represents signal strength
    - **Node size** indicates market influence
    
    Navigate to the **Network Analysis** page for interactive exploration!
    """)

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p>Built with ❤️ using Streamlit • Data updated in real-time from TimescaleDB</p>
</div>
""", unsafe_allow_html=True)
