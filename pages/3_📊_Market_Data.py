"""
Market Data Explorer Page
Interactive exploration of market features and raw data
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.dashboard_helpers import (
    ASSETS, ASSET_DISPLAY_NAMES,
    load_market_features_from_db,
    create_time_series_chart,
    get_available_assets
)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Market Data - FinLagX",
    page_icon="📊",
    layout="wide"
)

# ==================== HEADER ====================

st.markdown("# 📊 Market Data Explorer")
st.markdown("""
Explore raw market data and features directly from the TimescaleDB database. 
Visualize price movements, returns, volatility, and technical indicators across multiple assets.
""")

st.markdown("---")

# ==================== CONTROLS ====================

st.markdown("### 🎛️ Data Selection")

col1, col2, col3 = st.columns(3)

with col1:
    # Asset selection
    available_assets = get_available_assets()
    selected_assets = st.multiselect(
        "Select Assets",
        options=available_assets,
        format_func=lambda x: ASSET_DISPLAY_NAMES.get(x, x),
        default=[available_assets[0]] if available_assets else []
    )

with col2:
    # Date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    date_start = st.date_input(
        "Start Date",
        value=start_date,
        max_value=end_date
    )

with col3:
    date_end = st.date_input(
        "End Date",
        value=end_date,
        max_value=end_date
    )

# Feature selection
features = st.multiselect(
    "Select Features",
    options=['returns', 'return_5d', 'return_10d', 'volatility_20', 'sma_20', 'sma_50'],
    default=['returns', 'volatility_20']
)

st.markdown("---")

# ==================== LOAD DATA ====================

if selected_assets and features:
    with st.spinner("Loading data from database..."):
        # Map display names back to asset IDs
        asset_ids = selected_assets
        
        df = load_market_features_from_db(
            symbols=asset_ids,
            start_date=pd.Timestamp(date_start),
            end_date=pd.Timestamp(date_end)
        )
    
    if not df.empty:
        st.success(f"Loaded {len(df):,} data points for {len(selected_assets)} asset(s)")
        
        # ==================== TIME SERIES CHARTS ====================
        
        st.markdown("### 📈 Time Series Visualization")
        
        for feature in features:
            if feature in df.columns:
                st.markdown(f"#### {feature.replace('_', ' ').title()}")
                
                chart = create_time_series_chart(df, asset_ids, feature)
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning(f"Feature '{feature}' not found in data")
        
        st.markdown("---")
        
        # ==================== STATISTICS ====================
        
        st.markdown("### 📊 Statistical Summary")
        
        # Group by symbol and calculate stats
        stats_data = []
        
        for asset in asset_ids:
            asset_df = df[df['symbol'] == asset]
            
            for feature in features:
                if feature in asset_df.columns:
                    stats_data.append({
                        'Asset': ASSET_DISPLAY_NAMES.get(asset, asset),
                        'Feature': feature,
                        'Mean': asset_df[feature].mean(),
                        'Std': asset_df[feature].std(),
                        'Min': asset_df[feature].min(),
                        'Max': asset_df[feature].max(),
                        'Latest': asset_df[feature].iloc[-1] if len(asset_df) > 0 else None
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            st.dataframe(
                stats_df.style.format({
                    'Mean': '{:.4f}',
                    'Std': '{:.4f}',
                    'Min': '{:.4f}',
                    'Max': '{:.4f}',
                    'Latest': '{:.4f}'
                }).background_gradient(subset=['Mean'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("---")
        
        # ==================== RAW DATA TABLE ====================
        
        st.markdown("### 🗂️ Raw Data")
        
        # Pagination
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        
        total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
        page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Display columns
        display_cols = ['time', 'symbol'] + [f for f in features if f in df.columns]
        
        st.dataframe(
            df[display_cols].iloc[start_idx:end_idx].style.format({
                col: '{:.4f}' for col in features if col in df.columns
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.info(f"Showing rows {start_idx + 1} to {min(end_idx, len(df))} of {len(df):,}")
        
        # ==================== DOWNLOAD ====================
        
        st.markdown("### 💾 Export Data")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"market_data_{date_start}_to_{date_end}.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("No data found for the selected filters. Try expanding the date range or selecting different assets.")

else:
    st.info("👆 Please select at least one asset and one feature to visualize data")

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p><strong>Data Source:</strong> TimescaleDB • <strong>Update Frequency:</strong> Daily</p>
</div>
""", unsafe_allow_html=True)
