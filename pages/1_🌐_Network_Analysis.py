"""
Network Analysis Page
Displays Granger causality network visualizations and relationships
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.dashboard_helpers import (
    ASSET_DISPLAY_NAMES, CATEGORY_COLORS,
    load_granger_results_from_db
)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Network Analysis - FinLagX",
    page_icon="🌐",
    layout="wide"
)

# ==================== HEADER ====================

st.markdown("# 🌐 Granger Causality Network Analysis")
st.markdown("""
Explore lead-lag relationships between financial assets. The network shows which assets **predict** others,
revealing hidden market dynamics and systemic risk patterns.
""")

st.markdown("---")

# ==================== CONTROLS ====================

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("### 🎛️ Visualization Controls")

with col2:
    theme_mode = st.selectbox("Network Theme", ["Dark", "Light"], index=0)

st.markdown("---")

# ==================== NETWORK VISUALIZATION ====================

st.markdown("## 🕸️ Network Visualization")

col1, col2 = st.columns([3, 1])

with col1:
    network_file = "network_dark_premium.png" if theme_mode == "Dark" else "network_light_premium.png"
    network_path = Path("data") / network_file
    
    if network_path.exists():
        st.image(str(network_path), caption=f"Granger Causality Network - Top  Relationships")
    else:
        st.error(f"""
        **Network visualization not found!**
        
        Please generate the visualizations first by running:
        ```bash
        python src/visualization/create_premium_pngs.py
        ```
        """)

with col2:
    st.markdown("""
    ### 🔍 Reading the Network
    
    **Nodes (Bubbles):**
    - Size = Market influence
    - Color = Asset category
    
    **Edges (Arrows):**
    - Direction = Lead-lag
    - Thickness = Strength
    - Arrow: A → B means "A predicts B"
    
    **Categories:**
    """)
    
    for cat, color in CATEGORY_COLORS.items():
        st.markdown(f'<div style="display: flex; align-items: center; margin: 0.3rem 0;">'
                   f'<div style="width: 20px; height: 20px; background: {color}; border-radius: 3px; margin-right: 0.5rem;"></div>'
                   f'<span>{cat}</span></div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== TOP LEADERS ====================

st.markdown("## 👑 Top Market Leaders")

col1, col2 = st.columns([2, 2])

with col1:
    leaders_path = Path("data/top_leaders.png")
    if leaders_path.exists():
        st.image(str(leaders_path), caption="Assets with Strongest Lead-Lag Influence")
    else:
        st.warning("Top leaders chart not found.")

with col2:
    st.markdown("""
    ### Understanding Market Leaders
    
    These assets have the **strongest predictive power** over other markets:
    
    - **High influence** = Predicts many other assets
    - **Leading indicators** = Early warning signals
    - **Systemic importance** = Key to risk management
    
    Use these for:
    - Portfolio hedging
    - Risk monitoring
    - Trading signals
    - Market timing
    """)

st.markdown("---")

# ==================== GRANGER RESULTS TABLE ====================

st.markdown("## 📊 Granger Causality Results")

granger_df = load_granger_results_from_db()

if not granger_df.empty:
    st.info(f"Showing {len(granger_df)} significant relationships (p-value < 0.05)")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        asset_filter = st.multiselect(
            "Filter by Leading Asset",
            options=sorted(granger_df['asset_x'].unique()),
            default=[]
        )
    
    with col2:
        target_filter = st.multiselect(
            "Filter by Target Asset",
            options=sorted(granger_df['asset_y'].unique()),
            default=[]
        )
    
    with col3:
        min_score = st.slider(
            "Minimum Granger Score",
            min_value=0.0,
            max_value=float(granger_df['granger_score'].max()),
            value=0.0,
            step=0.1
        )
    
    # Apply filters
    filtered_df = granger_df.copy()
    
    if asset_filter:
        filtered_df = filtered_df[filtered_df['asset_x'].isin(asset_filter)]
    
    if target_filter:
        filtered_df = filtered_df[filtered_df['asset_y'].isin(target_filter)]
    
    filtered_df = filtered_df[filtered_df['granger_score'] >= min_score]
    
    # Display table
    st.dataframe(
        filtered_df.style.format({
            'granger_score': '{:.4f}',
            'p_value': '{:.6f}'
        }).background_gradient(subset=['granger_score'], cmap='YlOrRd'),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="granger_causality_results.csv",
        mime="text/csv"
    )
else:
    st.warning("""
    No Granger causality results found in the database. 
    
    Please run the Granger analysis first:
    ```bash
    python src/modeling/run_statistical_models.py
    ```
    """)

st.markdown("---")

# ==================== INTERACTIVE NETWORK ====================

st.markdown("## 🔄 Interactive Network")

interactive_path = Path("data/interactive_network.html")

if interactive_path.exists():
    with open(interactive_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    st.components.v1.html(html_content, height=800, scrolling=True)
else:
    st.info("Interactive network visualization not available. The static network above shows the key relationships.")

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p><strong>Tip:</strong> Combine network insights with LSTM predictions for powerful trading signals!</p>
</div>
""", unsafe_allow_html=True)
