"""
Premium Network Visualization for FinLagX
Beautiful, professional-grade lead-lag relationship visualization
"""

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import logging
import yaml
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PREMIUM COLOR PALETTE
CATEGORY_COLORS = {
    'EQUITIES': '#667EEA',      # Purple-Blue
    'COMMODITIES': '#F6AD55',   # Warm Orange
    'FX': '#48BB78',            # Fresh Green
    'VOL_BONDS': '#FC8181',     # Coral Red
    'CRYPTO': '#9F7AEA',        # Vibrant Purple
    'MACRO': '#ECC94B',         # Golden Yellow
    'OTHER': '#A0AEC0'          # Cool Gray
}

def load_asset_categories(config_path="configs/config_market.yaml"):
    """Load asset categories from config"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        symbol_map = {}
        for category, assets in config.items():
            if isinstance(assets, dict):
                for name in assets.keys():
                    symbol_map[name] = category
        return symbol_map
    except Exception as e:
        logger.warning(f"Could not load categories: {e}")
        return {}


def create_interactive_network(top_n=60):
    """
    Create a stunning interactive network visualization using Plotly
    """
    logger.info("🎨 Creating premium interactive network visualization...")
    
    # Load data
    fs = FeatureStore()
    df = fs.get_latest_granger_network()
    
    if df.empty:
        logger.error("No data found!")
        return
    
    # Filter top relationships
    df = df.sort_values('granger_score', ascending=False).head(top_n)
    logger.info(f"   Visualizing top {len(df)} relationships")
    
    # Build graph
    G = nx.DiGraph()
    asset_categories = load_asset_categories()
    
    for _, row in df.iterrows():
        G.add_edge(
            row['asset_x'], 
            row['asset_y'], 
            weight=row['granger_score'],
            lag=row.get('optimal_lag', 1)
        )
    
    # Calculate layout using force-directed algorithm
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    
    # Calculate node sizes based on influence (weighted out-degree)
    out_degrees = dict(G.out_degree(weight='weight'))
    in_degrees = dict(G.in_degree(weight='weight'))
    
    # Prepare edge traces
    edge_traces = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        # Create curved edges
        edge_trace = go.Scatter(
            x=[x0, (x0+x1)/2, x1, None],
            y=[y0, (y0+y1)/2 + 0.05, y1, None],
            mode='lines',
            line=dict(
                width=min(weight/2, 3),
                color=f'rgba(252, 129, 129, {min(weight/10, 0.8)})'
            ),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Prepare node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        category = asset_categories.get(node, 'OTHER')
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS['OTHER'])
        node_color.append(color)
        
        out_deg = out_degrees.get(node, 0)
        in_deg = in_degrees.get(node, 0)
        
        # Size by total influence
        size = 20 + (out_deg + in_deg) * 3
        node_size.append(size)
        
        # Hover text
        text = f"<b>{node}</b><br>"
        text += f"Category: {category}<br>"
        text += f"Leads {G.out_degree(node)} assets (strength: {out_deg:.1f})<br>"
        text += f"Led by {G.in_degree(node)} assets (strength: {in_deg:.1f})"
        node_text.append(text)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=10, color='white', family='Arial Black'),
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white'),
            opacity=0.95
        ),
        showlegend=False
    )
    
    # Create legend traces
    legend_traces = []
    for category, color in CATEGORY_COLORS.items():
        # Check if category exists in our data
        if any(asset_categories.get(node, 'OTHER') == category for node in G.nodes()):
            legend_traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=15, color=color, line=dict(width=2, color='white')),
                    name=category,
                    showlegend=True
                )
            )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace] + legend_traces)
    
    # Update layout with premium styling
    fig.update_layout(
        title=dict(
            text="<b>FinLagX Lead-Lag Influence Network</b><br><sub>Interactive Market Intelligence Map</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=28, color='white', family='Arial Black')
        ),
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(20, 20, 40, 0.9)',
            bordercolor='white',
            borderwidth=2,
            font=dict(color='white', size=12)
        ),
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        hovermode='closest',
        margin=dict(l=40, r=200, t=100, b=40),
        height=900,
        annotations=[
            dict(
                text="Node size = influence | Edge thickness = signal strength | Hover for details",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.05,
                xanchor='center',
                font=dict(size=11, color='#888888'),
            )
        ]
    )
    
    # Save
    output_path = "data/interactive_network.html"
    os.makedirs("data", exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"✅ Saved interactive network to {output_path}")
    
    # Also save as static image
    try:
        fig.write_image("data/premium_network.png", width=1920, height=1080, scale=2)
        logger.info(f"✅ Saved static image to data/premium_network.png")
    except:
        logger.warning("   ⚠️ Could not save static image (install kaleido: pip install kaleido)")
    
    return fig


def create_hierarchy_sunburst():
    """
    Create a beautiful hierarchical sunburst chart showing lead-lag relationships
    """
    logger.info("🎨 Creating sunburst hierarchy visualization...")
    
    fs = FeatureStore()
    df = fs.get_latest_granger_network()
    
    if df.empty:
        logger.error("No data found!")
        return
    
    # Top 50 relationships
    df = df.sort_values('granger_score', ascending=False).head(50)
    asset_categories = load_asset_categories()
    
    # Build hierarchy data
    hierarchy_data = []
    
    for _, row in df.iterrows():
        leader = row['asset_x']
        follower = row['asset_y']
        score = row['granger_score']
        
        leader_cat = asset_categories.get(leader, 'OTHER')
        follower_cat = asset_categories.get(follower, 'OTHER')
        
        hierarchy_data.append({
            'leader': leader,
            'follower': follower,
            'leader_category': leader_cat,
            'follower_category': follower_cat,
            'score': score
        })
    
    # Create sunburst
    labels = []
    parents = []
    values = []
    colors = []
    
    # Root
    labels.append("Market")
    parents.append("")
    values.append(len(df))
    colors.append("#0a0e27")
    
    # Categories
    for cat, color in CATEGORY_COLORS.items():
        cat_count = sum(1 for d in hierarchy_data if d['leader_category'] == cat)
        if cat_count > 0:
            labels.append(cat)
            parents.append("Market")
            values.append(cat_count)
            colors.append(color)
    
    # Add top leaders
    leader_counts = {}
    for d in hierarchy_data:
        leader = d['leader']
        if leader not in leader_counts:
            leader_counts[leader] = {'count': 0, 'category': d['leader_category']}
        leader_counts[leader]['count'] += 1
    
    for leader, data in sorted(leader_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:20]:
        labels.append(leader)
        parents.append(data['category'])
        values.append(data['count'])
        colors.append(CATEGORY_COLORS.get(data['category'], '#888888'))
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{label}</b><br>Influences: %{value}<extra></extra>',
        textfont=dict(size=14, color='white', family='Arial Black')
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Market Leadership Hierarchy</b><br><sub>Who Drives the Market?</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=26, color='white', family='Arial Black')
        ),
        paper_bgcolor='#0a0e27',
        height=800,
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    output_path = "data/leadership_sunburst.html"
    fig.write_html(output_path)
    logger.info(f"✅ Saved sunburst to {output_path}")
    
    return fig


def create_dashboard():
    """
    Create a comprehensive dashboard with multiple visualizations
    """
    logger.info("🎨 Creating comprehensive dashboard...")
    
    fs = FeatureStore()
    df = fs.get_latest_granger_network()
    
    if df.empty:
        logger.error("No data found!")
        return
    
    asset_categories = load_asset_categories()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Top Leading Assets",
            "Top Following Assets", 
            "Signal Strength Distribution",
            "Cross-Category Influence"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Top Leaders
    leaders = df.groupby('asset_x')['granger_score'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(10)
    leader_colors = [CATEGORY_COLORS.get(asset_categories.get(idx, 'OTHER'), '#888') for idx in leaders.index]
    
    fig.add_trace(
        go.Bar(
            x=leaders.index,
            y=leaders['sum'],
            marker=dict(color=leader_colors, line=dict(color='white', width=1)),
            text=[f"{v:.1f}" for v in leaders['sum']],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Top Followers
    followers = df.groupby('asset_y')['granger_score'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(10)
    follower_colors = [CATEGORY_COLORS.get(asset_categories.get(idx, 'OTHER'), '#888') for idx in followers.index]
    
    fig.add_trace(
        go.Bar(
            x=followers.index,
            y=followers['sum'],
            marker=dict(color=follower_colors, line=dict(color='white', width=1)),
            text=[f"{v:.1f}" for v in followers['sum']],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Score Distribution
    fig.add_trace(
        go.Histogram(
            x=df['granger_score'],
            nbinsx=30,
            marker=dict(color='#667EEA', line=dict(color='white', width=1)),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Cross-category influence
    df['leader_cat'] = df['asset_x'].map(lambda x: asset_categories.get(x, 'OTHER'))
    df['follower_cat'] = df['asset_y'].map(lambda x: asset_categories.get(x, 'OTHER'))
    
    cross_cat = df.groupby(['leader_cat', 'follower_cat'])['granger_score'].sum().reset_index()
    top_cross = cross_cat.sort_values('granger_score', ascending=False).head(10)
    top_cross['pair'] = top_cross['leader_cat'] + ' → ' + top_cross['follower_cat']
    
    fig.add_trace(
        go.Bar(
            x=top_cross['pair'],
            y=top_cross['granger_score'],
            marker=dict(
                color=[CATEGORY_COLORS.get(cat.split(' → ')[0], '#888') for cat in top_cross['pair']],
                line=dict(color='white', width=1)
            ),
            text=[f"{v:.0f}" for v in top_cross['granger_score']],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>FinLagX Analytics Dashboard</b><br><sub>Comprehensive Market Intelligence Overview</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=28, color='white', family='Arial Black')
        ),
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='white', size=11),
        height=900,
        showlegend=False,
        margin=dict(l=60, r=60, t=120, b=60)
    )
    
    # Update axes
    fig.update_xaxes(showgrid=False, gridcolor='#333', tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor='#333')
    
    output_path = "data/analytics_dashboard.html"
    fig.write_html(output_path)
    logger.info(f"✅ Saved dashboard to {output_path}")
    
    return fig


def main():
    """Generate all premium visualizations"""
    logger.info("\n" + "="*70)
    logger.info("🎨 FINLAGX PREMIUM VISUALIZATION SUITE")
    logger.info("="*70 + "\n")
    
    # Create all visualizations
    create_interactive_network(top_n=60)
    create_hierarchy_sunburst()
    create_dashboard()
    
    logger.info("\n" + "="*70)
    logger.info("✅ ALL PREMIUM VISUALIZATIONS COMPLETE!")
    logger.info("="*70)
    logger.info("\n📂 Check the 'data' folder for:")
    logger.info("   • interactive_network.html - Stunning interactive network")
    logger.info("   • leadership_sunburst.html - Hierarchical market view")
    logger.info("   • analytics_dashboard.html - Comprehensive analytics\n")


if __name__ == "__main__":
    main()
