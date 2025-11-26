"""
Premium PNG Visualizations for FinLagX
Beautiful, high-resolution network graphics for Streamlit dashboards
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
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

# PREMIUM COLOR PALETTE - Modern, vibrant, professional
CATEGORY_COLORS = {
    'EQUITIES': '#667EEA',      # Purple-Blue
    'COMMODITIES': '#F6AD55',   # Warm Orange
    'FX': '#48BB78',            # Fresh Green
    'VOL_BONDS': '#FC8181',     # Coral Red
    'CRYPTO': '#9F7AEA',        # Vibrant Purple
    'MACRO': '#ECC94B',         # Golden Yellow
    'OTHER': '#A0AEC0'          # Cool Gray
}

DARK_BG = '#0f172a'  # Deep Navy
LIGHT_BG = '#f8fafc'  # Soft White


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


def create_premium_network_dark(top_n=50):
    """
    Create stunning dark-mode network visualization
    """
    logger.info("🎨 Creating premium dark network (PNG)...")
    
    # Load data
    fs = FeatureStore()
    df = fs.get_latest_granger_network()
    
    if df.empty:
        logger.error("No data found!")
        return
    
    # Top relationships
    df = df.sort_values('granger_score', ascending=False).head(top_n)
    logger.info(f"   Visualizing top {len(df)} relationships")
    
    # Build graph
    G = nx.DiGraph()
    asset_categories = load_asset_categories()
    
    for _, row in df.iterrows():
        G.add_edge(row['asset_x'], row['asset_y'], weight=row['granger_score'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 16), facecolor=DARK_BG, dpi=150)
    ax.set_facecolor(DARK_BG)
    
    # Layout - spring with good spacing
    pos = nx.spring_layout(G, k=3.5, iterations=150, seed=42)
    
    # Calculate node metrics
    out_degrees = dict(G.out_degree(weight='weight'))
    in_degrees = dict(G.in_degree(weight='weight'))
    
    # Edge styling with gradient
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    
    # Custom colormap for edges (dark red to bright red)
    edge_colors = [(0.3, 0.05, 0.05), (1, 0.2, 0.3), (1, 0.4, 0.5)]
    edge_cmap = LinearSegmentedColormap.from_list('neon_red', edge_colors)
    
    # Draw edges with glow effect
    # First pass: thick, transparent (glow)
    nx.draw_networkx_edges(
        G, pos,
        width=[w/max_weight * 8 for w in weights],
        edge_color=weights,
        edge_cmap=edge_cmap,
        alpha=0.15,
        connectionstyle='arc3,rad=0.15',
        arrowsize=0,
        ax=ax
    )
    
    # Second pass: thin, bright (core)
    nx.draw_networkx_edges(
        G, pos,
        width=[w/max_weight * 3 for w in weights],
        edge_color=weights,
        edge_cmap=edge_cmap,
        alpha=0.8,
        arrowstyle='->',
        arrowsize=20,
        connectionstyle='arc3,rad=0.15',
        ax=ax
    )
    
    # Node styling
    node_sizes = []
    node_colors = []
    
    for node in G.nodes():
        # Size based on total influence
        total_influence = out_degrees.get(node, 0) + in_degrees.get(node, 0) * 0.5
        size = 2000 + total_influence * 200
        node_sizes.append(size)
        
        # Color by category
        category = asset_categories.get(node, 'OTHER')
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS['OTHER'])
        node_colors.append(color)
    
    # Draw nodes with halo effect
    # Outer glow
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[s + 800 for s in node_sizes],
        node_color=node_colors,
        alpha=0.15,
        ax=ax
    )
    
    # Main nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='white',
        linewidths=3,
        alpha=1.0,
        ax=ax
    )
    
    # Labels with strong outline
    labels_dict = {node: node for node in G.nodes()}
    text_items = nx.draw_networkx_labels(
        G, pos,
        labels_dict,
        font_size=11,
        font_weight='bold',
        font_family='sans-serif',
        font_color='white',
        ax=ax
    )
    
    # Add outline to text for readability
    for text in text_items.values():
        text.set_path_effects([
            path_effects.Stroke(linewidth=5, foreground=DARK_BG),
            path_effects.Normal()
        ])
    
    # Legend
    legend_elements = []
    for category, color in CATEGORY_COLORS.items():
        if any(asset_categories.get(node, 'OTHER') == category for node in G.nodes()):
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='white', linewidth=2, label=category)
            )
    
    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=13,
        frameon=True,
        facecolor=DARK_BG,
        edgecolor='white',
        framealpha=0.9,
        title='ASSET CLASSES',
        title_fontsize=14
    )
    legend.get_title().set_color('white')
    legend.get_title().set_weight('bold')
    
    for text in legend.get_texts():
        text.set_color('white')
        text.set_weight('600')
    
    # Title
    title = ax.text(
        0.5, 1.05,
        'FINLAGX MARKET INTELLIGENCE NETWORK',
        transform=ax.transAxes,
        fontsize=32,
        weight='bold',
        color='white',
        ha='center',
        va='top',
        family='sans-serif'
    )
    title.set_path_effects([
        path_effects.Stroke(linewidth=6, foreground=DARK_BG),
        path_effects.Normal()
    ])
    
    subtitle = ax.text(
        0.5, 1.01,
        f'Lead-Lag Relationship Map • Top {len(df)} Signals',
        transform=ax.transAxes,
        fontsize=16,
        color='#94a3b8',
        ha='center',
        va='top',
        style='italic'
    )
    
    # Footer annotation
    ax.text(
        0.5, -0.02,
        'Node size = Market influence | Edge thickness = Signal strength | Arrow direction = Lead-Lag relationship',
        transform=ax.transAxes,
        fontsize=11,
        color='#64748b',
        ha='center',
        va='top',
        style='italic'
    )
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save with high quality
    output_path = 'data/network_dark_premium.png'
    os.makedirs('data', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    logger.info(f"  Saved to {output_path}")
    plt.close()


def create_premium_network_light(top_n=50):
    """
    Create stunning light-mode network visualization
    """
    logger.info("🎨 Creating premium light network (PNG)...")
    
    # Load data
    fs = FeatureStore()
    df = fs.get_latest_granger_network()
    
    if df.empty:
        return
    
    df = df.sort_values('granger_score', ascending=False).head(top_n)
    
    # Build graph
    G = nx.DiGraph()
    asset_categories = load_asset_categories()
    
    for _, row in df.iterrows():
        G.add_edge(row['asset_x'], row['asset_y'], weight=row['granger_score'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 16), facecolor=LIGHT_BG, dpi=150)
    ax.set_facecolor(LIGHT_BG)
    
    # Layout
    pos = nx.spring_layout(G, k=3.5, iterations=150, seed=42)
    
    # Metrics
    out_degrees = dict(G.out_degree(weight='weight'))
    in_degrees = dict(G.in_degree(weight='weight'))
    
    # Edges
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    
    # Draw edges - subtle gray with varying opacity
    for (u, v), w in zip(G.edges(), weights):
        alpha = 0.3 + (w/max_weight) * 0.5
        width = 1 + (w/max_weight) * 4
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        ax.annotate(
            '',
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle='->',
                lw=width,
                color='#64748b',
                alpha=alpha,
                connectionstyle='arc3,rad=0.15',
                shrinkA=20,
                shrinkB=20
            )
        )
    
    # Nodes
    node_sizes = []
    node_colors = []
    
    for node in G.nodes():
        total_influence = out_degrees.get(node, 0) + in_degrees.get(node, 0) * 0.5
        size = 2000 + total_influence * 200
        node_sizes.append(size)
        
        category = asset_categories.get(node, 'OTHER')
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS['OTHER'])
        node_colors.append(color)
    
    # Draw nodes with shadow
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[s + 400 for s in node_sizes],
        node_color='#cbd5e1',
        alpha=0.3,
        ax=ax
    )
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='white',
        linewidths=4,
        alpha=0.95,
        ax=ax
    )
    
    # Labels
    labels_dict = {node: node for node in G.nodes()}
    text_items = nx.draw_networkx_labels(
        G, pos,
        labels_dict,
        font_size=11,
        font_weight='bold',
        font_family='sans-serif',
        font_color='#1e293b',
        ax=ax
    )
    
    for text in text_items.values():
        text.set_path_effects([
            path_effects.Stroke(linewidth=6, foreground='white'),
            path_effects.Normal()
        ])
    
    # Legend
    legend_elements = []
    for category, color in CATEGORY_COLORS.items():
        if any(asset_categories.get(node, 'OTHER') == category for node in G.nodes()):
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='#475569', linewidth=2, label=category)
            )
    
    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=13,
        frameon=True,
        facecolor='white',
        edgecolor='#cbd5e1',
        framealpha=0.95,
        title='ASSET CLASSES',
        title_fontsize=14,
        shadow=True
    )
    legend.get_title().set_color('#1e293b')
    legend.get_title().set_weight('bold')
    
    for text in legend.get_texts():
        text.set_color('#334155')
        text.set_weight('600')
    
    # Title
    ax.text(
        0.5, 1.05,
        'FINLAGX MARKET INTELLIGENCE NETWORK',
        transform=ax.transAxes,
        fontsize=32,
        weight='bold',
        color='#0f172a',
        ha='center',
        va='top',
        family='sans-serif'
    )
    
    ax.text(
        0.5, 1.01,
        f'Lead-Lag Relationship Map • Top {len(df)} Signals',
        transform=ax.transAxes,
        fontsize=16,
        color='#475569',
        ha='center',
        va='top',
        style='italic'
    )
    
    # Footer
    ax.text(
        0.5, -0.02,
        'Node size = Market influence | Edge thickness = Signal strength | Arrow direction = Lead-Lag relationship',
        transform=ax.transAxes,
        fontsize=11,
        color='#64748b',
        ha='center',
        va='top',
        style='italic'
    )
    
    ax.axis('off')
    plt.tight_layout()
    
    output_path = 'data/network_light_premium.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=LIGHT_BG)
    logger.info(f"  Saved to {output_path}")
    plt.close()


def create_top_leaders_chart():
    """
    Create a beautiful bar chart of top market leaders
    """
    logger.info("🎨 Creating top leaders chart (PNG)...")
    
    fs = FeatureStore()
    df = fs.get_latest_granger_network()
    
    if df.empty:
        return
    
    asset_categories = load_asset_categories()
    
    # Get top leaders
    leaders = df.groupby('asset_x').agg({
        'granger_score': 'sum',
        'asset_y': 'count'
    }).sort_values('granger_score', ascending=False).head(15)
    
    leaders.columns = ['Total_Influence', 'Assets_Influenced']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=DARK_BG, dpi=150)
    ax.set_facecolor(DARK_BG)
    
    # Get colors
    colors = [CATEGORY_COLORS.get(asset_categories.get(asset, 'OTHER'), CATEGORY_COLORS['OTHER']) 
              for asset in leaders.index]
    
    # Create bars
    bars = ax.barh(
        range(len(leaders)),
        leaders['Total_Influence'],
        color=colors,
        edgecolor='white',
        linewidth=2,
        alpha=0.9
    )
    
    # Add glow effect
    for i, bar in enumerate(bars):
        bar_height = bar.get_height()
        bar_width = bar.get_width()
        bar_y = bar.get_y()
        
        # Glow
        ax.barh(
            i,
            bar_width,
            height=bar_height * 1.5,
            color=colors[i],
            alpha=0.2,
            edgecolor='none'
        )
    
    # Labels
    ax.set_yticks(range(len(leaders)))
    ax.set_yticklabels(leaders.index, fontsize=14, weight='bold', color='white')
    
    # Add value labels
    for i, (idx, row) in enumerate(leaders.iterrows()):
        ax.text(
            row['Total_Influence'] + max(leaders['Total_Influence']) * 0.02,
            i,
            f"{row['Total_Influence']:.1f}  ({int(row['Assets_Influenced'])} assets)",
            va='center',
            fontsize=12,
            weight='600',
            color='white'
        )
    
    # Styling
    ax.set_xlabel('Total Influence Score', fontsize=16, weight='bold', color='white', labelpad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#475569')
    ax.spines['bottom'].set_color('#475569')
    ax.tick_params(colors='white', labelsize=12)
    ax.grid(axis='x', alpha=0.2, color='white', linestyle='--')
    
    # Title
    ax.text(
        0.5, 1.05,
        'TOP MARKET LEADERS',
        transform=ax.transAxes,
        fontsize=28,
        weight='bold',
        color='white',
        ha='center',
        va='top'
    )
    
    ax.text(
        0.5, 1.01,
        'Assets with Strongest Lead-Lag Influence',
        transform=ax.transAxes,
        fontsize=14,
        color='#94a3b8',
        ha='center',
        va='top',
        style='italic'
    )
    
    plt.tight_layout()
    
    output_path = 'data/top_leaders.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    logger.info(f"  Saved to {output_path}")
    plt.close()


def main():
    """Generate all premium PNG visualizations"""
    logger.info("\n" + "="*70)
    logger.info("🎨 FINLAGX PREMIUM PNG GENERATOR")
    logger.info("="*70 + "\n")
    
    create_premium_network_dark(top_n=50)
    create_premium_network_light(top_n=50)
    create_top_leaders_chart()
    
    logger.info("\n" + "="*70)
    logger.info("  ALL PNG VISUALIZATIONS COMPLETE!")
    logger.info("="*70)
    logger.info("\n📂 Saved to 'data/' folder:")
    logger.info("   • network_dark_premium.png - Dark mode network (perfect for dashboards)")
    logger.info("   • network_light_premium.png - Light mode network")
    logger.info("   • top_leaders.png - Top influencing assets\n")


if __name__ == "__main__":
    main()
