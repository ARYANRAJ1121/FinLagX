import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from src.feature_store import FeatureStore
import yaml
from matplotlib.colors import LinearSegmentedColormap

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NEON COLOR PALETTE ---
# Professional Dark Mode Colors
CATEGORY_COLORS = {
    'EQUITIES': '#00F0FF',    # Neon Cyan
    'COMMODITIES': '#FFA500', # Neon Orange
    'FX': '#39FF14',          # Neon Lime
    'VOL_BONDS': '#FF0055',   # Neon Pink
    'CRYPTO': '#9D00FF',      # Neon Purple
    'MACRO': '#FFFF00',       # Neon Yellow
    'OTHER': '#B0B0B0'        # Silver
}

BACKGROUND_COLOR = '#121212'  # Deep Charcoal (Better than pure black)
TEXT_COLOR = 'white'

def load_asset_categories(config_path="configs/config_market.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        symbol_map = {}
        for category, assets in config.items():
            if isinstance(assets, dict):
                for name in assets.keys():
                    symbol_map[name] = category
        return symbol_map
    except:
        return {}

def plot_dark_network():
    logger.info("🎨 Generating Professional Dark Mode Network...")
    fs = FeatureStore()
    
    # 1. Fetch & Filter Data
    try:
        df = fs.get_latest_granger_network()
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    if df.empty:
        logger.warning("No data found.")
        return

    # Keep top 40 strongest signals for clarity
    df = df.sort_values('granger_score', ascending=False).head(40)

    # 2. Build Graph
    G = nx.DiGraph()
    asset_categories = load_asset_categories()
    
    for _, row in df.iterrows():
        G.add_edge(row['asset_x'], row['asset_y'], weight=row['granger_score'])

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(20, 14), facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    # 4. Layout Algorithm (Physics)
    # k=5 pushes nodes apart, iterations=100 untangles them
    pos = nx.spring_layout(G, k=4.5, iterations=100, seed=42)

    # 5. Draw Edges (The "Glow" Effect)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Custom Gradient Colormap (Transparent Red -> Bright Red)
    colors = [(0.2, 0, 0, 0), (1, 0, 0.3, 1)] # Fade from black to neon red
    cmap_name = 'neon_reds'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

    # Draw wider, transparent lines first to create "glow"
    nx.draw_networkx_edges(
        G, pos, 
        width=4, 
        edge_color=weights, 
        edge_cmap=cm, 
        arrowstyle='-', 
        alpha=0.3,  # Low alpha for glow
        connectionstyle="arc3,rad=0.2",
        ax=ax
    )
    
    # Draw thin, bright lines on top for precision
    nx.draw_networkx_edges(
        G, pos, 
        width=1.5, 
        edge_color=weights, 
        edge_cmap=cm, 
        arrowstyle='-|>', 
        arrowsize=25, 
        connectionstyle="arc3,rad=0.2",
        ax=ax
    )

    # 6. Draw Nodes (Neon Bubbles)
    # Size by influence (Out-Degree)
    degrees = dict(G.out_degree(weight='weight'))
    node_sizes = [2500 + (degrees.get(n, 0) * 400) for n in G.nodes()]
    node_colors = [CATEGORY_COLORS.get(asset_categories.get(n, 'OTHER'), '#7f7f7f') for n in G.nodes()]

    # Halo effect for nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=[s + 500 for s in node_sizes], 
        node_color='white', 
        alpha=0.2, 
        ax=ax
    )
    
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color=node_colors, 
        alpha=1.0, 
        edgecolors='white', 
        linewidths=2,
        ax=ax
    )

    # 7. Labels
    # Draw text with a black outline so it pops
    import matplotlib.patheffects as path_effects
    text = nx.draw_networkx_labels(
        G, pos, 
        font_size=11, 
        font_weight='bold', 
        font_color='white', 
        font_family='sans-serif',
        ax=ax
    )
    for t in text.values():
        t.set_path_effects([path_effects.withStroke(linewidth=4, foreground=BACKGROUND_COLOR)])

    # 8. Custom Legend
    legend_handles = []
    for cat, color in CATEGORY_COLORS.items():
        if cat in set(asset_categories.values()):
            legend_handles.append(mpatches.Patch(color=color, label=cat))
            
    leg = plt.legend(
        handles=legend_handles, 
        title="ASSET CLASSES", 
        loc='upper left', 
        facecolor=BACKGROUND_COLOR, 
        edgecolor='white',
        labelcolor='white',
        fontsize=10
    )
    leg.get_title().set_color('white')
    leg.get_title().set_fontweight('bold')

    # 9. Title & Info
    plt.title("FINLAGX SYSTEMIC RISK MAP", fontsize=28, color='white', fontweight='bold', pad=30)
    plt.figtext(
        0.5, 0.05, 
        "NOTE: Edge brightness indicates predictive strength. Larger nodes represent dominant market leaders.", 
        ha="center", fontsize=12, style='italic', color='#888888'
    )

    # Remove axes
    ax.axis('off')

    # Save
    save_path = "finlagx_dark_network.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    logger.info(f"  Saved Dark Mode Graph to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_dark_network()