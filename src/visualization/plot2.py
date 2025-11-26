import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import logging
from src.feature_store import FeatureStore
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Color Configuration
CATEGORY_COLORS = {
    'EQUITIES': '#1f77b4', 'COMMODITIES': '#ff7f0e',
    'FX': '#2ca02c', 'VOL_BONDS': '#d62728',
    'CRYPTO': '#9467bd', 'MACRO': '#8c564b'
}

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

def plot_clean_network():
    logger.info("🎨 Generating Clean Network Graph...")
    fs = FeatureStore()
    
    # Fetch data
    try:
        df = fs.get_latest_granger_network()
    except Exception as e:
        logger.error(f"Could not fetch data: {e}")
        return

    if df.empty:
        logger.warning("⚠️ No data found in Granger results. Run the analysis first.")
        return
    
    # FILTER: Top 50 strongest links only
    df = df.sort_values('granger_score', ascending=False).head(50)

    G = nx.DiGraph()
    asset_categories = load_asset_categories()
    for _, row in df.iterrows():
        G.add_edge(row['asset_x'], row['asset_y'], weight=row['granger_score'])

    # --- FIX STARTS HERE ---
    # Create Figure and explicitly capture the Axes (ax)
    fig, ax = plt.subplots(figsize=(18, 14), facecolor='white')
    
    # Layout Physics
    pos = nx.spring_layout(G, k=3.5, iterations=100, seed=42)

    # Draw Nodes
    degrees = dict(G.degree)
    node_sizes = [2000 + (degrees.get(n, 1) * 300) for n in G.nodes()]
    node_colors = [CATEGORY_COLORS.get(asset_categories.get(n, 'OTHER'), '#7f7f7f') for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                           edgecolors='black', linewidths=1.5, alpha=0.9, ax=ax)

    # Draw Labels with outlining
    text = nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    for t in text.values():
        t.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

    # Draw Edges
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Handle case with no edges
    if not weights: 
        logger.warning("No edges to plot.")
        return

    nx.draw_networkx_edges(G, pos, width=2, edge_color=weights, edge_cmap=plt.cm.Reds, 
                           arrowstyle='-|>', arrowsize=25, 
                           connectionstyle="arc3,rad=0.1", ax=ax)

    # Add Colorbar (Explicitly pointing to 'ax')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])
    
    # This is the line that fixed your error:
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Causality Strength (Granger Score)", fontsize=12)

    plt.title("FinLagX: Lead-Lag Influence Map", fontsize=24)
    plt.axis('off')
    
    output_file = "clean_force_network.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Saved graph to {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_clean_network()