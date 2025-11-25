import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from src.feature_store import FeatureStore
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your config for colors
CATEGORY_COLORS = {
    'EQUITIES': '#1f77b4',    'COMMODITIES': '#ff7f0e',
    'FX': '#2ca02c',          'VOL_BONDS': '#d62728',
    'CRYPTO': '#9467bd',      'MACRO': '#8c564b'
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

def plot_circular_network():
    fs = FeatureStore()
    df = fs.get_latest_granger_network()
    
    # --- CRITICAL FIX: FILTER DATA ---
    # Only keep the top 30-40 strongest relationships. 
    # Showing everything creates the "hairball."
    df = df.sort_values('granger_score', ascending=False).head(40)
    
    G = nx.DiGraph()
    asset_categories = load_asset_categories()
    
    for _, row in df.iterrows():
        G.add_edge(row['asset_x'], row['asset_y'], weight=row['granger_score'])

    plt.figure(figsize=(15, 15), facecolor='white')
    ax = plt.gca()

    # 1. Layout: Circular
    pos = nx.circular_layout(G)
    
    # 2. Nodes
    node_colors = [CATEGORY_COLORS.get(asset_categories.get(n, 'OTHER'), '#7f7f7f') for n in G.nodes()]
    
    # Draw nodes larger
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=1.0, edgecolors='white', linewidths=2)

    # 3. Edges (The Arcs)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Draw edges with curvature (connectionstyle is key here)
    nx.draw_networkx_edges(G, pos, 
                           width=2, 
                           edge_color=weights, 
                           edge_cmap=plt.cm.Reds, 
                           edge_vmin=min(weights), 
                           edge_vmax=max(weights),
                           connectionstyle="arc3,rad=0.2", # CURVES THE LINES
                           arrowsize=20)

    # 4. Labels (Pushed slightly outward if possible, but centered is fine for circular)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="white")

    # 5. Legend
    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in CATEGORY_COLORS.items()]
    plt.legend(handles=legend_handles, loc='upper right', title="Asset Class")

    plt.title("FinLagX: Top 40 Market Signals (Circular View)", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("circular_network.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_circular_network()