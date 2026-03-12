import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROCESSED_DATA_PATH = "data/processed/market/aligned_market_data.parquet"
OUTPUT_PATH = "data/results/asset_clusters.csv"
PLOTS_PATH = "data/results/cluster_plot.png"

def run_asset_clustering(n_clusters=4):
    logger.info("🎬 Starting Asset Clustering...")
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error("Dataset not found!")
        return
        
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    
    # Pivot to get returns for each symbol
    pivot_df = df.pivot(index='time', columns='symbol', values='returns')
    pivot_df = pivot_df.dropna() # Cluster on common date range
    
    # Calculate features for clustering (Mean Return and Volatility)
    cluster_data = pd.DataFrame({
        'mean_return': pivot_df.mean(),
        'volatility': pivot_df.std()
    })
    
    # Standardize
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data)
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_data['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cluster_data.to_csv(OUTPUT_PATH)
    logger.info(f"✅ Asset clusters saved to {OUTPUT_PATH}")
    
    # Summary of clusters
    for i in range(n_clusters):
        symbols = cluster_data[cluster_data['cluster'] == i].index.tolist()
        logger.info(f"   Cluster {i}: {symbols}")
        
    # Optional: Visualization (scatter plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cluster_data, x='mean_return', y='volatility', hue='cluster', palette='viridis', s=100)
    for i, symbol in enumerate(cluster_data.index):
        plt.annotate(symbol, (cluster_data.mean_return[i], cluster_data.volatility[i]), fontsize=8)
    plt.title("Asset Clustering by Risk-Return Profile")
    plt.savefig(PLOTS_PATH)
    logger.info(f"✅ Cluster plot saved to {PLOTS_PATH}")

if __name__ == "__main__":
    run_asset_clustering()
