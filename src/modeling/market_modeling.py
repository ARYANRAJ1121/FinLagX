# src/modeling/market_modeling.py

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
PROCESSED_DATA_PATH = "data/processed/market/aligned_market_data.parquet"
OUTPUT_GRAPH_PATH = "data/processed/graphs/lead_lag_graph.png"
GRANGER_MAX_LAG = 4  # Number of days to look back for causality
SIGNIFICANCE_LEVEL = 0.05

# --- Core Functions ---

def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    """Loads the preprocessed and aligned market data."""
    print(f"🔄 Loading preprocessed data from '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Processed data file not found at '{file_path}'.")
        print("Please run the preprocessing script first.")
        return pd.DataFrame()
    
    df = pd.read_parquet(file_path)
    # Ensure data is stationary (using percentage change) for robust modeling
    df = df.pct_change().dropna()
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    return df

def run_granger_causality_analysis(data: pd.DataFrame, variables: list) -> list:
    """
    Performs pairwise Granger causality tests and returns significant relationships.
    """
    print(f"\n🔍 Running Granger causality tests for max lag of {GRANGER_MAX_LAG} days...")
    causal_pairs = []
    
    for col1 in variables:
        for col2 in variables:
            if col1 == col2:
                continue
            
            test_result = grangercausalitytests(data[[col2, col1]], maxlag=GRANGER_MAX_LAG, verbose=False)
            
            # Find the minimum p-value across all lags
            min_p_value = np.min([test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, GRANGER_MAX_LAG + 1)])
            
            if min_p_value < SIGNIFICANCE_LEVEL:
                print(f"  -> Found significant relationship: {col1} Granger-causes {col2} (p={min_p_value:.4f})")
                causal_pairs.append((col1, col2))
                
    print(f"✅ Found {len(causal_pairs)} significant causal relationships.")
    return causal_pairs

def run_var_model_analysis(data: pd.DataFrame):
    """
    Fits a Vector Autoregression (VAR) model and prints the summary.
    """
    print("\n📈 Fitting Vector Autoregression (VAR) model...")
    try:
        model = VAR(data)
        results = model.fit(GRANGER_MAX_LAG)
        print("✅ VAR model fitted successfully. Summary for the first variable:")
        # Display summary for one variable for brevity
        print(results.summary().tables[0])
    except Exception as e:
        print(f"❌ VAR model fitting failed: {e}")

def visualize_lead_lag_graph(causal_pairs: list, output_path: str):
    """
    Creates and saves a directed graph of the lead-lag relationships.
    """
    print(f"\n🌐 Generating lead-lag dependency graph...")
    G = nx.DiGraph()
    G.add_edges_from(causal_pairs)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="#4C8BF5", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_color="white")
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color="#7D8A9C",
                           arrowstyle="->", arrowsize=20, width=2)
    
    ax.set_title("Financial Asset Lead-Lag Dependency Graph", fontsize=20, fontweight="bold")
    plt.margins(0.1)
    plt.box(False)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Graph saved successfully to '{output_path}'")

# --- Main Execution ---

def main():
    """
    Main function to run the complete modeling and visualization pipeline.
    """
    print("🚀 Starting FinLagX Modeling Pipeline...")
    
    # 1. Load Data
    # Assuming preprocessed data contains returns of key assets
    # Create a dummy file for demonstration if it doesn't exist
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("--- Creating dummy preprocessed data for demonstration ---")
        dates = pd.to_datetime(pd.date_range("2023-01-01", periods=200))
        dummy_data = {
            'S&P 500': np.random.randn(200).cumsum(),
            'VIX': np.random.randn(200).cumsum(),
            'Gold': np.random.randn(200).cumsum(),
            'Oil': np.random.randn(200).cumsum(),
            'Sentiment': np.random.randn(200).cumsum()
        }
        dummy_df = pd.DataFrame(dummy_data, index=dates)
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        dummy_df.to_parquet(PROCESSED_DATA_PATH)
        print("--- Dummy data created. ---")

    market_data = load_preprocessed_data(PROCESSED_DATA_PATH)
    
    if market_data.empty:
        return
        
    # 2. Run Statistical Models
    assets_to_model = market_data.columns.tolist()
    significant_relations = run_granger_causality_analysis(market_data, assets_to_model)
    run_var_model_analysis(market_data[assets_to_model])

    # 3. Visualize Results
    if significant_relations:
        visualize_lead_lag_graph(significant_relations, OUTPUT_GRAPH_PATH)
    else:
        print("\n⚠️ No significant relationships found to visualize.")
        
    print("\n🎉 FinLagX Modeling Pipeline completed successfully!")


if __name__ == "__main__":
    main()