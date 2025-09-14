import os
import pandas as pd
import matplotlib.pyplot as plt

# Base directories
RAW_DATA_DIR = "data/raw/market"
PROCESSED_DATA_DIR = "data/processed/market"

def process_and_analyze_market_data():
    """
    Processes raw market data, performs EDA, and saves the cleaned data.
    """
    print("\n✨ Starting market data preprocessing & EDA...")
    if not os.path.exists(RAW_DATA_DIR):
        print(f"⚠️ Raw data directory not found at {RAW_DATA_DIR}.")
        return

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                raw_file_path = os.path.join(root, file)

                try:
                    # Correctly read the CSV file, specifying the 'Date' column as the index and parsing dates
                    df = pd.read_csv(raw_file_path, index_col='Date', parse_dates=True)

                    # --- EDA & Preprocessing Steps ---

                    print(f"\n📊 Analyzing {file}...")
                    
                    # Handle missing values using forward-fill
                    df.ffill(inplace=True)
                    print("\n✅ Missing values handled.")

                    # Feature Engineering: Calculate a 50-day Simple Moving Average (SMA)
                    if 'Close' in df.columns:
                        df['SMA_50'] = df['Close'].rolling(window=50).mean()
                        print("📈 SMA_50 calculated.")
                    else:
                        print(f"⚠️ 'Close' column not found in {file}. Skipping SMA calculation.")
                    
                    # Basic EDA: Plot Close Price vs. SMA
                    plt.figure(figsize=(10, 6))
                    plt.plot(df.index, df['Close'], label='Close Price')
                    plt.plot(df.index, df['SMA_50'], label='SMA 50')
                    plt.title(f'{file.replace(".csv", "")} Price and 50-Day SMA')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(True)
                    plt.show()

                    # --- Save the processed data ---
                    relative_path = os.path.relpath(raw_file_path, RAW_DATA_DIR)
                    processed_file_path = os.path.join(PROCESSED_DATA_DIR, relative_path)
                    
                    processed_category_dir = os.path.dirname(processed_file_path)
                    os.makedirs(processed_category_dir, exist_ok=True)
                    
                    df.to_csv(processed_file_path, index=False)
                    print(f"\n✅ Processed data saved to {processed_file_path}")

                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")

    print("\n✅ Market data preprocessing & EDA finished.")

if __name__ == "__main__":
    process_and_analyze_market_data()