import pandas as pd
import numpy as np
import glob
import os
import pickle

# --- CONFIGURATION ---
DATA_PATH = "./raw_data"  # Where your files are
OUTPUT_PATH = "./processed_data"
start_date = "2015-01-01"
end_date = "2025-01-01"

# Create output folder
os.makedirs(OUTPUT_PATH, exist_ok=True)

def preprocess_nifty_data():
    print("🚀 Starting Data Processing...")
    
    # 1. Get list of all stock files
    files = glob.glob(f"{DATA_PATH}/*.csv")
    print(f"Found {len(files)} files.")

    # 2. Create the MASTER Timeline (09:15 to 15:30, 5-min intervals)
    # We filter out weekends and non-trading hours later, 
    # but first let's define the ideal grid.
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    times = pd.date_range("09:15", "15:30", freq="5min").time
    
    # This creates a perfect grid of (Days x Times)
    full_index = []
    for d in dates:
        # Skip weekends
        if d.weekday() < 5: 
            for t in times:
                full_index.append(pd.Timestamp.combine(d, t))
    
    master_index = pd.DatetimeIndex(full_index)
    print(f"Master Timeline Created: {len(master_index)} timesteps.")

    # 3. Load and Align Each Stock
    processed_dfs = []
    valid_stocks = []

    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        try:
            # Load CSV (Reading specific columns from your Screenshot 2)
            df = pd.read_csv(file)
            
            # Parse Date (Handle the +05:30 timezone)
            df['date'] = pd.to_datetime(df['date'])
            
            # Remove Timezone info to match Master Index (or convert master to UTC)
            # Easiest is to make naive if they are all IST
            df['date'] = df['date'].dt.tz_localize(None)
            
            df = df.set_index('date')
            
            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Reindex to Master Timeline (Crucial Step!)
            # method='ffill' propagates the last valid price forward (if data is missing)
            df = df.reindex(master_index, method='ffill')
            
            # Fill remaining NaNs (e.g., IPOs that didn't exist in 2015) with 0
            df = df.fillna(0)
            
            processed_dfs.append(df)
            valid_stocks.append(stock_name)
            
        except Exception as e:
            print(f"⚠️ Error processing {stock_name}: {e}")

    # 4. Create the 3D Tensor (Time x Stocks x Features)
    # Stack along a new axis
    print("📦 Stacking data into 3D Tensor...")
    # Result shape: (Time, Num_Stocks, 5_Features)
    data_tensor = np.stack([df.values for df in processed_dfs], axis=1)
    
    print(f"Final Tensor Shape: {data_tensor.shape}")
    # Example: (375000 timesteps, 500 stocks, 5 features)

    # 5. Generate Adjacency Matrix (The "Graph") for BasicTS
    # We calculate correlation based on 'Close' price returns
    print("🕸️ Generating Graph (Adjacency Matrix)...")
    closes = pd.concat([df['close'] for df in processed_dfs], axis=1)
    closes.columns = valid_stocks
    
    # Calculate returns for correlation
    returns = closes.pct_change().fillna(0)
    corr_matrix = returns.corr().values
    
    # Thresholding: Only keep strong connections (e.g., > 0.6 correlation)
    # This makes the graph sparse and faster
    adj_mx = np.where(corr_matrix > 0.6, 1, 0)

    # 6. Save Files
    with open(f"{OUTPUT_PATH}/data.pkl", "wb") as f:
        pickle.dump(data_tensor, f)
        
    with open(f"{OUTPUT_PATH}/adj_mx.pkl", "wb") as f:
        pickle.dump(adj_mx, f)
        
    # Save stock names so you know which index is which stock
    with open(f"{OUTPUT_PATH}/stock_names.txt", "w") as f:
        for name in valid_stocks:
            f.write(f"{name}\n")

    print("✅ Done! Data ready for BasicTS and Liquid Networks.")

if __name__ == "__main__":
    preprocess_nifty_data()