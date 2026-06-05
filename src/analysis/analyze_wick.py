import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

def process_stock_wick(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    try:
        df = pd.read_csv(file_path, usecols=['date', 'open', 'high', 'low', 'close'])
        
        # 1. Rolling Stats
        window = 20 # As per user request/code
        
        # Log Returns
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Z-Ret (Momentum)
        roll_ret = df['return'].rolling(window)
        df['z_ret'] = (df['return'] - roll_ret.mean()) / roll_ret.std()
        
        # 2. CLV (Close Location Value)
        # Handling High=Low case
        df['range'] = df['high'] - df['low']
        df['clv'] = (df['close'] - df['low']) / df['range']
        
        # Fix NaNs (where range is 0) -> 0.5 (Neutral)
        df['clv'] = df['clv'].fillna(0.5)
        
        # 3. Z-Wick
        roll_clv = df['clv'].rolling(window)
        df['z_wick'] = (df['clv'] - roll_clv.mean()) / roll_clv.std()
        
        # 4. Response (Next 15 mins / 3 bars)
        df['response'] = np.log(df['close'].shift(-3) / df['close'])
        
        # Filter for Valid Data
        # We need Z-scores, so dropna
        valid = df.dropna(subset=['z_ret', 'z_wick', 'response'])
        
        # Return only essential columns to save memory
        return valid[['z_ret', 'z_wick', 'response']]
        
    except Exception as e:
        return None

def process_wick_analysis():
    print("Starting Wick Physics Analysis (Minnows)...")
    
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "minnows.csv"))
    symbols = minnows['symbol'].tolist()
    
    all_events = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_stock_wick, sym): sym for sym in symbols}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{len(symbols)}...")
            
            if res is not None:
                all_events.append(res)
                
    if not all_events:
        print("No data.")
        return
        
    print("Concatenating events...")
    merged = pd.concat(all_events)
    
    # Analysis 1: Win Rate Comparison
    print("\n--- QUALITY CONTROL TEST ---")
    
    # Filter for Significant Bullish Pushes (> 1.5 sigma)
    bullish = merged[merged['z_ret'] > 1.5]
    
    # Split
    hq = bullish[bullish['z_wick'] > 0.5]
    lq = bullish[bullish['z_wick'] < -0.5]
    
    wr_hq = (hq['response'] > 0).mean()
    wr_lq = (lq['response'] > 0).mean()
    
    print(f"High Quality (Strong Close) Win Rate: {wr_hq:.2%}")
    print(f"Low Quality (Weak Close) Win Rate:    {wr_lq:.2%}")
    
    # Analysis 2: Trap Zone data
    # Binning for Heatmap
    # X: z_ret, Y: z_wick
    merged['ret_bin'] = pd.cut(merged['z_ret'], bins=np.linspace(-4, 4, 40), labels=False)
    merged['wick_bin'] = pd.cut(merged['z_wick'], bins=np.linspace(-4, 4, 40), labels=False)
    
    # Group and Mean Response
    heatmap = merged.groupby(['ret_bin', 'wick_bin'])['response'].mean().reset_index()
    
    # Add bin centers for plotting
    bins_z = np.linspace(-4, 4, 40)
    centers = (bins_z[:-1] + bins_z[1:]) / 2
    
    heatmap['z_ret_val'] = heatmap['ret_bin'].apply(lambda x: centers[int(x)] if not pd.isna(x) else np.nan)
    heatmap['z_wick_val'] = heatmap['wick_bin'].apply(lambda x: centers[int(x)] if not pd.isna(x) else np.nan)
    
    out_file = os.path.join(RESULTS_DIR, "wick_heatmap_data.csv")
    heatmap.dropna().to_csv(out_file, index=False)
    print(f"Saved Heatmap Data to {out_file}")

if __name__ == "__main__":
    process_wick_analysis()
