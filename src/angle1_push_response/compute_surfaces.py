import pandas as pd
import numpy as np
import os
import glob
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from push_response_lib import prepare_push_response_data, get_surface_for_lag

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"

# Lags from plan:
# Short: 1 to 12 (5 mins to 60 mins)
# Medium: 24 to 75 (2 hours to ~6 hours - 1 trading day is 375 mins = 75 bars)
LAGS_SHORT = list(range(1, 13))
LAGS_MEDIUM = list(range(24, 76, 1)) # Step 1 or step 3? Plan said 24, 48... let's do continuous or steps?
# Plan said "24, 48, ... 75". Let's do a finer grid for better surface.
# 24 (2h), 36 (3h), 48 (4h), 60 (5h), 72 (6h), 75 (Full Day - 6h15m)
LAGS_MEDIUM = [24, 30, 36, 42, 48, 54, 60, 66, 72, 75]

ALL_LAGS = sorted(list(set(LAGS_SHORT + LAGS_MEDIUM)))


# Worker function for parallel processing
def process_stock_worker(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    results = {}
    
    try:
        # Load stock data
        df = pd.read_csv(file_path, usecols=['date', 'close'])
        # Sort if needed, usually sorted
        
        if df.empty:
            return None
            
        for lag in ALL_LAGS:
            # Prepare vectors
            res = prepare_push_response_data(df, lag)
            if not res.empty:
                results[lag] = {
                    'z_push': res['z_push'].values,
                    'z_resp': res['z_resp'].values
                }
        return results
        
    except Exception as e:
        # print(f"Error processing {symbol}: {e}")
        return None

def process_group(group_name, stock_list):
    print(f"Processing group: {group_name} with {len(stock_list)} stocks...")
    
    # Dictionary to accumulate pairs for each lag
    accumulator = {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS}
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Run in parallel
    max_workers = 8
    symbols = stock_list['symbol'].tolist()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_stock_worker, sym): sym for sym in symbols}
        
        count = 0 
        total = len(symbols)
        
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 10 == 0:
                print(f"[{group_name}] Processed {count}/{total} stocks...")
                
            if res:
                for lag, data in res.items():
                    accumulator[lag]['z_push'].append(data['z_push'])
                    accumulator[lag]['z_resp'].append(data['z_resp'])
            
    # Now Aggregate and Compute Surface for each Lag
    surface_data = []
    
    print(f"Aggregating data for {group_name}...")
    for lag in ALL_LAGS:
        pushes = accumulator[lag]['z_push']
        responses = accumulator[lag]['z_resp']
        
        if not pushes:
            continue
            
        # Concatenate all numpy arrays
        all_z_push = np.concatenate(pushes)
        all_z_resp = np.concatenate(responses)
        
        # Create a temp DF for binning
        panel = pd.DataFrame({'z_push': all_z_push, 'z_resp': all_z_resp})
        
        # Get surface (Mean Response per Bin)
        # Index is Bin (interval), Value is Mean Response
        avg_resp_per_bin = get_surface_for_lag(panel)
        
        if avg_resp_per_bin is None:
            continue
            
        # Store results
        # We need to serialize this.
        # Format: {'lag': L, 'bins': [centers], 'responses': [values]}
        # avg_resp_per_bin index is Intervals using categories.
        # We want the mid points of bins.
        
        # Extract bin centers
        # The index is the bin center (float) because we used labels=centers
        
        # Let's iterate index
        for center, value in avg_resp_per_bin.items():
            if pd.isna(value):
                continue
            # center is already the float mid point
            surface_data.append({
                'lag': int(lag),
                'push_bin': float(center),
                'avg_response': float(value)
            })
            
    # Save to CSV
    res_df = pd.DataFrame(surface_data)
    out_file = os.path.join(RESULTS_DIR, f"surface_{group_name}.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res_df.to_csv(out_file, index=False)
    print(f"Saved surface data to {out_file}")

if __name__ == "__main__":
    # Load Groups
    try:
        whales = pd.read_csv(os.path.join(PROCESSED_DIR, "whales.csv"))
        minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "minnows.csv"))
        
        process_group("whales", whales)
        process_group("minnows", minnows)
        
    except Exception as e:
        print(f"Failed to load groups or process: {e}")
