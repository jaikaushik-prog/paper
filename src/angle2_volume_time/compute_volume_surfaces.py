import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from push_response_lib import prepare_push_response_data
from volume_time_lib import create_adaptive_volume_bars

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

# Lags to compute (Same as Clock Time for comparison)
LAGS_SHORT = list(range(1, 21))
LAGS_MEDIUM = list(range(25, 80, 5))
ALL_LAGS = sorted(list(set(LAGS_SHORT + LAGS_MEDIUM)))

def process_stock_volume_worker(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    results = {}
    
    try:
        # Load raw 5-min data
        df = pd.read_csv(file_path, usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        if df.empty:
            return None
            
        # --- KEY STEP: Convert to Volume Time ---
        # Target: 50 bars per day
        vol_df = create_adaptive_volume_bars(df, bars_per_day=50)
        
        if vol_df.empty or len(vol_df) < 2000: # Need enough data for rolling Z-score
            return None
            
        # Ensure 'date' column exists for preparer (it uses index usually, but let's check lib)
        # prepare_push_response_data expects a DF. It computes returns on 'close'.
        # It handles 'date' column or index? creating a 'date' column from index is safer if lib expects it.
        vol_df['date'] = vol_df.index
        
        for lag in ALL_LAGS:
            # Prepare vectors on VOLUME BARS
            res = prepare_push_response_data(vol_df, lag)
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
    print(f"Processing VOLUME TIME for group: {group_name} with {len(stock_list)} stocks...")
    
    accumulator = {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS}
    
    max_workers = 8
    symbols = stock_list['symbol'].tolist()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_stock_volume_worker, sym): sym for sym in symbols}
        
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
            
    # Aggregate
    surface_data = []
    print(f"Aggregating data for {group_name}...")
    
    # Same binning logic as Clock Time
    bins = np.arange(-4.0, 4.05, 0.1) 
    centers = (bins[:-1] + bins[1:]) / 2
    
    for lag in ALL_LAGS:
        pushes = accumulator[lag]['z_push']
        responses = accumulator[lag]['z_resp']
        
        if not pushes:
            continue
            
        cat_push = np.concatenate(pushes)
        cat_resp = np.concatenate(responses)
        
        df_lag = pd.DataFrame({'z_push': cat_push, 'z_resp': cat_resp})
        
        # Binning
        df_lag['bin'] = pd.cut(df_lag['z_push'], bins=bins, labels=centers)
        avg_resp_per_bin = df_lag.groupby('bin', observed=True)['z_resp'].mean()
        
        for center, value in avg_resp_per_bin.items():
            if pd.isna(value):
                continue
            surface_data.append({
                'lag': int(lag),
                'push_bin': float(center),
                'avg_response': float(value)
            })
            
    # Save
    out_file = os.path.join(RESULTS_DIR, f"surface_{group_name}_voltime.csv")
    pd.DataFrame(surface_data).to_csv(out_file, index=False)
    print(f"Saved VOLUME TIME surface data to {out_file}")

if __name__ == "__main__":
    # Load Groups
    whales_path = os.path.join(PROCESSED_DIR, "whales.csv")
    minnows_path = os.path.join(PROCESSED_DIR, "minnows.csv")
    
    if os.path.exists(whales_path):
        whales = pd.read_csv(whales_path)
        process_group("whales", whales)
        
    if os.path.exists(minnows_path):
        minnows = pd.read_csv(minnows_path)
        process_group("minnows", minnows)
