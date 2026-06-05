import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
LAGS = list(range(1, 21)) + list(range(25, 80, 5))

def load_and_resample(args):
    symbol, data_dir = args
    file_path = os.path.join(data_dir, f"{symbol}.csv")
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        # Ensure unique index
        df = df[~df.index.duplicated(keep='first')]
        return df['close'].rename(symbol)
    except:
        return None

def create_synthetic_index(stock_list):
    """
    Creates an equal-weighted index from a list of symbols.
    Returns: Series of 5-min log returns.
    """
    print(f"Constructing Index from {len(stock_list)} stocks...")
    
    symbols = stock_list['symbol'].tolist()
    
    # Parallel load
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(load_and_resample, (sym, DATA_DIR)): sym for sym in symbols}
        
        series_list = []
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                series_list.append(res)
                
    if not series_list:
        return None
        
    # Combine into DataFrame (Outer Join)
    price_df = pd.concat(series_list, axis=1)
    
    # Forward Fill missing data (limit 1-2 bars?) OR just use available
    price_df = price_df.ffill(limit=3)
    
    # Calculate Log Returns for EACH stock
    log_rets = np.log(price_df / price_df.shift(1))
    
    # Average Return (Equal Weighted Index Return)
    index_ret = log_rets.mean(axis=1)
    
    return index_ret

def compute_rolling_z(series, window=1500):
    roll = series.rolling(window=window)
    mean = roll.mean()
    std = roll.std()
    z = (series - mean) / std
    return z

def compute_cross_surface(push_series, resp_series, name="whale_to_minnow"):
    print(f"Computing Cross-Surface: {name}...")
    
    # Align indices
    common_idx = push_series.index.intersection(resp_series.index)
    p = push_series.loc[common_idx]
    r = resp_series.loc[common_idx]
    
    # Compute Rolling Z
    z_p = compute_rolling_z(p)
    z_r = compute_rolling_z(r)
    
    # Drop NaNs
    valid = z_p.notna() & z_r.notna()
    z_p = z_p[valid]
    z_r = z_r[valid]
    
    # We need to shift Response by Lag L relative to Push
    # Push at t, Resp at t+L
    
    # Re-align for lags
    # It's faster to pull values as arrays
    # But we need to handle time gaps?
    # Index is timestamp.
    # Assuming continuous 5-min bars for "Index". Or at least aligned.
    
    # Robust Lagging: shift the RESPONSE back by L to align t+L with t
    
    surface_data = []
    bins = np.arange(-4.0, 4.05, 0.1) 
    centers = (bins[:-1] + bins[1:]) / 2
    
    for lag in LAGS:
        # Shift R backwards by lag => R[t] becomes R[t+L] effectively for the row
        # shift(-L) means the value at t is now the value from t+L
        z_r_lagged = z_r.shift(-lag)
        
        # Align
        mask = z_p.notna() & z_r_lagged.notna()
        cur_p = z_p[mask]
        cur_r = z_r_lagged[mask]
        
        if cur_p.empty: continue
        
        # Binning
        df_lag = pd.DataFrame({'p': cur_p, 'r': cur_r})
        df_lag['bin'] = pd.cut(df_lag['p'], bins=bins, labels=centers)
        
        grouped = df_lag.groupby('bin', observed=True)['r']
        mean = grouped.mean()
        
        for center in mean.index:
            m = mean[center]
            if pd.isna(m): continue
            surface_data.append({
                'lag': lag,
                'push_bin': float(center),
                'avg_response': float(m)
            })
            
    # Save
    out_file = os.path.join(RESULTS_DIR, f"surface_leadlag_{name}.csv")
    pd.DataFrame(surface_data).to_csv(out_file, index=False)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    # Load Lists
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "minnows.csv"))
    
    # Create Indices
    whale_idx = create_synthetic_index(whales)
    minnow_idx = create_synthetic_index(minnows)
    
    if whale_idx is not None and minnow_idx is not None:
        # Save indices for sanity check
        # whale_idx.to_csv(os.path.join(RESULTS_DIR, "whale_index_ret.csv"))
        
        # Run Cross Analysis
        compute_cross_surface(whale_idx, minnow_idx, "whales_lead_minnows")
