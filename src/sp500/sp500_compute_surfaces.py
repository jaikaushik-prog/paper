"""
S&P 500 Compute Surfaces
Computes Push-Response efficiency surfaces for Whales and Minnows groups.
"""

import pandas as pd
import numpy as np
import os
from sp500_push_response_lib import prepare_push_response_data, get_surface_for_lag

# Configuration
DATA_FILE = r"c:\Users\DELL\Desktop\project_nifty_liquid\datasets\data\all_stocks_5yr.csv"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"

# Lags in trading days
# Short-term: 1-5 days, Medium-term: 10-20 days
LAGS = [1, 2, 3, 5, 7, 10, 15, 20]

def load_stock_data(symbol, all_data):
    """Extract a single stock's data from the combined dataframe."""
    stock_df = all_data[all_data['Name'] == symbol].copy()
    stock_df = stock_df.sort_values('date').reset_index(drop=True)
    return stock_df

def process_stock(symbol, all_data, lags):
    """
    Process a single stock: compute z-scores for all lags.
    """
    stock_df = load_stock_data(symbol, all_data)
    
    if len(stock_df) < 100:  # Need enough data
        return None
    
    results = {}
    for lag in lags:
        z_data = prepare_push_response_data(stock_df, lag)
        if not z_data.empty:
            results[lag] = z_data
    
    return results if results else None

def process_group(group_name, stock_list, all_data):
    """
    Process all stocks in a group and compute aggregated surfaces.
    """
    print(f"\nProcessing group: {group_name} ({len(stock_list)} stocks)...")
    
    # Accumulator for each lag
    accumulator = {lag: [] for lag in LAGS}
    
    symbols = stock_list['symbol'].tolist()
    processed = 0
    
    for i, symbol in enumerate(symbols):
        results = process_stock(symbol, all_data, LAGS)
        
        if results:
            for lag, z_data in results.items():
                accumulator[lag].append(z_data)
            processed += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(symbols)} stocks...")
    
    print(f"  Successfully processed {processed} stocks")
    
    # Aggregate and compute surfaces
    surface_data = []
    
    for lag in LAGS:
        if not accumulator[lag]:
            continue
        
        # Concatenate all stock data for this lag
        combined = pd.concat(accumulator[lag], ignore_index=True)
        print(f"  Lag {lag}: {len(combined):,} data points")
        
        # Get surface
        surface = get_surface_for_lag(combined)
        
        if surface is None:
            continue
        
        # Store results
        for center, value in surface.items():
            if pd.notna(value):
                surface_data.append({
                    'lag': int(lag),
                    'push_bin': float(center),
                    'avg_response': float(value)
                })
    
    # Save to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_df = pd.DataFrame(surface_data)
    out_file = os.path.join(RESULTS_DIR, f"sp500_surface_{group_name}.csv")
    result_df.to_csv(out_file, index=False)
    print(f"  Saved surface to: {out_file}")
    
    return result_df

def main():
    # Load all S&P 500 data
    print("Loading S&P 500 data...")
    all_data = pd.read_csv(DATA_FILE)
    all_data['date'] = pd.to_datetime(all_data['date'])
    print(f"Loaded {len(all_data):,} rows")
    
    # Load groups
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_minnows.csv"))
    
    print(f"Whales: {len(whales)} stocks")
    print(f"Minnows: {len(minnows)} stocks")
    
    # Process each group
    whales_surface = process_group("whales", whales, all_data)
    minnows_surface = process_group("minnows", minnows, all_data)
    
    print("\n" + "="*60)
    print("SURFACE COMPUTATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
