"""
Nifty 500 Angle 1 Analysis - Same Period as S&P 500
Period: Feb 2015 to Feb 2018 (overlapping period)

This script runs the full Angle 1 pipeline for Nifty 500 on the same time period
as S&P 500 for fair comparison.
"""

import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
OUTPUT_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"

# Time period to match S&P 500
START_DATE = pd.Timestamp("2015-02-01", tz="UTC")
END_DATE = pd.Timestamp("2018-02-28", tz="UTC")

# Lags in 5-minute bars (to roughly match daily lags)
# 1 day = 75 bars (6h15m trading)
# Matching S&P 500 lags: 1, 2, 3, 5, 7, 10, 15, 20 days
LAGS = [75, 150, 225, 375, 525, 750, 1125, 1500]  # in 5-min bars
LAG_DAYS = [1, 2, 3, 5, 7, 10, 15, 20]  # for labels

def load_stock_data(file_path, start_date, end_date):
    """Load and filter stock data to the specified period."""
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        return None

def calculate_adt_worker(args):
    """Worker function for ADT calculation."""
    file_path, start_date, end_date = args
    symbol = os.path.basename(file_path).replace('.csv', '')
    
    df = load_stock_data(file_path, start_date, end_date)
    if df is None or len(df) < 1000:  # Need at least ~2 weeks of 5-min data
        return None
    
    df['turnover'] = df['close'] * df['volume']
    daily_turnover = df.groupby(df['date'].dt.date)['turnover'].sum()
    avg_daily_turnover = daily_turnover.mean()
    
    return {
        'symbol': symbol,
        'adt': avg_daily_turnover,
        'days_count': len(daily_turnover),
        'bars_count': len(df)
    }

def liquidity_stratification():
    """Segment stocks into Whales and Minnows."""
    print("="*60)
    print("NIFTY 500 LIQUIDITY STRATIFICATION")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print("="*60)
    
    stock_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(stock_files)} stock files")
    
    task_args = [(fp, START_DATE, END_DATE) for fp in stock_files]
    
    adt_stats = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(calculate_adt_worker, arg): arg for arg in task_args}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                adt_stats.append(res)
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count}/{len(stock_files)} files...")
    
    adt_df = pd.DataFrame(adt_stats)
    adt_df = adt_df.sort_values('adt', ascending=False).reset_index(drop=True)
    
    # Filter stocks with at least 100 trading days
    valid_stocks = adt_df[adt_df['days_count'] >= 100].copy()
    print(f"\nFiltered to {len(valid_stocks)} stocks with >= 100 trading days")
    
    whales = valid_stocks.head(50).copy()
    minnows = valid_stocks.tail(50).copy()
    
    print("\nTop 5 Whales:")
    print(whales[['symbol', 'adt']].head().to_string(index=False))
    print("\nTop 5 Minnows:")
    print(minnows[['symbol', 'adt']].tail().to_string(index=False))
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    whales.to_csv(os.path.join(OUTPUT_DIR, "nifty_sameperiod_whales.csv"), index=False)
    minnows.to_csv(os.path.join(OUTPUT_DIR, "nifty_sameperiod_minnows.csv"), index=False)
    
    return whales, minnows

def prepare_push_response_data(df, lag_l, window=1500):
    """Compute Push and Response Z-scores for 5-min data."""
    df = df.copy()
    
    df['push_raw'] = np.log(df['close'] / df['close'].shift(lag_l))
    df['resp_raw'] = np.log(df['close'].shift(-lag_l) / df['close'])
    df = df.dropna()
    
    if df.empty or len(df) < window:
        return pd.DataFrame()
    
    push_mean = df['push_raw'].rolling(window, min_periods=window//2).mean()
    push_std = df['push_raw'].rolling(window, min_periods=window//2).std()
    resp_mean = df['resp_raw'].rolling(window, min_periods=window//2).mean()
    resp_std = df['resp_raw'].rolling(window, min_periods=window//2).std()
    
    push_std = push_std.replace(0, np.nan)
    resp_std = resp_std.replace(0, np.nan)
    
    df['z_push'] = (df['push_raw'] - push_mean) / push_std
    df['z_resp'] = (df['resp_raw'] - resp_mean) / resp_std
    
    return df[['z_push', 'z_resp']].dropna()

def process_stock_surfaces(args):
    """Process one stock for all lags."""
    symbol, start_date, end_date = args
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    
    df = load_stock_data(file_path, start_date, end_date)
    if df is None or len(df) < 3000:
        return None
    
    results = {}
    for lag in LAGS:
        z_data = prepare_push_response_data(df, lag)
        if not z_data.empty:
            results[lag] = z_data
    
    return results if results else None

def compute_surfaces(group_name, stock_list):
    """Compute efficiency surfaces for a group."""
    print(f"\nProcessing group: {group_name} ({len(stock_list)} stocks)...")
    
    accumulator = {lag: [] for lag in LAGS}
    
    task_args = [(sym, START_DATE, END_DATE) for sym in stock_list['symbol'].tolist()]
    
    processed = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_stock_surfaces, arg): arg for arg in task_args}
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                for lag, z_data in res.items():
                    accumulator[lag].append(z_data)
                processed += 1
    
    print(f"  Successfully processed {processed} stocks")
    
    # Aggregate
    surface_data = []
    for i, lag in enumerate(LAGS):
        if not accumulator[lag]:
            continue
        
        combined = pd.concat(accumulator[lag], ignore_index=True)
        print(f"  Lag {LAG_DAYS[i]} days ({lag} bars): {len(combined):,} data points")
        
        # Bin and compute mean response
        bins = np.arange(-4.0, 4.25, 0.25)
        centers = (bins[:-1] + bins[1:]) / 2
        combined['bin'] = pd.cut(combined['z_push'], bins=bins, labels=centers)
        surface = combined.groupby('bin', observed=True)['z_resp'].mean()
        
        for center, value in surface.items():
            if pd.notna(value):
                surface_data.append({
                    'lag': LAG_DAYS[i],
                    'push_bin': float(center),
                    'avg_response': float(value)
                })
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_df = pd.DataFrame(surface_data)
    out_file = os.path.join(RESULTS_DIR, f"nifty_sameperiod_surface_{group_name}.csv")
    result_df.to_csv(out_file, index=False)
    print(f"  Saved: {out_file}")
    
    return result_df

def analyze_results():
    """Compare Nifty 500 results with S&P 500."""
    print("\n" + "="*60)
    print("NIFTY 500 vs S&P 500 COMPARISON (Same Period)")
    print("="*60)
    
    # Load surfaces
    nifty_whales = pd.read_csv(os.path.join(RESULTS_DIR, "nifty_sameperiod_surface_whales.csv"))
    nifty_minnows = pd.read_csv(os.path.join(RESULTS_DIR, "nifty_sameperiod_surface_minnows.csv"))
    sp500_whales = pd.read_csv(os.path.join(RESULTS_DIR, "sp500_surface_whales.csv"))
    sp500_minnows = pd.read_csv(os.path.join(RESULTS_DIR, "sp500_surface_minnows.csv"))
    
    # Calculate metrics
    def calc_metrics(df):
        return {
            'avg_abs': df['avg_response'].abs().mean(),
            'crash_resp': df[df['push_bin'] < -2]['avg_response'].mean()
        }
    
    nw = calc_metrics(nifty_whales)
    nm = calc_metrics(nifty_minnows)
    sw = calc_metrics(sp500_whales)
    sm = calc_metrics(sp500_minnows)
    
    print(f"\n{'Metric':<25} {'Nifty Whales':<15} {'Nifty Minnows':<15} {'S&P Whales':<15} {'S&P Minnows':<15}")
    print("-"*85)
    print(f"{'Avg |Response|':<25} {nw['avg_abs']:<15.4f} {nm['avg_abs']:<15.4f} {sw['avg_abs']:<15.4f} {sm['avg_abs']:<15.4f}")
    print(f"{'Crash Response (Z<-2)':<25} {nw['crash_resp']:<15.4f} {nm['crash_resp']:<15.4f} {sw['crash_resp']:<15.4f} {sm['crash_resp']:<15.4f}")
    
    nifty_ratio = nm['avg_abs'] / nw['avg_abs'] if nw['avg_abs'] > 0 else 0
    sp500_ratio = sm['avg_abs'] / sw['avg_abs'] if sw['avg_abs'] > 0 else 0
    
    print(f"\n{'Inefficiency Ratio':<25} {nifty_ratio:<15.2f}x {'':<15} {sp500_ratio:<15.2f}x")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if nifty_ratio > sp500_ratio * 1.5:
        print("✓ Nifty 500 shows STRONGER liquidity-driven inefficiency")
        print(f"  Nifty Ratio: {nifty_ratio:.2f}x vs S&P Ratio: {sp500_ratio:.2f}x")
    else:
        print("≈ Both markets show similar efficiency levels")

def main():
    # Step 1: Liquidity Stratification
    whales, minnows = liquidity_stratification()
    
    # Step 2: Compute Surfaces
    compute_surfaces("whales", whales)
    compute_surfaces("minnows", minnows)
    
    # Step 3: Analyze and Compare
    analyze_results()
    
    print("\n" + "="*60)
    print("NIFTY 500 ANGLE 1 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
