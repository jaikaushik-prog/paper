"""
Nifty 500 Angle 4: Z-Gap Analysis - Same Period as S&P 500
Period: Feb 2015 to Feb 2018

Analyzes overnight gaps and tests Fade vs Follow hypothesis for comparison with S&P 500.
"""

import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"

START_DATE = pd.Timestamp("2015-02-01", tz="UTC")
END_DATE = pd.Timestamp("2018-02-28", tz="UTC")

def load_stock_data(file_path, start_date, end_date):
    """Load and filter stock data."""
    try:
        df = pd.read_csv(file_path, usecols=['date', 'open', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except:
        return None

def process_stock_gaps(args):
    """Process one stock for gap analysis."""
    symbol, start_date, end_date = args
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    
    df = load_stock_data(file_path, start_date, end_date)
    if df is None or len(df) < 1000:
        return None
    
    # Get daily OHLC from 5-min data
    df['day'] = df['date'].dt.date
    
    daily = df.groupby('day').agg({
        'open': 'first',
        'close': 'last'
    }).reset_index()
    
    if len(daily) < 60:
        return None
    
    # Gap calculation
    daily['prev_close'] = daily['close'].shift(1)
    daily['gap'] = (daily['open'] / daily['prev_close'] - 1) * 100
    daily['intraday'] = (daily['close'] / daily['open'] - 1) * 100
    
    # Z-score
    gap_mean = daily['gap'].rolling(30).mean()
    gap_std = daily['gap'].rolling(30).std().replace(0, np.nan)
    daily['z_gap'] = (daily['gap'] - gap_mean) / gap_std
    
    daily = daily.dropna()
    
    if len(daily) < 30:
        return None
    
    return daily[['z_gap', 'gap', 'intraday']]

def compute_gap_analysis():
    """Compute gap analysis for Nifty 500."""
    print("="*70)
    print("NIFTY 500 ANGLE 4: Z-GAP ANALYSIS (Same Period as S&P 500)")
    print(f"Period: Feb 2015 - Feb 2018")
    print("="*70)
    
    # Load Whales and Minnows
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "nifty_sameperiod_whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "nifty_sameperiod_minnows.csv"))
    
    results = []
    
    for group_name, stock_list in [('whales', whales), ('minnows', minnows)]:
        print(f"\nProcessing {group_name}...")
        
        task_args = [(sym, START_DATE, END_DATE) for sym in stock_list['symbol'].tolist()]
        
        all_gaps = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_stock_gaps, arg): arg for arg in task_args}
            
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    all_gaps.append(res)
        
        if not all_gaps:
            continue
        
        combined = pd.concat(all_gaps, ignore_index=True)
        print(f"  Total data points: {len(combined):,}")
        
        # Extreme down gaps
        down_gaps = combined[combined['z_gap'] < -2]
        if len(down_gaps) > 10:
            avg_gap = down_gaps['gap'].mean()
            avg_intraday = down_gaps['intraday'].mean()
            fade_pct = -avg_intraday / avg_gap * 100 if avg_gap != 0 else 0
            
            results.append({
                'group': group_name,
                'direction': 'down',
                'avg_gap': avg_gap,
                'avg_intraday': avg_intraday,
                'fade_pct': fade_pct,
                'count': len(down_gaps)
            })
            
            print(f"\n  DOWN GAPS (Z < -2σ):")
            print(f"    Average Gap: {avg_gap:+.3f}%")
            print(f"    Average Intraday: {avg_intraday:+.3f}%")
            print(f"    Fade %: {fade_pct:.1f}% of gap recovered")
            print(f"    Count: {len(down_gaps)}")
        
        # Extreme up gaps
        up_gaps = combined[combined['z_gap'] > 2]
        if len(up_gaps) > 10:
            avg_gap = up_gaps['gap'].mean()
            avg_intraday = up_gaps['intraday'].mean()
            fade_pct = -avg_intraday / avg_gap * 100 if avg_gap != 0 else 0
            
            results.append({
                'group': group_name,
                'direction': 'up',
                'avg_gap': avg_gap,
                'avg_intraday': avg_intraday,
                'fade_pct': fade_pct,
                'count': len(up_gaps)
            })
            
            print(f"\n  UP GAPS (Z > 2σ):")
            print(f"    Average Gap: {avg_gap:+.3f}%")
            print(f"    Average Intraday: {avg_intraday:+.3f}%")
            print(f"    Fade %: {fade_pct:.1f}% of gap faded")
            print(f"    Count: {len(up_gaps)}")
    
    return pd.DataFrame(results)

def compare_markets(nifty_df):
    """Compare Nifty 500 vs S&P 500 gap results."""
    print("\n" + "="*70)
    print("CROSS-MARKET Z-GAP COMPARISON (Same Period)")
    print("="*70)
    
    # Load S&P 500 results
    sp500 = pd.read_csv(os.path.join(RESULTS_DIR, "sp500_angle4_extreme.csv"))
    
    print(f"\n{'Market':<12} {'Group':<10} {'Direction':<8} {'Gap':<10} {'Intraday':<12} {'Fade %':<10}")
    print("-"*70)
    
    for market, df in [('S&P 500', sp500), ('Nifty 500', nifty_df)]:
        for _, row in df.iterrows():
            print(f"{market:<12} {row['group'].capitalize():<10} {row['direction'].upper():<8} "
                  f"{row['avg_gap']:+.3f}%    {row['avg_intraday']:+.3f}%      {row['fade_pct']:.1f}%")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Compare fade ratios
    sp_whale_down = sp500[(sp500['group'] == 'whales') & (sp500['direction'] == 'down')]['fade_pct'].values
    sp_minnow_down = sp500[(sp500['group'] == 'minnows') & (sp500['direction'] == 'down')]['fade_pct'].values
    ni_whale_down = nifty_df[(nifty_df['group'] == 'whales') & (nifty_df['direction'] == 'down')]['fade_pct'].values
    ni_minnow_down = nifty_df[(nifty_df['group'] == 'minnows') & (nifty_df['direction'] == 'down')]['fade_pct'].values
    
    print("\nDown Gap Fade Comparison:")
    if len(sp_whale_down) > 0 and len(ni_whale_down) > 0:
        print(f"  Whales: S&P {sp_whale_down[0]:.1f}% vs Nifty {ni_whale_down[0]:.1f}%")
    if len(sp_minnow_down) > 0 and len(ni_minnow_down) > 0:
        print(f"  Minnows: S&P {sp_minnow_down[0]:.1f}% vs Nifty {ni_minnow_down[0]:.1f}%")
        ratio = ni_minnow_down[0] / sp_minnow_down[0] if sp_minnow_down[0] != 0 else 0
        print(f"\n  Nifty Minnows fade {ratio:.1f}x MORE than S&P Minnows")

def main():
    # Compute Nifty 500 gap analysis
    nifty_df = compute_gap_analysis()
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    nifty_df.to_csv(os.path.join(RESULTS_DIR, "nifty_sameperiod_angle4.csv"), index=False)
    
    # Compare markets
    compare_markets(nifty_df)
    
    print("\n" + "="*70)
    print("ANGLE 4 SAME-PERIOD COMPARISON COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
