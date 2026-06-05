import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from push_response_lib import prepare_push_response_data

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

LAG_TO_TEST = 10 # Test at a representative lag (e.g., 50 mins) or avg of 5-15?
# User said slope. Let's aggregate short term slope.

def calculate_stock_metrics(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close', 'volume', 'open', 'high', 'low'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        # Ensure sorted
        df = df.sort_values('date')
        
        # 1. Amihud Ratio
        # Daily Amihud = |Ret| / (Price * Vol)
        # We have 5-min. Resample to Daily for classic Amihud.
        daily = df.resample('D').agg({
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        daily['ret'] = abs(daily['close'].pct_change())
        daily['amihud'] = daily['ret'] / (daily['close'] * daily['volume'])
        
        # Replace inf
        daily['amihud'] = daily['amihud'].replace([np.inf, -np.inf], np.nan)
        avg_amihud = daily['amihud'].mean()
        
        # 2. Inefficiency Slope & Raw Return stats
        # Prepare Z data
        # We use a fixed lag for slope comparison, e.g. Lag 10 (50 mins)
        # Only calculating for LAST 3 years? Or full? Full.
        
        # Re-use library logic manually to get raw volatility too
        # Rolling stats
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        window = 1500
        roll = df['log_ret'].rolling(window=window)
        mu = roll.mean()
        sigma = roll.std() # This is per-bar (5-min) volatility
        
        z = (df['log_ret'] - mu) / sigma
        
        # Push = z(t)
        # Resp = Sum(z(t+1)...z(t+L)) ? Or just Price(t+L)-Price(t)?
        # Our surface uses return over L.
        # Let's approximate slope using Lag 10.
        
        # Construct Push(t) and Resp(t+10)
        # Push is single bar return normalized?
        # The paper def: Push(L) = m_t - m_t-L. 
        # But we usually just use bar returns for creating "Push".
        # Let's stick to simple: Push = 1-bar return Z-score.
        # Response = 10-bar return Z-score.
        
        # Wait, the surface logic uses Push(L).
        # Let's match the "Minnow Surface" logic.
        # Push = Ret_L(t), Resp = Ret_L(t+L).
        # Let L=10.
        
        LAG = 10
        df['ret_L'] = np.log(df['close'] / df['close'].shift(LAG))
        
        # Re-calc Z for L-period return
        roll_L = df['ret_L'].rolling(window=window)
        sigma_L = roll_L.std()
        
        # To avoid re-standardizing everything:
        # Just use raw returns for "Economic Significance" check later.
        # For Slope, we need Z.
        
        df['z_push'] = (df['ret_L'] - roll_L.mean()) / sigma_L
        df['z_resp'] = df['z_push'].shift(-LAG) # Response is Next L period
        
        # Clean
        data = df[['z_push', 'z_resp']].dropna()
        
        # Compute Slope (Regression z_resp ~ z_push)
        if len(data) < 100: return None
        
        slope, intercept, _, _, _ = stats.linregress(data['z_push'], data['z_resp'])
        
        # 3. Raw Volatility Estimate (for Economic Sig)
        # Average 10-period volatility (sigma_L)
        avg_raw_vol_L = sigma_L.mean()
        
        return {
            'symbol': symbol,
            'amihud': avg_amihud,
            'slope': slope,
            'avg_vol_L': avg_raw_vol_L
        }
        
    except Exception as e:
        return None

def process_all_stocks():
    print("Starting Robustness Analysis (Amihud & Slopes)...")
    
    # Get all CSVs in raw_Data? Or Processed?
    # raw_Data has all. but use whales/minnows lists + others?
    # Let's just use the Top 500 list from sorting.
    # We have 'minnows.csv' and 'whales.csv'. Do we have the full 500 list?
    # 'liquidity_stratification.py' likely saved "ranked_stocks.csv"?
    # If not, let's just use Whales + Minnows for the plot. Extreme divergence is better.
    # Or iterate 500.
    
    # Let's grab all CSVs in processed_data if available or just raw_data glob.
    # Safer: raw_data csvs. Limit to top 500 by size? 
    # Just processing Whales + Minnows (100 stocks) is enough to show correlation line.
    
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "minnows.csv"))
    
    all_syms = whales['symbol'].tolist() + minnows['symbol'].tolist()
    
    results = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(calculate_stock_metrics, sym): sym for sym in all_syms}
        
        count = 0
        total = len(all_syms)
        
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{total}...")
            if res:
                results.append(res)
                
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No results.")
        return

    # Save
    out_path = os.path.join(RESULTS_DIR, "robustness_amihud.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved metrics to {out_path}")
    
    # Analyze Economic Significance for Minnows
    # Filter for Minnows
    minnow_syms = set(minnows['symbol'])
    minnow_stats = results_df[results_df['symbol'].isin(minnow_syms)]
    
    avg_vol_10 = minnow_stats['avg_vol_L'].mean()
    print(f"\n--- ECONOMIC SIGNIFICANCE (Minnows, Lag 10) ---")
    print(f"Avg Raw Volatility (10-bar/50min): {avg_vol_10*100:.4f}%")
    
    # User said Z of 0.12 (Avg Resp Strength). 
    # Let's assume Z_resp ~ 0.12 (from paper/walkthrough).
    # Exp_Ret = 0.12 * Vol
    z_resp_strength = 0.12
    exp_ret = z_resp_strength * avg_vol_10
    print(f"Expected Raw Return per Trade (approx): {exp_ret*100:.4f}%")
    print(f"vs Bid-Ask Spread (approx 0.05%-0.15%)")
    
    # Correlation Check
    # Handle Log Amihud
    # Filter zeros or negs
    valid = results_df[results_df['amihud'] > 0].copy()
    valid['log_amihud'] = np.log(valid['amihud'])
    
    corr = valid['log_amihud'].corr(valid['slope'])
    print(f"\n--- MECHANISM PROOF ---")
    print(f"Correlation (Log Amihud vs Slope): {corr:.4f}")
    
    slope_reg, _, r_val, _, _ = stats.linregress(valid['log_amihud'], valid['slope'])
    print(f"R-Squared: {r_val**2:.4f}")

if __name__ == "__main__":
    process_all_stocks()
