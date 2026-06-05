import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import kurtosis
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from volume_time_lib import create_adaptive_volume_bars

DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

def analyze_stock_kurtosis(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    try:
        # Load Raw (Clock Time)
        df = pd.read_csv(file_path, usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            return None
            
        # Clock Time Returns (5-min)
        # We need to sort by date
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values('date').set_index('date')
        
        clock_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        k_clock = kurtosis(clock_returns) 
        # Note: Fisher kurtosis (normal = 0). Pearson (normal = 3). scipy defaults to Fisher (excess kurtosis).
        
        # Volume Time Returns
        vol_df = create_adaptive_volume_bars(df.reset_index(), bars_per_day=50) # reset_index because lib expects column or handles index
        if vol_df.empty:
            return None
            
        vol_returns = np.log(vol_df['close'] / vol_df['close'].shift(1)).dropna()
        k_volume = kurtosis(vol_returns)
        
        return {
            'symbol': symbol,
            'k_clock': k_clock,
            'k_volume': k_volume,
            'improvement': k_clock - k_volume # Positive means Volume Time is less leptokurtic (closer to normal if both positive high)
        }
        
    except Exception as e:
        # print(f"Error {symbol}: {e}")
        return None

if __name__ == "__main__":
    print("Analyzing Kurtosis (Clock vs Volume)...")
    
    # Load list of Whales (Focus on liquid stocks where "Physics" should apply best)
    whales_path = os.path.join(PROCESSED_DIR, "whales.csv")
    if not os.path.exists(whales_path):
        print("Whales list not found.")
        exit()
        
    whales = pd.read_csv(whales_path)
    # Sample top 20 whales to be representative but fast
    sample_whales = whales.head(20)['symbol'].tolist()
    
    results = []
    for i, sym in enumerate(sample_whales):
        print(f"Processing {sym} ({i+1}/{len(sample_whales)})...")
        res = analyze_stock_kurtosis(sym)
        if res:
            results.append(res)
            
    res_df = pd.DataFrame(results)
    
    print("\n--- KURTOSIS RESULTS (Excess Kurtosis, Normal=0) ---")
    print(res_df[['symbol', 'k_clock', 'k_volume']].describe())
    
    avg_clock = res_df['k_clock'].mean()
    avg_vol = res_df['k_volume'].mean()
    
    print(f"\nAverage Clock Kurtosis: {avg_clock:.2f}")
    print(f"Average Volume Kurtosis: {avg_vol:.2f}")
    
    if avg_vol < avg_clock:
        print("SUCCESS: Volume Time significantly reduces heavy tails (More Gaussian).")
        ratio = avg_clock / avg_vol
        print(f"Reduction Factor: {ratio:.2f}x")
    else:
        print("fail: No improvement observed.")
