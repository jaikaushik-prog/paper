import pandas as pd
import numpy as np
import os
from scipy.stats import kurtosis
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from volume_time_lib import create_adaptive_volume_bars

DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

def hill_estimator(data, tail_fraction=0.05):
    """
    Calculate Hill Estimator for Tail Index (Alpha).
    Lower Alpha = Fatter Tails (Power Law).
    """
    if len(data) == 0: return np.nan
    
    # Focus on absolute returns (magnitude)
    x = np.abs(data)
    x = np.sort(x)[::-1] # Descending
    
    # Selecting the tail
    k = int(len(x) * tail_fraction)
    if k < 2: return np.nan
    
    # Hill formula: 1 / ( (1/k) * sum(ln(xi/xmin)) )
    # x_min is the k-th order statistic
    x_tail = x[:k]
    x_min = x[k]
    
    if x_min == 0: return np.nan
    
    log_ratios = np.log(x_tail / x_min)
    hill_stat = np.mean(log_ratios)
    
    if hill_stat == 0: return np.nan
    alpha = 1 / hill_stat
    return alpha

def analyze_forensics(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    try:
        # Load Data
        df = pd.read_csv(file_path, usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty: return None
        
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values('date').set_index('date')
        
        # 1. Helper: Get Stats
        def get_dist_stats(returns, name):
            if len(returns) < 100: return {}
            
            # Raw Kurtosis
            k_raw = kurtosis(returns)
            
            # Filtered (Jump-Removed)
            # Remove top/bottom 0.1%
            q_low = returns.quantile(0.001)
            q_high = returns.quantile(0.999)
            trimmed = returns[(returns >= q_low) & (returns <= q_high)]
            k_trimmed = kurtosis(trimmed)
            
            # Hill Alpha (Tail Index)
            alpha = hill_estimator(returns)
            
            return {
                f'{name}_k_raw': k_raw,
                f'{name}_k_trimmed': k_trimmed,
                f'{name}_alpha': alpha
            }

        # 2. Clock Time
        clock_ret = np.log(df['close'] / df['close'].shift(1)).dropna()
        clock_stats = get_dist_stats(clock_ret, 'clock')
        
        # 3. Volume Time
        vol_df = create_adaptive_volume_bars(df.reset_index(), bars_per_day=50)
        if vol_df.empty: return None
        vol_ret = np.log(vol_df['close'] / vol_df['close'].shift(1)).dropna()
        vol_stats = get_dist_stats(vol_ret, 'vol')
        
        return {**{'symbol': symbol}, **clock_stats, **vol_stats}
        
    except Exception as e:
        return None

def run_analysis():
    print("Running Phase 9 Forensics (Jump Filter + Hill Estimator)...")
    
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "whales.csv"))
    sample_whales = whales.head(20)['symbol'].tolist()
    
    results = []
    for i, sym in enumerate(sample_whales):
        print(f"Processing {sym}...")
        res = analyze_forensics(sym)
        if res: results.append(res)
        
    df = pd.DataFrame(results)
    
    print("\n--- FORENSICS RESULTS (Averages) ---")
    print(f"Raw Kurtosis:      Clock={df['clock_k_raw'].mean():.2f} | Vol={df['vol_k_raw'].mean():.2f}")
    print(f"Trimmed Kurtosis:  Clock={df['clock_k_trimmed'].mean():.2f} | Vol={df['vol_k_trimmed'].mean():.2f}")
    print(f"Tail Alpha:        Clock={df['clock_alpha'].mean():.2f} | Vol={df['vol_alpha'].mean():.2f}")
    
    # Save for plotting
    df.to_csv(os.path.join(r"c:\Users\DELL\Desktop\project_nifty_liquid\results", "forensics_data.csv"), index=False)

if __name__ == "__main__":
    run_analysis()
