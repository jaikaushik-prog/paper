import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"

# Thresholds
ADT_WHALE_THRESH = 1e9  # 100 Crore
ADT_MINNOW_THRESH = 1e8 # 10 Crore

def process_gap_dynamics(file_path):
    try:
        df = pd.read_csv(file_path, usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 1. Rolling ADT
        df['turnover'] = df['close'] * df['volume']
        daily_stats = df['turnover'].resample('D').sum().to_frame(name='daily_turnover')
        daily_stats['rolling_adt'] = daily_stats['daily_turnover'].rolling(window=20).mean().shift(1)
        
        # Map ADT back to intraday
        df['day_date'] = df.index.date
        daily_stats['day_date'] = daily_stats.index.date
        df = df.reset_index().merge(daily_stats[['day_date', 'rolling_adt']], on='day_date', how='left').set_index('date')
        
        # 2. Daily Gaps & Volatility
        daily_agg = df.resample('D').agg({
            'open': 'first',
            'close': 'last',
            'rolling_adt': 'first'
        }).dropna(subset=['open'])
        
        daily_agg['prev_close'] = daily_agg['close'].shift(1)
        daily_agg['gap_ret'] = np.log(daily_agg['open'] / daily_agg['prev_close'])
        daily_agg['daily_ret'] = np.log(daily_agg['close'] / daily_agg['prev_close'])
        daily_agg['vol_20d'] = daily_agg['daily_ret'].rolling(20).std().shift(1)
        daily_agg['z_gap'] = daily_agg['gap_ret'] / daily_agg['vol_20d']
        
        valid_gaps = daily_agg.dropna(subset=['z_gap', 'rolling_adt'])
        
        results = {'high_liq': [], 'low_liq': []}
        
        # 3. Intraday Responses using Vectorized Lookup
        # We need intraday bars for the days in valid_gaps
        # Group intraday by day
        day_groups = df.groupby('day_date')
        
        # Iterate over valid gap days
        for day, gap_row in valid_gaps.iterrows():
            day_date = day.date()
            if day_date not in day_groups.groups: continue
            
            group = day_groups.get_group(day_date)
            # Ensure at least 30 mins (6 bars of 5 mins)
            if len(group) < 6: continue
            
            # Response Windows
            # 0-5m: Close of first bar / Open of first bar
            p_open = group.iloc[0]['open']
            p_5m = group.iloc[0]['close']
            p_15m = group.iloc[2]['close'] # 3rd bar end
            p_30m = group.iloc[5]['close'] # 6th bar end
            
            # Use raw price ratio or log diff
            r_0_5 = np.log(p_5m / p_open)
            r_5_15 = np.log(p_15m / p_5m)
            r_15_30 = np.log(p_30m / p_15m)
            
            record = {
                'z_gap': gap_row['z_gap'],
                'd_gap': gap_row['gap_ret'], # Directional check
                'r_0_5': r_0_5,
                'r_5_15': r_5_15,
                'r_15_30': r_15_30
            }
            
            adt = gap_row['rolling_adt']
            if adt > ADT_WHALE_THRESH:
                results['high_liq'].append(record)
            elif adt < ADT_MINNOW_THRESH:
                results['low_liq'].append(record)
                
        return results
        
    except Exception as e:
        return None

def main():
    print("Running Refined Z-Gap Analysis...")
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    acc_high = []
    acc_low = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_gap_dynamics, f): f for f in files}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 50 == 0: print(f"Processed {count}/{len(files)}...")
            
            if res:
                acc_high.extend(res['high_liq'])
                acc_low.extend(res['low_liq'])
    
    bins = np.linspace(-4, 4, 17) # 0.5 width
    centers = (bins[:-1] + bins[1:]) / 2
    
    for name, data in [('high_liq', acc_high), ('low_liq', acc_low)]:
        if not data: continue
        df = pd.DataFrame(data)
        
        df['bin'] = pd.cut(df['z_gap'], bins=bins, labels=centers)
        
        agg = df.groupby('bin', observed=True).agg({
            'r_0_5': ['mean', 'sem', 'count'],
            'r_5_15': ['mean', 'sem', 'count'],
            'r_15_30': ['mean', 'sem', 'count']
        })
        
        agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
        agg.reset_index(inplace=True)
        
        out = os.path.join(RESULTS_DIR, f"refined_gap_{name}.csv")
        agg.to_csv(out, index=False)
        print(f"Saved {out}")

if __name__ == "__main__":
    main()
