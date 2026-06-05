import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from push_response_lib import prepare_push_response_data

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
ALL_LAGS = list(range(1, 21)) + list(range(25, 80, 5))

def process_stock_volume_filter(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    results = {'high_vol': {}, 'low_vol': {}}
    
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values('date')
        
        # Calculate Volume Z-Score
        # Use simple log volume to normalize? Or raw?
        # Volume is log-normally distributed usually.
        # df['log_vol'] = np.log(df['volume'] + 1)
        # But user said "Volume Z-score". Let's stick to simple rolling std of raw or log.
        # Log is safer.
        df['log_vol'] = np.log(df['volume'].replace(0, 1))
        
        window = 1500 # Approx 20 days
        roll_vol = df['log_vol'].rolling(window=window)
        df['z_vol'] = (df['log_vol'] - roll_vol.mean()) / roll_vol.std()
        
        # Split Dataframes
        # We need continuous time for Lags?
        # prepare_push_response_data requires continuous dataframe?
        # It uses simple shifting. So we can calculate P-R for the WHOLE dataframe first, then filter.
        # This is better.
        
        # Pre-calculate Pushes and Responses for all lags? 
        # Or iterate lags and filter.
        
        # Optimization:
        # For each lag, get z_push and z_resp (already standardized in lib).
        # Then filter based on z_vol at time 't' (Push time).
        
        valid_df = df.dropna(subset=['z_vol'])
        
        if valid_df.empty: return None
        
        for lag in ALL_LAGS:
            # Get P-R
            res = prepare_push_response_data(valid_df, lag)
            
            if res.empty: continue
            
            # Align z_vol
            # res has 'z_push' and 'z_resp' and likely same index as valid_df (if lib preserves index)
            # define: prepare_push_response_data(df, lag) -> returns df with z_push, z_resp.
            # We assume it aligns.
            
            # Join z_vol
            res['z_vol'] = valid_df.loc[res.index, 'z_vol']
            
            # Split
            high_vol = res[res['z_vol'] > 1.5]
            low_vol = res[res['z_vol'] < -0.5]
            
            results['high_vol'][lag] = {
                'z_push': high_vol['z_push'].values,
                'z_resp': high_vol['z_resp'].values
            }
            results['low_vol'][lag] = {
                'z_push': low_vol['z_push'].values,
                'z_resp': low_vol['z_resp'].values
            }
            
        return results
        
    except Exception as e:
        return None

def process_volume_groups():
    print("Starting Volume-Conditional Analysis (Minnows)...")
    
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "minnows.csv"))
    symbols = minnows['symbol'].tolist()
    
    accumulator = {
        'high_vol': {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS},
        'low_vol': {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS}
    }
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_stock_volume_filter, sym): sym for sym in symbols}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{len(symbols)}...")
                
            if res:
                for regime in ['high_vol', 'low_vol']:
                    for lag, data in res[regime].items():
                        accumulator[regime][lag]['z_push'].append(data['z_push'])
                        accumulator[regime][lag]['z_resp'].append(data['z_resp'])
    
    # Aggregate and Save
    for regime in ['high_vol', 'low_vol']:
        print(f"Aggregating {regime}...")
        surface_data = []
        bins = np.arange(-4.0, 4.05, 0.1) 
        centers = (bins[:-1] + bins[1:]) / 2
        
        for lag in ALL_LAGS:
            pushes = accumulator[regime][lag]['z_push']
            responses = accumulator[regime][lag]['z_resp']
            
            if not pushes: continue
            
            cat_push = np.concatenate(pushes)
            cat_resp = np.concatenate(responses)
            
            df_lag = pd.DataFrame({'z_push': cat_push, 'z_resp': cat_resp})
            df_lag['bin'] = pd.cut(df_lag['z_push'], bins=bins, labels=centers)
            
            mean_resp = df_lag.groupby('bin', observed=True)['z_resp'].mean()
            
            for center, val in mean_resp.items():
                if pd.isna(val): continue
                surface_data.append({
                    'lag': int(lag),
                    'push_bin': float(center),
                    'avg_response': float(val)
                })
        
        out_file = os.path.join(RESULTS_DIR, f"surface_minnows_{regime}.csv")
        pd.DataFrame(surface_data).to_csv(out_file, index=False)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    process_volume_groups()
