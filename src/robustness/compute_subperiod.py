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

def process_stock_subperiod(args):
    symbol, start_year, end_year = args
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    results = {}
    
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        
        # Filter Date Range
        mask = (df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)
        sub_df = df.loc[mask].copy()
        
        if len(sub_df) < 2000: return None # Not enough data
        
        # Sort is crucial
        sub_df = sub_df.sort_values('date')
        
        for lag in ALL_LAGS:
            res = prepare_push_response_data(sub_df, lag)
            if not res.empty:
                results[lag] = {
                    'z_push': res['z_push'].values,
                    'z_resp': res['z_resp'].values
                }
        return results
    except Exception as e:
        return None

def process_subperiod_group(group_name, stock_list, period_name, start_year, end_year):
    print(f"Processing {group_name} for {period_name} ({start_year}-{end_year})...")
    
    accumulator = {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS}
    symbols = stock_list['symbol'].tolist()
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_stock_subperiod, (sym, start_year, end_year)): sym for sym in symbols}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 10 == 0:
                print(f"[{period_name}] Processed {count}/{len(symbols)}...")
                
            if res:
                for lag, data in res.items():
                    accumulator[lag]['z_push'].append(data['z_push'])
                    accumulator[lag]['z_resp'].append(data['z_resp'])
                    
    # Aggregate
    print(f"Aggregating {period_name}...")
    surface_data = []
    bins = np.arange(-4.0, 4.05, 0.1) 
    centers = (bins[:-1] + bins[1:]) / 2
    
    for lag in ALL_LAGS:
        pushes = accumulator[lag]['z_push']
        responses = accumulator[lag]['z_resp']
        
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
            
    out_file = os.path.join(RESULTS_DIR, f"surface_minnows_{period_name}.csv")
    pd.DataFrame(surface_data).to_csv(out_file, index=False)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "minnows.csv"))
    
    # Pre-COVID (2014-2019)
    process_subperiod_group("minnows", minnows, "precovid", 2014, 2019)
    # Post-COVID (2020-2024)
    process_subperiod_group("minnows", minnows, "postcovid", 2020, 2024)
