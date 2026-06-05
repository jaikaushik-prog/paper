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

# Lags
LAGS_SHORT = list(range(1, 21))
LAGS_MEDIUM = list(range(25, 80, 5))
ALL_LAGS = sorted(list(set(LAGS_SHORT + LAGS_MEDIUM)))

def process_stock_worker(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    results = {}
    
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close', 'volume']) # Added volume just in case, though only close used
        # Note: prepare_push_response_data handles date parsing? 
        # Actually it assumes date is there or index. 
        # Let's ensure date parsing for safety if lib relies on it (lib uses rolling, so order matters)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        # Sort is crucial
        df = df.sort_values('date')
        
        if df.empty: return None
        
        for lag in ALL_LAGS:
            res = prepare_push_response_data(df, lag)
            if not res.empty:
                results[lag] = {
                    'z_push': res['z_push'].values,
                    'z_resp': res['z_resp'].values
                }
        return results
    except Exception as e:
        return None

def process_refined_group(group_name, stock_list):
    print(f"Processing REFINED SURFACES for {group_name}...")
    
    accumulator = {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS}
    symbols = stock_list['symbol'].tolist()
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_stock_worker, sym): sym for sym in symbols}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 10 == 0:
                print(f"[{group_name}] Processed {count}/{len(symbols)}...")
                
            if res:
                for lag, data in res.items():
                    accumulator[lag]['z_push'].append(data['z_push'])
                    accumulator[lag]['z_resp'].append(data['z_resp'])
                    
    # Aggregate with Advanced Metrics
    print(f"Aggregating {group_name}...")
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
        
        # Groupby Bin
        grouped = df_lag.groupby('bin', observed=True)['z_resp']
        
        # 1. Mean (Original)
        mean_resp = grouped.mean()
        # 2. Probability > 0 (Win Rate)
        prob_resp = grouped.apply(lambda x: (x > 0).mean())
        # 3. Risk (Std Dev)
        risk_resp = grouped.std()
        
        for center in mean_resp.index:
            m = mean_resp[center]
            p = prob_resp[center]
            s = risk_resp[center]
            
            if pd.isna(m): continue
            
            surface_data.append({
                'lag': int(lag),
                'push_bin': float(center),
                'avg_response': float(m),
                'prob_positive': float(p),
                'risk_std': float(s)
            })
            
    out_file = os.path.join(RESULTS_DIR, f"surface_refined_{group_name}.csv")
    pd.DataFrame(surface_data).to_csv(out_file, index=False)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    # Load Groups
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "minnows.csv"))
    
    # Process
    process_refined_group("whales", whales)
    process_refined_group("minnows", minnows)
