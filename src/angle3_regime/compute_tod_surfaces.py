import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from push_response_lib import prepare_push_response_data

DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

LAGS_SHORT = list(range(1, 21))
LAGS_MEDIUM = list(range(25, 80, 5))
ALL_LAGS = sorted(list(set(LAGS_SHORT + LAGS_MEDIUM)))

def process_stock_tod(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    results = {'open': {}, 'midday': {}, 'close': {}}
    
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True) 
        df.sort_index(inplace=True)
        
        if df.empty: return None

        # Pre-calc masks using DatetimeIndex (FAST)
        for lag in ALL_LAGS:
            # 1. Continuous Logic
            res_df = prepare_push_response_data(df.reset_index(), lag)
            if res_df.empty: continue
            
            # Re-align index to get push timestamps
            push_timestamps = df.index[res_df.index]
            t = push_timestamps.time
            
            # Open: < 10:15
            mask_open = (t >= pd.Timestamp("09:15").time()) & (t < pd.Timestamp("10:15").time())
            
            # Midday: 11:00 - 14:00
            mask_mid = (t >= pd.Timestamp("11:00").time()) & (t < pd.Timestamp("14:00").time())
            
            # Close: > 14:15
            mask_close = (t >= pd.Timestamp("14:15").time())
            
            def add_res(key, mask):
                subset = res_df[mask]
                if not subset.empty:
                    results[key][lag] = {
                        'z_push': subset['z_push'].values,
                        'z_resp': subset['z_resp'].values
                    }

            add_res('open', mask_open)
            add_res('midday', mask_mid)
            add_res('close', mask_close)
            
        return results
        
    except Exception as e:
        return None

def save_surface(accumulator, name):
    print(f"Aggregating {name}...")
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
        avg_resp = df_lag.groupby('bin', observed=True)['z_resp'].mean()
        
        for center, value in avg_resp.items():
            if pd.isna(value): continue
            surface_data.append({'lag': int(lag), 'push_bin': float(center), 'avg_response': float(value)})
            
    out_file = os.path.join(RESULTS_DIR, f"surface_tod_{name}.csv")
    pd.DataFrame(surface_data).to_csv(out_file, index=False)
    print(f"Saved {out_file}")

def main():
    print("Running Experiment 3b: Time-of-Day Conditioning...")
    minnows_path = os.path.join(PROCESSED_DIR, "minnows.csv")
    if not os.path.exists(minnows_path): return
    
    symbols = pd.read_csv(minnows_path)['symbol'].tolist()
    
    acc = {
        'open': {l: {'z_push': [], 'z_resp': []} for l in ALL_LAGS},
        'midday': {l: {'z_push': [], 'z_resp': []} for l in ALL_LAGS},
        'close': {l: {'z_push': [], 'z_resp': []} for l in ALL_LAGS}
    }
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_stock_tod, sym): sym for sym in symbols}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 20 == 0: print(f"Processed {count}/{len(symbols)}...")
            
            if res:
                for period in ['open', 'midday', 'close']:
                    for lag, data in res[period].items():
                        acc[period][lag]['z_push'].append(data['z_push'])
                        acc[period][lag]['z_resp'].append(data['z_resp'])
    
    save_surface(acc['open'], 'open')
    save_surface(acc['midday'], 'midday')
    save_surface(acc['close'], 'close')

if __name__ == "__main__":
    main()
