import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

def process_truncation(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close'])
        if df.empty: return None

        df['ret'] = df['close'].pct_change()
        
        # Volatility: Rolling 2000-min std
        df['vol'] = df['ret'].rolling(2000).std().shift(1)
        df['z_push'] = df['ret'] / df['vol']
        
        # Future Prices
        p_t = df['close']
        p_t1 = df['close'].shift(-1)
        p_t2 = df['close'].shift(-2)
        p_t10 = df['close'].shift(-10)
        
        valid = df['vol'] > 1e-8
        
        # Responses
        resp_0 = np.log(p_t10 / p_t)
        resp_1 = np.log(p_t10 / p_t1)
        resp_2 = np.log(p_t10 / p_t2)
        
        # Z-Normalize
        z_r0 = resp_0 / (df['vol'] * np.sqrt(10))
        z_r1 = resp_1 / (df['vol'] * np.sqrt(9))
        z_r2 = resp_2 / (df['vol'] * np.sqrt(8))
        
        # Filter for Z=-4 (+/- 0.5)
        mask_crash = (df['z_push'] >= -4.5) & (df['z_push'] <= -3.5) & valid
        
        r0_vals = z_r0[mask_crash]
        r1_vals = z_r1[mask_crash]
        r2_vals = z_r2[mask_crash]
        
        return {
            'skip_0': r0_vals.dropna().values,
            'skip_1': r1_vals.dropna().values,
            'skip_2': r2_vals.dropna().values
        }
        
    except Exception as e:
        return None

def main():
    print("Running Experiment 12.1: Lag Truncation (Bid-Ask Bounce Test)...")
    minnows_path = os.path.join(PROCESSED_DIR, "minnows.csv")
    if not os.path.exists(minnows_path): 
        print("Minnows file not found.")
        return
        
    symbols = pd.read_csv(minnows_path)['symbol'].tolist()
    
    acc = {'skip_0': [], 'skip_1': [], 'skip_2': []}
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_truncation, sym): sym for sym in symbols}
        
        count = 0 
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 20 == 0: print(f"Processed {count}/{len(symbols)}...")
            
            if res:
                acc['skip_0'].extend(res['skip_0'])
                acc['skip_1'].extend(res['skip_1'])
                acc['skip_2'].extend(res['skip_2'])
                
    # Stats
    print("\n--- RESULTS ---")
    for k in ['skip_0', 'skip_1', 'skip_2']:
        vals = np.array(acc[k])
        if len(vals) == 0:
            print(f"{k}: No events found")
            continue
            
        mean = np.mean(vals)
        sem = np.std(vals) / np.sqrt(len(vals))
        print(f"{k}: Mean Z-Resp = {mean:.4f} +/- {sem:.4f} (N={len(vals)})")
        
        pd.DataFrame({'z_resp': vals}).to_csv(os.path.join(RESULTS_DIR, f"truncation_{k}.csv"), index=False)

if __name__ == "__main__":
    main()
