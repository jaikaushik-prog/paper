import pandas as pd
import numpy as np
import os
import sys
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Use naive import, assuming fix
try:
    from push_response_lib import prepare_push_response_data
except ImportError:
    pass

PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"

def process_asymmetry(symbol):
    try:
        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(file_path): 
            return f"Missing: {symbol}"
        
        df = pd.read_csv(file_path, usecols=['date', 'close'])
        if df.empty: return f"Empty: {symbol}"
        
        # FIX: Positional argument for lag
        res = prepare_push_response_data(df, 10)
        if res.empty: return f"No Res: {symbol}"
        
        mask_crash = (res['z_push'] >= -4.25) & (res['z_push'] <= -3.75)
        mask_spike = (res['z_push'] >= 3.75) & (res['z_push'] <= 4.25)
        
        c_vals = res.loc[mask_crash, 'z_resp'].values
        s_vals = res.loc[mask_spike, 'z_resp'].values
        
        return {
            'crash': c_vals,
            'spike': s_vals,
            'status': 'ok'
        }
        
    except Exception as e:
        return f"Error {symbol}: {str(e)}"

def main():
    print("Computing Asymmetry Statistics (Crash vs Spike) [Attempt 3]...")
    minnows_path = os.path.join(PROCESSED_DIR, "minnows.csv")
    if not os.path.exists(minnows_path): return
        
    minnow_symbols = pd.read_csv(minnows_path)['symbol'].tolist()
    available_files = set(os.path.basename(f).replace('.csv', '') for f in glob.glob(os.path.join(DATA_DIR, "*.csv")))
    valid_symbols = [s for s in minnow_symbols if s in available_files]
    
    acc = {'crash': [], 'spike': []}
    errors = []
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_asymmetry, sym): sym for sym in valid_symbols}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 20 == 0: print(f"Processed {count}...")
            
            if isinstance(res, dict) and res.get('status') == 'ok':
                acc['crash'].extend(res['crash'])
                acc['spike'].extend(res['spike'])
            else:
                errors.append(res)

    if len(errors) > 0:
        print(f"Errors: {len(errors)} (Sample: {errors[:2]})")

    print("\n--- ASYMMETRY RESULTS ---")
    vals_c = np.array(acc['crash'])
    vals_s = np.array(acc['spike'])
    
    if len(vals_c) > 0:
        mean_c = np.mean(vals_c); sem_c = np.std(vals_c)/np.sqrt(len(vals_c))
        print(f"CRASH (Z=-4): Mean={mean_c:.4f} +/- {sem_c:.4f} (N={len(vals_c)})")
    else: print("CRASH: N=0")
        
    if len(vals_s) > 0:
        mean_s = np.mean(vals_s); sem_s = np.std(vals_s)/np.sqrt(len(vals_s))
        print(f"SPIKE (Z=+4): Mean={mean_s:.4f} +/- {sem_s:.4f} (N={len(vals_s)})")
    else: print("SPIKE: N=0")

if __name__ == "__main__":
    main()
