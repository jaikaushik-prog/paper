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

def get_market_regimes():
    """
    Calculates Calm/Stress days based on Top 10 Whales volatility.
    """
    print("Calculating Market Volatility Proxy...")
    
    # 1. Load Whales List
    whales_path = os.path.join(PROCESSED_DIR, "whales.csv")
    if not os.path.exists(whales_path):
        raise FileNotFoundError("Whales list not found")
        
    whales = pd.read_csv(whales_path).head(10)['symbol'].tolist()
    
    daily_vols = []
    
    for sym in whales:
        file_path = os.path.join(DATA_DIR, f"{sym}.csv")
        try:
            df = pd.read_csv(file_path, usecols=['date', 'close'])
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df.set_index('date', inplace=True)
            
            # Daily Std Dev of 5-min returns
            # Resample to day, calc std of pct_change
            # First calc returns
            df['ret'] = df['close'].pct_change()
            # Group by date part (D)
            d_vol = df['ret'].resample('D').std()
            daily_vols.append(d_vol)
        except Exception as e:
            print(f"Skipping {sym} for vol proxy: {e}")
            
    if not daily_vols:
        raise ValueError("No data for volatility proxy")
        
    # Average volatility across top 10 stocks
    # Concat and mean(axis=1) handles alignment
    market_vol_df = pd.concat(daily_vols, axis=1)
    avg_daily_vol = market_vol_df.mean(axis=1).dropna()
    
    # Thresholds
    calm_thresh = avg_daily_vol.quantile(0.20)
    stress_thresh = avg_daily_vol.quantile(0.80)
    
    calm_days = avg_daily_vol[avg_daily_vol < calm_thresh].index.strftime('%Y-%m-%d').tolist()
    stress_days = avg_daily_vol[avg_daily_vol > stress_thresh].index.strftime('%Y-%m-%d').tolist()
    
    print(f"Defined Regimes: {len(calm_days)} Calm Days, {len(stress_days)} Stress Days.")
    return set(calm_days), set(stress_days)

def process_stock_regime_full(symbol, calm_days, stress_days):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    results = {'calm': {}, 'stress': {}}
    
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        # We need integer index for alignment, so reset if needed, but read_csv gives RangeIndex default.
        
        if df.empty: return None
        
        # Pre-calculate Date mapping
        # Maps index (int) to date string (for filtering)
        date_str_series = df['date'].dt.strftime('%Y-%m-%d')
        is_calm = date_str_series.isin(calm_days)
        is_stress = date_str_series.isin(stress_days)
        
        for lag in ALL_LAGS:
            # 1. Get standardized data for ALL time (preserving Rolling stats correctly)
            res = prepare_push_response_data(df, lag) 
            # res has index corresponding to df.
            # prepare_push_response_data typically returns df aligned with input df index.
            
            if res.empty: continue
            
            # Align masks
            # res index matches df index.
            # Safety: use .loc to align mask to res index
            mask_calm = is_calm.loc[res.index]
            mask_stress = is_stress.loc[res.index]
            
            res_calm = res[mask_calm]
            res_stress = res[mask_stress]
            
            if not res_calm.empty:
                results['calm'][lag] = {
                    'z_push': res_calm['z_push'].values,
                    'z_resp': res_calm['z_resp'].values
                }
            if not res_stress.empty:
                results['stress'][lag] = {
                    'z_push': res_stress['z_push'].values,
                    'z_resp': res_stress['z_resp'].values
                }
                
        return results
        
    except Exception as e:
        # print(e)
        return None

def main():
    # 1. Get Regimes
    calm_days, stress_days = get_market_regimes()
    
    # 2. Process Minnows
    minnows_path = os.path.join(PROCESSED_DIR, "minnows.csv")
    if not os.path.exists(minnows_path):
        print("Minnows file not found")
        return

    stock_list = pd.read_csv(minnows_path)
    symbols = stock_list['symbol'].tolist()
    
    print(f"Processing Regimes for {len(symbols)} Minnows...")
    
    acc_calm = {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS}
    acc_stress = {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS}
    
    max_workers = 8
    
    # Helper to unpack - CANNOT be local function for ProcessPool on Windows
    # use a partial or simple list comprehension with submit
    
    # We can't pickle `run_worker` if it's local.
    # Solution: Make process_stock_regime_full valid for submit, 
    # but it takes 3 args. executor.submit(func, arg1, arg2...) works fine.
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Pass args directly: symbol, calm_days, stress_days
        futures = {executor.submit(process_stock_regime_full, sym, calm_days, stress_days): sym for sym in symbols}
        
        count = 0 
        total = len(symbols)
        
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{total}...")
                
            if res:
                # Merge Calm
                for lag, data in res['calm'].items():
                    acc_calm[lag]['z_push'].append(data['z_push'])
                    acc_calm[lag]['z_resp'].append(data['z_resp'])
                # Merge Stress
                for lag, data in res['stress'].items():
                    acc_stress[lag]['z_push'].append(data['z_push'])
                    acc_stress[lag]['z_resp'].append(data['z_resp'])

    # 3. Aggregate and Save
    save_surface(acc_calm, "minnows_calm")
    save_surface(acc_stress, "minnows_stress")

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
            
    out_file = os.path.join(RESULTS_DIR, f"surface_{name}.csv")
    pd.DataFrame(surface_data).to_csv(out_file, index=False)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
