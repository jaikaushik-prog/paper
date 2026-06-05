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
ALL_LAGS = list(range(1, 21)) + list(range(25, 80, 5))

# Thresholds based on analysis
# Minnow Avg: 6 Crores (6e7). Threshold < 10 Crores (1e8)
# Whale Avg: 700 Crores (7e9). Threshold > 100 Crores (1e9)
ADT_MINNOW_THRESH = 1.0e8 
ADT_WHALE_THRESH = 1.0e9

def process_stock_dynamic(file_path):
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values('date')
        
        # Calculate Daily ADT (approx)
        # We need daily ADT to determine regime.
        # But we have 5-min data.
        # Group by day -> sum(turnover) -> rolling mean
        df['turnover'] = df['close'] * df['volume']
        df['day'] = df['date'].dt.date
        
        daily_turnover = df.groupby('day')['turnover'].sum().reset_index()
        daily_turnover['rolling_adt'] = daily_turnover['turnover'].rolling(window=20).mean()
        
        # Map back to 5-min bars
        # Merge rolling_adt back to df
        df = df.merge(daily_turnover[['day', 'rolling_adt']], on='day', how='left')
        
        # Determine Regime per bar
        # 0: Noise/Mid
        # 1: Whale
        # -1: Minnow
        
        # Vectorized check
        # Use simple boolean masks later
        
        # Prepare P-R
        # Optimization: Only process if there is sufficient data
        valid_rows = df.dropna(subset=['rolling_adt'])
        if valid_rows.empty: return None
        
        # Calculate P-R (Costly part, do only if needed)
        # We can pass the whole df to prepare_push_response, then filter by regime
        
        results = {'whales': {}, 'minnows': {}}
        
        # Pre-filter to save time? No, need contiguous for lags.
        
        # Run calculation
        # To save compute, let's limit to critical lags only? 
        # No, full surface needs full lags.
        
        # Let's run for ALL_LAGS but optimize accumulation
        
        for lag in ALL_LAGS:
            res = prepare_push_response_data(df, lag)
            if res.empty: continue
            
            # Align Regime
            # res has index from df
            # Join regime
            res['rolling_adt'] = df.loc[res.index, 'rolling_adt']
            
            # Filter Dynamic Whales
            dyn_whales = res[res['rolling_adt'] > ADT_WHALE_THRESH]
            if not dyn_whales.empty:
                results['whales'][lag] = {
                    'z_push': dyn_whales['z_push'].values,
                    'z_resp': dyn_whales['z_resp'].values
                }
                
            # Filter Dynamic Minnows
            dyn_minnows = res[res['rolling_adt'] < ADT_MINNOW_THRESH]
            if not dyn_minnows.empty:
                results['minnows'][lag] = {
                    'z_push': dyn_minnows['z_push'].values,
                    'z_resp': dyn_minnows['z_resp'].values
                }
                
        return results
        
    except Exception as e:
        return None

def process_dynamic_liquidity():
    print("Starting Expr 1: Dynamic Liquidity Stratification...")
    
    # Process ALL 500 stocks, not just pre-selected lists
    stock_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    accumulator = {
        'whales': {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS},
        'minnows': {lag: {'z_push': [], 'z_resp': []} for lag in ALL_LAGS}
    }
    
    print(f"Processing {len(stock_files)} stocks dynamically...")
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_stock_dynamic, fp): fp for fp in stock_files}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 20 == 0:
                print(f"Processed {count}/{len(stock_files)}...")
                
            if res:
                for grp in ['whales', 'minnows']:
                    for lag, data in res[grp].items():
                        accumulator[grp][lag]['z_push'].append(data['z_push'])
                        accumulator[grp][lag]['z_resp'].append(data['z_resp'])
                        
    # Aggregate and Save
    for grp in ['whales', 'minnows']:
        print(f"Aggregating Dynamic {grp.title()}...")
        surface_data = []
        bins = np.arange(-4.0, 4.05, 0.1) 
        centers = (bins[:-1] + bins[1:]) / 2
        
        for lag in ALL_LAGS:
            pushes = accumulator[grp][lag]['z_push']
            responses = accumulator[grp][lag]['z_resp']
            
            if not pushes: continue
            
            cat_push = np.concatenate(pushes)
            cat_resp = np.concatenate(responses)
            
            df_lag = pd.DataFrame({'z_push': cat_push, 'z_resp': cat_resp})
            df_lag['bin'] = pd.cut(df_lag['z_push'], bins=bins, labels=centers)
            
            # Key Change: Get Mean, SEM, and Count for Confidence Intervals
            agg_stats = df_lag.groupby('bin', observed=True)['z_resp'].agg(['mean', 'sem', 'count'])
            
            for center in agg_stats.index:
                val_mean = agg_stats.loc[center, 'mean']
                val_sem = agg_stats.loc[center, 'sem']
                val_count = agg_stats.loc[center, 'count']
                
                if pd.isna(val_mean): continue
                
                surface_data.append({
                    'lag': int(lag),
                    'push_bin': float(center),
                    'avg_response': float(val_mean),
                    'sem_response': float(val_sem) if not pd.isna(val_sem) else 0.0,
                    'count': int(val_count)
                })
        
        out_file = os.path.join(RESULTS_DIR, f"surface_dynamic_{grp}.csv")
        pd.DataFrame(surface_data).to_csv(out_file, index=False)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    process_dynamic_liquidity()
