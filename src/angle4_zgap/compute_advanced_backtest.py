import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"

SIZES = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]

def run_advanced_backtest(symbol):
    try:
        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(file_path): return None
        
        df = pd.read_csv(file_path, usecols=['date', 'close', 'volume'])
        if df.empty: return None
        df['date'] = pd.to_datetime(df['date'], utc=True)
        
        # 1. Calc ADT
        df['turnover'] = df['close'] * df['volume']
        daily = df.set_index('date')['turnover'].resample('D').sum()
        adt_series = daily.rolling(20).mean().shift(1)
        
        df['date_only'] = df['date'].dt.date
        date_map = adt_series.to_dict() # Map date to ADT
        # Fix index to date object
        date_map = {k.date(): v for k, v in date_map.items()}
        df['adt'] = df['date_only'].map(date_map)
        
        # 2. Signals
        df['ret'] = np.log(df['close'] / df['close'].shift(1))
        # Vol
        window = 1500
        push_mean = df['ret'].rolling(window).mean()
        push_std = df['ret'].rolling(window).std()
        df['z_push'] = (df['ret'] - push_mean) / push_std
        
        # Trigger
        valid_mask = (df['z_push'] < -4.0) & (df['adt'] > 0)
        signals = df[valid_mask].copy()
        
        if signals.empty: return []
        
        trades = []
        for idx in signals.index:
            if idx + 10 >= len(df): continue
            
            sig_time = df.at[idx, 'date']
            p_t = df.at[idx, 'close']
            p_t10 = df.at[idx+10, 'close']
            
            gross_ret = np.log(p_t10 / p_t)
            adt = df.at[idx, 'adt']
            hour = sig_time.hour
            
            trades.append({
                'date': sig_time,
                'symbol': symbol,
                'gross': gross_ret,
                'adt': adt,
                'hour': hour
            })
            
        return trades

    except Exception as e:
        return None

def main():
    print("Running Advanced Backtest (Capacity, ToD, Regime)...")
    minnows_path = os.path.join(PROCESSED_DIR, "minnows.csv")
    if not os.path.exists(minnows_path): return
    
    minnows = pd.read_csv(minnows_path)['symbol'].tolist()
    available = set(os.path.basename(f).replace('.csv','') for f in glob.glob(os.path.join(DATA_DIR, "*.csv")))
    valid = [s for s in minnows if s in available]
    
    all_trades = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_advanced_backtest, sym): sym for sym in valid}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 20 == 0: print(f"Processed {count}...")
            if res: all_trades.extend(res)
            
    if not all_trades:
        print("No trades.")
        return

    df = pd.DataFrame(all_trades)
    df.sort_values('date', inplace=True)
    
    # 1. Capacity
    # Cost = 5bps + 50bps * (Size / ADT)
    capacity_results = []
    
    for size in SIZES:
        rel_size = size / df['adt']
        impact = 0.0005 + 0.5 * rel_size
        net_ret = df['gross'] - impact
        
        mean_net = net_ret.mean()
        # Annualized Sharpe (assuming 252*75 trades per year? No, roughly 10 trades per day)
        # N=25000 / 10 years = 2500/yr.
        # Annual Ret = Mean * 2500. Annual Vol = Std * sqrt(2500).
        # Sharpe = (Mean * 2500) / (Std * 50) = 50 * Mean/Std.
        sharpe = (mean_net / net_ret.std()) * np.sqrt(2500)
        
        capacity_results.append({
            'size': size,
            'avg_net': mean_net,
            'sharpe': sharpe,
            'win_rate': (net_ret > 0).mean()
        })
        
    print("\n--- CAPACITY CURVE ---")
    print(pd.DataFrame(capacity_results).to_string(index=False))
    
    # 2. ToD
    tod_stats = df.groupby('hour')['gross'].agg(['count', 'mean'])
    print("\n--- TIME OF DAY BREAKDOWN (Gross) ---")
    print(tod_stats)
    
    # 3. Yearly
    df['year'] = df['date'].dt.year
    year_stats = df.groupby('year')['gross'].agg(['count', 'mean'])
    print("\n--- YEARLY BREAKDOWN ---")
    print(year_stats)
    
    df.to_csv(os.path.join(RESULTS_DIR, "advanced_backtest.csv"), index=False)
    pd.DataFrame(capacity_results).to_csv(os.path.join(RESULTS_DIR, "capacity_curve.csv"), index=False)

if __name__ == "__main__":
    main()
