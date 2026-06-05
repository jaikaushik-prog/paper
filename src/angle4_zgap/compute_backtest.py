import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"

# Costs
COST_BPS = 0.0005 # 5 bps

def run_backtest(symbol):
    try:
        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(file_path): return None
        
        df = pd.read_csv(file_path, usecols=['date', 'close'])
        if df.empty: return None
        
        # 1. Signals
        df['ret'] = np.log(df['close'] / df['close'].shift(1))
        
        window = 1500
        push_mean = df['ret'].rolling(window).mean()
        push_std = df['ret'].rolling(window).std()
        df['z_push'] = (df['ret'] - push_mean) / push_std
        
        # 2. Trigger: Z < -4
        # Need signals to be in valid range (after rolling window)
        valid_idx = df['z_push'].dropna().index
        signals = df.loc[valid_idx]
        signals = signals[signals['z_push'] < -4.0]
        
        if signals.empty: return []
        
        trades = []
        # Need integer locations
        # reset index to be safe?
        df = df.reset_index(drop=True)
        # Re-locate signals in new index
        # Better: just use iloc on full df
        
        # Map original index to integer position
        # Actually df index is default RangeIndex after read_csv unless set.
        # But dropna might mess it up? No, I assigned columns.
        # Let's force reset.
        
        # Signals might be sparse. Iterate them.
        for idx in signals.index:
            if idx + 10 >= len(df): continue
            
            # Close at t (Signal)
            p_t = df.at[idx, 'close']
            p_t1 = df.at[idx+1, 'close']
            p_t10 = df.at[idx+10, 'close']
            
            # Scenario A (Instant)
            gross_0 = np.log(p_t10 / p_t)
            net_0 = gross_0 - COST_BPS
            
            # Scenario B (Delayed)
            gross_1 = np.log(p_t10 / p_t1)
            net_1 = gross_1 - COST_BPS
            
            trades.append({
                'date': df.at[idx, 'date'],
                'symbol': symbol,
                'z': df.at[idx, 'z_push'],
                'gross_0': gross_0,
                'net_0': net_0,
                'gross_1': gross_1,
                'net_1': net_1
            })
            
        return trades
        
    except Exception as e:
        return None

def main():
    print("Simulating Execution-Aware Backtest...")
    minnows_path = os.path.join(PROCESSED_DIR, "minnows.csv")
    if not os.path.exists(minnows_path): 
        print("No minnows file.")
        return
    
    minnows = pd.read_csv(minnows_path)['symbol'].tolist()
    available = set(os.path.basename(f).replace('.csv','') for f in glob.glob(os.path.join(DATA_DIR, "*.csv")))
    valid = [s for s in minnows if s in available]
    
    all_trades = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_backtest, sym): sym for sym in valid}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 20 == 0: print(f"Processed {count}...")
            
            if res:
                all_trades.extend(res)
                
    if not all_trades:
        print("No trades.")
        return

    trade_df = pd.DataFrame(all_trades)
    trade_df['date'] = pd.to_datetime(trade_df['date'], utc=True)
    trade_df.sort_values('date', inplace=True)
    
    print(f"\nTotal Trades: {len(trade_df)}")
    
    # Metrics
    m0 = trade_df['gross_0'].mean()
    n0 = trade_df['net_0'].mean()
    w0 = (trade_df['net_0'] > 0).mean()
    
    m1 = trade_df['gross_1'].mean()
    n1 = trade_df['net_1'].mean()
    w1 = (trade_df['net_1'] > 0).mean()
    
    print(f"\n--- SCENARIO A: INSTANT (Ideal) ---")
    print(f"Gross: {m0*100:.4f}% | Net: {n0*100:.4f}% | Win: {w0*100:.1f}%")
    
    print(f"\n--- SCENARIO B: DELAYED (Real) ---")
    print(f"Gross: {m1*100:.4f}% | Net: {n1*100:.4f}% | Win: {w1*100:.1f}%")

    trade_df.to_csv(os.path.join(RESULTS_DIR, "backtest_trades.csv"), index=False)

if __name__ == "__main__":
    main()
